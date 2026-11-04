"""Shared pieces for the JEPA direct-force MLIP pilot.

* PaiNN encoder (Schutt et al. 2021) in plain PyTorch, with a learned [MASK] species token.
* Direct heads: per-atom force vector (gated equivariant block) and per-atom energy (MLP).
* Graph batching with precomputed neighbour lists (cutoff + skin so small position
  perturbations such as denoising noise can reuse the same edge list).
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.neighborlist import neighbor_list

SPECIES = {1: 0, 6: 1, 8: 2, 14: 3}  # H, C, O, Si
MASK_ID = len(SPECIES)
CUTOFF = 5.0
SKIN = 0.5
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
RES_DIR = "results"


# ----------------------------------------------------------------------------- data
def build_edges(pos, cell, pbc, cutoff=CUTOFF + SKIN):
    from ase import Atoms

    a = Atoms(positions=pos, cell=cell, pbc=pbc, numbers=np.ones(len(pos), int))
    i, j, S = neighbor_list("ijS", a, cutoff)
    return np.stack([i, j]).astype(np.int64), (S @ np.asarray(cell)).astype(np.float32)


class Frames:
    """All frames of one system kind, stored as arrays; labels optional."""

    def __init__(self, npz):
        d = np.load(npz)
        self.Z = d["Z"]
        self.pos = d["pos"].astype(np.float32)
        self.cell = d["cell"]
        self.pbc = d["pbc"]
        self.E = d["E"]
        self.F = d["F"].astype(np.float32)
        self.name = str(d["name"])
        self._edges = [None] * len(self.pos)
        self._load_edge_cache(npz)

    def _load_edge_cache(self, npz):
        """Neighbour lists are ~0.1 s/frame with ASE; build once and cache next to the data."""
        import os

        cache = npz.replace(".npz", ".edges.npz")
        if not os.path.exists(cache):
            es = [self.edges(k) for k in range(len(self))]
            n = np.array([e.shape[1] for e, _ in es])
            np.savez(cache, n=n, e=np.concatenate([e for e, _ in es], 1), s=np.concatenate([s for _, s in es]))
        d = np.load(cache)
        off = np.concatenate([[0], np.cumsum(d["n"])])
        e, s = d["e"], d["s"]
        self._edges = [(e[:, off[k]:off[k + 1]], s[off[k]:off[k + 1]]) for k in range(len(self))]

    def __len__(self):
        return len(self.pos)

    def edges(self, k):
        if self._edges[k] is None:
            self._edges[k] = build_edges(self.pos[k], self.cell, self.pbc)
        return self._edges[k]


def collate(items):
    """items: list of (Frames, idx). Returns dict of tensors for a batched graph."""
    Z, pos, E, Fo, ei, sh, batch, natoms = [], [], [], [], [], [], [], []
    off = 0
    for b, (fr, k) in enumerate(items):
        n = len(fr.Z)
        e, s = fr.edges(k)
        Z.append(np.vectorize(SPECIES.get)(fr.Z))
        pos.append(fr.pos[k])
        E.append(fr.E[k])
        Fo.append(fr.F[k])
        ei.append(e + off)
        sh.append(s)
        batch.append(np.full(n, b))
        natoms.append(n)
        off += n
    t = lambda x, dt=torch.float32: torch.as_tensor(np.concatenate(x) if isinstance(x, list) else x, dtype=dt)
    return dict(
        z=t(Z, torch.long), pos=t(pos), E=torch.tensor(E, dtype=torch.float64), F=t(Fo),
        edge_index=torch.as_tensor(np.concatenate(ei, 1)),
        shift=t(sh), batch=t(batch, torch.long), natoms=torch.tensor(natoms), nb=len(items),
    )


def single_graph(Z, pos, cell, pbc):
    e, s = build_edges(pos, cell, pbc)
    return dict(
        z=torch.as_tensor(np.vectorize(SPECIES.get)(Z), dtype=torch.long),
        pos=torch.as_tensor(pos, dtype=torch.float32), edge_index=torch.as_tensor(e),
        shift=torch.as_tensor(s), batch=torch.zeros(len(Z), dtype=torch.long),
        natoms=torch.tensor([len(Z)]), nb=1,
    )


# ----------------------------------------------------------------------------- model
class RBF(nn.Module):
    def __init__(self, n=20, cutoff=CUTOFF):
        super().__init__()
        self.register_buffer("freq", torch.arange(1, n + 1) * math.pi / cutoff)
        self.cutoff = cutoff

    def forward(self, d):
        d = d.unsqueeze(-1)
        rbf = torch.sin(self.freq * d) / d
        fc = 0.5 * (torch.cos(math.pi * d / self.cutoff) + 1.0) * (d < self.cutoff)
        return rbf * fc


class Message(nn.Module):
    def __init__(self, F_, nrbf):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(F_, F_), nn.SiLU(), nn.Linear(F_, 3 * F_))
        self.W = nn.Linear(nrbf, 3 * F_)
        self.F = F_

    def forward(self, s, v, ei, rbf, unit):
        i, j = ei
        x = self.phi(s)[j] * self.W(rbf)
        xs, xvv, xvs = torch.split(x, self.F, -1)
        dv = xvv.unsqueeze(-1) * v[j] + xvs.unsqueeze(-1) * unit.unsqueeze(1)
        s = s + torch.zeros_like(s).index_add_(0, i, xs)
        v = v + torch.zeros_like(v).index_add_(0, i, dv)
        return s, v


class Update(nn.Module):
    def __init__(self, F_):
        super().__init__()
        self.U = nn.Linear(F_, F_, bias=False)
        self.V = nn.Linear(F_, F_, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * F_, F_), nn.SiLU(), nn.Linear(F_, 3 * F_))
        self.F = F_

    def forward(self, s, v):
        Uv = self.U(v.transpose(1, 2)).transpose(1, 2)
        Vv = self.V(v.transpose(1, 2)).transpose(1, 2)
        Vn = torch.sqrt((Vv ** 2).sum(-1) + 1e-8)
        a = self.mlp(torch.cat([s, Vn], -1))
        avv, asv, ass = torch.split(a, self.F, -1)
        v = v + avv.unsqueeze(-1) * Uv
        s = s + asv * (Uv * Vv).sum(-1) + ass
        return s, v


class PaiNN(nn.Module):
    def __init__(self, F_=64, layers=3, nrbf=20, cutoff=CUTOFF):
        super().__init__()
        self.emb = nn.Embedding(len(SPECIES) + 1, F_)  # last row = [MASK]
        self.rbf = RBF(nrbf, cutoff)
        self.msg = nn.ModuleList(Message(F_, nrbf) for _ in range(layers))
        self.upd = nn.ModuleList(Update(F_) for _ in range(layers))
        self.F = F_

    def geometry(self, g, pos=None):
        pos = g["pos"] if pos is None else pos
        i, j = g["edge_index"]
        r = pos[j] - pos[i] + g["shift"]
        d = r.norm(dim=-1)
        return self.rbf(d), r / d.unsqueeze(-1)

    def forward(self, g, z=None, pos=None):
        z = g["z"] if z is None else z
        rbf, unit = self.geometry(g, pos)
        s = self.emb(z)
        v = torch.zeros(s.shape[0], self.F, 3, dtype=s.dtype)
        for m, u in zip(self.msg, self.upd):
            s, v = m(s, v, g["edge_index"], rbf, unit)
            s, v = u(s, v)
        return s, v


class GatedVector(nn.Module):
    """Gated equivariant block -> `out` vector channels per atom."""

    def __init__(self, F_, out=1):
        super().__init__()
        self.W1 = nn.Linear(F_, F_, bias=False)
        self.W2 = nn.Linear(F_, out, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * F_, F_), nn.SiLU(), nn.Linear(F_, out))

    def forward(self, s, v):
        v1 = self.W1(v.transpose(1, 2)).transpose(1, 2)
        n = torch.sqrt((v1 ** 2).sum(-1) + 1e-8)
        gate = self.mlp(torch.cat([s, n], -1))
        return gate.unsqueeze(-1) * self.W2(v.transpose(1, 2)).transpose(1, 2)  # (N,out,3)


class DirectMLIP(nn.Module):
    """Encoder + direct force head + per-atom energy head (no autograd needed for forces)."""

    def __init__(self, F_=64, layers=3):
        super().__init__()
        self.enc = PaiNN(F_, layers)
        self.force = GatedVector(F_, 1)
        self.energy = nn.Sequential(nn.Linear(F_, F_), nn.SiLU(), nn.Linear(F_, 1))
        self.register_buffer("e0", torch.zeros(len(SPECIES) + 1, dtype=torch.float64))
        self.register_buffer("escale", torch.ones((), dtype=torch.float64))
        self.register_buffer("fscale", torch.ones(()))

    def forward(self, g, pos=None):
        s, v = self.enc(g, pos=pos)
        Fpred = self.force(s, v)[:, 0] * self.fscale
        ea = self.energy(s)[:, 0].double() * self.escale + self.e0[g["z"]]
        E = torch.zeros(g["nb"], dtype=torch.float64).index_add_(0, g["batch"], ea)
        return E, Fpred


class ConservativeHead(nn.Module):
    """Same PaiNN, but forces = -dE/dR (used only for the timing comparison)."""

    def __init__(self, F_=64, layers=3):
        super().__init__()
        self.enc = PaiNN(F_, layers)
        self.energy = nn.Sequential(nn.Linear(F_, F_), nn.SiLU(), nn.Linear(F_, 1))

    def forward(self, g):
        pos = g["pos"].clone().requires_grad_(True)
        s, _ = self.enc(g, pos=pos)
        E = self.energy(s).sum()
        return E, -torch.autograd.grad(E, pos)[0]


def ema_update(target, online, m):
    with torch.no_grad():
        for pt, po in zip(target.parameters(), online.parameters()):
            pt.mul_(m).add_(po.detach(), alpha=1 - m)


def clone_frozen(m):
    t = copy.deepcopy(m)
    for p in t.parameters():
        p.requires_grad_(False)
    return t
