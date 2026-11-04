"""Pretraining (JEPA with EMA target, or coordinate denoising) and supervised fine-tuning
of a direct-force PaiNN student.

  python train.py pretrain --method jepa
  python train.py pretrain --method denoise
  python train.py finetune --init {scratch,jepa,denoise} --budget 64
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from common import (CKPT_DIR, DATA_DIR, MASK_ID, RES_DIR, SPECIES, DirectMLIP, Frames, GatedVector,
                    Message, PaiNN, Update, clone_frozen, collate, ema_update)

SYSTEMS = ["si", "water", "ethanol"]
N_VAL = 32


def load(split):
    return {s: Frames(f"{DATA_DIR}/{s}_{split}.npz") for s in SYSTEMS}


def splits(pool, budget, seed=0):
    """Labelled train / val indices per system (val is disjoint and fixed)."""
    out = {}
    for s, fr in pool.items():
        perm = np.random.default_rng(1000 + SYSTEMS.index(s)).permutation(len(fr))
        val = perm[:N_VAL]
        train = perm[N_VAL:N_VAL + budget]
        out[s] = (train, val)
    return out


def sampler(pool, idx_per_sys, bs, rng):
    keys = [(s, int(k)) for s in SYSTEMS for k in idx_per_sys[s]]
    while True:
        for b in np.array_split(rng.permutation(len(keys)), max(1, len(keys) // bs)):
            yield collate([(pool[keys[i][0]], keys[i][1]) for i in b])


def cosine_lr(opt, step, total, base, warm=100):
    lr = base * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
    for g in opt.param_groups:
        g["lr"] = lr


# ----------------------------------------------------------------------------- JEPA
def block_mask(g, rng, frac=(0.15, 0.4), r=2.6):
    """Spatial block masking: union of radius-r neighbourhoods of random centres, per graph,
    added until >= frac[0] of the atoms are masked (capped at frac[1])."""
    N = len(g["z"])
    i, j = g["edge_index"]
    close = (g["pos"][j] - g["pos"][i] + g["shift"]).norm(dim=-1) < r
    i, j = i[close], j[close]
    natoms = g["natoms"]
    offs = torch.cumsum(natoms, 0) - natoms
    mask = torch.zeros(N, dtype=torch.bool)
    for _ in range(50):
        cnt = torch.zeros(len(natoms)).index_add_(0, g["batch"], mask.float())
        need = cnt < frac[0] * natoms
        if not need.any():
            break
        centres = offs[need] + torch.as_tensor(rng.integers(0, natoms[need].numpy()))
        sel = torch.zeros(N, dtype=torch.bool)
        sel[centres] = True
        mask |= sel
        mask[j[sel[i]]] = True
    for b, n in enumerate(natoms.tolist()):  # cap
        idx = torch.nonzero(mask & (g["batch"] == b)).flatten()
        cap = max(1, int(frac[1] * n))
        if len(idx) > cap:
            mask[idx[torch.as_tensor(rng.permutation(len(idx))[cap:])]] = False
    return mask


class Predictor(nn.Module):
    """Small PaiNN stack on the context graph that maps context embeddings -> target embeddings."""

    def __init__(self, F_=64, layers=2, nrbf=20):
        super().__init__()
        self.msg = nn.ModuleList(Message(F_, nrbf) for _ in range(layers))
        self.upd = nn.ModuleList(Update(F_) for _ in range(layers))
        self.out_s = nn.Linear(F_, F_)
        self.out_v = nn.Linear(F_, F_, bias=False)

    def forward(self, s, v, g, enc):
        rbf, unit = enc.geometry(g)
        for m, u in zip(self.msg, self.upd):
            s, v = m(s, v, g["edge_index"], rbf, unit)
            s, v = u(s, v)
        return self.out_s(s), self.out_v(v.transpose(1, 2)).transpose(1, 2)


def norm_targets(s, v):
    s = Fn.layer_norm(s, s.shape[-1:])
    v = v / torch.sqrt((v ** 2).sum(-1).mean(-1, keepdim=True) + 1e-6).unsqueeze(-1)
    return s, v


def pretrain(args):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pool = load("pool")
    allidx = {s: np.arange(len(fr)) for s, fr in pool.items()}
    batches = sampler(pool, allidx, args.bs, rng)
    enc = PaiNN(args.F, args.layers)
    if args.method == "jepa":
        tgt = clone_frozen(enc)
        head = Predictor(args.F)
    else:
        head = GatedVector(args.F, 1)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4 if args.method == "denoise" else 0.04)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        g = next(batches)
        cosine_lr(opt, step, args.steps, args.lr)
        if args.method == "jepa":
            mask = block_mask(g, rng)
            zc = g["z"].clone()
            zc[mask] = MASK_ID
            s, v = enc(g, z=zc)
            ps, pv = head(s, v, g, enc)
            with torch.no_grad():
                ts, tv = tgt(g)
                raw_std = ts[mask].std(0).mean().item()
                ts, tv = norm_targets(ts, tv)
            ls = Fn.mse_loss(ps[mask], ts[mask])
            lv = Fn.mse_loss(pv[mask], tv[mask])
            loss = ls + lv
            extra = dict(ls=ls.item(), lv=lv.item(), tgt_std=raw_std, mfrac=mask.float().mean().item())
        else:
            eps = torch.randn_like(g["pos"])
            s, v = enc(g, pos=g["pos"] + args.sigma * eps)
            pred = head(s, v)[:, 0]
            loss = Fn.mse_loss(pred, eps)
            extra = {}
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if args.method == "jepa":
            m = 1 - (1 - args.ema) * (math.cos(math.pi * step / args.steps) + 1) / 2
            ema_update(tgt, enc, m)
        if step % 50 == 0 or step == args.steps - 1:
            rec = dict(step=step, loss=loss.item(), t=time.time() - t0, **extra)
            log.append(rec)
            print(json.dumps(rec), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    final = tgt if args.method == "jepa" else enc
    torch.save(final.state_dict(), f"{CKPT_DIR}/pre_{args.method}.pt")
    os.makedirs(RES_DIR, exist_ok=True)
    json.dump(dict(args=vars(args), log=log), open(f"{RES_DIR}/pretrain_{args.method}.json", "w"), indent=1)


# ----------------------------------------------------------------------------- fine-tune
def fit_e0(pool, idx):
    rows, ys = [], []
    for s, (tr, _) in idx.items():
        fr = pool[s]
        counts = np.zeros(len(SPECIES) + 1)
        for zz in fr.Z:
            counts[SPECIES[zz]] += 1
        for k in tr:
            rows.append(counts)
            ys.append(fr.E[k])
    e0, *_ = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)
    return torch.tensor(e0)


@torch.no_grad()
def evaluate(model, frames, idx_per_sys, bs=16):
    out = {}
    for s in SYSTEMS:
        fr, idx = frames[s], idx_per_sys[s]
        fe, ee, fn = [], [], []
        for b in np.array_split(idx, max(1, len(idx) // bs)):
            g = collate([(fr, int(k)) for k in b])
            E, Fp = model(g)
            fe.append((Fp - g["F"]).numpy())
            fn.append(g["F"].numpy())
            ee.append(((E - g["E"]) / g["natoms"]).numpy())
        fe, ee, fn = np.concatenate(fe), np.concatenate(ee), np.concatenate(fn)
        out[s] = dict(F_mae=float(np.abs(fe).mean()), F_rmse=float(np.sqrt((fe ** 2).mean())),
                      F_rel=float(np.sqrt((fe ** 2).sum() / (fn ** 2).sum())),
                      E_mae_meV_atom=float(1000 * np.abs(ee - ee.mean()).mean()))
    out["mean_F_mae"] = float(np.mean([out[s]["F_mae"] for s in SYSTEMS]))
    return out


def finetune(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pool = load("pool")
    idx = splits(pool, args.budget)
    model = DirectMLIP(args.F, args.layers)
    if args.init != "scratch":
        sd = torch.load(f"{CKPT_DIR}/pre_{args.init}.pt")
        model.enc.load_state_dict(sd)
    model.e0.copy_(fit_e0(pool, idx))
    fr = np.concatenate([pool[s].F[idx[s][0]].ravel() for s in SYSTEMS])
    model.fscale.fill_(float(np.sqrt((fr ** 2).mean())))
    model.escale.fill_(float(model.fscale))
    batches = sampler(pool, {s: idx[s][0] for s in SYSTEMS}, args.bs, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best, best_sd, log = 1e9, None, []
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        g = next(batches)
        cosine_lr(opt, step, args.steps, args.lr)
        E, Fp = model(g)
        lf = Fn.mse_loss(Fp / model.fscale, g["F"] / model.fscale)
        le = (((E - g["E"]) / g["natoms"] / model.escale) ** 2).mean().float()
        loss = lf + args.wE * le
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            val = evaluate(model, pool, {s: idx[s][1] for s in SYSTEMS})
            rec = dict(step=step, lf=lf.item(), le=le.item(), val_F_mae=val["mean_F_mae"], t=time.time() - t0)
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if val["mean_F_mae"] < best:
                best, best_sd = val["mean_F_mae"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    tag = f"{args.init}_b{args.budget}_s{args.seed}"
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{CKPT_DIR}/ft_{tag}.pt")
    test = load("test")
    res = evaluate(model, test, {s: np.arange(len(test[s])) for s in SYSTEMS})
    json.dump(dict(args=vars(args), best_val=best, test=res, log=log),
              open(f"{RES_DIR}/ft_{tag}.json", "w"), indent=1)
    print("TEST", json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pretrain", "finetune"])
    ap.add_argument("--method", choices=["jepa", "denoise"], default="jepa")
    ap.add_argument("--init", choices=["scratch", "jepa", "denoise"], default="scratch")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--F", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--wE", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(RES_DIR, exist_ok=True)
    pretrain(args) if args.mode == "pretrain" else finetune(args)
