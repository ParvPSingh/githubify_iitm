"""Physics-side evaluation of fine-tuned direct-force students.

  python analyze.py jac      # force-Jacobian asymmetry (0 = conservative) on test frames
  python analyze.py timing   # direct head vs. -dE/dR (autograd) cost for the same PaiNN
  python analyze.py md --system si --budget 256   # NVE drift + MTS stride sweep (teacher = conservative corrector)
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import torch
from ase import units
from ase.data import atomic_masses

from common import CKPT_DIR, RES_DIR, ConservativeHead, DirectMLIP, Frames, single_graph
from train import SYSTEMS, load

warnings.filterwarnings("ignore")
ARMS = ["scratch", "jepa", "denoise"]


def load_model(tag):
    m = DirectMLIP()
    m.load_state_dict(torch.load(f"{CKPT_DIR}/ft_{tag}.pt"))
    return m.eval()


# ----------------------------------------------------------------------------- conservativeness
def jac_asym(model, fr, k):
    g = single_graph(fr.Z, fr.pos[k], fr.cell, fr.pbc)
    pos = g["pos"].clone().requires_grad_(True)
    f = lambda p: model(g, pos=p)[1].reshape(-1)
    J = torch.autograd.functional.jacobian(f, pos, vectorize=True).reshape(pos.numel(), -1)
    A = 0.5 * (J - J.T)
    return float(A.norm() / J.norm())


def cmd_jac(args):
    test = load("test")
    out = {}
    for path in sorted(glob.glob(f"{CKPT_DIR}/ft_*.pt")):
        tag = os.path.basename(path)[3:-3]
        m = load_model(tag)
        out[tag] = {s: float(np.mean([jac_asym(m, test[s], k) for k in np.linspace(0, len(test[s]) - 1, args.n, dtype=int)]))
                    for s in SYSTEMS}
        out[tag]["mean"] = float(np.mean([out[tag][s] for s in SYSTEMS]))
        print(tag, json.dumps(out[tag]), flush=True)
    json.dump(out, open(f"{RES_DIR}/jacobian_asymmetry.json", "w"), indent=1)


# ----------------------------------------------------------------------------- timing
def cmd_timing(args):
    from ase.build import bulk

    torch.manual_seed(0)
    res = {}
    for reps in [(2, 2, 2), (3, 3, 3), (4, 4, 4)]:
        a = bulk("Si", "diamond", 5.43, cubic=True) * reps
        a.rattle(0.05, seed=0)
        g = single_graph(a.numbers, a.positions, np.array(a.cell), a.pbc)
        d, c = DirectMLIP().eval(), ConservativeHead().eval()
        times = {}
        for name, fn in [("direct", lambda: d(g)), ("conservative", lambda: c(g))]:
            for _ in range(3):
                fn()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                fn()
            times[name] = (time.perf_counter() - t0) / args.reps * 1000
        times["speedup"] = times["conservative"] / times["direct"]
        res[len(a)] = times
        print(len(a), json.dumps(times), flush=True)
    json.dump(res, open(f"{RES_DIR}/timing.json", "w"), indent=1)


# ----------------------------------------------------------------------------- MD
class Student:
    def __init__(self, model, fr):
        self.m, self.fr = model, fr

    @torch.no_grad()
    def forces(self, pos):
        g = single_graph(self.fr.Z, pos, self.fr.cell, self.fr.pbc)
        return self.m(g)[1].double().numpy()


class Teacher:
    def __init__(self, fr, calc):
        from ase import Atoms

        self.atoms = Atoms(numbers=fr.Z, positions=fr.pos[0], cell=fr.cell, pbc=fr.pbc)
        self.atoms.calc = calc

    def ef(self, pos):
        self.atoms.positions = pos
        return self.atoms.get_potential_energy(), self.atoms.get_forces()


MD_CFG = {"si": dict(T=600, dt=1.0), "water": dict(T=300, dt=0.5), "ethanol": dict(T=300, dt=0.5)}


def run_md(fr, teacher, student, stride, nsteps, seed=0):
    """Impulse MTS / r-RESPA (Bigi et al. 2025 style).

    stride = 0 : pure direct-force NVE (teacher only used to *measure* energy every 20 steps)
    stride = k : inner velocity-Verlet with student forces; every k steps a kick with
                 (F_teacher - F_student) * k*dt/2 at both ends of the outer step.
    stride = -1: pure teacher (conservative reference)
    Energy = teacher potential + kinetic (the 'true' energy that should be conserved).
    """
    c = MD_CFG[fr.name]
    dt = c["dt"] * units.fs
    m = atomic_masses[fr.Z][:, None]
    rng = np.random.default_rng(seed)
    x = fr.pos[0].astype(np.float64).copy()
    v = rng.normal(size=x.shape) * np.sqrt(units.kB * c["T"] / m)
    v -= (m * v).sum(0) / m.sum()
    n = len(x)
    ke = lambda v: 0.5 * (m * v ** 2).sum()
    mon = 20 if stride <= 0 else stride * max(1, 20 // stride)  # energy sampling interval
    tcalls, last = 0, {}

    def tf(x):  # teacher energy + forces
        nonlocal tcalls
        tcalls += 1
        last["E"], last["F"] = teacher.ef(x)
        return last["F"]

    fd = tf if stride == -1 else student.forces
    E, Tlog = [], []
    f = fd(x)
    if stride > 0:
        ft = tf(x)
        v += 0.5 * stride * dt * (ft - f) / m
    for step in range(1, nsteps + 1):
        v += 0.5 * dt * f / m
        x += dt * v
        f = fd(x)
        v += 0.5 * dt * f / m
        synced = True
        if stride > 0:
            synced = step % stride == 0
            if synced:
                ft = tf(x)
                v += 0.5 * stride * dt * (ft - f) / m  # closing kick
        if synced and step % mon == 0:
            if stride == 0:
                tf(x)  # measurement only
            E.append((step, (last["E"] + ke(v)) / n))
            Tlog.append(2 * ke(v) / (3 * n * units.kB))
        if stride > 0 and synced:
            v += 0.5 * stride * dt * (ft - f) / m  # opening kick of next outer step
        T = 2 * ke(v) / (3 * n * units.kB)
        if not np.isfinite(T) or T > 6 * c["T"]:
            return dict(failed=True, fail_step=step, E=E, T=Tlog, teacher_calls=tcalls)
    E = np.array(E)
    t_ps = E[:, 0] * c["dt"] / 1000
    slope = np.polyfit(t_ps, E[:, 1], 1)[0] * 1000  # meV/atom/ps
    return dict(failed=False, drift_meV_atom_ps=float(slope),
                max_dev_meV_atom=float(1000 * np.abs(E[:, 1] - E[0, 1]).max()),
                E=E.tolist(), T=Tlog, teacher_calls=tcalls)


def cmd_md(args):
    from gen_data import teacher as make_teacher

    torch.set_num_threads(args.threads)
    test = load("test")
    fr = test[args.system]
    T = Teacher(fr, make_teacher("float32"))
    out = {"reference_teacher": run_md(fr, T, None, -1, args.steps)}
    print("teacher", {k: v for k, v in out["reference_teacher"].items() if k not in ("E", "T")}, flush=True)
    for arm in ARMS:
        S = Student(load_model(f"{arm}_b{args.budget}_s0"), fr)
        for stride in [0] + args.strides:
            r = run_md(fr, T, S, stride, args.steps)
            out[f"{arm}_k{stride}"] = r
            print(arm, stride, {k: v for k, v in r.items() if k not in ("E", "T")}, flush=True)
    json.dump(out, open(f"{RES_DIR}/md_{args.system}_b{args.budget}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["jac", "timing", "md"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--system", default="si")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--strides", type=lambda s: [int(x) for x in s.split(",")], default=[2, 4, 8, 16])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(RES_DIR, exist_ok=True)
    {"jac": cmd_jac, "timing": cmd_timing, "md": cmd_md}[args.cmd](args)
