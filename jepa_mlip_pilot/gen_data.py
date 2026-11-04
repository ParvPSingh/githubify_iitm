"""Generate teacher-labelled MD data with MACE-MP-0 (small) for three systems.

pool : Langevin MD over a temperature ladder -> used unlabelled for pretraining, and
       small labelled subsets are drawn from it for fine-tuning.
test : an independent trajectory (different seed / temperatures) for evaluation.
"""
import argparse
import os
import time
import warnings

import numpy as np
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from common import DATA_DIR

warnings.filterwarnings("ignore")


def teacher(dtype="float32"):
    from mace.calculators import mace_mp

    return mace_mp(model="small", device="cpu", default_dtype=dtype)


def make_system(name, rng):
    if name == "si":
        return bulk("Si", "diamond", 5.43, cubic=True) * (2, 2, 2)  # 64 atoms
    if name == "water":
        n, L = 3, (27 * 29.91) ** (1 / 3)  # 27 H2O at ~1 g/cm3
        w = molecule("H2O")
        w.positions -= w.get_center_of_mass()
        atoms = Atoms(cell=[L] * 3, pbc=True)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    m = w.copy()
                    m.rotate(rng.uniform(0, 360), rng.normal(size=3))
                    m.positions += (np.array([a, b, c]) + 0.5) * L / n
                    atoms += m
        return atoms
    if name == "ethanol":
        a = molecule("CH3CH2OH")
        a.pbc = False
        return a
    raise ValueError(name)


CFG = {  # temperatures (K), dt (fs), sample stride, frames per temperature
    "si": dict(pool_T=[300, 600, 900, 1200], test_T=[450, 1000], dt=1.0, stride=4),
    "water": dict(pool_T=[250, 300, 350, 400], test_T=[320, 380], dt=0.5, stride=4),
    "ethanol": dict(pool_T=[200, 400, 600, 800], test_T=[300, 700], dt=0.5, stride=10),
}


def run(name, temps, n_per_T, seed, calc, equil=200):
    rng = np.random.default_rng(seed)
    atoms = make_system(name, rng)
    atoms.calc = calc
    c = CFG[name]
    MaxwellBoltzmannDistribution(atoms, temperature_K=temps[0], rng=rng)
    Stationary(atoms)
    pos, E, Fo = [], [], []
    for T in temps:
        dyn = Langevin(atoms, c["dt"] * units.fs, temperature_K=T, friction=0.02, rng=rng)
        dyn.run(equil)
        for _ in range(n_per_T):
            dyn.run(c["stride"])
            pos.append(atoms.get_positions())
            E.append(atoms.get_potential_energy())
            Fo.append(atoms.get_forces())
    return dict(Z=atoms.numbers, pos=np.array(pos), E=np.array(E), F=np.array(Fo),
                cell=np.array(atoms.cell), pbc=atoms.pbc, name=name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=800)
    ap.add_argument("--test", type=int, default=160)
    ap.add_argument("--systems", default="si,water,ethanol")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(DATA_DIR, exist_ok=True)
    calc = teacher()
    for name in args.systems.split(","):
        c = CFG[name]
        for split, temps, n, seed in [("pool", c["pool_T"], args.pool, 0), ("test", c["test_T"], args.test, 1)]:
            t0 = time.time()
            d = run(name, temps, n // len(temps), seed, calc)
            np.savez(f"{DATA_DIR}/{name}_{split}.npz", **d)
            print(f"{name:8s} {split:5s} {len(d['pos'])} frames, {len(d['Z'])} atoms, "
                  f"|F|rms={np.sqrt((d['F']**2).mean()):.3f}  {time.time()-t0:.0f}s", flush=True)
