#!/usr/bin/env bash
# Lighter pilot: 3 arms x 3 label budgets x 1 seed, CPU only. Run from this directory.
set -euo pipefail
mkdir -p logs results checkpoints
[ -f data/water_test.npz ] || python3 gen_data.py  # ~30 min with 3 parallel procs (see README)

PRE_STEPS=${PRE_STEPS:-2000}
FT_STEPS=${FT_STEPS:-1500}
python3 train.py pretrain --method jepa    --steps $PRE_STEPS --threads 2 > logs/pre_jepa.log 2>&1 &
python3 train.py pretrain --method denoise --steps $PRE_STEPS --threads 2 > logs/pre_denoise.log 2>&1 &
wait

for b in 16 64 256; do
  for arm in scratch jepa denoise; do
    python3 train.py finetune --init $arm --budget $b --steps $FT_STEPS --threads 1 > logs/ft_${arm}_b$b.log 2>&1 &
  done
  wait
done

python3 analyze.py jac --threads 4 > logs/jac.log 2>&1
for s in si water ethanol; do python3 analyze.py md --system $s --budget 256 > logs/md_$s.log 2>&1 & done
wait
python3 analyze.py timing --threads 4 > logs/timing.log 2>&1
python3 report.py
