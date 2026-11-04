# JEPA pretraining for direct-force MLIPs: CPU pilot (lighter plan)

**Question.** Direct-force MLIPs skip the backward pass that conservative models need
(F = −∂E/∂R), but their forces aren't conservative, so NVE MD drifts. Can JEPA-style
self-supervised pretraining help a direct-force student with (a) **force accuracy when labels
are scarce**, and (b) **being closer to conservative**: less curl, less NVE drift, and a larger
usable stride in multiple time-stepping (MTS)?

**Scope.** This is a *pilot*: CPU only, one seed, small models, short MD. It tells us whether
the idea is worth a proper GPU study. It does not settle the question.

## Setup

| | |
|---|---|
| Teacher / "ground truth" | MACE-MP-0 *small* (float32), used for labels, for measuring true energy in MD, and as the conservative corrector in MTS |
| Systems | bulk Si (64 atoms, PBC), liquid water (27 H₂O, 81 atoms, PBC), ethanol (9 atoms, gas phase) |
| Data | Langevin MD with the teacher over a temperature ladder. **Pool**: 800 frames/system (Si 300–1200 K, water 250–400 K, ethanol 200–800 K). **Test**: 160 frames/system from an independent run at different temperatures. |
| Student | PaiNN, 64 features, 3 interaction layers, 5 Å cutoff (~150k params). **Direct** force head (gated equivariant vector block) plus a per-atom energy head. Forces never use autograd. |
| Arms | **scratch**; **JEPA** (EMA target encoder); **denoising** (Zaidi et al.-style noise prediction, σ = 0.05 Å) |
| Pretraining | Whole unlabelled pool (2400 frames), 2000 steps, batch 8 |
| Fine-tuning | 16 / 64 / 256 labelled frames per system, 1500 steps, batch 8, AdamW 1e-3 cosine, loss = F-MSE + 0.1·E-MSE, best checkpoint on a fixed 32-frame/system validation set. Same recipe for every arm. |

### JEPA objective (as implemented)

* **Context view:** spatial *block masking*. Random centre atoms plus all atoms within 2.6 Å
  are masked until ≥15 % of each structure is covered (capped at 40 %). Masked atoms keep their
  positions, but their species becomes a learned `[MASK]` token, so the context encoder must
  infer what sits there from the surrounding geometry and chemistry.
* **Target:** the EMA copy of the encoder (momentum 0.996 → 1, cosine) sees the full,
  unmasked structure. Its final **scalar *and* vector (l = 1)** per-atom features are the targets.
  Scalars are LayerNorm-ed; vectors are normalised by their per-atom RMS norm. Keeping the
  equivariant vector channel is deliberate: forces are vectors, and scalar-only targets would
  throw away exactly the directional information a force head needs.
* **Predictor:** a 2-layer PaiNN on the context graph, loss = MSE on masked atoms only.
* Fine-tuning starts from the **EMA (target) encoder**, as in I-JEPA.

### Metrics

1. **Force MAE vs. number of labels**, per system, on the independent test trajectories.
2. **Non-conservativeness:** ‖½(J − Jᵀ)‖ / ‖J‖ with J = ∂F/∂R (full 3N×3N Jacobian by
   autograd, 4 test frames/system). A conservative field has a symmetric J (it is a Hessian),
   so this is 0. A random linear map gives ≈ 0.71.
3. **NVE drift** with direct forces only: 1000 steps (Si 1 fs, water/ethanol 0.5 fs). Energy is
   measured as *teacher* PE + KE, i.e. the quantity a perfect model would conserve.
4. **MTS stride:** impulse r-RESPA as in Bigi et al. (2025). Inner velocity-Verlet with
   student forces; every *k* steps a kick of (F_teacher − F_student)·k·dt/2 at both ends of
   the outer step. We report drift vs. *k* ∈ {2, 4, 8, 16}. The teacher stands in for the
   conservative fine-tuned model that would play this role in practice.
5. **Cost:** ms per force call, direct head vs. −∂E/∂R, same PaiNN, Si cells of 64–512 atoms.

## Reproduce

```bash
pip install torch mace-torch ase matplotlib
cd jepa_mlip_pilot
./run_all.sh          # data generation ≈ 35 min (3 parallel procs), everything else ≈ 1.5 h on 4 cores
```

`results/summary.md` holds all tables (auto-generated). Figures are in `results/figs/`.
Raw per-run JSON is in `results/`.
