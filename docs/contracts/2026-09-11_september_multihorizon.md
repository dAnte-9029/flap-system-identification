# September matched training-horizon experiment

Frozen before execution; user authorized three training objectives, no monitoring.
Read registered September v2 train/validation only; September 8 remains sealed.
No architecture, phase representation or data split changes. All three models
use the controlled 64-unit history GRU, 26 real history samples, seed 17, 40 epochs,
batch 256, AdamW lr=3e-4, weight decay=1e-5, gradient clip=5, GPU 1.

## Matched origins and normalization

Use registered core 2 s training windows with >=0.5 s real history and >=25
preceding samples. Every model uses the exact same rows, order, shuffle seed,
initial parameters, optimizer settings and train-only normalization. Fit common
normalization once using the common 2 s batch and train transitions; even the
one-second model uses this common normalization. The first-second target is the
prefix of the identical two-second record. Do not compare different training
cohorts as if only the objective had changed.

Validation parents must exactly equal Step 2's saved 6,835 five-second windows
(identity, source endpoints and order). All models roll continuously for five
seconds from the same real history and initial state, without intermediate resets.
Future real states are loss/metric targets only; controls are the same known tape.

## Objectives and budget

- `matched_1s`: original L(0:1 s), 50 rollout steps.
- `full_2s`: original L(0:2 s), 100 rollout steps.
- `joint_1s_2s`: 0.5 L(0:1 s) + 0.5 L(0:2 s), both calculated from the **same**
  100-step free-running prediction, without real-state resets. This gives first
  second samples three times the weight of second second samples; it is a
  predeclared short/long weighting, not a new physics loss.

L is the unchanged Main V1 position, velocity, quaternion attitude, body-rate,
relative-phase and frequency loss, with unchanged component scales and weights.
The three final scalar training losses are not directly comparable because their
objectives differ. Same number of optimizer updates; two-second models have twice
the nominal rollout transitions per update. Record wall time, updates, transition
count and peak allocated GPU memory. Do not call this a compute-matched experiment.

## Outputs and decision evidence

Reuse Step 2 evaluation functions: full 50 Hz error curves, per-log and equal-log
metrics, 1/2/3/5 s endpoints, per-window error traces and whole-prefix trajectory
RMS, representative predictions and plots. Large-error threshold remains position
>10 m OR attitude >60 deg. Nonfinite predictions count as failures, not dropped.
Matched gain tables compare full_2s and joint against matched_1s for all four
metrics and each log; report short-horizon tradeoffs and 3/5 s extrapolation.
The previous Step 2 controlled model is a contextual reference only because its
training set and normalization differ. Its saved curve is copied separately.

No validation fitting, early stopping, threshold adjustment or test opening. This
single-seed diagnostic cannot establish final scientific promotion. Checkpoints,
train histories and fixed manifest are saved automatically. Exceptions create
failed status and traceback; no auto retry.

Command:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -u scripts/run_september_multihorizon.py
```

Output: `artifacts/september_multihorizon_20260911/`; detached launch stdout/stderr:
`artifacts/september_multihorizon_20260911.run.log`; launch PID/command metadata:
`artifacts/september_multihorizon_20260911.launch.json`. `status.json` records
loading/training/evaluating/completed/failed. Existing output is never overwritten.
