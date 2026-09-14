# Frozen local angular predictability diagnostic

User authorized local angular prediction versus constant-rate baseline after the
body-z weight change failed. Freeze original history_26.pt, original normalization,
3,966 train and 6,669 validation parent windows. No new training, parameter fitting,
label/time edits or September 8 test access. This is not a claim of RL readiness.

For train parent offsets 0/1s and validation offsets 0/1/2/3/4s, reconstruct the
26-sample real causal history ending at each local origin. Evaluate a 25-step
(0.5s nominal) local free rollout with endpoints 1/5/10/25 steps (20/100/200/500ms).
Each local forecast starts from real state and history but never consumes future
truth after that local origin. This is not teacher forcing at every 20ms step,
and is not one uninterrupted five-second trajectory.

At the same targets compare:
- local_model: freshly encoded real history, state and known control tape;
- hold_rate: constant true angular velocity at that local origin;
- parent_free_run: original uninterrupted forecast from the original parent t0.
The third arm has a longer prediction horizon at nonzero offsets and is a drift
reference, not an equal-information local predictor. Require offset-zero local
prediction and captured parent rollout to reproduce saved baseline rates.

Use a passive forward hook on derivative_head to capture unclipped normalized
outputs without changing model computation. Save raw arrays and per-log axis
saturation fractions at abs(output)>=6 for local and full-parent rollouts. First
step saturation uses real local state; later steps use predicted state. Record
p99/max values too; a lack of clipping alone does not establish observability.

Report Euclidean angular-rate RMSE, x/y/z RMSE, signed biases and nonfinite counts
per log and equal-log. Speed and absolute body-z-rate groups use previously frozen
train-only tertile boundaries; no bin refitting from validation. Preserve offsets,
actual dt/durations, log/segment/local/parent identity. Overlapping local forecasts
are not independent samples. A lower local than parent error does not itself show
learned dynamics: local_model must be compared with hold_rate at equal horizons.

If local_model cannot reliably beat hold_rate, investigate angular signal timing,
representation and input coverage before more weighting. If local predictions
improve over hold_rate but free rollout drifts, prioritize recurrent feedback and
rotation accumulation. Do not automatically promote or launch architecture search.

Entry: scripts/run_september_angular_local.py. --preflight verifies GPU/checkpoint
and small batches. --launch performs preflight and launches detached without
monitoring; output artifacts/september_angular_local_20260912 with sibling .run.log
and .launch.json. Refuse overwrite, persist exception/failed status, no auto retry.
