# Step 8 complete — no candidate promoted

S0 completed GPU retraining with the original seeds, optimizer, 40 base epochs,
25 actuator epochs and fixed-final-epoch checkpoint selection. All parameters
and state tensors are bitwise equal to the historical baseline. The complete
722-origin, 250-step validation trajectories are also bitwise equal, including
hidden, actuator and phase-anchor state. See `baseline_reproduction.json`.

S0 diagnostics are saved in `S0/`: teacher-state errors, autonomous metrics,
error growth, variation, train/validation perturbations, and partial physical
state Jacobians. There are 0 numerical failures, 0 clipping failures and 255
support-envelope failures at 5 seconds. Sealed test was not opened.

The nine affected Main V2 test modules pass (40 tests). `git diff --check`
passes. No Step 1–7 checkpoint or result is intentionally modified.

## Approved scientific contract

The historical objective already supervises all 50 free-running steps, with
gradient propagation through predicted states. It is not a teacher-forced
one-step objective. The proposed six-model contract therefore preserves this
original objective and adds a uniformly averaged first-H-step loss on velocity,
attitude and body rate (H=5/10/20; lambda=0.1/0.5/1.0 per requested matrix).
This tests short-prefix reweighting, not the first introduction of closed-loop
training. Translation/rotation/all-state groups would be reported as loss
components rather than expanding the six-model training matrix.

The user explicitly approved this contract on 2026-09-18. GPU training and
722-origin evaluation completed for all five fixed candidates. R2 (H=10,
lambda=0.1) ranks best but is not promoted: 5s velocity/attitude improve
1.40%/1.62%, while 0.5–1s does not improve consistently and last-1s body-rate
variation remains 0.07454. See `report.md` for all seven answers, component
analysis, error growth, perturbations and Jacobian limitations.

All 398 protected historical/input files match their recorded hashes.
The affected Main V2 regression suite passes (40 tests). The full repository
suite has 633 passes and one external prerequisite failure: the PX4 checkout
does not contain `src/modules/rpm_pid/rpm_pid_params.c`. This failure is recorded
in `full_suite_result.json`; the full suite is not claimed to pass.

## Completed baseline evaluation command

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_main_v2_rollout_consistency.py --results docs/analysis/results/main_v2_rollout_consistency --artifacts artifacts/main_v2_rollout_consistency --experiment S0 --device cuda:1
```

Full reproduction in empty directories:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_rollout_consistency.py --results /tmp/main-v2-step8/results --artifacts /tmp/main-v2-step8/models --device cuda:1 --contract legacy50_plus_short_prefix
```
