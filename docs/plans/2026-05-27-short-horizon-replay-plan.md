# Short-Horizon Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a log-seeded short-horizon replay evaluation that first verifies oracle effective-wrench consistency, then compares DeLaurier prior and phase-structured B+C corrected wrench models without claiming closed-loop simulator validation.

**Architecture:** Add one standalone replay script with reusable rigid-body integration helpers, narrow tests, and artifact/report writers. The first execution stage is oracle-only sanity: use held-out log labels as the applied effective wrench, reset from real log states at many start times, and check whether short-horizon velocity, position, body-rate, and attitude drift remain small enough to justify later prior/corrected replay. Later stages reuse the same evaluator for prior and corrected B+C wrenches.

**Tech Stack:** Python, NumPy, pandas, PyYAML, matplotlib, pytest, parquet artifacts, existing canonical split and phase-structured wrench prediction artifacts.

---

## Paper Framing Constraints

- This is **open-loop, log-seeded short-horizon replay**, not closed-loop simulator validation.
- The first stage is an **oracle sanity check**. It asks whether the current labels, mass/inertia metadata, quaternion convention, time alignment, and integrator are self-consistent.
- Oracle replay may use the reconstructed effective-wrench labels (`fx_b...mz_b`) because it is diagnostic-only.
- Prior/corrected replay must not use target labels as model inputs.
- Claims should be limited to local simulator dynamics over short reset windows.
- If oracle replay fails, do not continue to prior/corrected comparison until the cause is diagnosed.

## Fixed Inputs

Use these defaults:

```text
split_root: dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
metadata_path: metadata/aircraft/flapper_01/aircraft_metadata.yaml
wrench_artifact_root: artifacts/20260527_phase_structured_wrench_correction_v1
```

The relevant data columns are present in `test_samples.parquet`:

```text
time_s
log_id
vehicle_local_position.x/y/z
vehicle_local_position.vx/vy/vz
vehicle_attitude.q[0:3]
vehicle_angular_velocity.xyz[0:2]
fx_b/fy_b/fz_b
mx_b/my_b/mz_b
```

Prediction columns for later model comparison are present in:

```text
artifacts/20260527_phase_structured_wrench_correction_v1/prediction_parquets/test_predictions.parquet
```

## Outputs

Oracle-only first-stage outputs:

```text
artifacts/20260527_short_horizon_replay_v1/oracle_sanity/replay_window_metrics.csv
artifacts/20260527_short_horizon_replay_v1/oracle_sanity/horizon_summary.csv
artifacts/20260527_short_horizon_replay_v1/oracle_sanity/log_summary.csv
artifacts/20260527_short_horizon_replay_v1/oracle_sanity/config.json
artifacts/20260527_short_horizon_replay_v1/oracle_sanity/README.md
docs/results/2026-05-27-oracle-short-horizon-replay.md
```

Later full-comparison outputs:

```text
artifacts/20260527_short_horizon_replay_v1/model_comparison/replay_window_metrics.csv
artifacts/20260527_short_horizon_replay_v1/model_comparison/horizon_summary.csv
artifacts/20260527_short_horizon_replay_v1/model_comparison/figures/replay_error_vs_horizon.pdf
docs/results/2026-05-27-short-horizon-replay-comparison.md
```

## Replay Modes

### Mode A: Oracle Teacher-Forced Consistency

Use log attitude for rotating oracle force into NED and log body rates in the gyroscopic term when reconstructing angular acceleration from oracle moment.

Purpose:

```text
Check label/time/quaternion/mass/inertia consistency before testing any model.
```

Equations:

```text
a_N(t) = R_NB(q_log(t)) F_B_oracle(t) / m + g_N
alpha_B(t) = I^{-1} (M_B_oracle(t) - omega_log(t) x I omega_log(t))
```

Integrate:

```text
v_N, p_N from a_N
omega_B from alpha_B
q_NB from integrated omega_B
```

This mode should be the first gate. If it cannot reproduce short windows reasonably, the replay setup is not trustworthy.

### Mode B: Coupled Oracle Upper Bound

Use simulated attitude for force rotation and simulated body rates in the gyroscopic term.

Purpose:

```text
Estimate the best achievable replay drift when the simulator is allowed to use the true effective wrench but has normal state-coupled integration.
```

This will be worse than Mode A but should still be the best full-dynamics curve.

### Mode C: Prior and Corrected Model Comparison

Use the same log-seeded windows and replay:

```text
prior wrench: prior_fx_b...prior_mz_b
corrected wrench: force_corr_fx_b...force_corr_fz_b and moment_corr_mx_b...moment_corr_mz_b
```

Primary paper pattern:

```text
oracle teacher-forced <= coupled oracle <= corrected B+C <= prior
```

If only the translational channels improve, report that explicitly and keep rotational replay as limited by moment labels/inertia/CG.

## Metrics

For every window and horizon:

```text
position_error_m = ||p_sim - p_log||
velocity_error_m_s = ||v_sim - v_log||
attitude_error_deg = 2 acos(|dot(q_sim, q_log)|)
body_rate_error_rad_s = ||omega_sim - omega_log||
```

Aggregate by horizon:

```text
n_windows
median
p25
p75
p90
p95
mean
```

Report per log as well, because samples within a log are correlated.

Suggested horizons:

```text
0.10 s, 0.25 s, 0.50 s, 1.00 s, 2.00 s
```

Suggested start stride:

```text
0.25 s
```

Reject windows that:

- cross a `log_id` boundary;
- cross a `segment_id` boundary, if present;
- have non-monotonic or large-gap timestamps;
- contain non-finite state or wrench values;
- do not have enough samples for the requested horizon.

## Task 1: Add Replay Helper Tests

**Files:**

- Create: `tests/test_short_horizon_replay.py`
- Create later: `scripts/evaluate_short_horizon_replay.py`

**Step 1: Write failing tests**

Add tests for:

```python
from scripts.evaluate_short_horizon_replay import (
    normalize_quaternion,
    quaternion_to_rotation_body_to_ned,
    integrate_oracle_teacher_forced_window,
    attitude_error_deg,
)
```

Test cases:

- identity quaternion rotates body force `[1, 2, 3]` to NED `[1, 2, 3]`;
- constant non-gravity body force with identity attitude produces analytically expected velocity drift;
- zero moment and zero initial body rate keep body-rate error near zero;
- quaternion sign flip gives zero attitude error.

**Step 2: Run red test**

```bash
pytest tests/test_short_horizon_replay.py -q
```

Expected: fail because the new module does not exist.

**Step 3: Implement minimal helper API**

Create `scripts/evaluate_short_horizon_replay.py`.

Implement:

```python
normalize_quaternion(q)
quaternion_to_rotation_body_to_ned(q)
attitude_error_deg(q_a, q_b)
load_mass_properties(metadata_path)
integrate_oracle_teacher_forced_window(window, mass_kg, inertia_b, gravity_m_s2)
```

Use the same quaternion convention as `src/system_identification/pipeline.py`: `vehicle_attitude.q[0:3]` is wxyz and represents body-to-NED rotation. Gravity is NED `[0, 0, +g]`.

**Step 4: Run green test**

```bash
pytest tests/test_short_horizon_replay.py -q
```

Expected: pass.

## Task 2: Implement Oracle Window Evaluation

**Files:**

- Modify: `scripts/evaluate_short_horizon_replay.py`
- Modify: `tests/test_short_horizon_replay.py`

**Step 1: Add tests**

Add a synthetic multi-window test that creates one log with known constant acceleration and verifies:

- window count is as expected;
- horizon summary contains all requested horizons;
- median velocity error is near zero for the oracle teacher-forced mode;
- output CSVs are written under a temporary output root.

**Step 2: Implement**

Implement:

```python
select_replay_windows(frame, horizons_s, stride_s)
evaluate_oracle_replay(samples, metadata, horizons_s, stride_s, mode="oracle_teacher_forced")
summarize_replay_metrics(window_metrics)
write_oracle_replay_artifacts(...)
```

Window metric rows must include:

```text
mode
split
log_id
segment_id
start_time_s
horizon_s
n_steps
position_error_m
velocity_error_m_s
attitude_error_deg
body_rate_error_rad_s
```

**Step 3: Run tests**

```bash
pytest tests/test_short_horizon_replay.py -q
```

Expected: pass.

## Task 3: Run Oracle Sanity Check on Held-Out Logs

**Files:**

- Generated: `artifacts/20260527_short_horizon_replay_v1/oracle_sanity/*`
- Create: `docs/results/2026-05-27-oracle-short-horizon-replay.md`

**Step 1: Run command**

```bash
python scripts/evaluate_short_horizon_replay.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-root artifacts/20260527_short_horizon_replay_v1/oracle_sanity \
  --split test \
  --modes oracle_teacher_forced,coupled_oracle \
  --horizons 0.10,0.25,0.50,1.00,2.00 \
  --stride-s 0.25 \
  --overwrite
```

**Step 2: Inspect output**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("artifacts/20260527_short_horizon_replay_v1/oracle_sanity")
summary = pd.read_csv(root / "horizon_summary.csv")
print(summary.to_string(index=False))
logs = pd.read_csv(root / "log_summary.csv")
print(logs.to_string(index=False))
PY
```

**Step 3: Write oracle report**

Create `docs/results/2026-05-27-oracle-short-horizon-replay.md` with:

- command used;
- number of logs/windows;
- horizon-wise median and p90 errors;
- per-log notes;
- whether oracle sanity passes;
- if it fails, the most likely failure source: quaternion convention, gravity sign, mass/inertia, timestamp gaps, label spikes, or integration scheme;
- paper-safe wording.

Sanity pass criteria should be conservative and diagnostic, not absolute:

```text
0.10 s and 0.25 s oracle teacher-forced velocity/body-rate drift should be much smaller than prior/corrected replay is expected to be.
2.00 s may drift and should not be used as a hard pass/fail gate.
```

**Step 4: Verify**

```bash
python -m py_compile scripts/evaluate_short_horizon_replay.py
pytest tests/test_short_horizon_replay.py -q
```

Expected: pass.

**Step 5: Commit first-stage oracle work**

```bash
git add scripts/evaluate_short_horizon_replay.py tests/test_short_horizon_replay.py docs/plans/2026-05-27-short-horizon-replay-plan.md docs/results/2026-05-27-oracle-short-horizon-replay.md
git commit -m "feat: add oracle short-horizon replay sanity check"
```

Do not force-add `artifacts/`.

## Task 4: Add Prior and Corrected Replay Comparison

Do this only after Task 3 is reviewed.

**Files:**

- Modify: `scripts/evaluate_short_horizon_replay.py`
- Modify: `tests/test_short_horizon_replay.py`
- Create: `docs/results/2026-05-27-short-horizon-replay-comparison.md`

**Implementation notes:**

- Join `test_samples.parquet` with `prediction_parquets/test_predictions.parquet` by row order after checking identical `timestamp_us`, `log_id`, and `split`.
- Add model modes:

```text
prior
phase_structured_corrected
```

- Use `prior_fx_b...prior_mz_b` and `force_corr_* / moment_corr_*` from the prediction parquet.
- Keep replay windows identical across oracle/prior/corrected modes.
- Report whether corrected reduces short-horizon drift relative to prior.

**Run command:**

```bash
python scripts/evaluate_short_horizon_replay.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --wrench-artifact-root artifacts/20260527_phase_structured_wrench_correction_v1 \
  --output-root artifacts/20260527_short_horizon_replay_v1/model_comparison \
  --split test \
  --modes oracle_teacher_forced,coupled_oracle,prior,phase_structured_corrected \
  --horizons 0.10,0.25,0.50,1.00,2.00 \
  --stride-s 0.25 \
  --overwrite
```

## Task 5: Add Paper-Facing Plot

Do this only after Task 4 has credible numbers.

**Files:**

- Modify: `scripts/evaluate_short_horizon_replay.py`
- Create: `artifacts/20260527_short_horizon_replay_v1/model_comparison/figures/replay_error_vs_horizon.pdf`

Figure layout:

- panel A: velocity error vs horizon;
- panel B: attitude error vs horizon;
- panel C: body-rate error vs horizon;
- panel D: position drift vs horizon;
- lines: oracle teacher-forced, coupled oracle, prior, corrected;
- band: p25-p75 or p10-p90.

Use conservative caption language:

```text
Open-loop log-seeded replay from held-out flight states. Each window is reset from the measured state; the curves measure local drift and do not constitute closed-loop simulator validation.
```

## Final Acceptance Checklist

- [ ] `pytest tests/test_short_horizon_replay.py -q` passes.
- [ ] `python -m py_compile scripts/evaluate_short_horizon_replay.py` passes.
- [ ] Oracle artifacts exist under `artifacts/20260527_short_horizon_replay_v1/oracle_sanity`.
- [ ] Oracle report states whether the replay setup passes sanity.
- [ ] If oracle sanity fails, prior/corrected comparison is not run.
- [ ] All claims distinguish oracle replay, open-loop model replay, and closed-loop simulation.
- [ ] Heavy artifacts remain untracked.
