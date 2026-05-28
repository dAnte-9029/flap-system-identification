# Rotational Oracle Replay Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose why held-out oracle six-degree-of-freedom short-horizon replay fails in rotation even when true reconstructed effective moment labels are used.

**Architecture:** Add a focused rotational diagnostic script that reuses the existing short-horizon replay helpers but separates kinematic attitude closure, rotational inverse/forward dynamics closure, integration-method sensitivity, time-lag sensitivity, smoothing/noise sensitivity, inertia sensitivity, reference-point transforms, and spike robustness. Each diagnostic writes CSV artifacts and a conservative report that decides whether full 6-DoF prior/corrected replay is justified or whether only translational replay should be used in the paper.

**Tech Stack:** Python, NumPy, pandas, PyYAML, pytest, optional matplotlib for figures, canonical parquet split, existing `scripts/evaluate_short_horizon_replay.py` helpers.

---

## Current Evidence and Problem Statement

The current oracle short-horizon replay result is:

```text
artifact root: artifacts/20260527_short_horizon_replay_v1/oracle_sanity
report: docs/results/2026-05-27-oracle-short-horizon-replay.md
```

The oracle teacher-forced translational channel is locally plausible:

```text
0.25 s median velocity error: 0.064 m/s
0.25 s median position error: 0.053 m
```

The rotational channel fails as a clean six-DoF oracle sanity check:

```text
0.10 s median attitude error: 2.70 deg
0.25 s median attitude error: 5.32 deg
0.25 s median body-rate error: 0.428 rad/s
```

This plan diagnoses the rotational mismatch. It must not be framed as a model-comparison experiment and must not claim closed-loop simulator validation.

## Fixed Inputs and Outputs

Default inputs:

```text
split_root: dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
metadata_path: metadata/aircraft/flapper_01/aircraft_metadata.yaml
oracle_replay_root: artifacts/20260527_short_horizon_replay_v1/oracle_sanity
```

New output root:

```text
artifacts/20260528_rotational_oracle_replay_diagnostics_v1
```

New code, tests, and report:

```text
scripts/diagnose_rotational_replay_oracle.py
tests/test_rotational_replay_diagnostics.py
docs/results/2026-05-28-rotational-oracle-replay-diagnostics.md
```

Do not force-add generated artifacts.

## Diagnostic Gates

Use these gates to prevent misleading later work:

1. **Kinematic attitude gate:** logged body rates integrated into attitude should match logged attitude over short windows. If this fails, first fix quaternion multiplication, body-rate frame/sign, timestamp alignment, or attitude convention.
2. **Moment-label closure gate:** moment recomputed from logged `omega`, logged `alpha`, and metadata inertia should match stored `mx_b,my_b,mz_b`. If this fails, diagnose label generation or column selection.
3. **Forward rotational replay gate:** oracle moment integrated into `omega` should reproduce logged `omega` over short windows. If this fails after the first two gates pass, focus on integration method, time lag, derivative noise, inertia, reference point, or spikes.
4. **6-DoF model-comparison gate:** do not run prior/corrected six-DoF replay until rotational oracle diagnostics are satisfactory. Translational replay comparison remains allowed if clearly separated.

## Metrics

Use the existing window protocol from `scripts/evaluate_short_horizon_replay.py`:

```text
horizons: 0.10,0.25,0.50,1.00,2.00 s
stride: 0.25 s
split: test
grouping: do not cross log_id or segment_id
```

Primary metrics:

```text
attitude_error_deg
body_rate_error_rad_s
alpha_error_rad_s2
moment_error_n_m
moment_rmse_by_axis
omega_rmse_by_axis
```

Aggregates:

```text
median, p25, p75, p90, p95, mean, n_windows
```

Always report per-log summaries as well as horizon summaries.

## Task 1: Add Rotational Diagnostic Unit Tests

**Files:**

- Create: `tests/test_rotational_replay_diagnostics.py`
- Create later: `scripts/diagnose_rotational_replay_oracle.py`

**Step 1: Write failing tests**

Add tests that import:

```python
from scripts.diagnose_rotational_replay_oracle import (
    integrate_attitude_from_logged_rates,
    recompute_moment_from_alpha,
    infer_alpha_from_moment,
    apply_moment_reference_transform,
    summarize_metric_table,
)
```

Test cases:

- constant yaw rate with identity initial attitude produces the expected quaternion after `dt`;
- `recompute_moment_from_alpha` returns `I alpha + omega x I omega`;
- `infer_alpha_from_moment` inverts `I alpha + omega x I omega`;
- `apply_moment_reference_transform` supports `none`, `minus_r_cross_f`, and `plus_r_cross_f`;
- `summarize_metric_table` returns median and p90 grouped by diagnostic and horizon.

**Step 2: Run red test**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: fail because `scripts.diagnose_rotational_replay_oracle` does not exist.

**Step 3: Implement minimal diagnostic API**

Create `scripts/diagnose_rotational_replay_oracle.py`.

Reuse helpers from `scripts/evaluate_short_horizon_replay.py`:

```python
attitude_error_deg
normalize_quaternion
_delta_quaternion_from_body_rate
_quat_multiply
load_mass_properties
select_replay_windows
QUATERNION_COLUMNS
OMEGA_COLUMNS
FORCE_COLUMNS
MOMENT_COLUMNS
```

Implement:

```python
integrate_attitude_from_logged_rates(window, rate_sign=1.0, multiply_side="right")
recompute_moment_from_alpha(omega_b, alpha_b, inertia_b)
infer_alpha_from_moment(omega_b, moment_b, inertia_b)
apply_moment_reference_transform(moment_b, force_b, r_b, mode)
summarize_metric_table(metrics, group_columns)
```

Keep implementation small and deterministic.

**Step 4: Run green test**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 2: Kinematic Attitude Closure Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Check whether logged `vehicle_angular_velocity.xyz[*]` can reproduce logged `vehicle_attitude.q[*]` without using moments, inertia, or force.

**Step 1: Add tests**

Add synthetic tests for:

- `evaluate_attitude_kinematic_closure(...)` returns near-zero attitude error for a synthetic log generated by a constant body rate;
- sign-flipped quaternion targets still give zero attitude error;
- the diagnostic writes rows for each horizon.

**Step 2: Implement**

Implement:

```python
evaluate_attitude_kinematic_closure(samples, horizons_s, stride_s, split)
```

Evaluate variants:

```text
right_multiply_rate_plus
right_multiply_rate_minus
left_multiply_rate_plus
left_multiply_rate_minus
```

For each window:

- initialize `q_sim` from the first logged quaternion;
- integrate using logged body rate only;
- compare final `q_sim` with final logged quaternion.

**Step 3: Add gate logic**

The report should answer:

```text
Does the nominal convention already work?
Does another sign/multiply convention work much better?
```

Do not silently switch conventions. If a non-nominal variant is best, report it as a possible convention mismatch.

**Step 4: Run tests**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 3: Moment-Label Closure Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Check whether stored moment labels match the rigid-body expression that the label pipeline claims:

```text
M_B = I_B alpha_B + omega_B x (I_B omega_B)
```

**Step 1: Add tests**

Synthetic test:

- create known `omega`, `alpha`, `I`;
- create `mx_b,my_b,mz_b` using the same expression;
- verify closure RMSE is zero.

**Step 2: Implement**

Implement:

```python
evaluate_moment_label_closure(samples, inertia_b)
```

Use columns:

```text
vehicle_angular_velocity.xyz[0:2]
vehicle_angular_velocity.xyz_derivative[0:2]
mx_b,my_b,mz_b
```

Output rows:

```text
diagnostic = "moment_label_closure"
axis = mx_b/my_b/mz_b/moment_norm
rmse
mae
bias
correlation
n
```

Also compute alpha closure:

```text
alpha_from_moment = I^{-1}(M - omega x I omega)
alpha_error = alpha_from_moment - logged_alpha
```

**Step 3: Run tests**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 4: Forward Omega Replay Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Check whether oracle moment can reproduce logged body rates independent of attitude integration.

**Step 1: Add tests**

Synthetic test:

- create a log with constant angular acceleration and diagonal inertia;
- compute moment labels;
- verify `evaluate_omega_replay(...)` produces near-zero final body-rate error.

**Step 2: Implement**

Implement:

```python
integrate_omega_from_moment(window, inertia_b, gyro_source="logged", method="euler", substeps=1)
evaluate_omega_replay(samples, inertia_b, horizons_s, stride_s)
```

Variants:

```text
logged_gyro_euler
sim_gyro_euler
logged_gyro_substep4
sim_gyro_substep4
trapezoid_alpha
```

For `trapezoid_alpha`, compute `alpha` at both ends from oracle moment and average across the step.

**Step 3: Run tests**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 5: Time-Lag Sensitivity Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Detect whether moment labels or angular acceleration are delayed/advanced relative to logged body rates.

**Step 1: Add tests**

Synthetic test:

- create a sinusoidal `omega` and delayed moment-derived alpha;
- verify lag sweep identifies the imposed lag within one step.

**Step 2: Implement**

Implement:

```python
evaluate_moment_lag_sweep(samples, inertia_b, lags_s, horizons_s, stride_s)
```

Default lags:

```text
-0.100,-0.080,-0.060,-0.040,-0.020,0.000,0.020,0.040,0.060,0.080,0.100
```

Apply lag only within each `log_id`/`segment_id` group using interpolation. Do not cross boundaries.

Report:

```text
best_lag_s by horizon and by log
body_rate_error_rad_s median/p90 at each lag
```

**Step 3: Run tests**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 6: Angular-Acceleration and Moment Smoothing Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Determine whether high-frequency angular-acceleration or moment spikes dominate rotational replay error.

**Step 1: Add tests**

Synthetic test:

- inject a one-sample angular-acceleration spike;
- verify a rolling/Savitzky-Golay smoothing variant reduces replay error or alpha RMS.

**Step 2: Implement**

Implement:

```python
smooth_grouped_signal(samples, columns, method, window_s, polyorder)
evaluate_smoothing_sensitivity(samples, inertia_b, horizons_s, stride_s)
```

Variants:

```text
raw_moment
moment_savgol_0p04
moment_savgol_0p08
moment_savgol_0p16
moment_savgol_0p32
alpha_savgol_0p04_recomputed_moment
alpha_savgol_0p08_recomputed_moment
alpha_savgol_0p16_recomputed_moment
alpha_savgol_0p32_recomputed_moment
```

Use group-wise smoothing by `log_id` and `segment_id`.

**Step 3: Run tests**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 7: Inertia Sensitivity and Fitted-Inertia Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Check whether provisional CAD-seeded inertia is a major cause of rotational replay mismatch.

**Step 1: Add tests**

Synthetic test:

- create data with known diagonal inertia;
- verify fitted diagonal inertia recovers the known values within tolerance.

**Step 2: Implement inertia scaling**

Implement:

```python
evaluate_inertia_scale_sensitivity(samples, inertia_b, scales, horizons_s, stride_s)
```

Default scales:

```text
0.25,0.50,0.75,1.00,1.25,1.50,2.00,3.00,4.00
```

Evaluate both:

```text
global_scale: I_scaled = s I
axis_scale: scale Ixx/Iyy/Izz independently, one axis at a time
```

**Step 3: Implement fitted inertia**

Implement:

```python
fit_diagonal_inertia_from_logs(train_samples_or_test_diagnostic_subset)
fit_symmetric_inertia_from_logs(...)
```

Treat fitted inertia as diagnostic only. Prefer fitting on train split and evaluating on test split when available.

Report:

- fitted inertia matrix;
- whether it is positive definite;
- improvement in omega replay;
- whether values are physically plausible relative to metadata.

**Step 4: Run tests**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 8: Reference-Point / CG Transform Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Check whether moment labels and replay dynamics are using inconsistent reference points, such as IMU origin versus CG.

**Step 1: Add tests**

Synthetic test:

- use known `r_b` and `F_b`;
- verify `M +/- r x F` transforms as expected.

**Step 2: Implement**

Implement:

```python
evaluate_reference_point_sensitivity(samples, inertia_b, cg_b_m, horizons_s, stride_s)
```

Candidate transforms:

```text
none
minus_cg_cross_force
plus_cg_cross_force
minus_2x_cg_cross_force
plus_2x_cg_cross_force
```

Use metadata:

```text
mass_properties.cg_b_m.value
frames.body_reference_origin
frames.cg_reference_origin
label_definition.moment_definition
```

Do not silently pick a transform as truth. Report which transform best closes replay and whether it contradicts metadata.

**Step 3: Run tests**

```bash
pytest tests/test_rotational_replay_diagnostics.py -q
```

Expected: pass.

## Task 9: Spike and Outlier Robustness Diagnostic

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Purpose:** Check whether rare moment spikes dominate window-level rotational replay drift.

**Step 1: Add tests**

Synthetic test:

- inject one large moment spike;
- verify spike removal or winsorization changes the diagnostic metric in the expected direction.

**Step 2: Implement**

Implement:

```python
evaluate_spike_robustness(samples, inertia_b, horizons_s, stride_s)
```

Variants:

```text
raw
drop_top_0p5_percent_moment_norm_windows
drop_top_1p0_percent_moment_norm_windows
winsorize_moment_99p0
winsorize_moment_99p5
```

Report both:

- error improvement;
- fraction of windows/samples removed or clipped.

Do not use spike removal for final model comparison unless the paper clearly calls it diagnostic.

## Task 10: Full CLI and Artifact Writer

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Modify: `tests/test_rotational_replay_diagnostics.py`

**Step 1: Add integration test**

Create a temporary synthetic split and run:

```python
run_rotational_diagnostics(...)
```

Assert these files exist:

```text
attitude_kinematic_closure.csv
moment_label_closure.csv
omega_replay_summary.csv
lag_sweep_summary.csv
smoothing_sensitivity_summary.csv
inertia_sensitivity_summary.csv
reference_point_sensitivity_summary.csv
spike_robustness_summary.csv
diagnostic_decision.json
README.md
```

**Step 2: Implement CLI**

Command:

```bash
python scripts/diagnose_rotational_replay_oracle.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-root artifacts/20260528_rotational_oracle_replay_diagnostics_v1 \
  --split test \
  --horizons 0.10,0.25,0.50,1.00,2.00 \
  --stride-s 0.25 \
  --lags-s -0.100,-0.080,-0.060,-0.040,-0.020,0.000,0.020,0.040,0.060,0.080,0.100 \
  --overwrite
```

**Step 3: Diagnostic decision JSON**

Write `diagnostic_decision.json` with:

```json
{
  "kinematic_attitude_gate": "pass|fail|conditional",
  "moment_label_closure_gate": "pass|fail|conditional",
  "forward_omega_replay_gate": "pass|fail|conditional",
  "likely_issue_order": [],
  "safe_next_experiments": [],
  "paper_claim_boundary": ""
}
```

Decision rules:

- If kinematic attitude gate fails, likely issue is quaternion/rate convention or timestamp alignment.
- If moment-label closure fails, likely issue is label pipeline or alpha column mismatch.
- If moment-label closure passes but omega replay fails, rank lag/noise/inertia/reference-point/spikes by observed improvement.
- If only smoothing or spike removal improves the result, paper should not claim clean six-DoF replay.

**Step 4: Run tests**

```bash
python -m py_compile scripts/diagnose_rotational_replay_oracle.py
pytest tests/test_rotational_replay_diagnostics.py tests/test_short_horizon_replay.py -q
```

Expected: pass.

## Task 11: Run Real Diagnostics and Write Report

**Files:**

- Generated: `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/*`
- Create: `docs/results/2026-05-28-rotational-oracle-replay-diagnostics.md`

**Step 1: Run command**

```bash
python scripts/diagnose_rotational_replay_oracle.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-root artifacts/20260528_rotational_oracle_replay_diagnostics_v1 \
  --split test \
  --horizons 0.10,0.25,0.50,1.00,2.00 \
  --stride-s 0.25 \
  --lags-s -0.100,-0.080,-0.060,-0.040,-0.020,0.000,0.020,0.040,0.060,0.080,0.100 \
  --overwrite
```

**Step 2: Inspect key outputs**

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
root = Path("artifacts/20260528_rotational_oracle_replay_diagnostics_v1")
print(json.dumps(json.loads((root / "diagnostic_decision.json").read_text()), indent=2))
for name in [
    "attitude_kinematic_closure.csv",
    "moment_label_closure.csv",
    "omega_replay_summary.csv",
    "lag_sweep_summary.csv",
    "inertia_sensitivity_summary.csv",
    "reference_point_sensitivity_summary.csv",
    "spike_robustness_summary.csv",
]:
    print("\\n==", name)
    print(pd.read_csv(root / name).head(20).to_string(index=False))
PY
```

**Step 3: Write report**

Create `docs/results/2026-05-28-rotational-oracle-replay-diagnostics.md` with:

- short statement of the original oracle replay failure;
- gate-by-gate diagnosis;
- table ranking likely causes;
- which checks passed and failed;
- whether six-DoF prior/corrected replay is currently allowed;
- what should be done after measured CG/inertia become available;
- paper-safe wording.

Recommended wording if rotation remains unresolved:

```text
The replay diagnostics indicate that the translational effective-force channel is locally self-consistent, whereas rotational replay remains sensitive to moment-label differentiation, inertia/reference-point assumptions, and short-duration moment spikes. We therefore do not use six-degree-of-freedom replay as a validation claim in this version; replay evidence is restricted to translational/local force consistency.
```

**Step 4: Verify**

```bash
python -m py_compile scripts/diagnose_rotational_replay_oracle.py
pytest tests/test_rotational_replay_diagnostics.py tests/test_short_horizon_replay.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/diagnose_rotational_replay_oracle.py tests/test_rotational_replay_diagnostics.py docs/plans/2026-05-28-rotational-oracle-replay-diagnostics-plan.md docs/results/2026-05-28-rotational-oracle-replay-diagnostics.md
git commit -m "feat: diagnose rotational oracle replay mismatch"
```

Do not force-add `artifacts/`.

## Optional Task 12: Paper-Facing Figure

Do this only after the numeric report is reviewed.

**Files:**

- Modify: `scripts/diagnose_rotational_replay_oracle.py`
- Generated: `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/figures/rotational_diagnostics_summary.pdf`

Figure panels:

1. attitude kinematic closure by horizon and convention;
2. omega replay error by integrator variant;
3. lag sweep error versus lag;
4. inertia/reference-point/spike sensitivity summary.

Caption boundary:

```text
Diagnostics for oracle rotational replay from held-out logs. These plots identify likely sources of rotational mismatch and do not validate closed-loop simulation.
```

## Final Acceptance Checklist

- [ ] `pytest tests/test_rotational_replay_diagnostics.py tests/test_short_horizon_replay.py -q` passes.
- [ ] `python -m py_compile scripts/diagnose_rotational_replay_oracle.py` passes.
- [ ] Artifacts exist under `artifacts/20260528_rotational_oracle_replay_diagnostics_v1`.
- [ ] `diagnostic_decision.json` ranks likely causes rather than hiding unresolved failures.
- [ ] Report clearly says whether six-DoF prior/corrected replay is allowed or blocked.
- [ ] Report separates translational replay from rotational replay.
- [ ] No generated artifacts are force-added.
- [ ] No paper text claims closed-loop simulator validation.
