# Smoothed and Time-Aligned Effective-Wrench Labels Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible preprocessing and diagnostic pipeline that smooths raw kinematic and actuator/airdata signals before inverse-dynamics reconstruction, then evaluates whether the resulting effective-wrench labels are less spike-dominated without erasing physically meaningful wingbeat-scale structure.

**Architecture:** Extend the existing split-rewriting path instead of replacing it. Add reusable signal-processing utilities for groupwise smoothing, derivative estimation, time-shift sweeps, and actuator/airdata filtering; create a clean-label split builder that writes a manifest and diagnostic tables; then rerun label-quality diagnostics, phase-structured correction, and translational replay checks on the new split. All lag/filter choices that affect the final dataset must be selected on train logs only and then applied unchanged to validation/test logs.

**Tech Stack:** Python, NumPy, pandas, SciPy signal/interpolate, PyYAML, pytest, parquet canonical split, existing `system_identification.pipeline`, existing B+C correction script, existing short-horizon replay script.

---

## Context and Design Rules

Existing relevant files:

- `src/system_identification/pipeline.py`
  - Already contains `compute_smoothed_kinematic_derivatives(...)` using Savitzky-Golay derivatives.
  - Already contains `_compute_effective_wrench_labels(...)`.
- `scripts/build_smoothed_label_split.py`
  - Already rewrites a split with smoothed force/moment labels.
  - Does not yet handle time-lag sweeps, actuator/airdata filtering, spline derivatives, or label-quality reports.
- `scripts/diagnose_target_generation_sensitivity.py`
  - Already diagnoses `fy_b` sensitivity to smoothing windows.
  - Needs to be generalized to six-axis wrench and signal-alignment variants.
- `scripts/train_phase_structured_wrench_correction.py`
  - Existing B+C model experiment to rerun on the new split.
- `scripts/evaluate_short_horizon_replay.py`
  - Existing replay checker. Use translational/local force replay first; do not claim full 6-DoF replay until rotational oracle diagnostics are resolved.
- `scripts/diagnose_rotational_replay_oracle.py`
  - Current diagnostics show rotational forward replay is still unresolved, so moment labels remain lower-confidence.

Default input split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
```

Default metadata:

```text
metadata/aircraft/flapper_01/aircraft_metadata.yaml
```

New output split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1
```

New artifact root:

```text
artifacts/20260528_smoothed_time_aligned_wrench_labels_v1
```

New report:

```text
docs/results/2026-05-28-smoothed-time-aligned-wrench-labels.md
```

Do not force-add generated dataset/artifact files.

## Method Basis

This pipeline follows the useful part of NeuroBEM's data processing idea: reduce inverse-dynamics noise by smoothing and synchronizing measured states before differentiating them, rather than filtering the final reconstructed wrench label directly. NeuroBEM uses fitted cubic splines to obtain less noisy derivatives and uses time synchronization between onboard data and motion-capture-derived states; it also low-pass filters motor-speed data according to motor response. Our setting lacks motion capture, so the equivalent strategy is to smooth onboard state estimates groupwise, evaluate time-lag sensitivity between state/airdata/phase/actuator streams, and filter only input signals whose noise is not physically meaningful.

The key reviewer-facing constraint is:

```text
Do not post-filter the effective wrench merely to make the target easier.
Smooth and time-align the measured signals used by inverse dynamics, then recompute the wrench from rigid-body equations.
```

## Acceptance Gates

The implementation is acceptable only if all gates pass:

1. **No leakage gate:** smoothing, lag selection, normalization, calibration, and model selection do not use validation/test labels to choose final parameters.
2. **No segment-crossing gate:** smoothing, derivatives, filtering, and lag operations never cross `log_id` or `segment_id` boundaries.
3. **Label plausibility gate:** cleaned labels reduce obvious spike/high-frequency artifacts but retain wingbeat-scale phase/frequency structure.
4. **Raw-comparison gate:** every cleaned label variant is reported against the raw label with correlation, RMSE, high-pass energy, p99/p999 magnitude, and top-percentile overlap.
5. **Model/replay gate:** rerun B+C correction and translational oracle replay on the chosen split; report improvements and regressions conservatively.
6. **Moment caution gate:** moment labels are reported separately because rotational oracle replay is not yet closed.

---

## Task 1: Add Reusable Signal-Preprocessing Tests

**Files:**

- Create: `tests/test_signal_preprocessing.py`
- Create later: `src/system_identification/signal_preprocessing.py`

**Step 1: Write failing tests**

Test the following behaviors:

- groupwise smoothing and derivatives do not cross `log_id`;
- Savitzky-Golay derivative recovers the derivative of a quadratic trajectory within tolerance;
- cubic-spline derivative recovers a smooth sinusoid derivative within tolerance;
- low-pass filtering reduces injected high-frequency noise while preserving a low-frequency signal;
- applying a lag shifts a signal within each group and does not wrap samples from one log into another.

Example test intent:

```python
def test_groupwise_lag_does_not_cross_logs():
    frame = make_two_log_frame()
    shifted = apply_groupwise_time_shift(frame, "cmd", lag_s=0.02, group_columns=["log_id"])
    assert shifted.loc[frame["log_id"] == "log_a", "cmd_shifted"].notna().any()
    assert shifted.groupby("log_id")["cmd_shifted"].first().isna().all()
```

**Step 2: Run red test**

```bash
pytest tests/test_signal_preprocessing.py -q
```

Expected: fail because `system_identification.signal_preprocessing` does not exist.

**Step 3: Implement minimal utilities**

Create `src/system_identification/signal_preprocessing.py` with:

```python
groupwise_savgol_derivative(...)
groupwise_cubic_spline_derivative(...)
groupwise_lowpass_filter(...)
apply_groupwise_time_shift(...)
finite_difference_quality_metrics(...)
```

Implementation requirements:

- operate groupwise over `log_id` and, if present, `segment_id`;
- sort by `time_s` inside each group but restore original index order;
- output `NaN` near unsupported edges instead of borrowing samples from neighboring logs;
- expose method metadata: window, polyorder, cutoff, lag, filter order.

**Step 4: Run green test**

```bash
pytest tests/test_signal_preprocessing.py -q
```

Expected: pass.

**Reason:** Most wrench spikes originate from differentiated kinematic states. A tested signal-processing layer prevents ad-hoc smoothing from being embedded directly in one-off scripts.

**Method basis:** Savitzky-Golay preserves low-order polynomial motion while estimating derivatives; cubic splines provide differentiable trajectory fits; groupwise processing prevents artificial continuity across log boundaries.

---

## Task 2: Implement State-Smoothing and Derivative Variants

**Files:**

- Modify: `src/system_identification/pipeline.py`
- Modify: `tests/test_signal_preprocessing.py`
- Create: `scripts/build_time_aligned_smoothed_label_split.py`

**Step 1: Add tests**

Add tests that build a small synthetic split and assert:

- raw, SG, and spline variants produce expected derivative columns;
- smoothed acceleration columns are used only when requested;
- the output split preserves train/val/test row counts and log manifests;
- manifest records derivative method and parameters.

**Step 2: Implement derivative variant API**

Add a wrapper:

```python
compute_kinematic_derivatives(
    samples,
    method: Literal["raw", "savgol", "cubic_spline"],
    window_s: float,
    polyorder: int,
    group_columns: list[str],
) -> pd.DataFrame
```

Keep the existing `compute_smoothed_kinematic_derivatives(...)` as a compatibility wrapper around the Savitzky-Golay path.

**Step 3: Implement split builder skeleton**

Create `scripts/build_time_aligned_smoothed_label_split.py` with CLI:

```bash
python scripts/build_time_aligned_smoothed_label_split.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1 \
  --artifact-dir artifacts/20260528_smoothed_time_aligned_wrench_labels_v1 \
  --derivative-method savgol \
  --window-s 0.12 \
  --polyorder 2
```

**Step 4: Recompute labels**

For each split:

- load `{split}_samples.parquet`;
- compute smoothed derivatives groupwise;
- call `_compute_effective_wrench_labels(...)` with selected acceleration columns;
- write `fx_b, fy_b, fz_b, mx_b, my_b, mz_b`;
- add provenance columns such as:

```text
label_variant
linear_derivative_source
angular_derivative_source
label_reconstruction_valid
```

**Step 5: Run tests**

```bash
pytest tests/test_signal_preprocessing.py tests/test_build_lateral_smoothed_label_split.py tests/test_target_generation_sensitivity.py -q
```

Expected: pass.

**Reason:** This creates the basic “smooth first, then compute wrench” path.

**Method basis:** This is the direct analog of NeuroBEM's spline/smoothed derivative strategy, adapted to onboard PX4 estimates.

---

## Task 3: Add Train-Only Lag Sweep Diagnostics

**Files:**

- Modify: `scripts/build_time_aligned_smoothed_label_split.py`
- Create: `tests/test_wrench_label_lag_sweep.py`

**Step 1: Add tests**

Test that:

- lag candidates are evaluated only on `train_samples.parquet`;
- selected lags are written to `lag_selection.json`;
- validation/test are transformed using the train-selected lags only;
- groupwise lagging creates `NaN` edges instead of circular shifts.

**Step 2: Implement lag candidate config**

Support a small YAML or JSON config:

```json
{
  "phase_raw_rad": [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04],
  "flap_frequency_hz": [-0.04, -0.02, 0.0, 0.02, 0.04],
  "airspeed_validated.true_airspeed_m_s": [-0.10, -0.05, 0.0, 0.05, 0.10],
  "servo_left_elevon": [-0.08, -0.04, 0.0, 0.04, 0.08],
  "servo_right_elevon": [-0.08, -0.04, 0.0, 0.04, 0.08],
  "servo_rudder": [-0.08, -0.04, 0.0, 0.04, 0.08]
}
```

**Step 3: Implement objective metrics**

For each candidate lag, compute train-only diagnostics:

- calibrated-prior residual RMSE if DeLaurier prior columns are available;
- correlation between phase-binned residual and wingbeat phase;
- high-pass energy and p99 jump metrics of reconstructed labels;
- optional translational one-step acceleration consistency.

Do not automatically pick a lag from validation/test. The final selection policy should be:

```text
Choose the smallest-magnitude lag that improves train objective materially and does not create worse label-quality diagnostics.
```

Write:

```text
artifacts/.../lag_sweep_train_metrics.csv
artifacts/.../lag_selection.json
```

**Step 4: Apply selected lags**

Create shifted feature columns:

```text
phase_raw_rad_aligned
phase_raw_unwrapped_rad_aligned
flap_frequency_hz_aligned
true_airspeed_m_s_aligned
servo_left_elevon_aligned
servo_right_elevon_aligned
servo_rudder_aligned
```

Keep original columns unchanged.

**Step 5: Run tests**

```bash
pytest tests/test_wrench_label_lag_sweep.py -q
```

Expected: pass.

**Reason:** Time misalignment produces artificial residual phase shifts and force/moment spikes even when each sensor is individually reasonable.

**Method basis:** NeuroBEM estimates offset/clock skew through correlation between IMU gyro and spline-derived angular rate. Our available data are different, so a train-only lag sweep is the safer equivalent.

---

## Task 4: Add Actuator, Frequency, and Airdata Filtering

**Files:**

- Modify: `src/system_identification/signal_preprocessing.py`
- Modify: `scripts/build_time_aligned_smoothed_label_split.py`
- Create: `tests/test_wrench_label_input_filtering.py`

**Step 1: Add tests**

Test:

- low-pass filtering reduces high-frequency noise in synthetic actuator command;
- filtering is groupwise and does not cross logs;
- filter settings are recorded in the manifest;
- original raw columns remain available.

**Step 2: Implement filter config**

Support:

```json
{
  "flap_frequency_hz": {"method": "butterworth", "order": 2, "cutoff_hz": 12.0},
  "airspeed_validated.true_airspeed_m_s": {"method": "butterworth", "order": 2, "cutoff_hz": 5.0},
  "servo_left_elevon": {"method": "first_order", "time_constant_s": 0.04},
  "servo_right_elevon": {"method": "first_order", "time_constant_s": 0.04},
  "servo_rudder": {"method": "first_order", "time_constant_s": 0.04}
}
```

Use conservative defaults. Treat them as diagnostics first, not final truth.

**Step 3: Implement filtered columns**

Add columns:

```text
flap_frequency_hz_filt
true_airspeed_m_s_filt
servo_left_elevon_filt
servo_right_elevon_filt
servo_rudder_filt
```

Do not overwrite raw columns.

**Step 4: Causality note**

The builder should mark each filter as:

```text
offline_zero_phase: false
causal_deployable: true/false
```

If using `filtfilt`, it is acceptable only for offline label diagnostics, not for deployable model features. Prefer causal `lfilter` or first-order actuator response for columns later used by a deployable predictor.

**Step 5: Run tests**

```bash
pytest tests/test_wrench_label_input_filtering.py tests/test_signal_preprocessing.py -q
```

Expected: pass.

**Reason:** Noisy frequency, airspeed, and actuator signals can make the simulator prior and residual features jitter even if the reconstructed force label is reasonable.

**Method basis:** NeuroBEM low-pass filters motor speed according to motor response. For flapping-wing logs, the analogous inputs are flapping frequency, airspeed, and control-surface commands.

---

## Task 5: Generalize Six-Axis Label-Quality Diagnostics

**Files:**

- Create: `scripts/diagnose_wrench_label_preprocessing.py`
- Create: `tests/test_wrench_label_preprocessing_diagnostics.py`

**Step 1: Add tests**

Synthetic tests should verify:

- metrics are computed for all six channels;
- high-pass energy decreases for a noisy variant;
- top-percentile overlap detects whether spikes are preserved or removed;
- report generation includes raw-vs-clean comparisons.

**Step 2: Implement metrics**

For each split and each target channel:

```text
std_raw
std_clean
rmse_clean_vs_raw
corr_clean_vs_raw
p95_abs_raw / p95_abs_clean
p99_abs_raw / p99_abs_clean
p999_abs_raw / p999_abs_clean
max_abs_raw / max_abs_clean
sample_to_sample_jump_p99_raw / clean
highpass_energy_fraction_raw / clean
top1pct_overlap_fraction
top5pct_overlap_fraction
finite_ratio
```

Use cutoff bands:

```text
8 Hz, 12 Hz, 20 Hz
```

because the flapping fundamental is near the operating flapping frequency and should not be accidentally erased.

**Step 3: Add per-log summaries**

Write:

```text
artifacts/.../label_quality_by_split_channel.csv
artifacts/.../label_quality_by_log_channel.csv
artifacts/.../label_quality_report.md
```

**Step 4: Add simple plots**

Use matplotlib only if already available. Generate:

```text
figures/label_quality_timeseries_example.pdf
figures/label_quality_spectrum_force.pdf
figures/label_quality_channel_boxplot.pdf
```

If plotting dependency fails, still write CSV/Markdown.

**Step 5: Run tests**

```bash
pytest tests/test_wrench_label_preprocessing_diagnostics.py -q
```

Expected: pass.

**Reason:** A smoother label is not automatically better. The diagnostics must show that the pipeline removes implausible derivative artifacts without destroying repeatable wingbeat structure.

**Method basis:** This is a quantitative version of the reviewer-facing “label credibility” argument: high-frequency spike reduction plus raw-label agreement and per-log stability.

---

## Task 6: Build the Candidate Clean Split

**Files:**

- Use: `scripts/build_time_aligned_smoothed_label_split.py`
- Use: `scripts/diagnose_wrench_label_preprocessing.py`
- Create/update: `docs/results/2026-05-28-smoothed-time-aligned-wrench-labels.md`

**Step 1: Run candidate build**

```bash
python scripts/build_time_aligned_smoothed_label_split.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1 \
  --artifact-dir artifacts/20260528_smoothed_time_aligned_wrench_labels_v1 \
  --derivative-method savgol \
  --window-s 0.12 \
  --polyorder 2
```

Expected:

```text
dataset_manifest.json written
train/val/test parquet written
lag_selection.json written if lag sweep enabled
```

**Step 2: Run diagnostics**

```bash
python scripts/diagnose_wrench_label_preprocessing.py \
  --raw-split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --clean-split-root dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1 \
  --output-dir artifacts/20260528_smoothed_time_aligned_wrench_labels_v1/label_quality
```

Expected:

```text
label_quality_by_split_channel.csv
label_quality_by_log_channel.csv
label_quality_report.md
```

**Step 3: Write result report**

Create:

```text
docs/results/2026-05-28-smoothed-time-aligned-wrench-labels.md
```

The report should include:

- exact input split and output split;
- derivative method and parameters;
- whether lag/filtering was enabled;
- label-quality metrics;
- representative examples;
- decision: use clean split, keep raw split, or run another variant.

**Reason:** This creates a reviewable artifact trail rather than a hidden preprocessing change.

**Method basis:** Reproducible preprocessing with a manifest is necessary because label reconstruction choices directly affect all downstream metrics.

---

## Task 7: Rerun B+C Correction on the Clean Split

**Files:**

- Use: `scripts/train_phase_structured_wrench_correction.py`
- Create/update: `docs/results/2026-05-28-smoothed-time-aligned-wrench-labels.md`

**Step 1: Run B+C on clean split**

```bash
python scripts/train_phase_structured_wrench_correction.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-dir artifacts/20260528_smoothed_time_aligned_wrench_labels_v1/phase_structured_correction
```

Use the actual CLI flags from the script if they differ; record the exact command in the report.

**Step 2: Compare with previous B+C artifact**

Compare against:

```text
artifacts/20260527_phase_structured_wrench_correction_v1
```

Report:

```text
force_mean RMSE/R2
moment_mean RMSE/R2
per-channel RMSE/MAE/bias/R2
selected force variant
selected moment variant
```

**Step 3: Interpret conservatively**

Use this decision logic:

- If RMSE improves and raw-vs-clean correlation remains high, the new labels likely reduce derivative artifacts.
- If RMSE improves but raw-vs-clean correlation drops sharply, suspect over-smoothing.
- If force improves but moment remains weak, keep the moment-caution framing.
- If performance is unchanged, the current model is probably limited by excitation/envelope/model structure, not only label spikes.

**Reason:** The point of cleaner labels is not just prettier signals; it should improve or clarify downstream correction behavior.

**Method basis:** A physics-guided residual model should benefit from less noisy supervised targets, but only if true dynamic content is preserved.

---

## Task 8: Rerun Translational Replay Sanity Check

**Files:**

- Use: `scripts/evaluate_short_horizon_replay.py`
- Create/update: `docs/results/2026-05-28-smoothed-time-aligned-wrench-labels.md`

**Step 1: Run oracle translational/local replay**

```bash
python scripts/evaluate_short_horizon_replay.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-dir artifacts/20260528_smoothed_time_aligned_wrench_labels_v1/oracle_replay
```

Use the actual CLI flags from the script if they differ.

**Step 2: Compare with previous oracle replay**

Compare against:

```text
artifacts/20260527_short_horizon_replay_v1/oracle_sanity
```

Focus on:

```text
0.10 s / 0.25 s / 0.50 s position error
0.10 s / 0.25 s / 0.50 s velocity error
```

Do not interpret full 6-DoF model comparison until rotational oracle diagnostics pass.

**Reason:** If the reconstructed force label is physically useful, local translational replay should remain plausible or improve.

**Method basis:** Replay is a stronger sanity check than samplewise RMSE because it tests whether the reconstructed force integrates consistently with motion.

---

## Task 9: Verification, Report, and Commit

**Files:**

- Update: `docs/results/2026-05-28-smoothed-time-aligned-wrench-labels.md`
- Modify/create as needed:
  - `src/system_identification/signal_preprocessing.py`
  - `src/system_identification/pipeline.py`
  - `scripts/build_time_aligned_smoothed_label_split.py`
  - `scripts/diagnose_wrench_label_preprocessing.py`
  - tests from previous tasks

**Step 1: Run unit tests**

```bash
pytest \
  tests/test_signal_preprocessing.py \
  tests/test_wrench_label_lag_sweep.py \
  tests/test_wrench_label_input_filtering.py \
  tests/test_wrench_label_preprocessing_diagnostics.py \
  tests/test_build_lateral_smoothed_label_split.py \
  tests/test_target_generation_sensitivity.py \
  tests/test_phase_structured_wrench_correction.py \
  tests/test_short_horizon_replay.py \
  -q
```

Expected: pass.

**Step 2: Compile scripts**

```bash
python -m py_compile \
  src/system_identification/signal_preprocessing.py \
  scripts/build_time_aligned_smoothed_label_split.py \
  scripts/diagnose_wrench_label_preprocessing.py
```

Expected: no output and exit code 0.

**Step 3: Review git diff**

```bash
git status --short
git diff -- src/system_identification scripts tests docs/results
```

Confirm generated datasets/artifacts are not accidentally staged.

**Step 4: Commit code and reports only**

```bash
git add \
  src/system_identification/signal_preprocessing.py \
  src/system_identification/pipeline.py \
  scripts/build_time_aligned_smoothed_label_split.py \
  scripts/diagnose_wrench_label_preprocessing.py \
  tests/test_signal_preprocessing.py \
  tests/test_wrench_label_lag_sweep.py \
  tests/test_wrench_label_input_filtering.py \
  tests/test_wrench_label_preprocessing_diagnostics.py \
  docs/results/2026-05-28-smoothed-time-aligned-wrench-labels.md

git commit -m "feat: add smoothed time-aligned wrench label pipeline"
```

**Reason:** The implementation changes the foundation of the supervised target. It needs a clear commit and reproducible report.

---

## Final Paper Interpretation Rules

If the pipeline improves label quality and downstream force correction:

```text
We reconstruct effective wrench from smoothed and time-aligned onboard state estimates, following the principle of estimating derivatives from filtered trajectory signals rather than post-filtering the wrench labels.
```

Avoid:

```text
We filter the wrench labels to make training easier.
```

If moment replay remains unresolved:

```text
The force channels are used for simulator-prior correction analysis, while moment channels are reported as lower-confidence effective rotational-wrench targets due to angular-acceleration, inertia, and reference-point sensitivity.
```

If translational replay improves but full 6-DoF replay still fails:

```text
The result supports local translational consistency of the reconstructed force labels, but does not yet constitute closed-loop six-degree-of-freedom simulator validation.
```

