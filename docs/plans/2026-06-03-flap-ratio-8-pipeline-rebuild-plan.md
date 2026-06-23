# Flap Ratio 8 Pipeline Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the flight-log, PX4, IsaacLab DeLaurier-prior, correction-model, and paper-result pipeline from the obsolete 7.5 motor-rev-per-flap ratio to the confirmed 8.0 ratio, then rerun the remaining checks needed before using the results in the AeroConf paper.

**Architecture:** Treat the aircraft metadata and PX4 `FLAP_RATIO` parameter as the source of truth for encoder-derived phase/frequency. Keep old datasets and artifacts immutable; write new ratio-8 datasets and artifacts with explicit versioned names. Make each generated artifact self-auditing by recording phase source, frequency source, ratio, input split, prior root, and model root.

**Tech Stack:** PX4 C/C++, Python 3.10 in `flap-train-gpu`, IsaacLab Python in `env_isaaclab`, pandas/parquet, pytest, LaTeX paper artifacts.

---

## Context and Non-Negotiable Decisions

- Confirmed mechanical ratio: `encoder_to_drive_ratio = 8.0`.
- Old ratio: `7.5`; any result generated from it is stale unless explicitly labeled as historical.
- Current paper-critical stale inputs include:
  - `dataset/canonical_v0.2_training_ready_split_measured_massprops_v1`
  - `dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1`
  - `dataset/delaurier_force_prior_moment_direct_measured_massprops_v1`
  - `artifacts/20260602_delaurier_prior_measured_massprops_v1`
  - `artifacts/20260603_delaurier_prior_measured_massprops_key_aligned_v1`
  - `artifacts/20260603_delaurier_force_recalibration_measured_massprops_key_aligned_v1`
  - `artifacts/20260602_bc_correction_measured_massprops_v1`
  - `artifacts/20260602_residual_diagnostics_measured_massprops_v1`
  - `artifacts/20260602_replay_measured_massprops_v1`
- Do not overwrite old artifacts. New outputs use `ratio8_v1` or `20260603_ratio8_*`.
- Canonical frequency must not blindly trust old `flap_frequency` topic values. The old topic matches `encoder_rpm_est / (60 * 7.5)`. New canonical frequency should prefer Hall-indexed `wing_phase.flap_frequency_hz` if present and valid, otherwise compute `abs(encoder_rpm_est) / (60 * metadata_ratio)`, otherwise fall back to the logged `flap_frequency` topic with a warning/source label.
- Canonical mechanical phase should prefer Hall-indexed `wing_phase.phase_rad` when present and valid. Encoder-derived `drive_phase_rad` remains useful as a diagnostic/fallback but is not the first-choice mechanical phase for DeLaurier export.
- DeLaurier log export must use the real-flight sine convention: `q = A sin(phi)`, `qdot = A omega cos(phi)`, `qddot = -A omega^2 sin(phi)`, where `phi = 0` is neutral wing position starting upstroke.

## Versioned Outputs

Use these new paths:

- Seed labels: `dataset/canonical_v0.2_seed_labels_measured_massprops_ratio8_v1`
- Whole-log split: `dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_v1`
- Smoothed/time-aligned split: `dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1`
- Phase/frequency audit: `artifacts/20260603_ratio8_phase_frequency_audit_v1`
- DeLaurier prior: `artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1`
- Prior convention diagnostics: `artifacts/20260603_delaurier_prior_convention_diagnostics_ratio8_v1`
- Force recalibration: `artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1`
- Residual split: `dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1`
- B+C correction: `artifacts/20260603_bc_correction_measured_massprops_ratio8_v1`
- Residual diagnostics: `artifacts/20260603_residual_diagnostics_measured_massprops_ratio8_v1`
- Replay diagnostics: `artifacts/20260603_replay_measured_massprops_ratio8_v1`
- Artifact lineage audit: `artifacts/20260603_ratio8_lineage_audit_v1`

---

### Task 1: Add a Cross-Repo PX4 Ratio Contract Test

**Files:**
- Create: `/home/zn/flap-system-identification/tests/test_px4_flap_ratio_contract.py`
- Read: `/home/zn/PX4-Autopilot/src/modules/rpm_pid/rpm_pid_params.c`
- Read: `/home/zn/PX4-Autopilot/src/drivers/encoder/as5600/AS5600.hpp`
- Read: `/home/zn/PX4-Autopilot/src/modules/wing_phase/WingPhase.cpp`

**Step 1: Write the failing test**

Create `tests/test_px4_flap_ratio_contract.py`:

```python
from pathlib import Path


PX4_ROOT = Path("/home/zn/PX4-Autopilot")


def test_px4_default_flap_ratio_is_current_aircraft_ratio():
    rpm_params = (PX4_ROOT / "src/modules/rpm_pid/rpm_pid_params.c").read_text()
    as5600_hpp = (PX4_ROOT / "src/drivers/encoder/as5600/AS5600.hpp").read_text()
    wing_phase_cpp = (PX4_ROOT / "src/modules/wing_phase/WingPhase.cpp").read_text()

    assert "PARAM_DEFINE_FLOAT(FLAP_RATIO, 8.0f);" in rpm_params
    assert "_flap_ratio{8.0f}" in as5600_hpp
    assert "_counts_per_cycle{kCountsPerRevolution * 8.0f}" in wing_phase_cpp
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_px4_flap_ratio_contract.py -q -o cache_dir=/tmp/pytest_cache_px4_ratio8
```

Expected: FAIL because PX4 still contains `7.5f`.

**Step 3: Update PX4 defaults**

Modify:

- `/home/zn/PX4-Autopilot/src/modules/rpm_pid/rpm_pid_params.c`
  - Change `PARAM_DEFINE_FLOAT(FLAP_RATIO, 7.5f);` to `PARAM_DEFINE_FLOAT(FLAP_RATIO, 8.0f);`
- `/home/zn/PX4-Autopilot/src/drivers/encoder/as5600/AS5600.hpp`
  - Change `_flap_ratio{7.5f};` to `_flap_ratio{8.0f};`
- `/home/zn/PX4-Autopilot/src/modules/wing_phase/WingPhase.cpp`
  - Change `_counts_per_cycle{kCountsPerRevolution * 7.5f};` to `_counts_per_cycle{kCountsPerRevolution * 8.0f};`

Do not change unrelated PX4 files.

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_px4_flap_ratio_contract.py -q -o cache_dir=/tmp/pytest_cache_px4_ratio8
```

Expected: PASS.

**Step 5: Operational check**

Record in the final report that source defaults are updated but aircraft-side parameter state still must be verified on hardware/logs:

```bash
param show FLAP_RATIO
```

Expected on aircraft: `FLAP_RATIO = 8.0`.

---

### Task 2: Finish System-Identification Metadata and Unit-Test Updates

**Files:**
- Modify: `/home/zn/flap-system-identification/metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Modify: `/home/zn/flap-system-identification/tests/test_metadata.py`
- Modify: `/home/zn/flap-system-identification/tests/test_phase.py`
- Modify: `/home/zn/flap-system-identification/tests/test_pipeline.py`

**Step 1: Verify metadata ratio is 8.0**

Expected metadata block:

```yaml
encoder_to_drive_ratio:
  value: 8.0
  unit: encoder_rad_per_drive_rad
  status: confirmed
  source: user_confirmed_updated_mechanical_ratio
```

**Step 2: Update current-aircraft tests**

- `tests/test_metadata.py` should assert `8.0`.
- Tests that are generic formula tests may use arbitrary ratios, but tests that represent `flapper_01` should use `8.0`.
- In `tests/test_phase.py`, keep one generic ratio test if useful, and add a current-aircraft example:

```python
def test_current_aircraft_encoder_ratio_eight_maps_encoder_to_drive_phase():
    encoder_phase_unwrapped = np.array([0.0, 8.0 * np.pi / 2.0, 8.0 * np.pi])
    drive_unwrapped, drive_wrapped = compute_drive_phase_rad(
        encoder_phase_unwrapped_rad=encoder_phase_unwrapped,
        encoder_to_drive_ratio=8.0,
        encoder_to_drive_sign=1.0,
        drive_phase_zero_offset_rad=0.0,
    )
    np.testing.assert_allclose(drive_unwrapped, [0.0, np.pi / 2.0, np.pi])
```

**Step 3: Run focused tests**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_metadata.py tests/test_phase.py tests/test_pipeline.py -q -o cache_dir=/tmp/pytest_cache_ratio8_core
```

Expected: PASS.

---

### Task 3: Make Canonical Phase/Frequency Selection Explicit in the Pipeline

**Files:**
- Modify: `/home/zn/flap-system-identification/src/system_identification/pipeline.py`
- Modify: `/home/zn/flap-system-identification/src/system_identification/phase.py` only if a helper is needed
- Modify: `/home/zn/flap-system-identification/tests/test_pipeline.py`

**Step 1: Write failing tests**

Add tests to `tests/test_pipeline.py` covering:

1. When `wing_phase` has valid `phase_rad`, canonical `phase_raw_rad` and `mechanical_phase_rad` use `wing_phase.phase_rad`.
2. When `wing_phase` has `flap_frequency_hz`, canonical `flap_frequency_hz` uses it.
3. When `wing_phase.flap_frequency_hz` is absent but `rpm.rpm_estimate` is present, canonical `flap_frequency_hz` equals `abs(encoder_rpm_est) / (60 * encoder_to_drive_ratio)`.
4. The logged `flap_frequency` topic is preserved as `flap_frequency_topic_hz` and does not silently override the ratio-8 canonical frequency.
5. `wing_stroke_angle_rad` uses the selected mechanical phase, not the stale encoder fallback when `wing_phase` is present.

Minimal expected assertions:

```python
assert samples["phase_source"].eq("wing_phase").all()
assert "flap_frequency_hz_source" in samples.columns
assert samples["flap_frequency_hz_source"].eq("wing_phase.flap_frequency_hz").all()
assert "flap_frequency_topic_hz" in samples.columns
np.testing.assert_allclose(samples["wing_stroke_angle_rad"], amplitude * np.sin(samples["phase_raw_rad"]))
```

**Step 2: Run tests to verify failure**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_pipeline.py -q -o cache_dir=/tmp/pytest_cache_pipeline_ratio8
```

Expected: FAIL because the current pipeline writes `flap_frequency_hz` directly from the `flap_frequency` topic and computes `wing_stroke_angle_rad` before canonical phase selection.

**Step 3: Implement canonical selection**

In `src/system_identification/pipeline.py`:

- Preserve logged topic:

```python
samples["flap_frequency_topic_hz"] = linear_resample(...)
```

- Preserve encoder rpm:

```python
samples["encoder_rpm_est"] = ...
```

- Choose canonical frequency:

```python
if wing_phase_frame has "flap_frequency_hz":
    samples["flap_frequency_hz"] = zoh_resampled_wing_phase_frequency
    samples["flap_frequency_hz_source"] = "wing_phase.flap_frequency_hz"
elif encoder_rpm_est exists:
    samples["flap_frequency_hz"] = np.abs(samples["encoder_rpm_est"]) / (60.0 * encoder_to_drive_ratio)
    samples["flap_frequency_hz_source"] = "encoder_rpm_est_metadata_ratio"
elif logged flap_frequency topic exists:
    samples["flap_frequency_hz"] = samples["flap_frequency_topic_hz"]
    samples["flap_frequency_hz_source"] = "flap_frequency_topic_fallback_unverified"
else:
    samples["flap_frequency_hz"] = np.nan
    samples["flap_frequency_hz_source"] = "missing"
```

- Add explicit mechanical phase columns:

```python
samples["mechanical_phase_rad"] = phase_raw
samples["mechanical_phase_unwrapped_rad"] = phase_raw_unwrapped
samples["mechanical_phase_source"] = phase_source
```

- Recompute wing stroke after canonical phase selection:

```python
samples["wing_stroke_angle_rad"] = compute_wing_stroke_angle_rad(
    drive_phase_rad=samples["mechanical_phase_rad"].to_numpy(),
    wing_stroke_amplitude_rad=...,
    wing_stroke_phase_offset_rad=...,
)
```

Keep `drive_phase_rad` and `drive_phase_unwrapped_rad` as encoder-derived diagnostic/fallback columns.

**Step 4: Run focused tests**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_pipeline.py tests/test_phase.py -q -o cache_dir=/tmp/pytest_cache_pipeline_ratio8
```

Expected: PASS.

---

### Task 4: Add a Ratio-8 Phase/Frequency Audit Script

**Files:**
- Create: `/home/zn/flap-system-identification/scripts/audit_ratio8_phase_frequency.py`
- Create: `/home/zn/flap-system-identification/tests/test_audit_ratio8_phase_frequency.py`

**Step 1: Write failing tests**

Create a small synthetic frame with:

- `log_id`
- `time_s`
- `encoder_phase_unwrapped_rad`
- `wing_phase.phase_rad`
- `flap_frequency_hz`
- `encoder_rpm_est`

Assert the audit:

- Fits `wing_phase.phase_rad ~= encoder_phase_unwrapped_rad / ratio + per-log offset`.
- Reports median circular residual.
- Reports frequency consistency between `flap_frequency_hz` and `encoder_rpm_est / (60 * ratio)`.
- Writes `phase_offset_by_log.csv`, `frequency_consistency.csv`, and `summary.json`.

**Step 2: Implement script**

CLI:

```bash
python scripts/audit_ratio8_phase_frequency.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --ratio 8.0 \
  --output-root artifacts/20260603_ratio8_phase_frequency_audit_v1
```

Script behavior:

- Read `train_samples.parquet`, `val_samples.parquet`, `test_samples.parquet`.
- Per log, compute circular offset between `wing_phase.phase_rad` and `encoder_phase_unwrapped_rad / 8.0`.
- Per log, compute derivative-based self-unwrapped `wing_phase` frequency.
- Compare canonical `flap_frequency_hz` against:
  - self-unwrapped `wing_phase`
  - `encoder_rpm_est / (60 * 8.0)`
  - old logged topic if `flap_frequency_topic_hz` exists
- Write summary with pass/fail gates.

Recommended gates:

- Median per-log circular resultant `R > 0.98`.
- Median circular residual `< 0.20 rad`.
- Median canonical frequency ratio to `encoder_rpm_est/(60*8)` within `[0.98, 1.02]` unless `wing_phase` derivative is the chosen source.
- No canonical `flap_frequency_hz_source == "flap_frequency_topic_fallback_unverified"` for paper-critical logs unless explicitly justified.

**Step 3: Run tests**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_audit_ratio8_phase_frequency.py -q -o cache_dir=/tmp/pytest_cache_ratio8_audit
```

Expected: PASS.

---

### Task 5: Regenerate Ratio-8 Seed Dataset, Split, and Smoothed Split

**Files/Outputs:**
- Input: `dataset/canonical_v0.2_seed_labels_measured_massprops_v1/accepted_logs.csv`
- Input: `metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Output: `dataset/canonical_v0.2_seed_labels_measured_massprops_ratio8_v1`
- Output: `dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_v1`
- Output: `dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1`

**Step 1: Regenerate seed labels**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/regenerate_canonical_from_accepted_logs.py \
  --accepted-logs-csv dataset/canonical_v0.2_seed_labels_measured_massprops_v1/accepted_logs.csv \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_seed_labels_measured_massprops_ratio8_v1 \
  --rate-hz 100 \
  --overwrite
```

Expected:

- `dataset_manifest.json` exists.
- `accepted_logs.csv` count matches old measured-mass seed dataset.
- New sample parquets contain `encoder_to_drive_ratio`-consistent `drive_phase_rad`, `mechanical_phase_rad`, and `flap_frequency_hz_source`.

**Step 2: Materialize the same whole-log assignment**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/materialize_split_from_log_assignment.py \
  --source-manifest dataset/canonical_v0.2_seed_labels_measured_massprops_ratio8_v1/dataset_manifest.json \
  --assignment-csv dataset/canonical_v0.2_training_ready_split_measured_massprops_v1/all_logs.csv \
  --output dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_v1 \
  --altitude-window-min-m 5 \
  --overwrite
```

Expected:

- Same train/val/test log IDs as old measured-mass split.
- Row counts may differ slightly only if the regenerated pipeline changes validity masks; any difference must be reported.

**Step 3: Build SG 0.03 smoothed/time-aligned split**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/build_time_aligned_smoothed_label_split.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --artifact-dir artifacts/20260603_ratio8_time_alignment_sg0p03_v1 \
  --derivative-method savgol \
  --window-s 0.03 \
  --polyorder 3 \
  --force-label-source smooth \
  --moment-label-source smooth \
  --enable-input-filtering
```

Expected:

- `train_samples.parquet`, `val_samples.parquet`, `test_samples.parquet` exist.
- Smoothing config is written under `artifacts/20260603_ratio8_time_alignment_sg0p03_v1`.

**Step 4: Run phase/frequency audit**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_ratio8_phase_frequency.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --ratio 8.0 \
  --output-root artifacts/20260603_ratio8_phase_frequency_audit_v1
```

Expected: audit passes or produces a blocking reason.

---

### Task 6: Restore or Create a Reproducible IsaacLab DeLaurier Prior Export

**Files:**
- Create or restore: `/home/zn/IsaacLab/scripts/flapping_px4/export_delaurier_prior_predictions.py`
- Create: `/home/zn/IsaacLab/tests/test_delaurier_log_export_phase_contract.py`
- Read/Use: `/home/zn/IsaacLab/source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Input split: `/home/zn/flap-system-identification/dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1`
- Output prior: `/home/zn/flap-system-identification/artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1`

**Step 1: Write failing contract tests**

The test should assert that the export script:

- Reads `mechanical_phase_rad` or `wing_phase.phase_rad` before falling back to `drive_phase_rad`.
- Reads canonical `flap_frequency_hz`.
- Uses sine convention:

```python
q = amp * sin(phi)
qd = amp * omega * cos(phi)
qdd = -amp * omega * omega * sin(phi)
```

- Writes keys: `log_id`, `time_s`, `timestamp_us` when available.
- Writes a manifest recording:
  - `phase_column`
  - `frequency_column`
  - `encoder_to_drive_ratio = 8.0`
  - `source_split_root`
  - IsaacLab git commit or dirty status if available

**Step 2: Run test to verify failure**

Run:

```bash
cd /home/zn/IsaacLab
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/test_delaurier_log_export_phase_contract.py -q -o cache_dir=/tmp/pytest_cache_delaurier_export_ratio8
```

Expected: FAIL until export script exists and obeys the contract.

**Step 3: Implement or restore export script**

Implement the minimal script that:

- Loads each split parquet.
- Selects phase/frequency columns using the contract.
- Reconstructs `q/qdot/qddot` from the log phase and frequency.
- Calls the DeLaurier model path used by `straight_flight_env.py` or a factored helper.
- Emits `train_predictions.parquet`, `val_predictions.parquet`, `test_predictions.parquet` with sample keys and prior force/moment columns.
- Emits `manifest.json`.

Do not reuse legacy row-order-only exports.

**Step 4: Run export contract tests**

Run:

```bash
cd /home/zn/IsaacLab
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/test_delaurier_log_export_phase_contract.py tests/test_delaurier_convention_contract.py -q -o cache_dir=/tmp/pytest_cache_delaurier_export_ratio8
```

Expected: PASS.

**Step 5: Export ratio-8 prior**

Run:

```bash
cd /home/zn/IsaacLab
./isaaclab.sh -p scripts/flapping_px4/export_delaurier_prior_predictions.py \
  --split-root /home/zn/flap-system-identification/dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --metadata /home/zn/flap-system-identification/metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-root /home/zn/flap-system-identification/artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1 \
  --overwrite
```

Expected:

- Prediction parquet row counts match input splits.
- Prediction parquets include keys for key-based joins.
- Manifest records ratio 8.0 and phase/frequency source.

---

### Task 7: Diagnose and Recalibrate the Ratio-8 DeLaurier Prior

**Files/Outputs:**
- Input split: `dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1`
- Input prior: `artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1`
- Output diagnostics: `artifacts/20260603_delaurier_prior_convention_diagnostics_ratio8_v1`
- Output calibration: `artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1`

**Step 1: Run convention diagnostics**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/diagnose_delaurier_prior_conventions.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --prior-root artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1 \
  --output-root artifacts/20260603_delaurier_prior_convention_diagnostics_ratio8_v1
```

Expected:

- No row-order fallback.
- Summary reports raw prior, signed permutations, and per-channel affine fits.
- If raw prior remains nearly uncorrelated, do not proceed as if the physics prior is validated; document this as a model/convention issue.

**Step 2: Run train-only force recalibration**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_delaurier_force_recalibration.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --prior-root artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1 \
  --output-root artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1 \
  --channel-weights fx_b=1,fy_b=1,fz_b=1
```

Expected:

- `parameters.csv`, `metrics_by_split.csv`, and calibrated prior parquets exist.
- Affine gains are reported clearly. Very small gains are not automatically a failure, but they must be interpreted as weak raw-prior utility.

---

### Task 8: Build the Ratio-8 Residual Split

**Files/Outputs:**
- Input split: `dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1`
- Input calibrated prior: `artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1`
- Output residual split: `dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1`

**Step 1: Build split with key-based joins only**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/build_delaurier_residual_split.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --prior-root artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1 \
  --output-root dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1 \
  --prior-name A1_per_channel_affine
```

Do not use `--allow-row-order-fallback`.

Expected:

- Residual split has same split names and sample keys.
- No key mismatch.
- Prior columns and residual target columns exist.

---

### Task 9: Rerun B+C Correction on Ratio-8 Artifacts

**Files/Outputs:**
- Input residual split: `dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1`
- Input prior: `artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1`
- Output: `artifacts/20260603_bc_correction_measured_massprops_ratio8_v1`

**Step 1: Train force structured correction**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/train_delaurier_greybox_force_correction.py \
  --split-root dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1 \
  --prior-root artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1 \
  --output-root artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/force_v1 \
  --variants additive,multiplicative,affine \
  --alphas 0,0.001,0.01,0.1,1,10,100
```

Expected:

- `metrics_by_split.csv`, `model_selection.csv`, and prediction parquets exist.
- Config records ratio-8 split/prior roots.

**Step 2: Train deployable correction**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/train_deployable_wrench_correction_v2.py \
  --split-root dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1 \
  --prior-root artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1 \
  --force-v1-root artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/force_v1 \
  --moment-v1-root artifacts/20260525_dynamic_arm_moment_head_v1 \
  --output-root artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/deployable_v2 \
  --alphas 0,0.001,0.01,0.1,1,10,100,1000 \
  --overwrite
```

Expected:

- Per-channel metrics exist.
- Model config records ratio-8 roots.
- If `moment-v1-root` is stale relative to the new split, either retrain a ratio-8 moment head or explicitly mark moment-v1 reuse as invalid. Do not silently reuse incompatible moment outputs.

**Step 3: Train phase-structured correction**

Run:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/train_phase_structured_wrench_correction.py \
  --split-root dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1 \
  --prior-root artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1 \
  --v2-reference-root artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/deployable_v2 \
  --output-root artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/phase_structured \
  --alphas 0,0.001,0.01,0.1,1,10,100,1000 \
  --overwrite
```

Expected:

- Prediction parquets exist.
- Config records ratio-8 roots.
- The distinction between deployable correction and phase-structured correction remains clear.

---

### Task 10: Rerun Evaluation, Residual Diagnostics, and Replay Diagnostics

**Files/Outputs:**
- Output evaluation: `artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/evaluation`
- Output residual diagnostics: `artifacts/20260603_residual_diagnostics_measured_massprops_ratio8_v1`
- Output replay diagnostics: `artifacts/20260603_replay_measured_massprops_ratio8_v1`

**Step 1: Recreate per-channel and per-log evaluation**

Use the existing evaluation/alignment scripts if present; if not, create a small evaluator that aligns prediction parquets by keys and writes:

- `per_channel_metrics.csv`
- `per_log_metrics.csv`
- `summary.md`
- `evaluation_manifest.json`
- aligned prediction parquets under `evaluation/aligned/`

Required inputs:

- calibrated prior predictions
- force-v1 predictions
- deployable-v2 predictions
- phase-structured predictions
- true labels from ratio-8 residual split

Expected:

- Metrics use the same rows for all compared models.
- Overall mixed-unit RMSE is not the main paper metric; use per-channel RMSE/MAE/RMSE-over-std.

**Step 2: Rerun residual diagnostics**

Run the existing diagnostics on the new aligned test predictions:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/analyze_delaurier_residual_phase.py \
  --aligned-parquet artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/evaluation/aligned/test_phase_structured_aligned.parquet \
  --output-dir artifacts/20260603_residual_diagnostics_measured_massprops_ratio8_v1/phase
```

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/analyze_delaurier_residual_conditions.py \
  --aligned-parquet artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/evaluation/aligned/test_phase_structured_aligned.parquet \
  --output-dir artifacts/20260603_residual_diagnostics_measured_massprops_ratio8_v1/conditions
```

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/analyze_delaurier_residual_frequency.py \
  --aligned-parquet artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/evaluation/aligned/test_phase_structured_aligned.parquet \
  --output-dir artifacts/20260603_residual_diagnostics_measured_massprops_ratio8_v1/frequency
```

Expected:

- Phase/frequency diagnostics use ratio-8 canonical phase/frequency.
- Figure/table inputs are regenerated; no `20260602` paths.

**Step 3: Rerun replay diagnostics**

Oracle translational replay:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_short_horizon_replay.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --split test \
  --output-root artifacts/20260603_replay_measured_massprops_ratio8_v1/oracle_sanity
```

Rotational diagnostics:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/diagnose_rotational_replay_oracle.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --split test \
  --output-root artifacts/20260603_replay_measured_massprops_ratio8_v1/rotational_oracle_diagnostics
```

Expected:

- Replay remains diagnostic-only.
- Six-DOF replay should not be claimed unless the rotational gates pass.

---

### Task 11: Add Artifact Lineage Audit

**Files:**
- Create: `/home/zn/flap-system-identification/scripts/audit_ratio8_artifact_lineage.py`
- Create: `/home/zn/flap-system-identification/tests/test_audit_ratio8_artifact_lineage.py`
- Output: `artifacts/20260603_ratio8_lineage_audit_v1`

**Step 1: Write failing tests**

Create a test with small JSON/config files that include:

- one valid ratio-8 path
- one invalid `20260602` path
- one invalid `7.5` string

Assert the audit detects the invalid references.

**Step 2: Implement audit script**

CLI:

```bash
python scripts/audit_ratio8_artifact_lineage.py \
  --roots \
    dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
    artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1 \
    artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1 \
    dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1 \
    artifacts/20260603_bc_correction_measured_massprops_ratio8_v1 \
  --forbidden-substrings 20260602,20260603_delaurier_prior_measured_massprops_key_aligned_v1,7.5 \
  --output-root artifacts/20260603_ratio8_lineage_audit_v1
```

Script should inspect JSON, YAML, TXT, MD, CSV configs/manifests, not large parquet binaries.

Expected:

- No stale paths in new ratio-8 artifacts.
- If a historical comparison intentionally mentions old paths, whitelist it explicitly and record why.

---

### Task 12: Update Paper Artifacts Only After Ratio-8 Results Are Verified

**Files:**
- Modify after verification: `/home/zn/paper/AeroConf_effective_aero/sections/02_dataset.tex`
- Modify after verification: `/home/zn/paper/AeroConf_effective_aero/sections/03_method.tex`
- Modify after verification: `/home/zn/paper/AeroConf_effective_aero/sections/04_experiments.tex`
- Modify after verification: `/home/zn/paper/AeroConf_effective_aero/sections/05_results.tex`
- Modify after verification: `/home/zn/paper/AeroConf_effective_aero/sections/06_conclusion.tex`
- Modify figures/tables only after new artifact paths are final.

**Step 1: Mark old paper numbers as stale in local notes**

Create a note:

`/home/zn/paper/AeroConf_effective_aero/research_notes/20260603_ratio8_result_rebuild_status.md`

Include:

- Old `20260602`/`20260603 key aligned` prior artifacts are stale.
- Paper results must not be updated until ratio-8 evaluation is complete.
- Current claim boundary remains log-based correction only.

**Step 2: Update method wording**

After verified:

- State `FLAP_RATIO = 8.0`.
- State phase comes from Hall-indexed `wing_phase` when available.
- State encoder-derived phase/frequency are retained as diagnostics/fallback.
- State smoothing applies only to label reconstruction and not as a network input leakage channel.

**Step 3: Update results**

Use only:

- `artifacts/20260603_bc_correction_measured_massprops_ratio8_v1/evaluation/per_channel_metrics.csv`
- `artifacts/20260603_residual_diagnostics_measured_massprops_ratio8_v1`
- `artifacts/20260603_replay_measured_massprops_ratio8_v1`

Do not cite stale `20260602` metrics.

**Step 4: Compile**

Run:

```bash
cd /home/zn/paper/AeroConf_effective_aero
latexmk -pdf main.tex
```

Expected: PDF compiles with no unresolved citations/refs relevant to changed sections.

---

## Final Verification Checklist

Run these before declaring the ratio-8 rebuild complete:

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest \
  tests/test_metadata.py \
  tests/test_phase.py \
  tests/test_pipeline.py \
  tests/test_px4_flap_ratio_contract.py \
  tests/test_audit_ratio8_phase_frequency.py \
  tests/test_audit_ratio8_artifact_lineage.py \
  tests/test_build_delaurier_residual_split.py \
  tests/test_delaurier_force_recalibration.py \
  -q -o cache_dir=/tmp/pytest_cache_ratio8_final
```

```bash
cd /home/zn/IsaacLab
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_delaurier_log_export_phase_contract.py \
  tests/test_delaurier_convention_contract.py \
  -q -o cache_dir=/tmp/pytest_cache_isaac_ratio8_final
```

```bash
cd /home/zn/flap-system-identification
PYTHONDONTWRITEBYTECODE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_ratio8_artifact_lineage.py \
  --roots \
    dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
    artifacts/20260603_delaurier_prior_measured_massprops_ratio8_v1 \
    artifacts/20260603_delaurier_force_recalibration_measured_massprops_ratio8_v1 \
    dataset/delaurier_force_prior_moment_direct_measured_massprops_ratio8_v1 \
    artifacts/20260603_bc_correction_measured_massprops_ratio8_v1 \
  --forbidden-substrings 20260602,20260603_delaurier_prior_measured_massprops_key_aligned_v1,7.5 \
  --output-root artifacts/20260603_ratio8_lineage_audit_v1
```

Success criteria:

- PX4 source defaults use `8.0`.
- Metadata uses `8.0`.
- New ratio-8 datasets are regenerated from logs, not patched in place.
- Canonical frequency no longer uses old `flap_frequency` topic silently.
- New DeLaurier prior export is reproducible and key-aligned.
- No ratio-8 artifact references stale `20260602` or old key-aligned prior roots.
- B+C correction and diagnostics are rerun on ratio-8 split/prior.
- Paper is updated only from ratio-8 verified metrics.
