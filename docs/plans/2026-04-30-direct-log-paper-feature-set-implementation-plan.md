# Direct Log Paper Feature Set Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a non-leaking paper-style input feature set that uses direct PX4 log fields plus derived attitude/aero features without acceleration or angular acceleration inputs.

**Architecture:** The canonical pipeline will resample extra fields already present in `airspeed_validated`. The training module will derive roll/pitch, body-frame velocity, relative air velocity, alpha/beta, dynamic pressure, and elevon command combinations from existing canonical columns.

**Tech Stack:** Python, pandas, NumPy, PyTorch training code, pytest.

---

### Task 1: Pipeline Direct Airspeed Fields

**Files:**
- Modify: `src/system_identification/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Add a pipeline test that provides an `airspeed_validated` topic with `indicated_airspeed_m_s`, `calibrated_airspeed_m_s`, `true_airspeed_m_s`, `calibrated_ground_minus_wind_m_s`, `true_ground_minus_wind_m_s`, `airspeed_derivative_filtered`, `throttle_filtered`, and `pitch_filtered`.

**Step 2: Run the focused test**

Run: `pytest tests/test_pipeline.py::test_assemble_canonical_samples_resamples_direct_airspeed_fields -q`

Expected: fail because the newly asserted canonical columns are missing.

**Step 3: Implement minimal extraction**

Add the extra `airspeed_validated` fields to the ZOH resampling list.

**Step 4: Verify**

Run the same focused test and confirm it passes.

### Task 2: Training Feature Set

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`
- Modify: `scripts/train_baseline_torch.py`

**Step 1: Write the failing tests**

Add tests that assert `paper_no_accel_v2` excludes acceleration and angular acceleration columns, includes direct airspeed fields, and derives roll/pitch/body velocity/relative air velocity/alpha/beta/dynamic pressure/elevon combinations.

**Step 2: Run focused tests**

Run: `pytest tests/test_training.py::test_paper_no_accel_v2_feature_set_excludes_label_derivative_inputs tests/test_training.py::test_prepare_feature_target_frames_derives_paper_no_accel_v2_features -q`

Expected: fail because the feature set and derived columns do not exist.

**Step 3: Implement minimal training changes**

Add helper derivations in `_with_derived_columns`, add `PAPER_NO_ACCEL_V2_FEATURE_COLUMNS`, register it in `DEFAULT_FEATURE_SETS`, and update CLI help text.

**Step 4: Verify**

Run focused training tests, then full `pytest -q`.

### Task 3: Rebuild Dataset And Smoke Train

**Files:**
- No tracked code changes expected.

**Step 1: Rebuild canonical/split data**

Use accepted HQ log source paths to regenerate canonical samples into a new dataset root, then materialize a whole-log split.

**Step 2: Smoke train**

Run a short baseline with `--feature-set paper_no_accel_v2` on capped samples and inspect `metrics.json`.
