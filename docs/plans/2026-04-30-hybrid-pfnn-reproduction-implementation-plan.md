# Hybrid PFNN Reproduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce the paper-style Hybrid PFNN point-prediction model for equivalent force and moment regression using the current whole-log dataset and non-leaking flight inputs.

**Architecture:** Add a `paper_pfnn_10` feature set matching the paper inputs plus raw phase as a special unscaled feature. Add a PyTorch `HybridPFNNRegressor` with cyclic Catmull-Rom interpolation over six phase control points, a 10-to-45 input expansion, hidden layers of 40 and 40 units, skip concatenation of the original state inputs, and a six-output wrench head.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `scripts/train_baseline_torch.py` training entry point.

---

### Task 1: Paper Feature Set

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing test**

Assert `resolve_feature_set_columns("paper_pfnn_10")` returns raw `phase_corrected_rad` plus the 10 paper-style state/control inputs:
`velocity_b.x/y/z`, `pitch_rad`, `roll_rad`, `alpha_rad`, `beta_rad`, `cycle_flap_frequency_hz`, `elevator_like`, and `servo_rudder`.

**Step 2: Verify RED**

Run: `pytest tests/test_training.py::test_paper_pfnn_10_feature_set_matches_paper_inputs -q`

Expected: fail with unknown feature set or missing constant.

**Step 3: Implement**

Add `PAPER_PFNN_10_FEATURE_COLUMNS` and register `paper_pfnn_10`.

**Step 4: Verify GREEN**

Run the same focused test.

### Task 2: Phase Functioned Model

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add tests for cyclic Catmull-Rom interpolation periodicity and `HybridPFNNRegressor` forward-pass shape/periodicity.

**Step 2: Verify RED**

Run focused tests and confirm import/name failures.

**Step 3: Implement**

Add `cyclic_catmull_rom_weights`, `PhaseFunctionedLinear`, and `HybridPFNNRegressor`.

**Step 4: Verify GREEN**

Run focused tests.

### Task 3: Training Integration

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`
- Test: `tests/test_training.py`

**Step 1: Write failing test**

Add a smoke test that calls `run_training_job(..., feature_set_name="paper_pfnn_10", model_type="pfnn")` and verifies saved config/bundle metadata.

**Step 2: Verify RED**

Run the focused smoke test and confirm `model_type` is unsupported.

**Step 3: Implement**

Add `model_type` to `fit_torch_regressor` and `run_training_job`, preserve raw phase during feature scaling, rebuild models from bundles by type, and add CLI options.

**Step 4: Verify GREEN**

Run focused tests, then full `pytest -q`.

### Task 4: Baseline Run

**Files:**
- No tracked code changes expected.

**Step 1: Train PFNN baseline**

Run `scripts/train_baseline_torch.py` on `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1` with `--feature-set paper_pfnn_10 --model-type pfnn`.

**Step 2: Compare**

Read `metrics.json` and compare against the existing `paper_no_accel_v2` MLP baseline.
