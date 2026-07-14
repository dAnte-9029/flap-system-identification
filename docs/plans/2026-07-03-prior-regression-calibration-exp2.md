# Prior Regression Calibration Experiment 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace discrete prior-setting selection with a bounded regression calibration of low-dimensional DeLaurier effective parameters, then evaluate the calibrated prior and its gain-bias correction under the same six-fold nested whole-log protocol.

**Architecture:** Use a linearized local DeLaurier basis around the current shaped prior. Export finite-difference prior predictions for a small parameter vector, fit bounded ridge-style parameter updates inside each outer fold, re-export the exact calibrated prior per fold, and evaluate prior-only plus gain-bias correction on held-out logs.

**Tech Stack:** Python, pandas, NumPy, SciPy `lsq_linear`, parquet artifacts, existing IsaacLab DeLaurier exporter, existing nested Experiment 1 alignment and gain-bias helpers.

---

### Task 1: Add Calibration Math Tests

**Files:**
- Create: `tests/test_prior_regression_calibration.py`
- Create: `scripts/prior_regression_calibration.py`

**Steps:**
1. Write failing tests for bounded ridge calibration recovering a known parameter update.
2. Write failing tests that bounds clip an otherwise too-large update.
3. Run `pytest tests/test_prior_regression_calibration.py -q` and confirm the tests fail because the production module is missing behavior.
4. Implement the minimal math utility needed to pass.
5. Re-run the same tests and confirm they pass.

### Task 2: Implement Local Prior Basis Export

**Files:**
- Modify: `scripts/run_nested_prior_regression_calibration_exp2.py`
- Reuse: `scripts/sweep_delaurier_original_parameters.py`

**Steps:**
1. Create Experiment 2 runner with defaults for `theta0`, bounds, finite-difference steps, nested splits, and output root.
2. Add logic to export or reuse the base shaped prior and each finite-difference perturbation.
3. Record `prior_basis_records.csv`.
4. Compile-check the script with `python -m py_compile`.

### Task 3: Fit Fold-Level Bounded Prior Calibration

**Files:**
- Modify: `scripts/run_nested_prior_regression_calibration_exp2.py`

**Steps:**
1. Load each prior basis and align it to nested train/val/test samples using stable sample keys.
2. For each outer fold and lambda candidate, fit `delta_theta` on inner train.
3. Select lambda on inner validation RMSE.
4. Refit selected lambda on train+val.
5. Write `experiment2_calibrated_parameters_by_fold.csv`, selection metrics, and linearized prior diagnostics.

### Task 4: Export Exact Calibrated Priors and Evaluate Gain-Bias

**Files:**
- Modify: `scripts/run_nested_prior_regression_calibration_exp2.py`

**Steps:**
1. For each outer fold, export exact DeLaurier predictions at the selected calibrated `theta`.
2. Align exact calibrated prior to nested fold samples.
3. Evaluate calibrated prior-only on outer test.
4. Train gain-bias correction on train+val with inner validation-selected ridge alpha.
5. Evaluate final gain-bias model on outer test.
6. Write `experiment2_outer_test_metrics.csv`, `experiment2_table_fx_fz_rmse.csv`, and selected alpha files.

### Task 5: Run Smoke and Full Experiments

**Files:**
- Output: `artifacts/20260703_prior_regression_calibration_exp2/`

**Steps:**
1. Run fold 0 smoke with `--folds 0 --force`.
2. Inspect calibrated parameters, linearization error, and final metrics.
3. If smoke passes, run all six folds with `--force`.
4. Verify shaped Experiment 1 baseline remains unchanged.

### Task 6: Diagnostics and Summary

**Files:**
- Create: `scripts/analyze_prior_regression_calibration_exp2_diagnostics.py`
- Output: `artifacts/20260703_prior_regression_calibration_exp2/diagnostics/`

**Steps:**
1. Reuse Experiment 1 diagnostic structure for phase peak-to-peak, flap fundamental energy, gain/bias decomposition, and fold parameter stability.
2. Produce a combined fold report comparable to Experiment 1.
3. Summarize calibrated prior versus nominal and current shaped prior.
4. Verify scripts compile and key outputs exist.
