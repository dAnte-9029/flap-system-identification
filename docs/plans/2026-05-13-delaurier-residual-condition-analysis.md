# DeLaurier Residual Flight-Condition Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Identify flight-condition regimes where the calibrated DeLaurier prior has larger effective-wrench residuals, and check whether the residual Transformer reduces those errors across the safe flight envelope.

**Architecture:** Add one standalone analysis script that reads the residual Transformer aligned prediction parquet, bins samples by selected flight-condition variables, and reports true DeLaurier residual RMSE versus remaining residual RMSE after neural correction. The analysis uses quantile bins by default so each regime has enough samples for stable held-out-log statistics.

**Tech Stack:** Python 3.11, pandas, NumPy, matplotlib, pytest.

---

### Task 1: Condition-Bin Metrics

**Files:**
- Create: `scripts/analyze_delaurier_residual_conditions.py`
- Create: `tests/test_analyze_delaurier_residual_conditions.py`

**Steps:**
1. Write a failing test with one condition column and one target.
2. Verify quantile bins are formed from finite condition values only.
3. Verify each bin reports sample count, value range, median condition value, true residual RMSE, remaining residual RMSE, and RMSE reduction.
4. Implement the minimum functions needed to pass:
   - `condition_bin_table(frame, condition_columns, targets, quantile_bins, min_samples)`
   - `condition_summary_table(condition_table)`
5. Run the targeted test.

### Task 2: CLI And Figures

**Files:**
- Modify: `scripts/analyze_delaurier_residual_conditions.py`

**Steps:**
1. Add CLI arguments: `--aligned-parquet`, `--output-dir`, `--quantile-bins`, and optional `--min-samples`.
2. Default condition columns:
   - `airspeed_validated.true_airspeed_m_s`
   - `dynamic_pressure_pa`
   - `alpha_rad`
   - `cycle_flap_frequency_hz`
3. Write:
   - `condition_residual_bins.csv`
   - `condition_residual_summary.csv`
   - `condition_residual_config.json`
4. Plot key targets (`fx_b`, `fz_b`, `my_b`) for each condition:
   - true DeLaurier residual RMSE
   - remaining residual RMSE after residual Transformer
5. Save both PNG and PDF.

### Task 3: Run On Residual Transformer

**Command:**

```bash
conda run -n flap-train-gpu python scripts/analyze_delaurier_residual_conditions.py \
  --aligned-parquet artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_combined_eval/test_aligned_residual_predictions.parquet \
  --output-dir artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_condition_analysis \
  --quantile-bins 5 \
  --min-samples 500
```

Expected outputs:
- `condition_residual_bins.csv`
- `condition_residual_summary.csv`
- `condition_residual_rmse_key_targets.png`
- `condition_residual_rmse_key_targets.pdf`

### Task 4: Result Note

**Files:**
- Modify: `docs/results/2026-05-13-delaurier-residual-nn-gpu.md`

**Steps:**
1. Add a short flight-condition residual section.
2. Report which variables show the strongest residual variation across bins.
3. Report the worst bins for `fx_b`, `fz_b`, and `my_b`.
4. Keep the claim bounded: this identifies empirical applicability limits within the tested safe flight envelope, not outside-envelope extrapolation.

### Task 5: Verification

Run:

```bash
conda run -n flap-train-gpu python -m pytest tests/test_analyze_delaurier_residual_conditions.py tests/test_analyze_delaurier_residual_phase.py tests/test_evaluate_delaurier_residual_model.py -q
conda run -n flap-train-gpu python -m py_compile scripts/analyze_delaurier_residual_conditions.py
```

Inspect the generated CSV summaries before reporting conclusions.
