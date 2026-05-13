# DeLaurier Residual Phase Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Analyze whether the DeLaurier residual has repeatable wingbeat-phase structure and whether the residual Transformer captures it.

**Architecture:** Add one focused analysis script in `scripts/` that reads an aligned residual prediction parquet file and writes phase-binned tables, summary metrics, and publication-oriented plots. The calculation is independent of model training and uses existing aligned prediction columns: `label_*`, `prior_*`, `pred_*`, and `phase_corrected_rad`.

**Tech Stack:** Python 3.11, pandas, NumPy, matplotlib, pytest.

---

### Task 1: Phase-Bin Metric Functions

**Files:**
- Create: `scripts/analyze_delaurier_residual_phase.py`
- Create: `tests/test_analyze_delaurier_residual_phase.py`

**Step 1: Write failing tests**

Create tests that build a tiny aligned frame with `phase_corrected_rad`, `label_fx_b`, `prior_fx_b`, and `pred_fx_b`.

Verify:
- `true_residual = label - prior`
- `remaining_residual = true_residual - pred`
- phase bins cover `[0, 2*pi)` and wrap phase values into that interval
- summary reports `phase_r2_true_residual`, `pred_phase_pattern_r2`, and RMSE reduction.

**Step 2: Verify RED**

Run:

```bash
conda run -n flap-train-gpu python -m pytest tests/test_analyze_delaurier_residual_phase.py -q
```

Expected: import/function failure.

**Step 3: Implement minimal computation**

Implement:
- `phase_bin_table(frame, targets, phase_bins=36)`
- `phase_summary_table(frame, phase_table, targets)`
- internal helpers for channel metrics and phase-bin-median prediction.

**Step 4: Verify GREEN**

Run the same pytest command and confirm it passes.

### Task 2: CLI And Outputs

**Files:**
- Modify: `scripts/analyze_delaurier_residual_phase.py`

**Steps:**
1. Add CLI arguments: `--aligned-parquet`, `--output-dir`, `--phase-bins`.
2. Write `phase_binned_residuals.csv`, `phase_residual_summary.csv`, and `phase_residual_config.json`.
3. Create `phase_residual_medians.png` and `.pdf` with one subplot per wrench channel.
4. Keep the plot readable: true residual median, predicted residual median, and remaining residual median, using a colorblind-safe palette.

### Task 3: Run On Residual Transformer

**Command:**

```bash
conda run -n flap-train-gpu python scripts/analyze_delaurier_residual_phase.py \
  --aligned-parquet artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_combined_eval/test_aligned_residual_predictions.parquet \
  --output-dir artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_phase_analysis \
  --phase-bins 36
```

Expected outputs:
- `phase_binned_residuals.csv`
- `phase_residual_summary.csv`
- `phase_residual_config.json`
- `phase_residual_medians.png`
- `phase_residual_medians.pdf`

### Task 4: Result Note

**Files:**
- Modify: `docs/results/2026-05-13-delaurier-residual-nn-gpu.md`

**Steps:**
1. Add a short phase-analysis section.
2. Report which channels have the largest phase-locked residual structure.
3. Report whether the Transformer captures or reduces that phase-conditioned residual.
4. Keep interpretation bounded: phase structure indicates systematic wingbeat-correlated model discrepancy, not a uniquely identified aerodynamic mechanism.

### Task 5: Verification

Run:

```bash
conda run -n flap-train-gpu python -m pytest tests/test_analyze_delaurier_residual_phase.py tests/test_build_delaurier_residual_split.py tests/test_evaluate_delaurier_residual_model.py -q
```

Confirm the generated CSV files exist and inspect their top rows before reporting conclusions.
