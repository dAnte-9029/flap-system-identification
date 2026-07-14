# Prior vs TCN Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compare DeLaurier prior plus gain-bias corrections against prior plus TCN residual and pure TCN baselines under the same whole-log folds.

**Architecture:** Add one experiment script that reuses existing prior alignment, feature construction, linear ridge fitting, and PyTorch sequence model utilities. The script writes fold-level predictions, overall metrics, condition-binned metrics, residual diagnostics, plots, and a manifest under one artifact root.

**Tech Stack:** Python, pandas, numpy, scipy/sklearn ridge utilities already used by the repo, PyTorch TCN training through `system_identification.training`, matplotlib.

---

### Task 1: Add Tests for Reporting and Baselines

**Files:**
- Create: `tests/test_prior_vs_tcn_comparison.py`

**Steps:**
1. Write tests for overall metric rows, condition-bin assignment, and affine baseline fitting.
2. Run `pytest tests/test_prior_vs_tcn_comparison.py -q` and confirm import failure because the new script is absent.

### Task 2: Add Experiment Script

**Files:**
- Create: `scripts/run_prior_vs_tcn_comparison.py`

**Steps:**
1. Implement prior alignment and fold loading by reusing helpers from `scripts/run_nested_prior_shaping_ablation_exp1.py`.
2. Implement Raw prior, Global gain-bias, Matrix gain-bias, and Conditioned gain-bias.
3. Implement TCN residual and Pure TCN training by calling `fit_torch_sequence_regressor`.
4. Implement unified predictions, overall metrics, binned metrics, residual spectra, and diagnostic plots.
5. Run the new unit tests until they pass.

### Task 3: Run Full Experiment

**Files:**
- Output: `artifacts/20260707_prior_vs_tcn_comparison/`

**Steps:**
1. Run the comparison with CUDA, six folds, TCN history 64, and 12 phase bins.
2. Verify the manifest, metric CSVs, fold predictions, and plots exist.
3. Summarize the overall table and the main binned residual findings for review.
