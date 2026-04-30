# Training Diagnostics And Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the current PyTorch training pipeline with reproducible training diagnostics and a feature-ablation runner, so the first baseline is no longer a black box.

**Architecture:** Keep the current `training.py` as the single training entry point, but add three layers around it: (1) richer epoch history and final prediction diagnostics, (2) plotting helpers that export static PNG artifacts with matplotlib, and (3) a separate ablation runner that executes multiple named feature-set variants against the same split and aggregates their results into comparison tables and plots.

**Tech Stack:** Python, pandas, numpy, PyTorch, matplotlib, pytest

---

### Task 1: Add failing tests for richer training outputs

**Files:**
- Modify: `tests/test_training.py`
- Modify: `src/system_identification/training.py`

**Step 1: Write the failing test**

Add a test that runs `run_training_job(...)` on a tiny synthetic split and asserts:
- `history.csv` is written
- `history.csv` includes at least `epoch`, `train_loss`, `val_loss`, `val_overall_rmse`, `val_overall_r2`
- prediction diagnostics files are written for the test split

**Step 2: Run test to verify it fails**

Run: `env PYTHONNOUSERSITE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_training.py::test_run_training_job_writes_history_and_diagnostics -v`
Expected: FAIL because those outputs do not exist yet.

**Step 3: Write minimal implementation**

Implement:
- per-epoch history capture in `fit_torch_regressor`
- history export in `run_training_job`
- diagnostic plotting/export helpers

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

### Task 2: Add failing tests for feature-ablation definitions

**Files:**
- Modify: `tests/test_training.py`
- Modify: `src/system_identification/training.py`

**Step 1: Write the failing test**

Add a test that verifies:
- default feature groups are stable
- default ablation variants resolve to concrete feature-column lists
- `full` equals the default feature set
- `no_phase` excludes all phase-group features

**Step 2: Run test to verify it fails**

Run: `env PYTHONNOUSERSITE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_training.py::test_default_ablation_variants_resolve_expected_feature_sets -v`
Expected: FAIL because feature-group and ablation helpers do not exist yet.

**Step 3: Write minimal implementation**

Add:
- named feature groups
- named default ablation variants
- helpers to resolve variant names into explicit feature-column lists

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

### Task 3: Add failing tests for the ablation runner

**Files:**
- Modify: `tests/test_training.py`
- Modify: `src/system_identification/training.py`
- Create: `scripts/run_feature_ablation.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

Add a test that creates a tiny synthetic split, runs an ablation study on two variants, and asserts:
- per-variant artifact directories are written
- `ablation_summary.csv` is written
- `ablation_summary.png` is written
- summary rows contain `variant_name`, `val_overall_r2`, and `test_overall_r2`

**Step 2: Run test to verify it fails**

Run: `env PYTHONNOUSERSITE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_training.py::test_run_ablation_study_writes_summary_outputs -v`
Expected: FAIL because the ablation runner does not exist yet.

**Step 3: Write minimal implementation**

Implement:
- ablation-study orchestration
- summary CSV/JSON writing
- comparison bar plot export
- CLI wrapper
- add `matplotlib` to project dependencies

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

### Task 4: Run the focused diagnostics/ablation test suite

**Files:**
- Test: `tests/test_training.py`

**Step 1: Run targeted tests**

Run: `env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-flap-train /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_training.py -v`

**Step 2: Verify all tests pass**

Confirm exit code 0.

### Task 5: Generate real diagnostics for the current baseline

**Files:**
- Use: `dataset/canonical_v0.2_training_ready_split_v1/*`
- Modify/Create: `artifacts/baseline_torch_v1/*`

**Step 1: Re-run baseline training with diagnostics enabled**

Run the training CLI so `baseline_torch_v1` contains:
- `history.csv`
- `training_curves.png`
- `pred_vs_true_test.png`
- `residual_hist_test.png`

**Step 2: Verify outputs**

Check the files exist and the history file has the expected columns.

### Task 6: Run the first ablation study on the real split

**Files:**
- Use: `dataset/canonical_v0.2_training_ready_split_v1/*`
- Create: `artifacts/ablation_torch_v1/*`

**Step 1: Run the ablation CLI**

Use the default ablation variant set against the current split.

**Step 2: Verify outputs**

Check:
- per-variant metrics exist
- `ablation_summary.csv` exists
- `ablation_summary.png` exists

### Task 7: Final verification and report

**Files:**
- Read: `artifacts/baseline_torch_v1/history.csv`
- Read: `artifacts/baseline_torch_v1/metrics.json`
- Read: `artifacts/ablation_torch_v1/ablation_summary.csv`

**Step 1: Run final verification commands**

Inspect fresh outputs and confirm:
- diagnostics are written
- ablation summary is written
- the repository now supports repeatable diagnosis of model behavior

**Step 2: Report status**

Summarize:
- where the new diagnostics live
- which ablation variants were run
- which feature removals hurt performance most
