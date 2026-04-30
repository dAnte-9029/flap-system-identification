# Baseline Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first end-to-end PyTorch baseline training pipeline for effective wrench regression using the current training-ready split.

**Architecture:** Add a self-contained training module based on plain PyTorch that prepares features from split parquet files, scales inputs and targets, trains a multi-output MLP on GPU when available, evaluates on train/val/test, and saves reproducible artifacts. Keep it separate from the canonical data pipeline.

**Tech Stack:** Python, pandas, numpy, PyTorch, pytest

---

### Task 1: Add failing tests for feature preparation

**Files:**
- Create: `tests/test_training.py`
- Create: `src/system_identification/training.py`

**Step 1: Write the failing test**

Add a test that verifies:
- default feature columns are stable
- derived phase sine/cosine columns are added
- feature/target arrays are numeric and aligned

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_prepare_feature_target_arrays_adds_phase_encoding -v`
Expected: FAIL because the training module does not exist yet.

**Step 3: Write minimal implementation**

Implement the smallest feature-preparation helpers needed to make the test pass.

**Step 4: Run test to verify it passes**

Run the same command and confirm PASS.

### Task 2: Add failing tests for training and evaluation

**Files:**
- Modify: `tests/test_training.py`
- Modify: `src/system_identification/training.py`

**Step 1: Write the failing test**

Add a synthetic-data smoke test that:
- trains a baseline PyTorch model
- evaluates it
- verifies metric keys and prediction shape

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_fit_and_evaluate_baseline_mlp_smoke -v`
Expected: FAIL because fit/evaluate functions are not implemented yet.

**Step 3: Write minimal implementation**

Add baseline fit, predict, and metric helpers.

**Step 4: Run test to verify it passes**

Run the same command and confirm PASS.

### Task 3: Add artifact saving and CLI script

**Files:**
- Create: `scripts/train_baseline_mlp.py`
- Modify: `src/system_identification/training.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

Add a temporary-directory integration test that:
- writes tiny split parquet files
- runs the training entry point
- checks that model and metrics artifacts are created

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_train_baseline_run_writes_artifacts -v`
Expected: FAIL because the training runner is missing.

**Step 3: Write minimal implementation**

Implement:
- split loading
- training run orchestration
- artifact serialization
- CLI wrapper
- environment-facing PyTorch usage

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

### Task 4: Run focused training tests

**Files:**
- Test: `tests/test_training.py`

**Step 1: Run targeted tests**

Run: `pytest tests/test_training.py -v`

**Step 2: Verify all tests pass**

Confirm exit code 0.

### Task 5: Run a real baseline training job

**Files:**
- Use: `dataset/canonical_v0.2_training_ready_split_v1/*`
- Create: `artifacts/baseline_mlp_v1/*`

**Step 1: Run training**

Train the first baseline model against the current split.

**Step 2: Verify outputs**

Check that:
- metrics JSON exists
- serialized model bundle exists
- train/val/test metrics are reported

### Task 6: Final readiness verification

**Files:**
- Read: `artifacts/baseline_mlp_v1/metrics.json`
- Read: `artifacts/baseline_mlp_v1/training_config.json`

**Step 1: Run final verification commands**

Inspect fresh outputs and confirm the training pipeline completed end to end.

**Step 2: Report status**

State that the repository now supports the first baseline training run, and list the main next improvements:
- better feature ablations
- inclusion strategy for encoder-fallback cohort
