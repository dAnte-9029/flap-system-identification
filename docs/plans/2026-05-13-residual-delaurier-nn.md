# DeLaurier Residual NN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a first residual neural network that predicts the log-derived effective-wrench residual left by a calibrated DeLaurier prior.

**Architecture:** Keep physics-prior generation in `/home/zn/IsaacLab` and residual learning in `/home/zn/flap-system-identification`. IsaacLab exports per-split calibrated DeLaurier predictions. This repository builds a residual split where the standard target columns `fx_b..mz_b` are replaced by `true_wrench - prior_wrench`, then reuses the existing `train_baseline_torch.py` training pipeline.

**Tech Stack:** Python 3.11, pandas/parquet, PyTorch training pipeline already present in `system_identification.training`, pytest.

---

### Task 1: Residual Split Builder

**Files:**
- Create: `scripts/build_delaurier_residual_split.py`
- Create: `tests/test_build_delaurier_residual_split.py`

**Steps:**
1. Write a failing test that creates tiny train/val/test samples and matching prior prediction parquet files.
2. Assert that output split keeps all non-target columns, stores original target columns as `true_<target>`, stores prior columns as `prior_<target>`, and replaces `fx_b..mz_b` with residual targets.
3. Implement `build_residual_frame` and CLI.
4. Run `pytest tests/test_build_delaurier_residual_split.py`.

### Task 2: Physics Prior Export

**Files:**
- IsaacLab side: `scripts/flapping_px4/export_delaurier_prior_predictions.py`

**Steps:**
1. Export physically calibrated DeLaurier predictions for train/val/test into `artifacts/delaurier_physical_prior_v1`.
2. Keep this script in IsaacLab because it imports IsaacLab flapping physics code.

### Task 3: First Residual Training

**Files:**
- Output only: `artifacts/20260513_delaurier_residual_nn_v1/*`

**Steps:**
1. Build residual split under `dataset/delaurier_residual_physical_v1`.
2. Train one small MLP first for a fast sanity check.
3. Train one residual Transformer using the same final Transformer settings if the sanity check works.
4. Evaluate combined prediction as `prior + residual_nn` against the original true targets.

### Task 4: Residual Analysis

**Files:**
- Create: `scripts/evaluate_delaurier_residual_model.py`

**Steps:**
1. Load original split, prior predictions, and residual model predictions.
2. Report DeLaurier prior, residual NN, and combined metrics.
3. Save residual diagnostic tables for phase, airspeed, controls, and frequency-domain summaries.
