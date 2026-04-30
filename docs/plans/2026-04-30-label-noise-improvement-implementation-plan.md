# Label Noise Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve training against noisy effective-wrench labels by creating a smoothed-derivative label variant and adding target-weighted training loss.

**Architecture:** Keep the existing paper-style feature sets and whole-log split unchanged. Add a small relabeling CLI that reads an existing split, recomputes acceleration/angular-acceleration from smoothed velocity/gyro per log, rewrites only `fx_b...mz_b` and `label_valid`, then reuse the current training CLI. Add optional target weights to the training loss so weak/noisy targets can be downweighted without changing metrics.

**Tech Stack:** Python, pandas/parquet, NumPy/SciPy, PyTorch, pytest.

---

### Task 1: Add reusable smoothed-derivative label helpers

**Files:**
- Modify: `src/system_identification/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing tests**

Add tests that verify:
- A noisy angular velocity sequence can be smoothed and differentiated per `log_id`.
- `_compute_effective_wrench_labels()` can use replacement derivative columns to produce moment labels different from raw logged derivatives.

**Step 2: Verify red**

Run:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_pipeline.py::test_compute_smoothed_kinematic_derivatives_by_log tests/test_pipeline.py::test_compute_effective_wrench_labels_accepts_replacement_derivatives -q
```

Expected: fail because helper/replacement support does not exist yet.

**Step 3: Implement minimal helpers**

Add:
- `compute_smoothed_kinematic_derivatives(samples, group_column="log_id", window_s=0.12, polyorder=2)`
- optional keyword arguments to `_compute_effective_wrench_labels()` for `linear_acceleration_columns` and `angular_acceleration_columns`.

**Step 4: Verify green**

Run the same pytest command. Expected: pass.

---

### Task 2: Add smoothed-label split materializer

**Files:**
- Create: `scripts/build_smoothed_label_split.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing test**

Add a small temporary split with `train/val/test_samples.parquet` and log CSVs, run the script function directly, and assert:
- output split files exist,
- `mx_b/mz_b` change when smoothed derivatives differ,
- `dataset_manifest.json` records the source split and smoothing config,
- split assignments are preserved.

**Step 2: Verify red**

Run:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_pipeline.py::test_build_smoothed_label_split_rewrites_labels_and_preserves_split -q
```

Expected: fail because the script/module does not exist yet.

**Step 3: Implement CLI**

Implement CLI arguments:
- `--split-root`
- `--metadata`
- `--output`
- `--window-s`
- `--polyorder`

For each split parquet:
- compute smoothed derivatives per `log_id`,
- call label computation with replacement derivative columns,
- overwrite six label columns and `label_valid`,
- copy `*_logs.csv` and `all_logs.csv`,
- write `dataset_manifest.json`.

**Step 4: Verify green**

Run the same test. Expected: pass.

---

### Task 3: Add target-weighted training loss

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add tests that verify:
- a weight string like `fx_b=1,fy_b=0.5,fz_b=1,mx_b=0.5,my_b=1,mz_b=0.5` resolves into target-order weights,
- `_evaluate_scaled_loss()` and training use weighted scaled MSE,
- the saved bundle/config include the weights.

**Step 2: Verify red**

Run:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_training.py::test_resolve_target_loss_weights_from_mapping tests/test_training.py::test_run_training_job_records_target_loss_weights -q
```

Expected: fail because weight support does not exist yet.

**Step 3: Implement weighted MSE**

Add:
- `resolve_target_loss_weights(target_columns, target_loss_weights)`
- `_weighted_scaled_mse(predictions, targets, weights)`
- `target_loss_weights` plumbing through `fit_torch_regressor()` and `run_training_job()`
- `--target-loss-weights` CLI option.

**Step 4: Verify green**

Run the same pytest command. Expected: pass.

---

### Task 4: Materialize smoothed-label dataset and run baseline

**Files:**
- Output: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_smooth_labels`
- Output: `artifacts/baseline_torch_hq_v3_direct_airspeed_logsplit_paper_no_accel_v2_smooth_labels_weighted`

**Step 1: Build dataset**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-flap-noise /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/build_smoothed_label_split.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_smooth_labels \
  --window-s 0.12 \
  --polyorder 2
```

Expected: split parquet files and manifest are written.

**Step 2: Run diagnostics**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-flap-noise /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/diagnose_label_noise.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_smooth_labels \
  --output-dir artifacts/label_noise_diagnostics_hq_v3_smooth_labels
```

Expected: diagnostic summary shows lower high-frequency content for moment labels.

**Step 3: Train weighted baseline**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-flap-noise /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/train_baseline_torch.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_smooth_labels \
  --output-dir artifacts/baseline_torch_hq_v3_direct_airspeed_logsplit_paper_no_accel_v2_smooth_labels_weighted \
  --feature-set paper_no_accel_v2 \
  --model-type mlp \
  --hidden-sizes 256,256 \
  --batch-size 4096 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --target-loss-weights fx_b=1,fy_b=0.5,fz_b=1,mx_b=0.5,my_b=1,mz_b=0.5 \
  --device auto
```

Expected: metrics and plots are written.

---

### Task 5: Final verification and comparison

**Files:**
- Read: old/new `metrics.json`
- Read: old/new `label_noise_diagnostic_summary.csv`

**Step 1: Run tests**

Run:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q
```

Expected: all tests pass.

**Step 2: Compare old vs new**

Compare:
- per-target test R2,
- high-frequency fraction,
- derivative consistency,
- primary target average (`fx_b`, `fz_b`, `my_b`),
- secondary target average (`fy_b`, `mx_b`, `mz_b`).

**Step 3: Report**

Summarize whether smoothing improved labels and whether weighted training preserved primary targets while reducing noisy-target dominance.
