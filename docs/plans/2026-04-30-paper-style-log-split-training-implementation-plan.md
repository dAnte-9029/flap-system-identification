# Paper-Style Log Split Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a leakage-resistant training path that keeps acceleration channels in the canonical dataset for label generation and auditing, but excludes linear acceleration and angular acceleration from model inputs while evaluating on whole held-out flight logs.

**Architecture:** Keep canonical parquet files unchanged. Add a whole-log split materializer alongside the existing cycle-block split, and add an explicit `no_accel_no_alpha` training feature set for paper-style aerodynamic surrogate training. The new split writes train/val/test log manifests plus filtered sample parquet files, guaranteeing that no `(dataset_id, log_id)` appears in more than one split.

**Tech Stack:** Python, pandas, numpy, PyTorch, pytest

---

### Task 1: Add tests for whole-log split behavior

**Files:**
- Modify: `tests/test_dataset_split.py`
- Modify: `src/system_identification/dataset_split.py`
- Create: `scripts/build_log_split.py`

**Step 1: Write the failing test**

Add a test that creates several tiny accepted logs, calls `materialize_log_split(...)`, and asserts:
- `train_logs.csv`, `val_logs.csv`, and `test_logs.csv` are written
- all output sample rows pass the existing valid-row mask
- each `(dataset_id, log_id)` appears in exactly one split
- output sample rows include `dataset_id`, `log_id`, `source_samples_path`, and `split`

**Step 2: Run test to verify it fails**

Run:
`env PYTHONNOUSERSITE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_dataset_split.py::test_materialize_log_split_keeps_logs_disjoint -v`

Expected: FAIL because `materialize_log_split` does not exist yet.

**Step 3: Write minimal implementation**

Implement:
- deterministic log assignment using the existing split-count policy
- sample filtering with `_valid_row_mask`
- per-split sample parquet files
- per-split log CSV files
- dataset manifest with split log/sample counts
- CLI wrapper `scripts/build_log_split.py`

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

### Task 2: Add tests for no-accel/no-alpha feature set

**Files:**
- Modify: `tests/test_training.py`
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`

**Step 1: Write the failing test**

Add tests that assert:
- `NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS` excludes `vehicle_local_position.ax/ay/az`
- it excludes `vehicle_angular_velocity.xyz_derivative[0/1/2]`
- it keeps phase, actuator, velocity, angular velocity, attitude/gravity, and aero inputs
- `run_training_job(feature_set_name="no_accel_no_alpha")` writes a config whose feature list excludes acceleration channels

**Step 2: Run test to verify it fails**

Run:
`env PYTHONNOUSERSITE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_training.py::test_no_accel_no_alpha_feature_set_excludes_label_derivative_inputs -v`

Expected: FAIL because the feature-set constant does not exist yet.

**Step 3: Write minimal implementation**

Implement:
- named feature-set resolver
- `NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS`
- optional `feature_set_name` in `run_training_job`
- `--feature-set` CLI argument for `scripts/train_baseline_torch.py`

**Step 4: Run test to verify it passes**

Run the targeted training tests and confirm PASS.

### Task 3: Generate the real paper-style split

**Files:**
- Use: `dataset/canonical_v0.2_seed_labels_hq_v1/dataset_manifest.json`
- Create ignored artifact: `dataset/canonical_v0.2_training_ready_split_hq_v2_airborne_logsplit_paper_v1/*`

**Step 1: Run the log-split CLI**

Run:
`env PYTHONNOUSERSITE=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/build_log_split.py --manifest dataset/canonical_v0.2_seed_labels_hq_v1/dataset_manifest.json --output dataset/canonical_v0.2_training_ready_split_hq_v2_airborne_logsplit_paper_v1 --seed 42 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15`

**Step 2: Verify the split**

Confirm:
- no log overlap across train/val/test
- sample files are nonempty
- manifests are written

### Task 4: Run a smoke paper-style training job

**Files:**
- Use: `dataset/canonical_v0.2_training_ready_split_hq_v2_airborne_logsplit_paper_v1/*`
- Create ignored artifact: `artifacts/baseline_torch_hq_v2_logsplit_no_accel_no_alpha_smoke/*`

**Step 1: Run a bounded training job**

Run a small GPU training pass with `--feature-set no_accel_no_alpha` and capped sample counts so verification finishes quickly.

**Step 2: Inspect metrics**

Read `metrics.json` and `training_config.json` to confirm:
- feature list excludes acceleration channels
- train/val/test metrics exist
- test performance is now a held-out-log estimate, not a same-log cycle-block estimate

### Task 5: Final verification

**Files:**
- Test: `tests/test_dataset_split.py`
- Test: `tests/test_training.py`
- Test: full test suite

**Step 1: Run focused tests**

Run:
`env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-flap-train /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest tests/test_dataset_split.py tests/test_training.py -q`

**Step 2: Run full tests**

Run:
`env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-flap-train /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q`

**Step 3: Report**

Summarize:
- files changed
- split produced
- feature-set behavior
- smoke training metrics
- remaining limitations
