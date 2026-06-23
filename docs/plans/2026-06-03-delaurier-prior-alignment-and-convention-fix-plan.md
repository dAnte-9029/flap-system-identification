# DeLaurier Prior Alignment and Convention Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make DeLaurier prior calibration and residual/correction datasets robust to split row reordering, then diagnose remaining low-correlation behavior as a coordinate/sign/phase or model-adequacy issue.

**Architecture:** Treat row alignment as a data-integrity boundary. Every script that combines effective-wrench labels with exported DeLaurier predictions must either prove that the two tables share the same row identity or merge by stable sample keys. After this deterministic bug is fixed, run isolated convention tests for body-frame sign, axis, and phase assumptions before changing model claims.

**Tech Stack:** Python, pandas/parquet, pytest, existing `flap-system-identification` scripts and artifacts.

---

## Root-Cause Summary

The measured-mass rerun used `dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1` but reused prior predictions exported for `dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1`. The row sets match, but row order does not. Existing scripts only check row counts, so they can fit or subtract mismatched prior/label rows. This explains the near-zero positive affine gains in `artifacts/20260602_delaurier_prior_measured_massprops_v1/parameters.csv`.

After key-based reordering, the prior is still weak and partly sign-inverted. That is a second-stage physical/convention issue, not the same bug.

## Task 1: Add Stable Sample-Key Alignment Utilities

**Files:**
- Modify: `scripts/build_delaurier_residual_split.py`
- Test: `tests/test_build_delaurier_residual_split.py`

**Steps:**
1. Add helper functions that construct a stable key from available columns: `dataset_id`, `log_id`, `segment_id`, and rounded `time_s`.
2. If a prior parquet already contains all key columns, align by one-to-one merge.
3. If the prior parquet lacks keys, keep row-index alignment only when the caller explicitly allows it and record this in the manifest.
4. Raise a clear error if row counts match but keys are missing and strict alignment is requested.
5. Add tests for shuffled keyed prior rows and duplicate/missing keys.

**Reason:** Row count equality is not a valid identity check after split materialization or smoothing can reorder rows.

## Task 2: Use Key Alignment in Residual Split Builders

**Files:**
- Modify: `scripts/build_delaurier_residual_split.py`
- Modify: `scripts/build_delaurier_residual_kfold_splits.py`
- Test: `tests/test_build_delaurier_residual_split.py`
- Test: `tests/test_build_delaurier_residual_kfold_splits.py`

**Steps:**
1. Route residual construction through the new alignment helper.
2. Preserve `prior_*`, `label_*`, and `true_*` columns after alignment.
3. Add manifest fields: `alignment_mode`, `alignment_key_columns`, `allow_row_order_fallback`.
4. Ensure k-fold residual source construction uses the same helper.

**Reason:** B/C correction datasets depend on residual targets. A single row-order bug contaminates all downstream training.

## Task 3: Use Key Alignment in Force Recalibration

**Files:**
- Modify: `scripts/run_delaurier_force_recalibration.py`
- Test: `tests/test_delaurier_force_recalibration.py`

**Steps:**
1. Replace `_load_split_frames` with a loader that aligns prior predictions to sample rows.
2. Add CLI flag `--allow-row-order-fallback` for legacy artifacts that have no keys and are known to be same-order.
3. Write alignment mode and key columns into `manifest.json`.
4. Add a test where keyed prior rows are intentionally shuffled; expected affine coefficients must match the unshuffled case.

**Reason:** The affine wrapper is where the observed tiny diagonal gains were produced.

## Task 4: Materialize a Keyed Prior for the Measured-Mass Split

**Files:**
- Create output artifact: `artifacts/20260603_delaurier_prior_measured_massprops_key_aligned_v1`

**Steps:**
1. Read old hq_v4 sample keys and old prior predictions.
2. Attach keys to the old prior predictions.
3. Merge keyed prior predictions onto the measured-mass split.
4. Run force recalibration with strict key alignment.
5. Compare coefficients and metrics against the broken 20260602 artifact.

**Reason:** This gives a clean replacement for the measured-mass DeLaurier affine artifact without re-exporting IsaacLab prior yet.

## Task 5: Add Convention Diagnostics

**Files:**
- Create: `scripts/diagnose_delaurier_prior_conventions.py`
- Create output artifact: `artifacts/20260603_delaurier_prior_convention_diagnostics_v1`

**Steps:**
1. Evaluate sample-level and phase-binned correlations for candidate transforms: identity, force sign flip, FLU-to-FRD `fx,-fy,-fz`, `fz` only sign flip, `fy` only sign flip, and phase-shifted variants.
2. Report per-channel correlation, RMSE after train-only affine fit, and phase-bin median correlation.
3. Write `summary.csv` and `README.md`.

**Reason:** Correct row alignment does not solve the weak/negative correlations. The next likely causes are axis/sign/phase convention mismatches.

## Task 6: Rerun B+C Only After Task 4 Passes

**Files:**
- Existing scripts: `scripts/train_deployable_wrench_correction_v2.py`, `scripts/train_phase_structured_wrench_correction.py`

**Steps:**
1. Build corrected residual split using the keyed prior artifact.
2. Rerun deployable correction.
3. Rerun phase-structured correction.
4. Compare per-channel metrics to the previous sg0p03 and measured-mass artifacts.

**Reason:** Model training should not be used to judge physical correction until residual targets are known to be aligned.

## Stop Criteria

Stop and report after Task 4 if:
- strict key alignment fails;
- the corrected measured-mass prior still has correlations near zero and convention diagnostics have not been run;
- force recalibration changes paper-facing numbers materially.

Do not update paper claims until the corrected artifact and convention diagnostics are reviewed.
