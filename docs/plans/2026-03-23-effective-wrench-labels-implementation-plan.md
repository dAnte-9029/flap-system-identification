# Effective Wrench Labels Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-pass rigid-body effective force and moment labels to the canonical parquet pipeline while preserving `NaN` fallback when label metadata is incomplete.

**Architecture:** Keep the current resampling pipeline intact and add a narrow label-computation stage inside `assemble_canonical_samples()`. Compute force from `vehicle_local_position` acceleration rotated from NED to body FRD, compute moment from `vehicle_angular_velocity` and metadata inertia, and gate outputs with a simple finite-data validity mask.

**Tech Stack:** Python, NumPy, pandas, pytest

---

### Task 1: Lock label behavior with tests

**Files:**
- Modify: `tests/test_pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Add a test that supplies complete metadata plus `vehicle_local_position`, `vehicle_attitude`, and `vehicle_angular_velocity` samples with analytically known force and moment labels.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -q`
Expected: FAIL because labels are still all `NaN`.

### Task 2: Implement minimal rigid-body label computation

**Files:**
- Modify: `src/system_identification/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 3: Write minimal implementation**

Add helper logic to:
- convert PX4 `vehicle_attitude.q` (`wxyz`, body->NED) into `R_BN`
- compute `F_eff_B = m * R_BN * (a_N - g_N)`
- compute `M_eff_B = I_B * alpha_B + omega_B x (I_B * omega_B)`
- emit `label_valid` only where required inputs are finite

Keep `fx_b...mz_b = NaN` when metadata is incomplete.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -q`
Expected: PASS.

### Task 3: Full verification

**Files:**
- Modify: `src/system_identification/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Step 5: Run project verification**

Run: `pytest -q`
Expected: all tests pass.

Run: `python scripts/ulg_to_canonical_parquet.py --ulg /home/zn/QgcLogs/2026.3.22/log_6_2026-3-22-18-46-24good.ulg --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml --output dataset/canonical_v0.1`
Expected: command succeeds and rewrites parquet/report outputs.

**Step 6: Commit**

Skip. Current workspace is not a git repository.
