# ULog To Canonical Parquet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a first-pass offline pipeline that converts a PX4 `.ulg` flight log plus aircraft metadata into canonical parquet outputs for flapping-wing system identification.

**Architecture:** Implement a small Python package under `src/system_identification/` with pure utility functions for metadata loading, topic extraction, resampling, phase reconstruction, and label generation. Add a thin CLI entrypoint under `scripts/` that runs the pipeline on one log and writes `samples.parquet`, `segments.parquet`, `source_manifest.json`, and `preprocessing_report.json`.

**Tech Stack:** Python, `pyulog`, `numpy`, `pandas`, `pyarrow`, `PyYAML`, `scipy`, `pytest`

---

### Task 1: Project Skeleton

**Files:**
- Create: `/home/zn/system-identification/pyproject.toml`
- Create: `/home/zn/system-identification/src/system_identification/__init__.py`
- Create: `/home/zn/system-identification/src/system_identification/metadata.py`
- Create: `/home/zn/system-identification/src/system_identification/resample.py`
- Create: `/home/zn/system-identification/src/system_identification/phase.py`
- Create: `/home/zn/system-identification/src/system_identification/pipeline.py`
- Create: `/home/zn/system-identification/scripts/ulg_to_canonical_parquet.py`

**Step 1: Write the failing test**

- Add tests that import the target modules and call core helpers.

**Step 2: Run test to verify it fails**

Run: `pytest -q`

Expected: import failures because package files do not exist yet.

**Step 3: Write minimal implementation**

- Add package skeleton and CLI entrypoint.

**Step 4: Run test to verify it passes**

Run: `pytest -q`

Expected: import-related failures disappear and helper tests can execute.

### Task 2: Core Math And Metadata

**Files:**
- Create: `/home/zn/system-identification/tests/test_metadata.py`
- Create: `/home/zn/system-identification/tests/test_phase.py`
- Modify: `/home/zn/system-identification/src/system_identification/metadata.py`
- Modify: `/home/zn/system-identification/src/system_identification/phase.py`

**Step 1: Write the failing test**

- Verify metadata loader preserves confirmed values and exposes missing placeholders.
- Verify drive phase reconstruction from encoder phase.
- Verify wing stroke angle and stroke direction follow the agreed convention.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metadata.py tests/test_phase.py -q`

Expected: failures because functions are not implemented yet.

**Step 3: Write minimal implementation**

- Implement metadata loading and placeholder detection.
- Implement `wrap_to_2pi`, drive phase mapping, wing stroke angle, and stroke direction helpers.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metadata.py tests/test_phase.py -q`

Expected: all tests pass.

### Task 3: Time Grid And Resampling

**Files:**
- Create: `/home/zn/system-identification/tests/test_resample.py`
- Modify: `/home/zn/system-identification/src/system_identification/resample.py`

**Step 1: Write the failing test**

- Verify `build_uniform_grid_us`.
- Verify bin-mean aggregation for actuator-like samples.
- Verify ZOH freshness invalidation.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_resample.py -q`

Expected: failures because resampling helpers are missing.

**Step 3: Write minimal implementation**

- Implement grid construction, bin-mean, linear interpolation, ZOH, and freshness age helpers.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_resample.py -q`

Expected: all tests pass.

### Task 4: Pipeline Assembly

**Files:**
- Create: `/home/zn/system-identification/tests/test_pipeline.py`
- Modify: `/home/zn/system-identification/src/system_identification/pipeline.py`
- Modify: `/home/zn/system-identification/scripts/ulg_to_canonical_parquet.py`

**Step 1: Write the failing test**

- Verify pipeline can emit canonical samples from synthetic topic frames.
- Verify missing mass/inertia yields NaN label columns instead of crashing.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -q`

Expected: failures because pipeline assembly is not implemented.

**Step 3: Write minimal implementation**

- Implement topic extraction, canonical frame assembly, placeholder label generation, report output, and CLI argument parsing.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -q`

Expected: all tests pass.

### Task 5: Smoke Verification On Real Log

**Files:**
- None

**Step 1: Run the pipeline on the real good log**

Run:

```bash
python scripts/ulg_to_canonical_parquet.py \
  --ulg /home/zn/QgcLogs/2026.3.22/log_6_2026-3-22-18-46-24good.ulg \
  --metadata /home/zn/system-identification/metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output /home/zn/system-identification/dataset/canonical_v0.1
```

Expected:

- Files are written
- Canonical samples include phase and actuator columns
- Label columns exist and are NaN when metadata is still placeholder

**Step 2: Run the full test suite**

Run: `pytest -q`

Expected: all tests pass.
