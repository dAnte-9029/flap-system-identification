# Cycle-Block Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a training-ready train/validation/test split from the accepted canonical parquet cohorts using a global shuffled cycle-block policy with purge gaps.

**Architecture:** Add a new split module that reads accepted log manifests, extracts valid cycle blocks from each `samples.parquet`, assigns blocks to splits with a fixed seed, purges neighboring train cycles around validation/test blocks, and materializes split manifests plus merged parquet files. Keep this work isolated from the canonical conversion pipeline.

**Tech Stack:** Python, pandas, parquet, pytest

---

### Task 1: Add a failing unit test for cycle block construction

**Files:**
- Create: `tests/test_dataset_split.py`
- Create: `src/system_identification/dataset_split.py`

**Step 1: Write the failing test**

Write a test that builds a synthetic sample table with valid and invalid cycles and asserts that block extraction:
- keeps only rows satisfying the valid-row mask
- groups consecutive valid cycles
- emits the expected cycle ranges for a fixed block size

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_split.py::test_build_cycle_blocks_groups_valid_cycles -v`
Expected: FAIL because `system_identification.dataset_split` or the target function does not exist yet.

**Step 3: Write minimal implementation**

Add the smallest block-construction function needed for the test to pass.

**Step 4: Run test to verify it passes**

Run the same pytest command and verify PASS.

### Task 2: Add a failing unit test for split assignment and purge behavior

**Files:**
- Modify: `tests/test_dataset_split.py`
- Modify: `src/system_identification/dataset_split.py`

**Step 1: Write the failing test**

Add a test that constructs synthetic blocks from one or more logs and asserts:
- train/validation/test block ids do not overlap
- blocks assigned to validation/test cause nearby train cycles in the same log to be purged

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_split.py::test_assign_blocks_purges_nearby_train_cycles -v`
Expected: FAIL because assignment/purge logic is not implemented yet.

**Step 3: Write minimal implementation**

Implement deterministic split assignment and purge handling to satisfy the test.

**Step 4: Run test to verify it passes**

Run the same pytest command and verify PASS.

### Task 3: Add the CLI script for materializing the split

**Files:**
- Create: `scripts/build_cycle_block_split.py`
- Modify: `src/system_identification/dataset_split.py`

**Step 1: Write the failing integration-oriented test**

Add a test that builds tiny temporary parquet inputs and a mock accepted-log manifest, runs the materialization function, and checks that output manifests/parquets are created.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_split.py::test_materialize_split_writes_manifests_and_parquets -v`
Expected: FAIL because the materialization entry point is missing.

**Step 3: Write minimal implementation**

Implement:
- accepted-log loading
- split materialization
- dataset manifest writing
- CLI wrapper in `scripts/build_cycle_block_split.py`

**Step 4: Run test to verify it passes**

Run the same pytest command and verify PASS.

### Task 4: Run focused test suite

**Files:**
- Test: `tests/test_dataset_split.py`

**Step 1: Run targeted tests**

Run: `pytest tests/test_dataset_split.py -v`

**Step 2: Verify all tests pass**

Confirm exit code 0 and no failures.

### Task 5: Build the first training-ready split artifact

**Files:**
- Use: `dataset/canonical_v0.2_seed_labels_2026_4_12/dataset_manifest.json`
- Use: `dataset/canonical_v0.2_seed_labels_2026_4_14_3algorithms_weakwing/dataset_manifest.json`
- Use: `dataset/canonical_v0.2_seed_labels_2026_4_14_3algorithms/dataset_manifest.json`
- Use: `dataset/canonical_v0.2_seed_labels_2026_4_15/dataset_manifest.json`
- Create: `dataset/canonical_v0.2_training_ready_split_v1/*`

**Step 1: Run the builder**

Run the new split-building CLI against the accepted wing-phase manifests.

**Step 2: Verify outputs**

Check that:
- split parquets exist
- split CSV manifests exist
- manifest counts are internally consistent
- there is no split overlap

### Task 6: Final verification and readiness statement

**Files:**
- Read: `dataset/canonical_v0.2_training_ready_split_v1/dataset_manifest.json`
- Read: `docs/plans/2026-04-14-dataset-log-admission-tracker.md`

**Step 1: Run final verification commands**

Run fresh commands to inspect counts and split consistency.

**Step 2: Report readiness**

State clearly whether the dataset is ready for baseline training now, and identify what remains outside the current scope:
- encoder-fallback cohort handling
- future data growth
- downstream trainer implementation
