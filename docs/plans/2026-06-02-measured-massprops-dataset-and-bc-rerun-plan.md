# Measured Mass Properties Dataset and B+C Rerun Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Regenerate the flight-log effective-wrench dataset using the measured whole-aircraft mass, CG, and inertia metadata, then rerun the B+C correction pipeline and diagnostics without overwriting previous paper artifacts.

**Architecture:** Treat the measured mass properties as a new dataset version, not as an in-place update of old labels. The pipeline should proceed from canonical ULog conversion to whole-log split, smoothed/time-aligned labels, DeLaurier prior calibration, residual/correction datasets, B+C model training, diagnostics, replay sanity checks, and old-vs-new comparison. Every output directory uses the tag `measured_massprops_v1`.

**Tech Stack:** Python, pandas, NumPy, PyYAML, PyTorch scripts already in this repo, parquet datasets, CSV/JSON metrics, pytest, existing PX4 ULog conversion pipeline.

---

## Paper and Claim Boundaries

- Do not overwrite old dataset or artifact directories.
- Do not report mixed old-label and new-label results.
- The new reported values must all trace back to `metadata/aircraft/flapper_01/aircraft_metadata.yaml` after the measured mass-property update.
- Treat this as a new label generation pass because mass affects `fx_b, fy_b, fz_b` and inertia affects `mx_b, my_b, mz_b`.
- Keep six-degree-of-freedom replay as diagnostic only unless oracle checks pass clearly.
- In the paper, phrase replay as log-seeded or local sanity checking, not closed-loop simulator validation.
- Preserve the whole-log train/val/test protocol. Prefer reusing the previous log assignment so old-vs-new comparison isolates the metadata/label change.

## Fixed Version Tags

Use these names consistently:

```text
version_tag: measured_massprops_v1
metadata_path: metadata/aircraft/flapper_01/aircraft_metadata.yaml
old_seed_root: dataset/canonical_v0.2_seed_labels_hq_v2_direct_airspeed
old_split_root: dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1
new_seed_root: dataset/canonical_v0.2_seed_labels_measured_massprops_v1
new_split_root: dataset/canonical_v0.2_training_ready_split_measured_massprops_v1
new_smoothed_split_root: dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1
new_prior_root: artifacts/20260602_delaurier_prior_measured_massprops_v1
new_residual_split_root: dataset/delaurier_force_prior_moment_direct_measured_massprops_v1
new_bc_root: artifacts/20260602_bc_correction_measured_massprops_v1
new_diagnostics_root: artifacts/20260602_residual_diagnostics_measured_massprops_v1
new_replay_root: artifacts/20260602_replay_measured_massprops_v1
comparison_root: artifacts/20260602_old_vs_measured_massprops_comparison
```

Current measured metadata values expected:

```text
mass_kg = 0.90415
cg_b_m = [-0.12154, 0.00541, -0.04298]  # relative to imu_origin, FRD
inertia_b_kg_m2 = diag([0.02329, 0.02573, 0.04270])
```

## Success Criteria

- `pytest -q tests/test_metadata.py` passes.
- New seed dataset has 29 accepted logs or any difference is explicitly explained.
- New split has the same train/val/test log IDs as the selected old paper split, unless a log fails validation.
- New smoothed split has finite label ratio near 1.0 for train/val/test.
- Prior calibration, B+C training, residual diagnostics, and replay scripts complete without unresolved missing-file or unresolved-column errors.
- A final comparison report lists old vs new per-channel metrics and states what changed because of measured mass properties.

## Task 1: Verify Metadata and Preserve a Snapshot

**Files:**

- Read: `metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Modify only if needed: `metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Read/Test: `tests/test_metadata.py`
- Create: `artifacts/20260602_measured_massprops_metadata_snapshot/aircraft_metadata.yaml`
- Create: `artifacts/20260602_measured_massprops_metadata_snapshot/README.md`

**Step 1: Run metadata test**

```bash
cd /home/zn/flap-system-identification
pytest -q tests/test_metadata.py
```

Expected: `2 passed`.

**Step 2: Save metadata snapshot**

```bash
mkdir -p artifacts/20260602_measured_massprops_metadata_snapshot
cp metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  artifacts/20260602_measured_massprops_metadata_snapshot/aircraft_metadata.yaml
```

**Step 3: Write a short README**

The README must record:

```text
mass_kg = 0.90415
cg_b_m = [-0.12154, 0.00541, -0.04298]
inertia_b_kg_m2 = diag([0.02329, 0.02573, 0.04270])
coordinate convention = FRD, cg relative to imu_origin
IMU approximation = r_imu^U_FRD = [0, 0, 0.030] m
```

**Step 4: Commit metadata-only checkpoint if appropriate**

```bash
git status --short
git add metadata/aircraft/flapper_01/aircraft_metadata.yaml tests/test_metadata.py
git commit -m "chore: update flapper mass properties metadata"
```

Skip the commit if the user wants the metadata bundled with the full rerun.

## Task 2: Build a Batch ULog Regeneration Helper if Needed

**Files:**

- Read: `dataset/canonical_v0.2_seed_labels_hq_v2_direct_airspeed/accepted_logs.csv`
- Read: `scripts/ulg_to_canonical_parquet.py`
- Create if no existing equivalent: `scripts/regenerate_canonical_from_accepted_logs.py`
- Test: `tests/test_regenerate_canonical_from_accepted_logs.py`

**Rationale:** `scripts/ulg_to_canonical_parquet.py` converts one ULog at a time. The accepted-log CSV already contains `source_log_path`, so a small batch helper prevents manual mistakes.

**Step 1: Check whether a batch helper already exists**

```bash
find scripts -maxdepth 1 -type f | sort | rg "canonical|ulog|accepted|batch"
```

If an existing helper can read `accepted_logs.csv` and regenerate all logs, reuse it. If not, create the helper.

**Step 2: Add helper behavior**

The helper should:

- accept `--accepted-logs-csv`;
- accept `--metadata`;
- accept `--output`;
- read `source_log_path` from the CSV;
- call `run_ulog_to_canonical()` or the existing CLI behavior for each log;
- write a manifest with the exact metadata path and input CSV path;
- fail if `--output` exists unless `--overwrite` is provided.

**Step 3: Run a dry smoke test on one log**

```bash
python scripts/regenerate_canonical_from_accepted_logs.py \
  --accepted-logs-csv dataset/canonical_v0.2_seed_labels_hq_v2_direct_airspeed/accepted_logs.csv \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output /tmp/canonical_measured_massprops_smoke \
  --limit 1 \
  --overwrite
```

Expected: one `samples.parquet`, one `segments.parquet`, and a manifest are created.

**Step 4: Run the new helper tests**

```bash
pytest -q tests/test_regenerate_canonical_from_accepted_logs.py
```

Expected: pass.

## Task 3: Regenerate Canonical Seed Labels

**Files:**

- Input: `dataset/canonical_v0.2_seed_labels_hq_v2_direct_airspeed/accepted_logs.csv`
- Input: `metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Output: `dataset/canonical_v0.2_seed_labels_measured_massprops_v1`

**Step 1: Run the full accepted-log regeneration**

```bash
python scripts/regenerate_canonical_from_accepted_logs.py \
  --accepted-logs-csv dataset/canonical_v0.2_seed_labels_hq_v2_direct_airspeed/accepted_logs.csv \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_seed_labels_measured_massprops_v1
```

If no helper was created, run equivalent per-log conversion from the CSV `source_log_path` values with:

```bash
python scripts/ulg_to_canonical_parquet.py \
  --ulg <source_log_path> \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_seed_labels_measured_massprops_v1
```

**Step 2: Verify accepted log count**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("dataset/canonical_v0.2_seed_labels_measured_massprops_v1")
accepted = pd.read_csv(root / "accepted_logs.csv")
print("accepted logs:", len(accepted))
print("label_valid_ratio min:", accepted["label_valid_ratio"].min())
print("active_label_valid_ratio min:", accepted.get("active_label_valid_ratio", accepted["label_valid_ratio"]).min())
PY
```

Expected: accepted log count should match old selected set unless a specific log fails.

**Step 3: Compare label scales with old seed labels**

Create a small report:

```text
artifacts/20260602_old_vs_measured_massprops_comparison/seed_label_scale_delta.csv
```

Include per channel:

```text
old mean/std/range
new mean/std/range
new/old std ratio
```

Expected pattern:

- force scale changes roughly with `0.90415 / 0.95`;
- moment scale changes strongly because inertia changed from CAD seed to measured diagonal values.

## Task 4: Rebuild the Whole-Log Split with the Same Log Assignment

**Files:**

- Input: `dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1/all_logs.csv`
- Input: `dataset/canonical_v0.2_seed_labels_measured_massprops_v1/dataset_manifest.json`
- Output: `dataset/canonical_v0.2_training_ready_split_measured_massprops_v1`

**Step 1: Prefer exact split reuse**

Do not rely on a fresh random split if an exact old `all_logs.csv` exists. The new split should reuse each old log's `split` assignment.

If no existing script supports this, add a small helper:

```text
scripts/materialize_split_from_log_assignment.py
```

Inputs:

```text
--source-manifest dataset/canonical_v0.2_seed_labels_measured_massprops_v1/dataset_manifest.json
--assignment-csv dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1/all_logs.csv
--output dataset/canonical_v0.2_training_ready_split_measured_massprops_v1
```

**Step 2: Materialize split**

```bash
python scripts/materialize_split_from_log_assignment.py \
  --source-manifest dataset/canonical_v0.2_seed_labels_measured_massprops_v1/dataset_manifest.json \
  --assignment-csv dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1/all_logs.csv \
  --output dataset/canonical_v0.2_training_ready_split_measured_massprops_v1
```

Fallback only if exact reuse is not feasible:

```bash
python scripts/build_log_split.py \
  --manifest dataset/canonical_v0.2_seed_labels_measured_massprops_v1/dataset_manifest.json \
  --output dataset/canonical_v0.2_training_ready_split_measured_massprops_v1 \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --altitude-window-min-m 5
```

If fallback is used, record that old-vs-new comparison is no longer perfectly controlled.

**Step 3: Verify same log IDs**

```bash
python - <<'PY'
import pandas as pd
old = pd.read_csv("dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1/all_logs.csv")
new = pd.read_csv("dataset/canonical_v0.2_training_ready_split_measured_massprops_v1/all_logs.csv")
merged = old[["log_id","split"]].merge(new[["log_id","split"]], on="log_id", suffixes=("_old","_new"))
print("old logs", len(old), "new logs", len(new), "merged", len(merged))
print("split mismatches", (merged["split_old"] != merged["split_new"]).sum())
print(merged[merged["split_old"] != merged["split_new"]].head(20))
PY
```

Expected: zero split mismatches.

## Task 5: Recompute Smoothed and Time-Aligned Effective-Wrench Labels

**Files:**

- Input: `dataset/canonical_v0.2_training_ready_split_measured_massprops_v1`
- Output: `dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1`
- Reference: old manifest at `dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1/dataset_manifest.json`

**Step 1: Identify the previous smoothing script**

Check old manifest:

```text
method = savgol
window_s = 0.03
polyorder = 2
force_label_source = smooth
moment_label_source = smooth
selected lags = 0 for all listed inputs
```

Use the same script and settings. Candidate scripts:

```text
scripts/build_time_aligned_smoothed_label_split.py
scripts/build_smoothed_label_split.py
```

Inspect their CLI before running.

**Step 2: Run the same smoothing/time-alignment policy**

Expected command shape:

```bash
python scripts/build_time_aligned_smoothed_label_split.py \
  --input-root dataset/canonical_v0.2_training_ready_split_measured_massprops_v1 \
  --output-root dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1 \
  --window-s 0.03 \
  --polyorder 2 \
  --force-label-source smooth \
  --moment-label-source smooth \
  --overwrite
```

Adjust flags to the actual CLI after inspection.

**Step 3: Verify finite ratios and sample counts**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1")
for split in ["train","val","test"]:
    df = pd.read_parquet(root / f"{split}_samples.parquet")
    cols = ["fx_b","fy_b","fz_b","mx_b","my_b","mz_b"]
    print(split, len(df), df[cols].notna().all(axis=1).mean())
PY
```

Expected: finite ratio near 1.0.

## Task 6: Recalibrate or Regenerate the DeLaurier Prior

**Files:**

- Input: `dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1`
- Output: `artifacts/20260602_delaurier_prior_measured_massprops_v1`

**Step 1: Run prior calibration**

Use:

```bash
python scripts/run_delaurier_force_recalibration.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1 \
  --output-root artifacts/20260602_delaurier_prior_measured_massprops_v1
```

If the script requires a nominal prior root:

```bash
python scripts/run_delaurier_force_recalibration.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1 \
  --prior-root <nominal_prior_root> \
  --output-root artifacts/20260602_delaurier_prior_measured_massprops_v1
```

**Step 2: Verify prior output columns**

Ensure the output contains columns needed by residual split building:

```text
prior_fx_b, prior_fy_b, prior_fz_b
prior_mx_b, prior_my_b, prior_mz_b
```

If force-only prior is intentional, document which moment channels use direct prediction.

**Step 3: Save calibration summary**

Record:

```text
calibration parameters
train/val/test prior metrics
per-channel RMSE/MAE/R2/RMSE-over-std
```

## Task 7: Build Residual / Hybrid Dataset for B+C

**Files:**

- Input: `dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1`
- Input: `artifacts/20260602_delaurier_prior_measured_massprops_v1`
- Output: `dataset/delaurier_force_prior_moment_direct_measured_massprops_v1`

**Step 1: Build aligned residual split**

```bash
python scripts/build_delaurier_residual_split.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1 \
  --prior-root artifacts/20260602_delaurier_prior_measured_massprops_v1 \
  --output-root dataset/delaurier_force_prior_moment_direct_measured_massprops_v1 \
  --prior-name delaurier_physical_calibrated_measured_massprops_v1
```

**Step 2: Verify row counts and columns**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("dataset/delaurier_force_prior_moment_direct_measured_massprops_v1")
for split in ["train","val","test"]:
    df = pd.read_parquet(root / f"{split}_samples.parquet")
    print(split, len(df))
    required = ["fx_b","fy_b","fz_b","mx_b","my_b","mz_b"]
    missing = [c for c in required if c not in df.columns]
    print("missing required:", missing)
PY
```

Expected: no missing required targets.

## Task 8: Train B+C Correction Models

**Files:**

- Input: `dataset/delaurier_force_prior_moment_direct_measured_massprops_v1`
- Output: `artifacts/20260602_bc_correction_measured_massprops_v1`

**B+C definition for this rerun:**

- B: phase/condition/frequency-structured residual correction, keeping the low-dimensional prior explicit.
- C: deployable effective-wrench correction model suitable for later simulator embedding.
- A remains out of scope for this clean rerun.

**Step 1: Run deployable correction**

```bash
python scripts/train_deployable_wrench_correction_v2.py \
  --split-root dataset/delaurier_force_prior_moment_direct_measured_massprops_v1 \
  --prior-root artifacts/20260602_delaurier_prior_measured_massprops_v1 \
  --output-root artifacts/20260602_bc_correction_measured_massprops_v1/deployable_v2 \
  --overwrite
```

**Step 2: Run phase-structured correction**

```bash
python scripts/train_phase_structured_wrench_correction.py \
  --split-root dataset/delaurier_force_prior_moment_direct_measured_massprops_v1 \
  --prior-root artifacts/20260602_delaurier_prior_measured_massprops_v1 \
  --v2-reference-root artifacts/20260602_bc_correction_measured_massprops_v1/deployable_v2 \
  --output-root artifacts/20260602_bc_correction_measured_massprops_v1/phase_structured \
  --overwrite
```

**Step 3: Run neural baseline comparison only if needed for the table**

If the paper still includes direct neural and sequence baselines:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/delaurier_force_prior_moment_direct_measured_massprops_v1 \
  --output-dir artifacts/20260602_bc_correction_measured_massprops_v1/baseline_comparison \
  --model-types mlp,gru,transformer \
  --device auto
```

If the revised paper focuses on B+C only, skip full baseline retraining and state that old baseline tables are not carried forward.

## Task 9: Evaluate Per-Channel Metrics on Aligned Rows

**Files:**

- Input: trained model bundle from `artifacts/20260602_bc_correction_measured_massprops_v1`
- Input: `dataset/delaurier_force_prior_moment_direct_measured_massprops_v1`
- Output: `artifacts/20260602_bc_correction_measured_massprops_v1/evaluation`

**Step 1: Evaluate the selected model on test split**

Command shape:

```bash
python scripts/evaluate_delaurier_residual_model.py \
  --residual-split-root dataset/delaurier_force_prior_moment_direct_measured_massprops_v1 \
  --model-bundle artifacts/20260602_bc_correction_measured_massprops_v1/<selected_model>/model_bundle.pt \
  --output-dir artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/test \
  --split test \
  --device auto
```

Use the actual selected model path after Task 8.

**Step 2: Produce paper-ready metric tables**

Create:

```text
artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/per_channel_metrics.csv
artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/per_log_metrics.csv
artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/summary.md
```

Metrics:

```text
RMSE
MAE
R2
RMSE/std
median absolute error
per-log RMSE mean/std
```

Separate force and moment tables. Do not use mixed-unit overall RMSE as the main claim.

## Task 10: Rerun Residual Diagnostics and Figures

**Files:**

- Input: aligned prediction parquet from Task 9
- Output: `artifacts/20260602_residual_diagnostics_measured_massprops_v1`

**Step 1: Locate aligned parquet**

The evaluation output should include a parquet with labels, prior predictions, corrected predictions, and residual columns. If not, add an export option or create it from the model predictions.

Expected file:

```text
artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/test/test_predictions_aligned.parquet
```

**Step 2: Phase residual analysis**

```bash
python scripts/analyze_delaurier_residual_phase.py \
  --aligned-parquet artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/test/test_predictions_aligned.parquet \
  --output-dir artifacts/20260602_residual_diagnostics_measured_massprops_v1/phase \
  --phase-bins 36
```

**Step 3: Condition residual analysis**

```bash
python scripts/analyze_delaurier_residual_conditions.py \
  --aligned-parquet artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/test/test_predictions_aligned.parquet \
  --output-dir artifacts/20260602_residual_diagnostics_measured_massprops_v1/conditions \
  --quantile-bins 5 \
  --min-samples 500
```

**Step 4: Frequency residual analysis**

```bash
python scripts/analyze_delaurier_residual_frequency.py \
  --aligned-parquet artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/test/test_predictions_aligned.parquet \
  --output-dir artifacts/20260602_residual_diagnostics_measured_massprops_v1/frequency
```

**Step 5: Generate representative curves**

Use the existing plotting script if compatible:

```bash
python scripts/plot_prediction_curves.py \
  --aligned-parquet artifacts/20260602_bc_correction_measured_massprops_v1/evaluation/test/test_predictions_aligned.parquet \
  --output-dir artifacts/20260602_residual_diagnostics_measured_massprops_v1/curves
```

Adjust flags to the actual CLI. If the script cannot consume the aligned parquet, create a narrow plotting helper.

## Task 11: Rerun Replay Sanity Checks

**Files:**

- Input: `dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1`
- Input: `metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Output: `artifacts/20260602_replay_measured_massprops_v1`

**Step 1: Translational/oracle short-horizon replay**

```bash
python scripts/evaluate_short_horizon_replay.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1 \
  --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-root artifacts/20260602_replay_measured_massprops_v1/oracle_sanity \
  --split test \
  --modes oracle_teacher_forced \
  --horizons 0.10,0.25,0.50,1.00,2.00 \
  --stride-s 0.25 \
  --overwrite
```

**Step 2: Rotational diagnostics**

```bash
python scripts/diagnose_rotational_replay_oracle.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_sg0p03_v1 \
  --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output-root artifacts/20260602_replay_measured_massprops_v1/rotational_oracle_diagnostics \
  --split test \
  --horizons 0.10,0.25,0.50,1.00,2.00 \
  --stride-s 0.25 \
  --overwrite
```

**Step 3: Decide replay claim level**

Use this rule:

- If oracle translational replay is stable but rotation remains poor, use only translational/local replay language.
- If both translation and rotation pass, consider a small prior-vs-corrected replay comparison.
- If oracle replay fails, do not use replay as evidence for simulator deployment.

## Task 12: Build Old-vs-New Comparison Report

**Files:**

- Input old results: previous paper artifacts and old split
- Input new results: all `measured_massprops_v1` artifacts
- Output: `artifacts/20260602_old_vs_measured_massprops_comparison/summary.md`

**Step 1: Compare labels**

Report:

```text
force target scale old/new
moment target scale old/new
per-channel target standard deviation old/new
label finite ratios
sample counts
```

**Step 2: Compare model metrics**

Report:

```text
calibrated prior old/new
B+C corrected model old/new
per-channel RMSE old/new
per-channel RMSE/std old/new
per-log RMSE old/new
```

**Step 3: Compare diagnostics**

Report:

```text
phase residual peak-to-peak old/new
frequency retained energy old/new
condition-binned worst-bin RMSE old/new
replay oracle metrics old/new
```

**Step 4: Write interpretation**

Use conservative wording:

```text
Measured mass properties primarily rescale force labels through mass and substantially change moment labels through inertia. The force residual structure should remain the core evidence if phase/frequency/condition patterns persist. Moment results should be described as effective rotational-wrench fitting under measured but still approximate neutral-wing inertia.
```

## Task 13: Update Paper Artifacts and Notes

**Files:**

- Update if results support it: `/home/zn/paper/AeroConf_effective_aero/sections/*.tex`
- Update figure files under: `/home/zn/paper/AeroConf_effective_aero/figures/`
- Update notes: `/home/zn/paper/AeroConf_effective_aero/research_notes/`

**Step 1: Copy only final paper figures**

Candidate figures:

```text
operating envelope, if split/log counts changed
per-channel result table
representative prediction curves
phase residual figure
condition residual figure
frequency residual figure
optional translational replay sanity figure
```

**Step 2: Update paper text**

Mandatory text changes:

- Sec. 2 mass/CG/inertia source.
- Effective-wrench reconstruction assumptions.
- Results tables and numbers.
- Limitations: neutral-wing lumped inertia, off-diagonal products neglected, wing time-varying inertia ignored.
- Discussion: moment labels improved in metadata realism but still sensitive to derivative and reference-point assumptions.

**Step 3: Compile the paper**

```bash
cd /home/zn/paper/AeroConf_effective_aero
latexmk -pdf main.tex
```

Expected: PDF builds without unresolved citations or missing figures.

## Task 14: Final Verification Checklist

Run:

```bash
cd /home/zn/flap-system-identification
pytest -q tests/test_metadata.py
```

Run any tests added in this plan:

```bash
pytest -q tests/test_regenerate_canonical_from_accepted_logs.py tests/test_materialize_split_from_log_assignment.py
```

Check git state:

```bash
git status --short
```

Final report must include:

```text
new seed dataset path
new split path
new smoothed split path
new prior/calibration path
new B+C artifact path
new diagnostics path
new replay path
old-vs-new summary path
whether paper PDF compiled
which results are ready for paper use
which results remain diagnostic only
```
