# Phase-Structured Wrench Correction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run a clean B+C experiment for the paper: phase-structured force correction around the DeLaurier-like prior plus wrench-consistent force-arm moment correction, using only deployable features and validation-only model selection.

**Architecture:** Create a new standalone experiment script that reuses the existing v2 data-loading, metrics, ridge, and feature-derivation helpers, but exposes a cleaner paper-facing candidate set. Force candidates must separate slow-only, phase-only, phase-structured, and phase-structured-plus-rate/control/lateral terms. Moment candidates must compare direct residual, force-arm-only, force-arm-plus-free-torque, and prior-referenced hybrid forms using the selected deployable force candidate. Results go to a new artifact root and a concise docs/results report; old exploratory experiments remain untouched.

**Tech Stack:** Python, pandas, NumPy, matplotlib for optional plots, pytest, existing `scripts/train_deployable_wrench_correction_v2.py` helpers, whole-log split parquet data.

---

## Paper Framing Constraints

Use these constraints throughout implementation and reporting:

- The method is **Phase-Structured Wrench Correction**, not a generic neural benchmark.
- The force method is B:
  \[
  F_{\mathrm{corr}} = F_{\mathrm{prior}} + \Delta F_{\mathrm{phase}}(\phi,z)
  \]
  where the correction uses a low-order phase basis and condition-dependent amplitudes.
- The moment method is C:
  \[
  M_{\mathrm{corr}} = \hat r(\phi,z)\times F_{\mathrm{corr}} + \hat\tau_{\mathrm{free}}(\phi,z)
  \]
  plus direct and hybrid ablations.
- Do **not** use `true_force`, label force, target wrench, future samples, or non-deployable features as inference inputs.
- Selection of force family, moment form, feature family, and alpha must use validation metrics only.
- Test metrics are final reporting only.
- Claims must say held-out log prediction, not closed-loop simulator validation.
- Do not modify old artifacts:
  - `artifacts/20260525_delaurier_greybox_force_correction_v1`
  - `artifacts/20260525_dynamic_arm_moment_head_v1`
  - `artifacts/20260526_deployable_wrench_correction_v2`

## Fixed Inputs and Outputs

Default inputs:

```text
split_root: dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
prior_root: artifacts/delaurier_physical_prior_v1
v2_reference_root: artifacts/20260526_deployable_wrench_correction_v2
```

New output root:

```text
artifacts/20260527_phase_structured_wrench_correction_v1
```

New report:

```text
docs/results/2026-05-27-phase-structured-wrench-correction.md
```

New code and tests:

```text
scripts/train_phase_structured_wrench_correction.py
tests/test_phase_structured_wrench_correction.py
```

## Candidate Definitions

### Force Candidate Families

Use ridge linear models and evaluate each family with alpha grid:

```text
alphas: 0, 0.001, 0.01, 0.1, 1, 10, 100, 1000
```

Candidate families:

1. `prior`
   No learned correction.

2. `slow_only`
   No phase harmonics. Use deployable slow flight-condition variables:
   `alpha_rad`, `flap_frequency_hz`, `true_airspeed_m_s`, `dynamic_pressure_pa`, `alpha_rad_x_flap_frequency_hz`, and available lateral slow proxies such as `beta_proxy_rad`, `v_air_b_y`, `q_dyn_x_beta_proxy`.

3. `phase_only`
   Use only low-order phase basis:
   `phase_sin_1`, `phase_cos_1`, `phase_sin_2`, `phase_cos_2`.

4. `phase_structured`
   Use slow variables, phase basis, and condition-dependent phase amplitude terms:
   slow variables plus `slow_variable_x_phase_sin_1`, `slow_variable_x_phase_cos_1`, and where already available `phase_sin_2`, `phase_cos_2`. Keep the dictionary compact and deterministic.

5. `phase_structured_plus_rates_controls`
   Use `phase_structured` plus body rates, controls, lateral proxies, and first-harmonic interactions:
   rates, controls, lateral, and interactions from the existing v2 feature builder.

For each non-prior family, evaluate two variants:

```text
additive: delta_F = Phi(x) B
affine:   delta_F = Phi(x, F_prior * Phi(x)) B
```

This yields a clean ablation:

```text
prior -> slow_only -> phase_only -> phase_structured -> phase_structured_plus_rates_controls
```

### Moment Candidate Forms

Use the validation-selected deployable force candidate as `F_corr`.

1. `prior_moment`
   Use prior moment without learned correction.

2. `direct_residual`
   \[
   M_{\mathrm{corr}} = M_{\mathrm{prior}} + \Delta M_\theta(x)
   \]
   This is an ablation, not the preferred paper method.

3. `force_arm_only`
   \[
   M_{\mathrm{corr}} = \hat r_\theta(x)\times F_{\mathrm{corr}}
   \]

4. `force_arm_plus_free`
   \[
   M_{\mathrm{corr}} = \hat r_\theta(x)\times F_{\mathrm{corr}} + \hat\tau_\theta(x)
   \]
   This is the preferred C candidate.

5. `hybrid_prior_arm_free`
   \[
   M_{\mathrm{corr}} = M_{\mathrm{prior}} + \hat r_\theta(x)\times F_{\mathrm{corr}} + \hat\tau_\theta(x)
   \]
   This tests whether retaining prior moment helps.

The moment feature families should include at least:

```text
phase_structured
phase_structured_plus_rates_controls
```

Selection target: validation `moment_mean` RMSE. Tie policy: prefer `force_arm_plus_free`, then `hybrid_prior_arm_free`, then `force_arm_only`, then `direct_residual`; within ties prefer smaller feature family and larger regularization.

## Required Outputs

The script must write:

```text
model_config.json
force_metrics_by_split.csv
moment_metrics_by_split.csv
per_log_metrics.csv
force_model_selection.csv
moment_model_selection.csv
prediction_parquets/train_predictions.parquet
prediction_parquets/val_predictions.parquet
prediction_parquets/test_predictions.parquet
inference_model_state.json
README.md
```

Metrics must include for each split/model/channel:

```text
n, mae, rmse, bias, r2
```

Aggregate rows:

```text
force_mean
moment_mean
```

Do not create a combined force+moment RMSE that mixes N and N m.

`inference_model_state.json` must include:

- selected force family, variant, features, alpha, normalization, coefficients;
- selected moment form, feature family, alpha, normalization, coefficients;
- `uses_true_force_for_inference: false`;
- required feature columns;
- prior root and split root;
- statement that validation was used for selection and test was final reporting.

## Task 1: Add Tests for Phase-Structured Feature Families

**Files:**

- Create: `tests/test_phase_structured_wrench_correction.py`
- Create later: `scripts/train_phase_structured_wrench_correction.py`

**Step 1: Write failing tests**

Add tests that import:

```python
from scripts.train_phase_structured_wrench_correction import (
    build_phase_structured_feature_families,
    phase_structured_force_family_specs,
)
```

Test with a small synthetic frame containing:

```python
phase_corrected_rad
cycle_flap_frequency_hz
airspeed_validated.true_airspeed_m_s
vehicle_air_data.rho
alpha_rad
vehicle_angular_velocity.xyz[0]
vehicle_angular_velocity.xyz[1]
vehicle_angular_velocity.xyz[2]
servo_left_elevon
servo_right_elevon
servo_rudder
air_relative_velocity_b_x
air_relative_velocity_b_y
air_relative_velocity_b_z
```

Assertions:

- `slow_only` contains no `phase_sin_*` or `phase_cos_*`.
- `phase_only` contains only phase harmonic features.
- `phase_structured` contains phase features and slow-by-phase interaction features.
- `phase_structured_plus_rates_controls` contains body-rate, control, lateral, and phase-interaction features.
- Returned metadata has `uses_true_force` false.
- No family contains label targets (`fx_b`, `fy_b`, `fz_b`, `mx_b`, `my_b`, `mz_b`) or columns prefixed by `label_`.

**Step 2: Run failing test**

Run:

```bash
pytest tests/test_phase_structured_wrench_correction.py::test_phase_structured_feature_families_are_disjoint_and_deployable -q
```

Expected: fail because the new module does not exist.

**Step 3: Implement minimal feature-family API**

Create `scripts/train_phase_structured_wrench_correction.py`.

Reuse:

```python
from scripts.train_deployable_wrench_correction_v2 import (
    build_v2_feature_frame,
    v2_feature_groups,
)
```

Implement:

```python
def build_phase_structured_feature_families(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, object]]:
    features, spec = build_v2_feature_frame(frame)
    groups = v2_feature_groups(features.columns)
    families = phase_structured_force_family_specs(features.columns, groups)
    return features, families, {**spec, "uses_true_force": False}
```

Implement `phase_structured_force_family_specs(columns, groups)` with the five families above. Keep ordering deterministic and remove duplicates.

**Step 4: Run passing tests**

Run:

```bash
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/train_phase_structured_wrench_correction.py tests/test_phase_structured_wrench_correction.py
git commit -m "feat: add phase-structured feature families"
```

## Task 2: Implement Force Candidate Training and Selection

**Files:**

- Modify: `scripts/train_phase_structured_wrench_correction.py`
- Modify: `tests/test_phase_structured_wrench_correction.py`

**Step 1: Write tests**

Add tests for:

- `_force_design_for_family(features, prior_force, family, variant)` returns original feature columns for `additive`.
- `affine` appends `prior_fx_b_x_*`, `prior_fy_b_x_*`, `prior_fz_b_x_*` interactions.
- `select_force_candidate(metrics)` selects lowest validation `force_mean` RMSE.
- Tie policy prefers smaller family order: `slow_only`, `phase_only`, `phase_structured`, `phase_structured_plus_rates_controls`; then `additive` before `affine`; then smaller alpha.

**Step 2: Run tests**

```bash
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: fail until implementation exists.

**Step 3: Implement**

Reuse from v2 script where possible:

```python
RidgeMultiOutputModel
ForceCorrectionModel
fit_ridge_multi_output
_fit_ridge_frame
_append_prior_force_interactions
force_metrics
_array_from_prefixed_columns
```

Implement `fit_phase_structured_force_models(split_frames, families, alphas, variants=("additive", "affine"))`.

Required behavior:

- Add prior baseline rows for train/val/test.
- Fit only on train.
- Evaluate train/val/test for every family/variant/alpha.
- Select by validation `force_mean` RMSE only.
- Mark selected rows with `is_selected=True`.
- Return metrics DataFrame and selected force model.

**Step 4: Run tests**

```bash
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/train_phase_structured_wrench_correction.py tests/test_phase_structured_wrench_correction.py
git commit -m "feat: train phase-structured force correction"
```

## Task 3: Implement Moment Candidate Training and Wrench Consistency

**Files:**

- Modify: `scripts/train_phase_structured_wrench_correction.py`
- Modify: `tests/test_phase_structured_wrench_correction.py`

**Step 1: Write tests**

Add synthetic tests for:

- `cross_arm_force(r_hat, force)` obeys `r x F`.
- `force_arm_only` predicts `r_hat x F_corr` without prior moment.
- `force_arm_plus_free` predicts `r_hat x F_corr + tau_free`.
- `hybrid_prior_arm_free` predicts `M_prior + r_hat x F_corr + tau_free`.
- `direct_residual` predicts `M_prior + delta_M`.
- All moment candidate configs have `uses_true_force_for_inference=False`.
- Selection uses validation `moment_mean` RMSE and tie policy prefers `force_arm_plus_free` over direct residual.

**Step 2: Run tests**

```bash
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: fail until moment implementation exists.

**Step 3: Implement**

Reuse from v2 script where possible:

```python
FeatureTransform
MomentCorrectionModel
cross_arm_force
moment_metrics
```

If reused names do not match the needed forms, wrap them with a clean local API rather than mutating v2 behavior. Implement:

```python
fit_phase_structured_moment_models(split_frames, selected_force_model, families, alphas)
```

Required behavior:

- First apply selected force model to each split and add `force_corr_fx_b`, `force_corr_fy_b`, `force_corr_fz_b` or `force_phase_structured_*` columns.
- Candidate forms: `direct_residual`, `force_arm_only`, `force_arm_plus_free`, `hybrid_prior_arm_free`.
- Fit on train, select on val `moment_mean`, evaluate train/val/test.
- Include prior moment baseline.
- For `force_arm_only`, solve coefficients for an effective arm design without free torque.
- For `force_arm_plus_free`, solve a joint linear least-squares/ridge system for arm coefficients and free-torque coefficients.
- For `hybrid_prior_arm_free`, target is `M_label - M_prior`.

**Step 4: Run tests**

```bash
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/train_phase_structured_wrench_correction.py tests/test_phase_structured_wrench_correction.py
git commit -m "feat: train wrench-consistent moment correction"
```

## Task 4: Add Full Experiment CLI and Artifact Writer

**Files:**

- Modify: `scripts/train_phase_structured_wrench_correction.py`
- Modify: `tests/test_phase_structured_wrench_correction.py`

**Step 1: Write tests**

Add a small temporary-directory integration test that:

- Creates tiny train/val/test frames with required label/prior columns and deployable features.
- Runs an internal `run_phase_structured_experiment(...)` function with a small alpha tuple.
- Asserts output files exist:
  - `force_metrics_by_split.csv`
  - `moment_metrics_by_split.csv`
  - `model_config.json`
  - `inference_model_state.json`
  - prediction parquet for `test`
- Asserts `inference_model_state.json` has `uses_true_force_for_inference` false.

Do not run the full real dataset inside unit tests.

**Step 2: Run tests**

```bash
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: fail until writer is implemented.

**Step 3: Implement**

Implement CLI:

```bash
python scripts/train_phase_structured_wrench_correction.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --prior-root artifacts/delaurier_physical_prior_v1 \
  --v2-reference-root artifacts/20260526_deployable_wrench_correction_v2 \
  --output-root artifacts/20260527_phase_structured_wrench_correction_v1 \
  --alphas 0,0.001,0.01,0.1,1,10,100,1000 \
  --overwrite
```

Implement `run_phase_structured_experiment(...)`:

- load train/val/test samples and prior predictions using v2 script conventions;
- build phase-structured features;
- train force candidates;
- train moment candidates using selected force;
- write metrics and predictions;
- write model config and inference state;
- write README with selected candidates and top-line val/test numbers.

If existing v2 loading helpers are not public, copy the minimal loading logic from `train_deployable_wrench_correction_v2.py` with clear comments.

**Step 4: Run tests**

```bash
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/train_phase_structured_wrench_correction.py tests/test_phase_structured_wrench_correction.py
git commit -m "feat: add phase-structured experiment runner"
```

## Task 5: Run the Full B+C Experiment

**Files:**

- Generated under: `artifacts/20260527_phase_structured_wrench_correction_v1/`

**Step 1: Run unit tests**

```bash
pytest tests/test_phase_structured_wrench_correction.py tests/test_deployable_wrench_correction_v2.py -q
```

Expected: pass.

**Step 2: Run full experiment**

```bash
python scripts/train_phase_structured_wrench_correction.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --prior-root artifacts/delaurier_physical_prior_v1 \
  --v2-reference-root artifacts/20260526_deployable_wrench_correction_v2 \
  --output-root artifacts/20260527_phase_structured_wrench_correction_v1 \
  --alphas 0,0.001,0.01,0.1,1,10,100,1000 \
  --overwrite
```

Expected:

- command finishes without exception;
- writes all required artifacts;
- `force_metrics_by_split.csv` has all force families;
- `moment_metrics_by_split.csv` has all moment forms;
- selected force and moment models are validation-selected;
- test rows are present but not used in selection.

**Step 3: Inspect key outputs**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
root = Path("artifacts/20260527_phase_structured_wrench_correction_v1")
cfg = json.loads((root / "model_config.json").read_text())
print(json.dumps(cfg.get("selected", cfg), indent=2)[:2000])
force = pd.read_csv(root / "force_metrics_by_split.csv")
moment = pd.read_csv(root / "moment_metrics_by_split.csv")
print(force[(force.split=="test") & (force.target=="force_mean")][["model","rmse","r2","is_selected"]].sort_values("rmse").head(12).to_string(index=False))
print(moment[(moment.split=="test") & (moment.target=="moment_mean")][["model","rmse","r2","is_selected"]].sort_values("rmse").head(12).to_string(index=False))
PY
```

Expected: printed selected models and top test metrics.

**Step 4: Do not commit artifacts unless requested**

`artifacts/` is gitignored. Do not force-add heavy artifacts.

## Task 6: Write Paper-Facing Result Report

**Files:**

- Create: `docs/results/2026-05-27-phase-structured-wrench-correction.md`

**Step 1: Read experiment outputs**

Use:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("artifacts/20260527_phase_structured_wrench_correction_v1")
force = pd.read_csv(root / "force_metrics_by_split.csv")
moment = pd.read_csv(root / "moment_metrics_by_split.csv")
per_log = pd.read_csv(root / "per_log_metrics.csv")
print(force[(force.split=="test") & force.target.isin(["fx_b","fy_b","fz_b","force_mean"])].to_string(index=False))
print(moment[(moment.split=="test") & moment.target.isin(["mx_b","my_b","mz_b","moment_mean"])].to_string(index=False))
print(per_log.head().to_string(index=False))
PY
```

**Step 2: Write report**

Report must include:

- command used;
- selected force candidate and validation metric;
- selected moment candidate and validation metric;
- force test table: prior, slow-only, phase-only, phase-structured, phase-structured-plus-rate/control/lateral, selected candidate;
- moment test table: prior, direct residual, force-arm-only, force-arm-plus-free, hybrid;
- per-log stability notes;
- `r_hat` magnitude summary if available;
- `tau_free` fraction summary if available;
- explicit limitations for `fy_b` and `mz_b`;
- paper-safe wording block.

Use conservative language:

```text
The results demonstrate held-out log effective-wrench prediction, not simulator rollout validation.
The learned arm is an effective wrench parameterization, not an identified center of pressure.
Residual attribution motivates features but does not prove aerodynamic causality.
```

**Step 3: Run documentation sanity**

```bash
python -m py_compile scripts/train_phase_structured_wrench_correction.py
pytest tests/test_phase_structured_wrench_correction.py -q
```

Expected: pass.

**Step 4: Commit code and report**

```bash
git add scripts/train_phase_structured_wrench_correction.py tests/test_phase_structured_wrench_correction.py docs/plans/2026-05-27-phase-structured-wrench-correction.md docs/results/2026-05-27-phase-structured-wrench-correction.md
git commit -m "feat: add phase-structured wrench correction experiment"
```

## Final Acceptance Checklist

Before reporting completion:

- [ ] `pytest tests/test_phase_structured_wrench_correction.py -q` passes.
- [ ] `python -m py_compile scripts/train_phase_structured_wrench_correction.py` passes.
- [ ] Full experiment artifacts exist under `artifacts/20260527_phase_structured_wrench_correction_v1`.
- [ ] `inference_model_state.json` says `uses_true_force_for_inference: false`.
- [ ] Test metrics are reported but not used for selection.
- [ ] Result report clearly states held-out log prediction only.
- [ ] `fy_b` and `mz_b` limitations are not hidden.
- [ ] No old artifacts were modified.
- [ ] Code/report commit exists, while heavy artifacts remain untracked.
