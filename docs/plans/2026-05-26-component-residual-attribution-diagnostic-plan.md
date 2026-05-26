# Component Residual Attribution Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a held-out-log residual attribution diagnostic that identifies candidate mismatch sources for the DeLaurier/grey-box flapping-wing wrench models without claiming strict aerodynamic causality.

**Architecture:** Add one standalone analysis script that builds a split-aligned residual frame, constructs physically motivated proxy variables, runs binning/correlation/feature-group residual regressions, and writes tables, plots, and a short interpretation README. Keep the analysis separate from training code; reuse existing DeLaurier/grey-box prediction artifacts and helper conventions where possible.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, scipy if available, matplotlib, pytest.

---

## Non-Negotiable Method Boundary

This diagnostic uses observational flight-log residuals. It can identify structured model-mismatch patterns and prioritize correction directions, but it must not claim isolated aerodynamic causality. Any generated README or result note must use language such as `associated with`, `consistent with`, `candidate source`, and `held-out-log residual explainability`; avoid `caused by` unless a controlled intervention exists.

Do not modify training scripts, model definitions, canonical parquet generation, or split files. Do not commit unless the user asks for a commit after review.

## Current Inputs

Use these defaults:

- Split root: `dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1`
- Force correction root: `artifacts/20260525_delaurier_greybox_force_correction_v1`
- Moment head root: `artifacts/20260525_dynamic_arm_moment_head_v1`
- Physical prior root: `artifacts/delaurier_physical_prior_v1`
- Output root: `artifacts/20260526_component_residual_attribution_v1`

Relevant existing files:

- `scripts/train_delaurier_greybox_force_correction.py`
- `scripts/train_dynamic_arm_moment_head.py`
- `scripts/analyze_delaurier_residual_conditions.py`
- `tests/test_delaurier_greybox_force_correction.py`
- `tests/test_dynamic_arm_moment_head.py`
- `tests/test_analyze_delaurier_residual_conditions.py`
- `docs/results/2026-05-25-residual-guided-delaurier-greybox-force-arm.md`

## Residual Definitions

The script must support at least two residual baselines:

```text
force_prior_residual     = label_force - prior_force
force_corrected_residual = label_force - corrected_force
moment_prior_residual    = label_moment - prior_moment
moment_current_residual  = label_moment - current_moment_prediction
```

For the present paper question, the most important targets are:

```text
fy_b, mx_b, my_b, mz_b
```

`fx_b` and `fz_b` should still be included in tables for completeness.

## Expected Output Files

The script should write:

```text
artifacts/20260526_component_residual_attribution_v1/
  residual_frame.parquet
  residual_variable_bins.csv
  residual_variable_bin_summary.csv
  residual_variable_rankings.csv
  per_log_residual_variable_rankings.csv
  residual_feature_group_ablation.csv
  per_log_residual_feature_group_ablation.csv
  residual_attribution_config.json
  figures/
    residual_group_ablation_key_targets.png
    residual_rank_heatmap_key_targets.png
    residual_bins_key_targets.png
  README.md
```

If a plot is impractical because a dependency is missing, the script should still write the CSV files and clearly state the skipped plot in `README.md`.

---

## Task 1: Tests for Candidate Variable Construction

**Files:**

- Create: `tests/test_component_residual_attribution.py`
- Create later: `scripts/analyze_component_residual_attribution.py`

**Step 1: Write failing tests**

Add tests that call functions that do not exist yet:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_component_residual_attribution import build_candidate_variables, build_residual_frame


def test_build_candidate_variables_derives_phase_lateral_and_control_proxies() -> None:
    frame = pd.DataFrame(
        {
            "phase_corrected_rad": [0.0, np.pi / 2.0],
            "cycle_flap_frequency_hz": [4.0, 5.0],
            "airspeed_validated.true_airspeed_m_s": [10.0, 20.0],
            "vehicle_air_data.rho": [1.2, 1.0],
            "alpha_rad": [0.1, 0.2],
            "vehicle_angular_velocity.xyz[0]": [0.01, 0.02],
            "vehicle_angular_velocity.xyz[1]": [0.03, 0.04],
            "vehicle_angular_velocity.xyz[2]": [0.05, 0.06],
            "actuator_servos.servo[0]": [0.1, 0.2],
            "actuator_servos.servo[1]": [0.3, 0.4],
            "actuator_servos.servo[2]": [0.5, 0.6],
            "air_relative_velocity_b_x": [10.0, 20.0],
            "air_relative_velocity_b_y": [1.0, -2.0],
            "air_relative_velocity_b_z": [-0.5, -1.0],
        }
    )

    variables, spec = build_candidate_variables(frame)

    assert "phase_sin_1" in variables
    assert "phase_cos_1" in variables
    assert "dynamic_pressure_pa" in variables
    assert "beta_proxy_rad" in variables
    assert "q_dyn_x_beta_proxy" in variables
    assert "body_rate_p" in variables
    assert "body_rate_q" in variables
    assert "body_rate_r" in variables
    assert "servo_0" in variables
    assert "servo_1" in variables
    assert "servo_2" in variables
    assert variables.loc[0, "dynamic_pressure_pa"] == 0.5 * 1.2 * 10.0**2
    assert np.isclose(variables.loc[0, "beta_proxy_rad"], np.arctan2(1.0, 10.0))
    assert spec["phase_column"] == "phase_corrected_rad"


def test_build_residual_frame_keeps_prior_and_corrected_residuals() -> None:
    samples = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "log_id": ["log_a", "log_a"],
            "fx_b": [10.0, 12.0],
            "fy_b": [1.0, 2.0],
            "fz_b": [-5.0, -6.0],
            "mx_b": [0.1, 0.2],
            "my_b": [0.3, 0.4],
            "mz_b": [0.5, 0.6],
            "phase_corrected_rad": [0.0, 1.0],
            "cycle_flap_frequency_hz": [4.0, 4.0],
            "airspeed_validated.true_airspeed_m_s": [10.0, 11.0],
        }
    )
    force_predictions = pd.DataFrame(
        {
            "label_fx_b": [10.0, 12.0],
            "label_fy_b": [1.0, 2.0],
            "label_fz_b": [-5.0, -6.0],
            "prior_fx_b": [8.0, 10.0],
            "prior_fy_b": [0.0, 1.0],
            "prior_fz_b": [-4.0, -5.0],
            "corrected_fx_b": [9.0, 11.5],
            "corrected_fy_b": [0.5, 1.8],
            "corrected_fz_b": [-4.5, -5.8],
        }
    )
    prior_predictions = pd.DataFrame(
        {
            "prior_mx_b": [0.0, 0.1],
            "prior_my_b": [0.1, 0.2],
            "prior_mz_b": [0.4, 0.5],
        }
    )
    current_moment_predictions = pd.DataFrame(
        {
            "pred_mx_b": [0.05, 0.15],
            "pred_my_b": [0.25, 0.35],
            "pred_mz_b": [0.45, 0.55],
        }
    )

    residual = build_residual_frame(
        split="train",
        samples=samples,
        force_predictions=force_predictions,
        prior_predictions=prior_predictions,
        current_moment_predictions=current_moment_predictions,
    )

    assert residual["split"].tolist() == ["train", "train"]
    assert residual["force_prior_residual_fx_b"].tolist() == [2.0, 2.0]
    assert residual["force_corrected_residual_fy_b"].tolist() == [0.5, 0.2]
    assert np.allclose(residual["moment_prior_residual_my_b"], [0.2, 0.2])
    assert np.allclose(residual["moment_current_residual_mz_b"], [0.05, 0.05])
```

**Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py -q
```

Expected: fail because `scripts.analyze_component_residual_attribution` does not exist.

---

## Task 2: Implement Candidate Variables and Residual Frame

**Files:**

- Create: `scripts/analyze_component_residual_attribution.py`
- Modify: `tests/test_component_residual_attribution.py`

**Step 1: Add minimal implementation**

Implement:

```python
FORCE_TARGETS = ("fx_b", "fy_b", "fz_b")
MOMENT_TARGETS = ("mx_b", "my_b", "mz_b")
ALL_TARGETS = FORCE_TARGETS + MOMENT_TARGETS

def build_candidate_variables(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    ...

def build_residual_frame(
    *,
    split: str,
    samples: pd.DataFrame,
    force_predictions: pd.DataFrame,
    prior_predictions: pd.DataFrame,
    current_moment_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ...
```

Implementation requirements:

- Use existing column search style from `scripts/train_delaurier_greybox_force_correction.py`.
- Use `phase_corrected_rad` if present, otherwise try `wing_phase.phase_rad`, `drive_phase_rad`, `encoder_phase_rad`, `phase_raw_rad`.
- Use `cycle_flap_frequency_hz`, `flap_frequency_hz`, or `encoder_rpm_est / 60`.
- Use `airspeed_validated.true_airspeed_m_s`, `airspeed_validated.calibrated_airspeed_m_s`, or `airspeed_validated.indicated_airspeed_m_s`.
- Compute `dynamic_pressure_pa = 0.5 * rho * V^2`, unless already present.
- Compute `alpha_rad` from existing `alpha_rad` if present; otherwise reuse the body-velocity AoA fallback from `train_delaurier_greybox_force_correction.py`.
- Compute `beta_proxy_rad` from available body relative air velocity if available:

```python
beta_proxy_rad = np.arctan2(v_air_y, np.maximum(np.abs(v_air_x), 1.0e-6))
```

- If exact body-air columns are absent, try deriving body velocity using quaternion/local velocity/wind as in `train_delaurier_greybox_force_correction.py`; otherwise fill beta proxy with zero and record the fallback in `spec`.
- Add phase harmonics, body rates, servo channels, elevon sum/diff when identifiable, and physically motivated interactions:

```text
phase_sin_1, phase_cos_1, phase_sin_2, phase_cos_2
flap_frequency_hz
true_airspeed_m_s
dynamic_pressure_pa
alpha_rad
beta_proxy_rad
v_air_b_x, v_air_b_y, v_air_b_z
body_rate_p, body_rate_q, body_rate_r
servo_0 ... servo_N
elevon_sum_proxy, elevon_diff_proxy
q_dyn_x_beta_proxy
q_dyn_x_body_rate_p/q/r
q_dyn_x_servo_0/1/2...
alpha_rad_x_phase_sin_1/cos_1
beta_proxy_x_phase_sin_1/cos_1
flap_frequency_hz_x_phase_sin_1/cos_1
```

Residual-frame requirements:

- Preserve metadata columns if present: `timestamp_us`, `time_s`, `log_id`, `segment_id`, `cycle_id`, `phase_corrected_rad`, `split`.
- Add candidate variables to the same frame.
- Add label, prior, corrected/current prediction, and residual columns with explicit names.
- Validate equal row counts across split samples and prediction parquets.

**Step 2: Run tests to verify GREEN**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py -q
```

Expected: tests pass.

---

## Task 3: Add Binning and Ranking Diagnostics

**Files:**

- Modify: `scripts/analyze_component_residual_attribution.py`
- Modify: `tests/test_component_residual_attribution.py`

**Step 1: Write failing tests**

Add tests for:

```python
def residual_variable_bin_table(...): ...
def residual_variable_ranking_table(...): ...
```

Expected behavior:

- Quantile bins ignore non-finite values.
- Each row reports `split`, `residual_kind`, `target`, `variable`, `bin`, `sample_count`, `variable_min`, `variable_max`, `variable_median`, `residual_bias`, `residual_mae`, `residual_rmse`.
- Ranking table reports Pearson, Spearman, and mutual information if available.
- The ranking table can group by all data or per log.

**Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py -q
```

Expected: fail because these functions are missing.

**Step 3: Implement diagnostics**

Implement robust helpers:

```python
def residual_variable_bin_table(
    frame: pd.DataFrame,
    *,
    residual_columns: tuple[str, ...],
    variable_columns: tuple[str, ...],
    quantile_bins: int,
    min_samples: int,
) -> pd.DataFrame:
    ...

def summarize_residual_variable_bins(bin_table: pd.DataFrame) -> pd.DataFrame:
    ...

def residual_variable_ranking_table(
    frame: pd.DataFrame,
    *,
    residual_columns: tuple[str, ...],
    variable_columns: tuple[str, ...],
    group_columns: tuple[str, ...] = (),
    min_samples: int,
) -> pd.DataFrame:
    ...
```

Interpretation rules:

- Use Pearson/Spearman as effect-size diagnostics, not p-value-driven claims.
- If `sklearn.feature_selection.mutual_info_regression` is available, compute MI; otherwise fill `mutual_information` with `NaN` and continue.
- Include `abs_pearson`, `abs_spearman`, and a simple combined rank score so tables are easy to sort.

**Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py -q
```

Expected: pass.

---

## Task 4: Add Feature-Group Held-Out Residual Regression

**Files:**

- Modify: `scripts/analyze_component_residual_attribution.py`
- Modify: `tests/test_component_residual_attribution.py`

**Step 1: Write failing tests**

Add tests for:

```python
def default_feature_groups(variable_columns: Iterable[str]) -> dict[str, list[str]]: ...
def residual_feature_group_ablation(...): ...
```

Synthetic test should verify:

- Zero-residual baseline is included.
- A group containing the true explanatory variable improves validation/test RMSE.
- Feature groups skip missing columns instead of failing.
- Alpha selection uses validation split, not test split.

**Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py -q
```

Expected: fail because feature-group functions are missing.

**Step 3: Implement feature-group regression**

Implement Ridge-based held-out residual validation:

```python
def default_feature_groups(variable_columns: Iterable[str]) -> dict[str, list[str]]:
    return {
        "phase": ...,
        "longitudinal": ...,
        "lateral_body": ...,
        "body_rates": ...,
        "tail_controls": ...,
        "phase_interactions": ...,
        "lateral_tail": ...,
        "all_candidate": ...,
    }
```

Implement:

```python
def residual_feature_group_ablation(
    frame: pd.DataFrame,
    *,
    residual_columns: tuple[str, ...],
    feature_groups: dict[str, list[str]],
    alphas: tuple[float, ...],
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Output one aggregate table and one per-log table. For each residual column and feature group, report:

```text
selected_alpha
train_rmse
val_rmse
test_rmse
zero_train_rmse
zero_val_rmse
zero_test_rmse
test_rmse_reduction_fraction
test_residual_r2
n_features
feature_columns
```

Use a small, transparent model:

```text
Pipeline-like behavior:
  finite-value median fill from train
  train mean/std standardization
  Ridge regression solved with NumPy or sklearn
```

No large NN in this diagnostic.

**Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py -q
```

Expected: pass.

---

## Task 5: CLI, Plots, and README

**Files:**

- Modify: `scripts/analyze_component_residual_attribution.py`
- Modify: `tests/test_component_residual_attribution.py`

**Step 1: Add CLI**

Add argparse options:

```text
--split-root
--force-prediction-root
--moment-prediction-root
--prior-root
--output-root
--quantile-bins
--min-samples
--alphas
--key-targets
```

Default values should match this plan.

**Step 2: Load split artifacts**

For each split `train`, `val`, `test`:

- Read `{split_root}/{split}_samples.parquet`
- Read `{force_prediction_root}/prediction_parquets/{split}_predictions.parquet`, falling back to direct `{split}_predictions.parquet`
- Read `{prior_root}/{split}_predictions.parquet`
- Read `{moment_prediction_root}/prediction_parquets/{split}_predictions.parquet`, falling back to direct `{split}_predictions.parquet`
- Build one residual frame per split and concatenate.

**Step 3: Write outputs**

Write all output files listed above. The README should contain:

- Residual definitions.
- Data roots used.
- Top variable associations for key targets.
- Best feature groups by held-out test RMSE reduction.
- A limitations paragraph that explicitly says this is observational residual attribution, not strict causal identification.

**Step 4: Add simple plot helpers**

Use matplotlib with `Agg`. Keep plots practical:

- Bar plot of best feature-group test RMSE reduction for `fy_b`, `mx_b`, `my_b`, `mz_b`.
- Heatmap-like plot of top absolute Spearman/Pearson scores.
- Bin plot for key target-variable pairs if enough data exists.

Tests should not require pixel-perfect plotting; only verify the CLI can run on a tiny synthetic fixture and write expected files.

---

## Task 6: Execute the Diagnostic on Current Artifacts

**Files:**

- Create: `docs/results/2026-05-26-component-residual-attribution-diagnostic.md`

**Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py -q
```

Expected: pass.

**Step 2: Run related regression tests**

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py tests/test_delaurier_greybox_force_correction.py tests/test_dynamic_arm_moment_head.py tests/test_analyze_delaurier_residual_conditions.py -q
```

Expected: pass.

**Step 3: Run the full diagnostic**

Run:

```bash
python scripts/analyze_component_residual_attribution.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --force-prediction-root artifacts/20260525_delaurier_greybox_force_correction_v1 \
  --moment-prediction-root artifacts/20260525_dynamic_arm_moment_head_v1 \
  --prior-root artifacts/delaurier_physical_prior_v1 \
  --output-root artifacts/20260526_component_residual_attribution_v1 \
  --quantile-bins 10 \
  --min-samples 500
```

Expected: writes the artifact directory without modifying training data.

**Step 4: Write result summary**

Create `docs/results/2026-05-26-component-residual-attribution-diagnostic.md` with:

- What was diagnosed.
- Which residual baseline is being discussed.
- Top variable groups for `fy_b`, `mx_b`, `my_b`, `mz_b`.
- Whether held-out-log residual regression supports each candidate source.
- Clear limitations and how this should guide next model revision.

Keep the writing conservative:

```text
This suggests ...
This is consistent with ...
This motivates ...
This does not establish strict causality ...
```

---

## Final Verification

Run:

```bash
python -m pytest tests/test_component_residual_attribution.py tests/test_delaurier_greybox_force_correction.py tests/test_dynamic_arm_moment_head.py tests/test_analyze_delaurier_residual_conditions.py -q
```

If runtime is acceptable, also run:

```bash
python -m pytest -q
```

Report:

- changed files
- generated artifacts
- commands run
- key diagnostic conclusions
- limitations or failures

