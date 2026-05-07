# Lateral Diagnostics Two-Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose why lateral wrench targets (`fy_b`, `mx_b`, `mz_b`) underperform by running two batches of targeted analyses, with a required user-facing checkpoint after Batch 1.

**Architecture:** Add a dedicated diagnostics script that loads an existing model bundle, aligns predictions with true targets and metadata, and writes CSV/Markdown reports. Batch 1 covers target signal scale, per-log lateral metrics, and per-regime lateral bins. Batch 2 covers phase/lag diagnosis and residual-correlation analysis. Reuse existing `predict_model_bundle`, `_targets_for_bundle`, and sequence metadata helpers instead of duplicating model inference.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `src/system_identification/training.py`, existing best model bundle under `artifacts/20260507_transformer_focused_final`.

---

## Inputs

Default model bundle:

```text
artifacts/20260507_transformer_focused_final/runs/transformer_focused_final_hist128_d64_l2_h4_do050/causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
```

Default split root:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
```

Default split:

```text
test
```

Default output dir:

```text
artifacts/20260507_lateral_diagnostics_best_transformer
```

Lateral targets:

```text
fy_b
mx_b
mz_b
```

Reference targets for comparison:

```text
fx_b
fz_b
my_b
```

---

## Required Checkpoint

After completing Batch 1, stop and inform the user before starting Batch 2.

The Batch 1 message must include:

```text
1. Whether lateral targets have worse RMSE/std than reference targets.
2. Which logs are worst for fy_b, mx_b, mz_b.
3. Which regime bins are worst.
4. Whether the evidence points more toward low signal/noise, specific bad logs, or specific flight regimes.
5. Ask for "继续" before Batch 2.
```

Do not continue to Batch 2 until the user says "继续" or equivalent.

---

## Batch 1: Scale, Per-Log, Per-Regime Diagnostics

### Task 1: Add Prediction Alignment Helper

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing test**

Add a test near existing prediction/evaluation tests:

```python
def test_prediction_metadata_frame_for_sequence_bundle_aligns_rows(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 410, "train_log"), ("val", 411, "val_log"), ("test", 412, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "model",
        feature_set_name="paper_no_accel_v2",
        model_type="causal_transformer",
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_dim_feedforward=32,
        device="cpu",
        random_seed=410,
        num_workers=0,
        use_amp=False,
    )
    bundle = torch.load(outputs["model_bundle_path"], map_location="cpu", weights_only=False)
    test_frame = pd.read_parquet(split_root / "test_samples.parquet")

    aligned = training_module.prediction_metadata_frame_for_bundle(bundle, test_frame, split_name="test", batch_size=16)

    assert len(aligned) == 77
    assert {"log_id", "segment_id", "time_s"}.issubset(aligned.columns)
    assert {"true_fy_b", "pred_fy_b", "resid_fy_b"}.issubset(aligned.columns)
```

**Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_training.py::test_prediction_metadata_frame_for_sequence_bundle_aligns_rows -q
```

Expected: fail because `prediction_metadata_frame_for_bundle` does not exist.

**Step 3: Implement helper**

Add to `src/system_identification/training.py` near `predict_model_bundle`:

```python
def prediction_metadata_frame_for_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)
    targets_df = _targets_for_bundle(bundle, frame)

    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        _, _, _, metadata = _sequence_arrays_with_metadata_for_bundle(bundle, frame)
    elif _is_rollout_model_type(bundle.get("model_type", "mlp")):
        _, _, _, _, metadata = _rollout_arrays_for_bundle(bundle, frame)
    else:
        features_df, _ = prepare_windowed_feature_target_frames(
            frame,
            bundle.get("base_feature_columns", bundle["feature_columns"]),
            bundle["target_columns"],
            window_mode=bundle.get("window_mode", "single"),
            window_radius=int(bundle.get("window_radius", 0)),
            window_feature_columns=bundle.get("window_feature_columns"),
        )
        metadata = frame.loc[features_df.index].reset_index(drop=True)

    aligned = metadata.reset_index(drop=True).copy()
    for target in bundle["target_columns"]:
        aligned[f"true_{target}"] = targets_df[target].to_numpy()
        aligned[f"pred_{target}"] = predictions_df[target].to_numpy()
        aligned[f"resid_{target}"] = aligned[f"true_{target}"] - aligned[f"pred_{target}"]
    aligned["split"] = split_name
    return aligned
```

If rollout metadata shape differs, handle it explicitly or raise `NotImplementedError` for rollout models. This lateral diagnostic targets sequence Transformer models first.

**Step 4: Run test to verify GREEN**

Run:

```bash
pytest tests/test_training.py::test_prediction_metadata_frame_for_sequence_bundle_aligns_rows -q
```

Expected: pass.

---

### Task 2: Add Lateral Diagnostics Script Skeleton

**Files:**
- Create: `scripts/run_lateral_diagnostics.py`
- Test: `tests/test_training.py`

**Step 1: Write failing CLI/import test**

Add:

```python
def test_lateral_diagnostics_compute_target_scale_table():
    from scripts.run_lateral_diagnostics import compute_target_scale_table

    frame = pd.DataFrame(
        {
            "true_fy_b": [0.0, 1.0, -1.0],
            "pred_fy_b": [0.0, 0.5, -0.5],
            "true_fx_b": [0.0, 2.0, -2.0],
            "pred_fx_b": [0.0, 1.0, -1.0],
        }
    )

    table = compute_target_scale_table(frame, target_columns=["fy_b", "fx_b"])

    assert set(table["target"]) == {"fy_b", "fx_b"}
    assert {"true_std", "rmse", "rmse_over_std", "mean_abs_true", "p95_abs_true"}.issubset(table.columns)
```

**Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_training.py::test_lateral_diagnostics_compute_target_scale_table -q
```

Expected: fail because the script/helper does not exist.

**Step 3: Implement script skeleton**

Create `scripts/run_lateral_diagnostics.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import prediction_metadata_frame_for_bundle

LATERAL_TARGETS = ("fy_b", "mx_b", "mz_b")
DEFAULT_TARGETS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
```

Implement:

```python
def compute_target_scale_table(aligned: pd.DataFrame, *, target_columns: list[str]) -> pd.DataFrame:
    rows = []
    for target in target_columns:
        true = aligned[f"true_{target}"].to_numpy(dtype=float)
        pred = aligned[f"pred_{target}"].to_numpy(dtype=float)
        resid = true - pred
        true_std = float(np.std(true, ddof=0))
        rmse = float(np.sqrt(np.mean(np.square(resid))))
        rows.append({
            "target": target,
            "sample_count": int(len(true)),
            "true_mean": float(np.mean(true)),
            "true_std": true_std,
            "mean_abs_true": float(np.mean(np.abs(true))),
            "p95_abs_true": float(np.percentile(np.abs(true), 95)),
            "rmse": rmse,
            "rmse_over_std": float(rmse / true_std) if true_std > 0 else np.nan,
            "resid_mean": float(np.mean(resid)),
            "resid_std": float(np.std(resid, ddof=0)),
        })
    return pd.DataFrame(rows)
```

Add CLI args:

```text
--model-bundle
--split-root
--split test
--output-dir
--batch-size
--device
--batch first|second|all
```

Only implement `--batch first` in Batch 1. For `second`, print a clear error until Batch 2 is implemented.

**Step 4: Run helper test**

Run:

```bash
pytest tests/test_training.py::test_lateral_diagnostics_compute_target_scale_table -q
```

Expected: pass.

---

### Task 3: Implement Per-Log Lateral Metrics

**Files:**
- Modify: `scripts/run_lateral_diagnostics.py`
- Test: `tests/test_training.py`

**Step 1: Write failing test**

Add:

```python
def test_lateral_diagnostics_compute_per_log_table():
    from scripts.run_lateral_diagnostics import compute_per_log_lateral_table

    frame = pd.DataFrame(
        {
            "log_id": ["a", "a", "b", "b"],
            "true_fy_b": [0.0, 1.0, 0.0, 2.0],
            "pred_fy_b": [0.0, 0.5, 0.0, 1.0],
            "true_mx_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mx_b": [0.0, 0.5, 0.0, 0.5],
            "true_mz_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mz_b": [0.0, 0.5, 0.0, 0.5],
        }
    )

    table = compute_per_log_lateral_table(frame, lateral_targets=["fy_b", "mx_b", "mz_b"])

    assert set(table["log_id"]) == {"a", "b"}
    assert {"fy_b_rmse", "fy_b_r2", "lateral_rmse_mean", "lateral_r2_mean"}.issubset(table.columns)
```

**Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_training.py::test_lateral_diagnostics_compute_per_log_table -q
```

**Step 3: Implement per-log table**

Implement:

```python
def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(true - pred))))

def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(np.square(true - np.mean(true))))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum(np.square(true - pred)) / denom)

def compute_per_log_lateral_table(aligned: pd.DataFrame, *, lateral_targets: list[str]) -> pd.DataFrame:
    rows = []
    for log_id, group in aligned.groupby("log_id", sort=True):
        row = {"log_id": str(log_id), "sample_count": int(len(group))}
        rmses = []
        r2s = []
        for target in lateral_targets:
            true = group[f"true_{target}"].to_numpy(dtype=float)
            pred = group[f"pred_{target}"].to_numpy(dtype=float)
            rmse = _rmse(true, pred)
            r2 = _r2(true, pred)
            row[f"{target}_rmse"] = rmse
            row[f"{target}_r2"] = r2
            rmses.append(rmse)
            if np.isfinite(r2):
                r2s.append(r2)
        row["lateral_rmse_mean"] = float(np.mean(rmses))
        row["lateral_r2_mean"] = float(np.mean(r2s)) if r2s else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
```

**Step 4: Run test**

Run:

```bash
pytest tests/test_training.py::test_lateral_diagnostics_compute_per_log_table -q
```

Expected: pass.

---

### Task 4: Implement Per-Regime Lateral Bins

**Files:**
- Modify: `scripts/run_lateral_diagnostics.py`
- Test: `tests/test_training.py`

**Step 1: Write failing test**

Add:

```python
def test_lateral_diagnostics_compute_regime_table():
    from scripts.run_lateral_diagnostics import compute_regime_lateral_table

    frame = pd.DataFrame(
        {
            "cycle_flap_frequency_hz": [1.0, 2.0, 4.0, 5.0],
            "true_fy_b": [0.0, 1.0, 0.0, 2.0],
            "pred_fy_b": [0.0, 0.5, 0.0, 1.0],
            "true_mx_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mx_b": [0.0, 0.5, 0.0, 0.5],
            "true_mz_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mz_b": [0.0, 0.5, 0.0, 0.5],
        }
    )

    table = compute_regime_lateral_table(
        frame,
        lateral_targets=["fy_b", "mx_b", "mz_b"],
        bin_specs={"cycle_flap_frequency_hz": [0.0, 3.0, 6.0]},
        min_samples=1,
    )

    assert len(table) == 2
    assert {"regime_column", "bin_label", "lateral_rmse_mean", "lateral_r2_mean"}.issubset(table.columns)
```

**Step 2: Run RED**

Run:

```bash
pytest tests/test_training.py::test_lateral_diagnostics_compute_regime_table -q
```

**Step 3: Implement regime bins**

Use bin specs:

```python
DEFAULT_LATERAL_BIN_SPECS = {
    "airspeed_validated.true_airspeed_m_s": [0.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "cycle_flap_frequency_hz": [0.0, 3.0, 5.0, 7.0, 10.0],
    "phase_corrected_rad": [0.0, 1.5707963268, 3.1415926536, 4.7123889804, 6.2831853072],
    "servo_rudder": [-1.0, -0.05, 0.05, 1.0],
    "elevon_diff": [-1.0, -0.05, 0.05, 1.0],
}
```

Before binning, add derived columns:

```python
aligned["elevon_diff"] = aligned["servo_left_elevon"] - aligned["servo_right_elevon"]
```

Skip bins with fewer than `min_samples`.

**Step 4: Run test**

Run:

```bash
pytest tests/test_training.py::test_lateral_diagnostics_compute_regime_table -q
```

Expected: pass.

---

### Task 5: Implement Batch 1 CLI Outputs

**Files:**
- Modify: `scripts/run_lateral_diagnostics.py`
- Create at runtime: `artifacts/20260507_lateral_diagnostics_best_transformer/`

**Step 1: Implement CLI batch first**

The script should:

```text
1. load model bundle with torch.load(..., weights_only=False)
2. read <split>_samples.parquet
3. call prediction_metadata_frame_for_bundle(...)
4. write aligned_predictions.parquet
5. write target_scale.csv
6. write per_log_lateral_metrics.csv
7. write per_regime_lateral_metrics.csv
8. write batch1_summary.md
9. write diagnostics_config.json
```

`batch1_summary.md` should include:

```text
top 6 targets sorted by rmse_over_std
worst 5 logs sorted by lateral_r2_mean ascending
worst 8 regime bins sorted by lateral_r2_mean ascending
short interpretation bullets
```

**Step 2: Smoke run on best Transformer**

Run:

```bash
python scripts/run_lateral_diagnostics.py \
  --model-bundle artifacts/20260507_transformer_focused_final/runs/transformer_focused_final_hist128_d64_l2_h4_do050/causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --split test \
  --output-dir artifacts/20260507_lateral_diagnostics_best_transformer \
  --batch first \
  --batch-size 8192 \
  --device cuda:0
```

Expected files:

```text
aligned_predictions.parquet
target_scale.csv
per_log_lateral_metrics.csv
per_regime_lateral_metrics.csv
batch1_summary.md
diagnostics_config.json
```

**Step 3: Inspect outputs**

Run:

```bash
python - <<'PY'
import pandas as pd
root = "artifacts/20260507_lateral_diagnostics_best_transformer"
print(pd.read_csv(f"{root}/target_scale.csv").sort_values("rmse_over_std", ascending=False).to_string(index=False))
print(pd.read_csv(f"{root}/per_log_lateral_metrics.csv").sort_values("lateral_r2_mean").head(8).to_string(index=False))
print(pd.read_csv(f"{root}/per_regime_lateral_metrics.csv").sort_values("lateral_r2_mean").head(8).to_string(index=False))
PY
```

**Step 4: Commit Batch 1 implementation**

Run:

```bash
pytest tests/test_training.py::test_prediction_metadata_frame_for_sequence_bundle_aligns_rows \
       tests/test_training.py::test_lateral_diagnostics_compute_target_scale_table \
       tests/test_training.py::test_lateral_diagnostics_compute_per_log_table \
       tests/test_training.py::test_lateral_diagnostics_compute_regime_table -q
python -m py_compile scripts/run_lateral_diagnostics.py src/system_identification/training.py
git diff --check
git add scripts/run_lateral_diagnostics.py src/system_identification/training.py tests/test_training.py docs/plans/2026-05-07-lateral-diagnostics-two-batch-plan.md
git commit -m "feat: add lateral diagnostics batch one"
```

**Step 5: Required User Checkpoint**

Stop after Batch 1. Inform the user with:

```text
Batch 1 is complete.
Summarize target scale, worst logs, and worst regimes.
State where the evidence points.
Ask the user to reply "继续" before Batch 2.
```

Do not proceed to Batch 2 without user confirmation.

---

## Batch 2: Phase/Lag and Residual Correlation

### Task 6: Add Phase/Lag Diagnostics

**Files:**
- Modify: `scripts/run_lateral_diagnostics.py`
- Test: `tests/test_training.py`

**Step 1: Write failing lag test**

Add:

```python
def test_lateral_diagnostics_estimates_integer_lag():
    from scripts.run_lateral_diagnostics import estimate_best_lag

    true = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    pred = np.array([0, 0, 0, 1, 0, 0], dtype=float)

    result = estimate_best_lag(true, pred, max_lag=2)

    assert result["best_lag"] == 1
```

**Step 2: Implement lag helpers**

Implement:

```python
def estimate_best_lag(true: np.ndarray, pred: np.ndarray, *, max_lag: int) -> dict[str, float]:
    # positive lag means prediction lags true and should be shifted earlier
```

For each target/log:

```text
compute best lag in samples
compute best correlation
compute zero-lag correlation
compute rmse at zero lag and best lag
```

Write:

```text
phase_lag_lateral_metrics.csv
```

Group by `log_id`, and optionally by cycle if `cycle_index` exists. If cycle ID does not exist, per-log lag is enough.

---

### Task 7: Add Residual Correlation Diagnostics

**Files:**
- Modify: `scripts/run_lateral_diagnostics.py`
- Test: `tests/test_training.py`

**Step 1: Write failing correlation test**

Add:

```python
def test_lateral_diagnostics_residual_correlation_table():
    from scripts.run_lateral_diagnostics import compute_residual_correlation_table

    frame = pd.DataFrame(
        {
            "resid_fy_b": [0.0, 1.0, 2.0, 3.0],
            "servo_rudder": [0.0, 1.0, 2.0, 3.0],
            "vehicle_angular_velocity.xyz[2]": [3.0, 2.0, 1.0, 0.0],
        }
    )

    table = compute_residual_correlation_table(
        frame,
        lateral_targets=["fy_b"],
        candidate_columns=["servo_rudder", "vehicle_angular_velocity.xyz[2]"],
    )

    assert set(table["feature"]) == {"servo_rudder", "vehicle_angular_velocity.xyz[2]"}
    assert {"target", "feature", "pearson_corr", "abs_corr"}.issubset(table.columns)
```

**Step 2: Implement residual correlation**

Candidate features:

```text
servo_rudder
servo_left_elevon
servo_right_elevon
elevon_diff
motor_cmd_0
vehicle_local_position.vy
vehicle_local_position.vx
vehicle_local_position.vz
vehicle_angular_velocity.xyz[0]
vehicle_angular_velocity.xyz[1]
vehicle_angular_velocity.xyz[2]
airspeed_validated.true_airspeed_m_s
cycle_flap_frequency_hz
phase_corrected_rad
sin(phase)
cos(phase)
```

Write:

```text
residual_correlations.csv
```

Sort by `abs_corr` descending.

---

### Task 8: Implement Batch 2 CLI Outputs

**Files:**
- Modify: `scripts/run_lateral_diagnostics.py`
- Create at runtime: `artifacts/20260507_lateral_diagnostics_best_transformer/`

**Step 1: Extend CLI**

`--batch second` should require an existing `aligned_predictions.parquet` or regenerate it if missing.

Write:

```text
phase_lag_lateral_metrics.csv
residual_correlations.csv
batch2_summary.md
```

**Step 2: Run Batch 2**

Run:

```bash
python scripts/run_lateral_diagnostics.py \
  --model-bundle artifacts/20260507_transformer_focused_final/runs/transformer_focused_final_hist128_d64_l2_h4_do050/causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --split test \
  --output-dir artifacts/20260507_lateral_diagnostics_best_transformer \
  --batch second \
  --batch-size 8192 \
  --device cuda:0
```

**Step 3: Inspect outputs**

Run:

```bash
python - <<'PY'
import pandas as pd
root = "artifacts/20260507_lateral_diagnostics_best_transformer"
print(pd.read_csv(f"{root}/phase_lag_lateral_metrics.csv").sort_values("best_lag_abs", ascending=False).head(12).to_string(index=False))
print(pd.read_csv(f"{root}/residual_correlations.csv").sort_values("abs_corr", ascending=False).head(20).to_string(index=False))
PY
```

**Step 4: Commit Batch 2**

Run:

```bash
pytest tests/test_training.py::test_lateral_diagnostics_estimates_integer_lag \
       tests/test_training.py::test_lateral_diagnostics_residual_correlation_table -q
python -m py_compile scripts/run_lateral_diagnostics.py
git diff --check
git add scripts/run_lateral_diagnostics.py tests/test_training.py
git commit -m "feat: add lateral lag and residual diagnostics"
```

---

## Final Report

### Task 9: Write Consolidated Lateral Diagnostics Report

**Files:**
- Create: `docs/results/2026-05-07-lateral-diagnostics-best-transformer.md`

Include:

```text
1. target scale and RMSE/std
2. worst logs
3. worst regime bins
4. phase/lag findings
5. residual correlation findings
6. recommended next modeling/data actions
```

Run:

```bash
git add docs/results/2026-05-07-lateral-diagnostics-best-transformer.md
git commit -m "docs: summarize lateral diagnostics"
```

---

## Final Verification

Run:

```bash
pytest tests/test_training.py::test_prediction_metadata_frame_for_sequence_bundle_aligns_rows \
       tests/test_training.py::test_lateral_diagnostics_compute_target_scale_table \
       tests/test_training.py::test_lateral_diagnostics_compute_per_log_table \
       tests/test_training.py::test_lateral_diagnostics_compute_regime_table \
       tests/test_training.py::test_lateral_diagnostics_estimates_integer_lag \
       tests/test_training.py::test_lateral_diagnostics_residual_correlation_table -q
pytest -q
python -m py_compile scripts/run_lateral_diagnostics.py src/system_identification/training.py
git diff --check
git status --short --branch
```

Expected:

```text
all tests pass
compile passes
diff check passes
```
