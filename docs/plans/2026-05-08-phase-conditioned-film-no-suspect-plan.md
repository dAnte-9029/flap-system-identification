# Phase-Conditioned FiLM No-Suspect Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and evaluate phase-conditioned Transformer FiLM variants using a filtered no-suspect whole-log split, then compare them against the existing Transformer baseline without using the suspect log for model selection.

**Architecture:** First create a reproducible filtered split that removes `log_4_2026-4-12-17-43-30` from train/val/test parquet and log metadata files. Then add two causal Transformer variants: head/output FiLM and input-sequence FiLM. Both use the existing phase actuator sequence features as the conditioner source and keep the leakage-resistant protocol: whole-log split, no acceleration inputs, no centered windows, no past wrench targets, validation-only model selection, and test only for locked final models.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `src/system_identification/training.py`, existing `scripts/run_temporal_backbone_screen.py`, existing `scripts/run_lateral_diagnostics.py`, whole-log parquet split dataset.

---

## Fixed Protocol

Use this suspect log id everywhere:

```text
log_4_2026-4-12-17-43-30
```

Source split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
```

Filtered split to create and use for all model selection:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log
```

Selection rule:

```text
screen stage:
  train/val from no-suspect split
  skip test eval
  rank by validation metrics only

final stage:
  train/val/test from no-suspect split
  include test eval only after configs are locked

diagnostics:
  main report: no-suspect split
  optional robustness report: original split with suspect included, marked as diagnostic only
```

Do not silently compare against the previous all-test numbers as the main result. Report the old all-test baseline only as legacy context.

Primary model-comparison metrics:

```text
val_overall_rmse
val_overall_r2
val_fx_b_r2
val_fz_b_r2
val_my_b_r2
val_mx_b_r2
val_mz_b_r2
val_fy_b_r2
```

Control-priority interpretation:

```text
priority 1: fx_b, fz_b, my_b
priority 2: mx_b
priority 3: mz_b
priority 4: fy_b
```

Default-model acceptance rule:

```text
Adopt FiLM only if it improves or matches overall test RMSE/R2 on the no-suspect split
and does not degrade mx_b/mz_b compared with the existing phase-actuator Transformer.
```

---

## Design Summary

### Existing Baseline

Existing best recipe:

```text
causal_transformer_paper_no_accel_v2_phase_actuator_airdata
```

Hyperparameters:

```text
history: 128
d_model: 64
layers: 2
heads: 4
dropout: 0.05
dim_feedforward: 128
loss: Huber, delta 1.5
```

Sequence mode:

```text
phase_actuator_airdata
```

This sequence mode already includes:

```text
phase_corrected_sin
phase_corrected_cos
phase_corrected_rad
wing_stroke_angle_rad
flap_frequency_hz
cycle_flap_frequency_hz
motor_cmd_0
servo_left_elevon
servo_right_elevon
servo_rudder
elevator_like
aileron_like
airdata columns
```

### New Variant 1: Head/Output FiLM

Model type:

```text
causal_transformer_head_film
```

Computation:

```text
embedded = input_projection(sequence_inputs) + position_embedding
encoded = TransformerEncoder(embedded, causal_mask)
h = encoded[:, -1, :]
c = sequence_inputs[:, -1, phase_conditioning_indices]
gamma, beta = film_net(c)
h_film = h * (1 + film_scale * tanh(gamma)) + film_scale * tanh(beta)
prediction = head(concat(h_film, current_inputs))
```

Meaning:

```text
The Transformer first summarizes the causal history.
The current wingbeat phase then modulates how the final hidden representation is read out.
```

### New Variant 2: Input-Sequence FiLM

Model type:

```text
causal_transformer_input_film
```

Computation:

```text
z_t = input_projection(sequence_inputs[:, t, :])
c_t = sequence_inputs[:, t, phase_conditioning_indices]
gamma_t, beta_t = film_net(c_t)
z_t_film = z_t * (1 + film_scale * tanh(gamma_t)) + film_scale * tanh(beta_t)
embedded = z_t_film + position_embedding
encoded = TransformerEncoder(embedded, causal_mask)
h = encoded[:, -1, :]
prediction = head(concat(h, current_inputs))
```

Meaning:

```text
Each history sample is interpreted under its own wingbeat phase before attention.
This is more physical than head FiLM, but also higher risk.
```

Shared FiLM details:

```text
phase_conditioning_columns: phase_corrected_sin, phase_corrected_cos
film_hidden_size: 32
film_scale: 0.1
final film layer initialized to zero
```

Zero initialization is important. At initialization, FiLM is almost identity:

```text
gamma = 0
beta = 0
output = input
```

This makes the variants start close to the existing Transformer and lowers the risk of destroying a strong baseline.

---

## Task 1: Add Reproducible No-Suspect Split Builder

**Files:**
- Create: `scripts/filter_split_logs.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing test**

Add a test near existing split/parquet pipeline tests:

```python
def test_filter_split_logs_removes_excluded_log_from_all_split_files(tmp_path: Path):
    from scripts.filter_split_logs import filter_split_logs

    input_root = tmp_path / "input_split"
    output_root = tmp_path / "filtered_split"
    input_root.mkdir()

    frames = {
        "train": pd.DataFrame({"log_id": ["good_train", "bad_log"], "value": [1.0, 2.0]}),
        "val": pd.DataFrame({"log_id": ["good_val", "bad_log"], "value": [3.0, 4.0]}),
        "test": pd.DataFrame({"log_id": ["good_test", "bad_log"], "value": [5.0, 6.0]}),
    }
    for split, frame in frames.items():
        frame.to_parquet(input_root / f"{split}_samples.parquet", index=False)
        pd.DataFrame({"log_id": frame["log_id"].unique(), "split": split}).to_csv(
            input_root / f"{split}_logs.csv",
            index=False,
        )

    pd.concat(
        [pd.DataFrame({"log_id": frame["log_id"].unique(), "split": split}) for split, frame in frames.items()],
        ignore_index=True,
    ).to_csv(input_root / "all_logs.csv", index=False)
    (input_root / "dataset_manifest.json").write_text(
        json.dumps({"split_policy": "whole_log", "note": "source"}, indent=2),
        encoding="utf-8",
    )

    outputs = filter_split_logs(
        input_root=input_root,
        output_root=output_root,
        exclude_log_ids=["bad_log"],
        reason="unit-test bad log",
    )

    assert Path(outputs["dataset_manifest_path"]).exists()
    for split in ("train", "val", "test"):
        filtered = pd.read_parquet(output_root / f"{split}_samples.parquet")
        assert "bad_log" not in set(filtered["log_id"])
        logs = pd.read_csv(output_root / f"{split}_logs.csv")
        assert "bad_log" not in set(logs["log_id"])

    all_logs = pd.read_csv(output_root / "all_logs.csv")
    assert "bad_log" not in set(all_logs["log_id"])
    manifest = json.loads((output_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_policy"] == "whole_log"
    assert manifest["filtered_from_split_root"].endswith("input_split")
    assert manifest["excluded_log_ids"] == ["bad_log"]
    assert manifest["removed_sample_counts_by_split"]["train"] == 1
    assert manifest["removed_sample_counts_by_split"]["val"] == 1
    assert manifest["removed_sample_counts_by_split"]["test"] == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py::test_filter_split_logs_removes_excluded_log_from_all_split_files -q
```

Expected: fail because `scripts/filter_split_logs.py` does not exist.

**Step 3: Implement `scripts/filter_split_logs.py`**

Create:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_frame_by_log_id(frame: pd.DataFrame, exclude_log_ids: set[str]) -> tuple[pd.DataFrame, int]:
    if "log_id" not in frame.columns:
        raise ValueError("split frame must contain log_id")
    mask = ~frame["log_id"].astype(str).isin(exclude_log_ids)
    return frame.loc[mask].reset_index(drop=True), int((~mask).sum())


def _filter_csv_if_exists(input_path: Path, output_path: Path, exclude_log_ids: set[str]) -> int:
    if not input_path.exists():
        return 0
    frame = pd.read_csv(input_path)
    if "log_id" not in frame.columns:
        frame.to_csv(output_path, index=False)
        return 0
    filtered, removed = _filter_frame_by_log_id(frame, exclude_log_ids)
    filtered.to_csv(output_path, index=False)
    return removed


def filter_split_logs(
    *,
    input_root: str | Path,
    output_root: str | Path,
    exclude_log_ids: list[str],
    reason: str,
    overwrite: bool = False,
) -> dict[str, str]:
    input_path = Path(input_root)
    output_path = Path(output_root)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"output_root already exists and is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    exclude_set = {str(value) for value in exclude_log_ids}
    if not exclude_set:
        raise ValueError("exclude_log_ids must not be empty")

    removed_sample_counts: dict[str, int] = {}
    kept_sample_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_input = input_path / f"{split}_samples.parquet"
        if not split_input.exists():
            raise FileNotFoundError(split_input)
        frame = pd.read_parquet(split_input)
        filtered, removed = _filter_frame_by_log_id(frame, exclude_set)
        filtered.to_parquet(output_path / f"{split}_samples.parquet", index=False)
        removed_sample_counts[split] = removed
        kept_sample_counts[split] = int(len(filtered))
        _filter_csv_if_exists(input_path / f"{split}_logs.csv", output_path / f"{split}_logs.csv", exclude_set)

    _filter_csv_if_exists(input_path / "all_logs.csv", output_path / "all_logs.csv", exclude_set)

    manifest = _read_manifest(input_path / "dataset_manifest.json")
    manifest.update(
        {
            "filtered_from_split_root": str(input_path),
            "excluded_log_ids": sorted(exclude_set),
            "excluded_log_reason": reason,
            "removed_sample_counts_by_split": removed_sample_counts,
            "kept_sample_counts_by_split": kept_sample_counts,
            "split_policy": manifest.get("split_policy", "whole_log"),
        }
    )
    manifest_path = output_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "dataset_manifest_path": str(manifest_path),
        "output_root": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter one or more log ids out of an existing train/val/test split")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exclude-log-ids", nargs="+", required=True)
    parser.add_argument("--reason", default="excluded suspect log")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = filter_split_logs(
        input_root=args.input_root,
        output_root=args.output_root,
        exclude_log_ids=args.exclude_log_ids,
        reason=args.reason,
        overwrite=args.overwrite,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline.py::test_filter_split_logs_removes_excluded_log_from_all_split_files -q
```

Expected: pass.

**Step 5: Build the actual no-suspect split**

```bash
python scripts/filter_split_logs.py \
  --input-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log \
  --exclude-log-ids log_4_2026-4-12-17-43-30 \
  --reason "suspect lateral-directional log identified by diagnostics" \
  --overwrite
```

**Step 6: Verify the actual filtered split**

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path("dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log")
bad = "log_4_2026-4-12-17-43-30"
for split in ("train", "val", "test"):
    frame = pd.read_parquet(root / f"{split}_samples.parquet")
    print(split, len(frame), sorted(frame["log_id"].astype(str).unique()))
    assert bad not in set(frame["log_id"].astype(str))
manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
print(manifest["removed_sample_counts_by_split"])
assert manifest["excluded_log_ids"] == [bad]
PY
```

Expected:

```text
bad log absent from train/val/test
test removed sample count should be 17721 if the current split matches previous diagnostics
```

**Step 7: Commit**

```bash
git add scripts/filter_split_logs.py tests/test_pipeline.py
git commit -m "feat: add split log filtering utility"
```

Do not commit the dataset parquet files unless the repository already tracks generated dataset files. If they are ignored, leave them as local artifacts.

---

## Task 2: Add Phase FiLM Module And Transformer Modes

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing forward-shape tests**

Add near `test_causal_transformer_regressor_forward_shape_with_current_features`:

```python
def test_causal_transformer_head_film_forward_shape_with_current_features():
    model = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="head",
        phase_conditioning_indices=(0, 1),
        film_hidden_size=16,
        film_scale=0.1,
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)


def test_causal_transformer_input_film_forward_shape_with_current_features():
    model = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="input",
        phase_conditioning_indices=(0, 1),
        film_hidden_size=16,
        film_scale=0.1,
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)
```

**Step 2: Write failing identity-init test**

Add:

```python
def test_phase_film_zero_initialized_as_identity():
    torch.manual_seed(12)
    base = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="none",
    )
    film = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="head",
        phase_conditioning_indices=(0, 1),
        film_hidden_size=16,
        film_scale=0.1,
    )

    base_state = base.state_dict()
    compatible = {key: value for key, value in base_state.items() if key in film.state_dict()}
    film.load_state_dict({**film.state_dict(), **compatible})

    sequence = torch.randn(2, 16, 5)
    current = torch.randn(2, 3)
    torch.testing.assert_close(film(sequence, current), base(sequence, current), atol=1e-6, rtol=1e-6)
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_training.py::test_causal_transformer_head_film_forward_shape_with_current_features \
       tests/test_training.py::test_causal_transformer_input_film_forward_shape_with_current_features \
       tests/test_training.py::test_phase_film_zero_initialized_as_identity -q
```

Expected: fail because `film_mode` and FiLM module are not implemented.

**Step 4: Implement `_PhaseFiLM`**

Add near sequence model classes:

```python
class _PhaseFiLM(nn.Module):
    def __init__(self, *, conditioner_dim: int, feature_dim: int, hidden_size: int = 32, scale: float = 0.1):
        super().__init__()
        if conditioner_dim <= 0:
            raise ValueError("conditioner_dim must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if scale < 0.0:
            raise ValueError("scale must be non-negative")
        self.conditioner_dim = int(conditioner_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(self.conditioner_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2 * self.feature_dim),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, conditioner: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(conditioner)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return features * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * torch.tanh(beta)
```

**Step 5: Extend `CausalTransformerRegressor`**

Add constructor args:

```python
film_mode: str = "none",
phase_conditioning_indices: tuple[int, ...] | None = None,
film_hidden_size: int = 32,
film_scale: float = 0.1,
```

Validate:

```python
self.film_mode = (film_mode or "none").lower()
if self.film_mode not in {"none", "head", "input"}:
    raise ValueError(f"Unknown film_mode: {film_mode}")
self.phase_conditioning_indices = tuple(int(v) for v in (phase_conditioning_indices or ()))
if self.film_mode != "none" and not self.phase_conditioning_indices:
    raise ValueError("phase_conditioning_indices are required when film_mode is enabled")
for index in self.phase_conditioning_indices:
    if index < 0 or index >= self.sequence_input_dim:
        raise ValueError(f"phase conditioning index {index} out of bounds")
```

Create modules:

```python
self.input_film = None
self.head_film = None
if self.film_mode == "input":
    self.input_film = _PhaseFiLM(
        conditioner_dim=len(self.phase_conditioning_indices),
        feature_dim=self.d_model,
        hidden_size=int(film_hidden_size),
        scale=float(film_scale),
    )
elif self.film_mode == "head":
    self.head_film = _PhaseFiLM(
        conditioner_dim=len(self.phase_conditioning_indices),
        feature_dim=self.d_model,
        hidden_size=int(film_hidden_size),
        scale=float(film_scale),
    )
```

In `forward`, after `embedded = self.input_projection(sequence_inputs)`:

```python
if self.input_film is not None:
    conditioner = sequence_inputs[:, :, self.phase_conditioning_indices]
    embedded = self.input_film(embedded, conditioner)
embedded = embedded + self.position_embedding[:, :history_size, :]
```

After `representation = encoded[:, -1, :]`:

```python
if self.head_film is not None:
    conditioner = sequence_inputs[:, -1, self.phase_conditioning_indices]
    representation = self.head_film(representation, conditioner)
```

**Step 6: Run tests to verify they pass**

```bash
pytest tests/test_training.py::test_causal_transformer_head_film_forward_shape_with_current_features \
       tests/test_training.py::test_causal_transformer_input_film_forward_shape_with_current_features \
       tests/test_training.py::test_phase_film_zero_initialized_as_identity -q
```

Expected: pass.

**Step 7: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add phase-conditioned transformer film"
```

---

## Task 3: Wire FiLM Model Types Through Training And Bundle Loading

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`
- Test: `tests/test_training.py`

**Step 1: Write failing model-type tests**

Add near `test_run_training_job_supports_temporal_sequence_model_types`:

```python
@pytest.mark.parametrize("model_type", ["causal_transformer_head_film", "causal_transformer_input_film"])
def test_run_training_job_supports_phase_film_transformers(tmp_path: Path, model_type: str):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 520, "train_log"), ("val", 521, "val_log"), ("test", 522, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / model_type,
        feature_set_name="paper_no_accel_v2",
        model_type=model_type,
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
        num_workers=0,
        use_amp=False,
    )

    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    assert cfg["model_type"] == model_type
    assert cfg["film_mode"] in {"head", "input"}
    assert cfg["phase_conditioning_columns"] == ["phase_corrected_sin", "phase_corrected_cos"]
    assert metrics["test"]["sample_count"] > 0
```

Add a CLI parse test:

```python
def test_train_baseline_torch_cli_supports_phase_film_model_type(monkeypatch):
    from scripts import train_baseline_torch

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_baseline_torch.py",
            "--split-root",
            "split",
            "--output-dir",
            "runs",
            "--model-type",
            "causal_transformer_head_film",
        ],
    )

    args = train_baseline_torch.parse_args()
    assert args.model_type == "causal_transformer_head_film"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training.py::test_run_training_job_supports_phase_film_transformers \
       tests/test_training.py::test_train_baseline_torch_cli_supports_phase_film_model_type -q
```

Expected: fail because model types are unknown.

**Step 3: Add model types**

In `_normalized_model_type`, include:

```python
"causal_transformer_head_film",
"causal_transformer_input_film",
```

In `_is_sequence_model_type`, include the same two model types.

In `scripts/train_baseline_torch.py`, add both choices to `--model-type`.

**Step 4: Add phase conditioning resolver**

Add:

```python
PHASE_CONDITIONING_COLUMNS = ["phase_corrected_sin", "phase_corrected_cos"]


def resolve_phase_conditioning_indices(sequence_feature_columns: list[str]) -> tuple[int, ...]:
    missing = [column for column in PHASE_CONDITIONING_COLUMNS if column not in sequence_feature_columns]
    if missing:
        raise ValueError(f"Phase FiLM requires sequence columns: {missing}")
    return tuple(sequence_feature_columns.index(column) for column in PHASE_CONDITIONING_COLUMNS)
```

**Step 5: Extend `_build_sequence_regressor`**

Add args:

```python
phase_conditioning_indices: tuple[int, ...] | None = None
film_hidden_size: int = 32
film_scale: float = 0.1
```

For transformer model types:

```python
if model_type in {"causal_transformer", "causal_transformer_head_film", "causal_transformer_input_film"}:
    transformer_head_sizes = tuple(int(v) for v in hidden_sizes[1:]) or (int(transformer_d_model),)
    film_mode = "none"
    if model_type == "causal_transformer_head_film":
        film_mode = "head"
    elif model_type == "causal_transformer_input_film":
        film_mode = "input"
    return CausalTransformerRegressor(
        sequence_input_dim=sequence_input_dim,
        current_input_dim=current_input_dim,
        output_dim=output_dim,
        d_model=int(transformer_d_model),
        num_layers=int(transformer_num_layers),
        num_heads=int(transformer_num_heads),
        dim_feedforward=int(transformer_dim_feedforward),
        dropout=dropout,
        head_hidden_sizes=transformer_head_sizes,
        film_mode=film_mode,
        phase_conditioning_indices=phase_conditioning_indices,
        film_hidden_size=film_hidden_size,
        film_scale=film_scale,
    )
```

**Step 6: Pass FiLM config from training**

In `fit_torch_sequence_regressor`, after `sequence_feature_columns` are resolved:

```python
phase_conditioning_indices = None
phase_conditioning_columns: list[str] = []
film_mode = "none"
if resolved_model_type == "causal_transformer_head_film":
    film_mode = "head"
elif resolved_model_type == "causal_transformer_input_film":
    film_mode = "input"
if film_mode != "none":
    phase_conditioning_indices = resolve_phase_conditioning_indices(sequence_feature_columns)
    phase_conditioning_columns = [sequence_feature_columns[index] for index in phase_conditioning_indices]
```

Pass `phase_conditioning_indices` to `_build_sequence_regressor`.

Store in bundle:

```python
"film_mode": film_mode,
"phase_conditioning_columns": phase_conditioning_columns,
"phase_conditioning_indices": list(phase_conditioning_indices or []),
"film_hidden_size": 32,
"film_scale": 0.1,
```

Also pass these fields in `_build_sequence_model_from_bundle`.

**Step 7: Run tests to verify they pass**

```bash
pytest tests/test_training.py::test_run_training_job_supports_phase_film_transformers \
       tests/test_training.py::test_train_baseline_torch_cli_supports_phase_film_model_type -q
```

Expected: pass.

**Step 8: Commit**

```bash
git add src/system_identification/training.py scripts/train_baseline_torch.py tests/test_training.py
git commit -m "feat: wire phase film transformer models"
```

---

## Task 4: Add FiLM Recipes And Screening Stage

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/run_temporal_backbone_screen.py`
- Test: `tests/test_training.py`

**Step 1: Write failing recipe test**

Add near `test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe`:

```python
def test_run_baseline_comparison_supports_phase_film_transformer_recipes(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 540, "train_log"), ("val", 541, "val_log"), ("test", 542, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "runs",
        recipe_names=[
            "causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata",
            "causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata",
        ],
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
        num_workers=0,
        use_amp=False,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert set(summary["model_type"]) == {"causal_transformer_head_film", "causal_transformer_input_film"}
    assert set(summary["sequence_feature_mode"]) == {"phase_actuator_airdata"}
```

**Step 2: Write failing stage test**

Add near temporal screen tests:

```python
def test_temporal_screen_phase_film_grid_contains_baseline_head_and_input():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="phase_film")

    assert {
        "phase_film_baseline",
        "phase_film_head",
        "phase_film_input",
    } == {config.config_id for config in configs}
    assert {config.stage for config in configs} == {"phase_film"}
    assert all(config.sequence_history_size == 128 for config in configs)
    assert all(config.dropout == 0.05 for config in configs)
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_training.py::test_run_baseline_comparison_supports_phase_film_transformer_recipes \
       tests/test_training.py::test_temporal_screen_phase_film_grid_contains_baseline_head_and_input -q
```

Expected: fail because recipes and stage do not exist.

**Step 4: Add recipes**

In `BASELINE_COMPARISON_RECIPES`, add:

```python
"causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "causal_transformer_head_film",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "window_mode": "single",
    "window_radius": 0,
    "window_feature_mode": "all",
    "sequence_history_size": 64,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
    "transformer_d_model": 64,
    "transformer_num_layers": 1,
    "transformer_num_heads": 4,
    "transformer_dim_feedforward": 128,
},
"causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "causal_transformer_input_film",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "window_mode": "single",
    "window_radius": 0,
    "window_feature_mode": "all",
    "sequence_history_size": 64,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
    "transformer_d_model": 64,
    "transformer_num_layers": 1,
    "transformer_num_heads": 4,
    "transformer_dim_feedforward": 128,
},
```

**Step 5: Add `phase_film` screen stage**

In `scripts/run_temporal_backbone_screen.py`, add:

```python
def _phase_film_configs(*, final: bool = False) -> list[ScreenConfig]:
    stage = "phase_film_final" if final else "phase_film"
    max_epochs = 50 if final else 20
    patience = 8 if final else 5
    specs = [
        ("baseline", "causal_transformer_paper_no_accel_v2_phase_actuator_airdata"),
        ("head", "causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata"),
        ("input", "causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata"),
    ]
    return [
        _config(
            config_id=f"{stage}_{label}",
            stage=stage,
            recipe_name=recipe_name,
            hidden_sizes=(64, 128),
            sequence_history_size=128,
            max_epochs=max_epochs,
            early_stopping_patience=patience,
            dropout=0.05,
            extra_args={
                "transformer_d_model": 64,
                "transformer_num_layers": 2,
                "transformer_num_heads": 4,
                "transformer_dim_feedforward": 128,
            },
        )
        for label, recipe_name in specs
    ]
```

Wire it into:

```python
build_screen_configs
parse_args stage choices
_stage_sample_defaults
skip_test_eval
```

Use:

```python
if stage in {"sweep", "tcn_gru_focused", "transformer_focused", "phase_harmonic", "phase_film"}:
    return 131072, 65536, 65536

skip_test_eval = args.stage in {"transformer_focused", "phase_harmonic", "phase_film"} and not args.include_test_eval
```

**Step 6: Run tests to verify they pass**

```bash
pytest tests/test_training.py::test_run_baseline_comparison_supports_phase_film_transformer_recipes \
       tests/test_training.py::test_temporal_screen_phase_film_grid_contains_baseline_head_and_input -q
```

Expected: pass.

**Step 7: Commit**

```bash
git add src/system_identification/training.py scripts/run_temporal_backbone_screen.py tests/test_training.py
git commit -m "feat: add phase film screening recipes"
```

---

## Task 5: Validation Screen On No-Suspect Split

**Files:**
- Runtime output: `artifacts/20260508_phase_film_screen_no_suspect/`

**Step 1: Dry-run the stage**

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log \
  --output-dir artifacts/20260508_phase_film_screen_no_suspect \
  --stage phase_film \
  --dry-run
```

Inspect:

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("artifacts/20260508_phase_film_screen_no_suspect/temporal_backbone_screen_summary.csv")
print(summary[["config_id", "recipe_name", "sequence_history_size", "max_epochs"]].to_string(index=False))
PY
```

Expected configs:

```text
phase_film_baseline
phase_film_head
phase_film_input
```

**Step 2: Run validation-only screen**

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log \
  --output-dir artifacts/20260508_phase_film_screen_no_suspect \
  --stage phase_film \
  --batch-size 512 \
  --device cuda:0 \
  --random-seed 42
```

Do not pass `--include-test-eval`.

**Step 3: Verify no test columns exist**

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("artifacts/20260508_phase_film_screen_no_suspect/temporal_backbone_screen_summary.csv")
print("test columns:", [c for c in summary.columns if c.startswith("test_")])
assert not [c for c in summary.columns if c.startswith("test_")]
PY
```

Expected:

```text
test columns: []
```

**Step 4: Rank by validation**

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("artifacts/20260508_phase_film_screen_no_suspect/temporal_backbone_screen_summary.csv")
cols = [
    "config_id",
    "recipe_name",
    "val_overall_rmse",
    "val_overall_r2",
    "val_fx_b_r2",
    "val_fy_b_r2",
    "val_fz_b_r2",
    "val_mx_b_r2",
    "val_my_b_r2",
    "val_mz_b_r2",
    "best_epoch",
]
print(summary[cols].sort_values("val_overall_rmse").to_string(index=False))
PY
```

Selection interpretation:

```text
If baseline wins validation:
  final still runs baseline, head, and input once, because there are only three configs
  but default remains baseline unless final proves otherwise

If head or input wins validation:
  final runs all three configs, then default selection uses final no-suspect test metrics
```

**Step 5: Commit code before final training**

Run verification first:

```bash
pytest tests/test_training.py::test_causal_transformer_head_film_forward_shape_with_current_features \
       tests/test_training.py::test_causal_transformer_input_film_forward_shape_with_current_features \
       tests/test_training.py::test_phase_film_zero_initialized_as_identity \
       tests/test_training.py::test_run_training_job_supports_phase_film_transformers \
       tests/test_training.py::test_run_baseline_comparison_supports_phase_film_transformer_recipes \
       tests/test_training.py::test_temporal_screen_phase_film_grid_contains_baseline_head_and_input -q
python -m py_compile src/system_identification/training.py scripts/run_temporal_backbone_screen.py scripts/run_baseline_comparison.py scripts/train_baseline_torch.py scripts/filter_split_logs.py
git diff --check
```

Expected: pass.

Commit any remaining code changes before final/test:

```bash
git status --short
```

If there are uncommitted code changes:

```bash
git add src/system_identification/training.py scripts/run_temporal_backbone_screen.py scripts/run_baseline_comparison.py scripts/train_baseline_torch.py scripts/filter_split_logs.py tests/test_training.py tests/test_pipeline.py
git commit -m "feat: prepare phase film no-suspect screen"
```

---

## Task 6: Full-Data Final On No-Suspect Split

**Files:**
- Runtime output: `artifacts/20260508_phase_film_final_no_suspect/`

**Step 1: Run final configs**

Run all three final configs to avoid ambiguity:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log \
  --output-dir artifacts/20260508_phase_film_final_no_suspect \
  --stage phase_film_final \
  --config-ids phase_film_final_baseline phase_film_final_head phase_film_final_input \
  --batch-size 512 \
  --device cuda:0 \
  --random-seed 42 \
  --include-test-eval
```

**Step 2: Inspect final results**

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("artifacts/20260508_phase_film_final_no_suspect/temporal_backbone_screen_summary.csv")
cols = [
    "config_id",
    "recipe_name",
    "val_overall_rmse",
    "val_overall_r2",
    "test_overall_rmse",
    "test_overall_r2",
    "test_fx_b_r2",
    "test_fy_b_r2",
    "test_fz_b_r2",
    "test_mx_b_r2",
    "test_my_b_r2",
    "test_mz_b_r2",
    "best_epoch",
]
print(summary[cols].sort_values("test_overall_rmse").to_string(index=False))
PY
```

**Step 3: Decide default model**

Default selection:

```text
Use no-suspect final test metrics.
Prefer lower test_overall_rmse.
Reject a FiLM variant if mx_b or mz_b drops materially versus baseline.
Mention fy_b gains only as secondary evidence.
```

Adoption thresholds:

```text
strong accept:
  test_overall_rmse improves by >= 1 percent
  and mx_b/mz_b are not worse

weak accept:
  test_overall_rmse improves by < 1 percent
  and control-priority targets improve or stay matched

reject:
  overall similar/worse
  or roll/yaw degrade while only fy improves
```

---

## Task 7: Diagnostics And Robustness Reports

**Files:**
- Runtime output: `artifacts/20260508_phase_film_final_no_suspect/diagnostics/`

**Step 1: Find final model bundle paths**

Expected:

```text
artifacts/20260508_phase_film_final_no_suspect/runs/phase_film_final_baseline/causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
artifacts/20260508_phase_film_final_no_suspect/runs/phase_film_final_head/causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
artifacts/20260508_phase_film_final_no_suspect/runs/phase_film_final_input/causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
```

Verify:

```bash
find artifacts/20260508_phase_film_final_no_suspect/runs -name model_bundle.pt | sort
```

**Step 2: Run lateral diagnostics on no-suspect split**

For each config id and bundle path:

```bash
python scripts/run_lateral_diagnostics.py \
  --model-bundle <MODEL_BUNDLE> \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log \
  --split test \
  --output-dir artifacts/20260508_phase_film_final_no_suspect/diagnostics/<CONFIG_ID> \
  --batch first \
  --batch-size 8192 \
  --device cuda:0

python scripts/run_lateral_diagnostics.py \
  --model-bundle <MODEL_BUNDLE> \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log \
  --split test \
  --output-dir artifacts/20260508_phase_film_final_no_suspect/diagnostics/<CONFIG_ID> \
  --batch second \
  --batch-size 8192 \
  --device cuda:0
```

**Step 3: Optional robustness diagnostics on original split**

This is diagnostic only, not model selection:

```bash
python scripts/run_lateral_diagnostics.py \
  --model-bundle <MODEL_BUNDLE> \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --split test \
  --output-dir artifacts/20260508_phase_film_final_no_suspect/diagnostics_original_split/<CONFIG_ID> \
  --batch first \
  --batch-size 8192 \
  --device cuda:0
```

Use this only to show how much the old suspect log would distort each model.

**Step 4: Summarize diagnostics**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("artifacts/20260508_phase_film_final_no_suspect/diagnostics")
for cfg in ["phase_film_final_baseline", "phase_film_final_head", "phase_film_final_input"]:
    print("\\n==", cfg, "==")
    p = base / cfg / "per_log_lateral_metrics.csv"
    if p.exists():
        df = pd.read_csv(p)
        print(df[["log_id", "sample_count", "lateral_rmse_mean", "lateral_r2_mean", "fy_b_r2", "mx_b_r2", "mz_b_r2"]].to_string(index=False))
PY
```

---

## Task 8: Write Result Report

**Files:**
- Create: `docs/results/2026-05-08-phase-conditioned-film-no-suspect.md`
- Modify: `docs/insights/research-narrative-notes.md`

**Step 1: Create result report**

Report must include:

```text
1. Why suspect log is excluded from main comparison
2. Filtered split path and removed sample counts
3. Method definitions:
   baseline Transformer
   Transformer + head/output FiLM
   Transformer + input-sequence FiLM
4. Validation screen ranking with no test columns
5. Full-data no-suspect final/test table
6. Per-target R2 table
7. Lateral diagnostics summary
8. Decision:
   default model accepted or rejected
9. Paper/meeting wording
```

Use this wording if FiLM wins:

```text
Phase-conditioned FiLM improves the no-suspect full-data test result under the locked whole-log protocol, suggesting that wingbeat phase is better treated as a conditioning variable than as a plain concatenated input.
```

Use this wording if FiLM does not win:

```text
Phase-conditioned FiLM was tested under the no-suspect whole-log protocol, but did not outperform the existing phase-actuator Transformer baseline. This suggests that the current Transformer already captures much of the useful phase dependence from concatenated phase-actuator history.
```

**Step 2: Update insights**

Append a short section to `docs/insights/research-narrative-notes.md`:

```text
Phase-conditioned FiLM result:
  protocol: no-suspect whole-log split
  baseline vs head FiLM vs input FiLM
  default decision
  control-priority interpretation
```

**Step 3: Run final verification**

```bash
pytest tests/test_pipeline.py::test_filter_split_logs_removes_excluded_log_from_all_split_files \
       tests/test_training.py::test_causal_transformer_head_film_forward_shape_with_current_features \
       tests/test_training.py::test_causal_transformer_input_film_forward_shape_with_current_features \
       tests/test_training.py::test_phase_film_zero_initialized_as_identity \
       tests/test_training.py::test_run_training_job_supports_phase_film_transformers \
       tests/test_training.py::test_run_baseline_comparison_supports_phase_film_transformer_recipes \
       tests/test_training.py::test_temporal_screen_phase_film_grid_contains_baseline_head_and_input -q
python -m py_compile src/system_identification/training.py scripts/run_temporal_backbone_screen.py scripts/run_baseline_comparison.py scripts/train_baseline_torch.py scripts/filter_split_logs.py scripts/run_lateral_diagnostics.py
git diff --check
```

Expected: all pass.

**Step 4: Commit**

```bash
git add docs/results/2026-05-08-phase-conditioned-film-no-suspect.md docs/insights/research-narrative-notes.md
git commit -m "docs: report phase-conditioned film no-suspect results"
```

---

## Final Deliverables

Required commits:

```text
feat: add split log filtering utility
feat: add phase-conditioned transformer film
feat: wire phase film transformer models
feat: add phase film screening recipes
docs: report phase-conditioned film no-suspect results
```

Required artifacts:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log/
artifacts/20260508_phase_film_screen_no_suspect/
artifacts/20260508_phase_film_final_no_suspect/
docs/results/2026-05-08-phase-conditioned-film-no-suspect.md
```

Required final answer:

```text
1. Confirm no-suspect split was created and bad log removed.
2. Give validation ranking.
3. Give final no-suspect test comparison.
4. State whether Head FiLM or Input FiLM beats baseline.
5. State default model decision.
6. Mention verification commands and commits.
```

## Risks And Guardrails

Risk: FiLM overfits validation.
Guardrail: final/default decision uses no-suspect full-data test only after configs are locked.

Risk: Input FiLM destabilizes Transformer.
Guardrail: zero-init final FiLM layer and `film_scale=0.1`.

Risk: Removing suspect log hides data issues.
Guardrail: filtered split is explicit, manifest records removed log, old original-split diagnostics remain available but are not used for selection.

Risk: A FiLM variant only improves `fy_b`.
Guardrail: control-priority decision rejects variants that degrade `mx_b` or `mz_b` materially.

