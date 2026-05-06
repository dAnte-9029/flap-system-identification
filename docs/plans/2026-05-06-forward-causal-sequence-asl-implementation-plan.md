# Forward Causal Sequence ASL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add leakage-resistant forward sequence models inspired by Sharvit et al. 2025: first a causal GRU baseline, then a causal GRU with an Adaptive Spectrum Layer (ASL), and compare them against the current single-frame MLP baseline.

**Architecture:** Keep the current whole-log/no-acceleration protocol as the comparison anchor. Add a sequence data path that builds causal windows within each `log_id`/`segment_id`, sorts by `time_s`, feeds historical `phase + actuator + airdata` signals to a GRU encoder, optionally applies ASL over the causal history, and predicts the current six-axis wrench target. To avoid derivative leakage, historical velocity/angular-velocity/attitude-derived kinematic columns must not be included in the sequence branch by default; current-time features may still be concatenated separately as point features.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `src/system_identification/training.py`, existing `scripts/train_baseline_torch.py`, existing `scripts/run_baseline_comparison.py`.

---

## Method Guardrails

This is a forward adaptation of Sharvit et al. 2025, not an inverse reproduction.

Use:

```text
past/current inputs -> current wrench
x_seq[t-H+1:t] = phase, actuator, airdata history
x_now[t]       = current non-history point features, if enabled
y[t]           = fx_b, fy_b, fz_b, mx_b, my_b, mz_b
```

Do not use:

```text
future force windows
centered windows
acceleration feature inputs
historical velocity / angular-velocity histories in the primary model
same-log train/val/test splits
```

The first credible comparison should be:

```text
mlp_paper_no_accel_v2
causal_gru_paper_no_accel_v2_phase_actuator_airdata
causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata
```

---

### Task 1: Causal Sequence Frame Builder

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add tests for a new helper:

```python
def prepare_causal_sequence_feature_target_frames(
    frame: pd.DataFrame,
    sequence_feature_columns: list[str],
    current_feature_columns: list[str],
    target_columns: list[str],
    *,
    history_size: int,
    group_columns: tuple[str, ...] = ("log_id", "segment_id"),
    sort_column: str = "time_s",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    ...
```

Minimum tests:

```python
def test_prepare_causal_sequence_feature_target_frames_keeps_windows_inside_log_and_segment():
    frame = _synthetic_frame(n_rows=12, seed=1)
    frame["log_id"] = ["a"] * 6 + ["b"] * 6
    frame["segment_id"] = [0] * 3 + [1] * 3 + [0] * 6
    frame["time_s"] = list(reversed(range(6))) + list(reversed(range(6)))

    seq, current, targets, meta = prepare_causal_sequence_feature_target_frames(
        frame,
        sequence_feature_columns=["phase_corrected_sin", "phase_corrected_cos", "motor_cmd_0"],
        current_feature_columns=["velocity_b.x"],
        target_columns=["fx_b", "fz_b"],
        history_size=3,
    )

    assert seq.shape == (6, 3, 3)
    assert current.shape == (6, 1)
    assert list(targets.columns) == ["fx_b", "fz_b"]
    assert set(meta.columns) >= {"log_id", "segment_id", "time_s"}
    assert not meta.duplicated(["log_id", "segment_id", "time_s"]).any()
```

```python
def test_prepare_causal_sequence_feature_target_frames_aligns_target_to_last_timestep():
    frame = _synthetic_frame(n_rows=5, seed=2)
    frame["log_id"] = "a"
    frame["segment_id"] = 0
    frame["time_s"] = np.arange(5, dtype=float)
    frame["fx_b"] = frame["time_s"] * 10.0

    seq, _, targets, meta = prepare_causal_sequence_feature_target_frames(
        frame,
        sequence_feature_columns=["phase_corrected_sin"],
        current_feature_columns=[],
        target_columns=["fx_b"],
        history_size=3,
    )

    np.testing.assert_allclose(meta["time_s"].to_numpy(), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(targets["fx_b"].to_numpy(), [20.0, 30.0, 40.0])
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_prepare_causal_sequence_feature_target_frames_keeps_windows_inside_log_and_segment \
       tests/test_training.py::test_prepare_causal_sequence_feature_target_frames_aligns_target_to_last_timestep -q
```

Expected: fail because the helper is not defined.

**Step 3: Implement**

Implement the helper near `prepare_windowed_feature_target_frames(...)`.

Implementation requirements:
- call `_with_derived_columns(frame)` before selecting columns
- require `history_size >= 1`
- validate all sequence/current/target columns exist after derived columns are added
- group by available columns among `("log_id", "segment_id")`
- sort each group by `time_s` when present
- only emit windows fully contained within one group
- align target and current features to the last timestep of each causal window
- return:
  - `sequence_features`: `np.ndarray` with shape `[N, H, S]`
  - `current_features`: `np.ndarray` with shape `[N, C]`, possibly zero columns
  - `targets`: `pd.DataFrame`
  - `metadata`: `pd.DataFrame` with `log_id`, `segment_id`, `time_s` when available

**Step 4: Verify GREEN**

Run the focused tests again.

**Step 5: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: build causal sequence training windows"
```

---

### Task 2: Sequence Feature Modes

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add tests for:

```python
def resolve_sequence_feature_columns(
    base_feature_columns: list[str],
    sequence_feature_mode: str = "phase_actuator_airdata",
) -> list[str]:
    ...
```

Required behavior:

```python
def test_resolve_sequence_feature_columns_defaults_to_leakage_resistant_history():
    columns = resolve_feature_set_columns("paper_no_accel_v2")
    sequence_columns = resolve_sequence_feature_columns(columns, "phase_actuator_airdata")

    assert "phase_corrected_sin" in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns
    assert "velocity_b.x" not in sequence_columns
    assert "vehicle_angular_velocity.xyz[0]" not in sequence_columns
    assert "alpha_rad" not in sequence_columns
```

```python
def test_resolve_sequence_feature_columns_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown sequence_feature_mode"):
        resolve_sequence_feature_columns(resolve_feature_set_columns("paper_no_accel_v2"), "bad")
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_resolve_sequence_feature_columns_defaults_to_leakage_resistant_history \
       tests/test_training.py::test_resolve_sequence_feature_columns_rejects_unknown_mode -q
```

Expected: fail because the helper is not defined.

**Step 3: Implement**

Add:

```python
SEQUENCE_FEATURE_MODE_COLUMNS = {
    "phase_actuator": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"],
    "phase_actuator_airdata": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
}
```

Implement modes:
- `phase_actuator`
- `phase_actuator_airdata`
- `all`, for explicitly leakage-prone diagnostics only
- `none`, for tests and ablations

Add a constant:

```python
SEQUENCE_HISTORY_DANGEROUS_COLUMNS = KINEMATIC_WINDOW_EXCLUDED_COLUMNS
```

The default comparison recipe must use `phase_actuator_airdata`.

**Step 4: Verify GREEN**

Run focused tests.

**Step 5: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add leakage-resistant sequence feature modes"
```

---

### Task 3: Causal GRU Model

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add a forward-shape test:

```python
def test_causal_gru_regressor_forward_shape_with_current_features():
    model = CausalGRURegressor(
        sequence_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    seq = torch.randn(5, 8, 4)
    current = torch.randn(5, 3)

    out = model(seq, current)

    assert out.shape == (5, 6)
```

Add a no-current-features test:

```python
def test_causal_gru_regressor_forward_shape_without_current_features():
    model = CausalGRURegressor(
        sequence_input_dim=4,
        current_input_dim=0,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    out = model(torch.randn(5, 8, 4), None)

    assert out.shape == (5, 6)
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_causal_gru_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_gru_regressor_forward_shape_without_current_features -q
```

Expected: fail because `CausalGRURegressor` is not defined.

**Step 3: Implement**

Add `CausalGRURegressor` near the existing model classes:

```python
class CausalGRURegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        ...

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        ...
```

Implementation requirements:
- use `batch_first=True`
- use the final GRU hidden state as sequence representation
- concatenate current features only when `current_input_dim > 0`
- use an MLP head with ReLU and optional dropout
- validate dimensions in `__init__`

**Step 4: Verify GREEN**

Run focused tests.

**Step 5: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add causal GRU regressor"
```

---

### Task 4: Sequence Training and Evaluation Path

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing smoke test**

Add a training smoke test:

```python
def test_run_training_job_supports_causal_gru_model(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 10, "train_log"), ("val", 11, "val_log"), ("test", 12, "test_log")]:
        frame = _synthetic_frame(n_rows=48, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "run",
        feature_set_name="paper_no_accel_v2",
        model_type="causal_gru",
        sequence_history_size=8,
        sequence_feature_mode="phase_actuator_airdata",
        hidden_sizes=(16,),
        batch_size=16,
        max_epochs=1,
        device="cpu",
        use_amp=False,
    )

    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))

    assert cfg["model_type"] == "causal_gru"
    assert cfg["sequence_history_size"] == 8
    assert cfg["sequence_feature_mode"] == "phase_actuator_airdata"
    assert "velocity_b.x" not in cfg["sequence_feature_columns"]
    assert metrics["test"]["sample_count"] == 41
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_run_training_job_supports_causal_gru_model -q
```

Expected: fail because `model_type="causal_gru"` and sequence config are unsupported.

**Step 3: Implement**

Add a sequence branch without disrupting existing MLP/PFNN behavior.

New helpers:

```python
def _normalized_model_type(model_type: str | None) -> str:
    # include causal_gru and causal_gru_asl
```

```python
def fit_torch_sequence_regressor(...):
    ...
```

```python
def evaluate_sequence_model_bundle(...):
    ...
```

```python
def _build_sequence_model_from_bundle(...):
    ...
```

Implementation requirements:
- keep `fit_torch_regressor(...)` for point models
- route `run_training_job(...)` to `fit_torch_sequence_regressor(...)` when `model_type in {"causal_gru", "causal_gru_asl"}`
- normalize sequence features from train only; compute sequence stats across both sample and time dimensions
- normalize current features separately from sequence features
- normalize targets exactly as point models do
- save model bundle with enough metadata to evaluate train/val/test later
- save `training_config.json` fields:
  - `model_type`
  - `sequence_history_size`
  - `sequence_feature_mode`
  - `sequence_feature_columns`
  - `current_feature_columns`
  - `sequence_sample_count_train`
  - `sequence_sample_count_val`
  - `sequence_group_columns`
  - `sequence_sort_column`
- preserve existing output files:
  - `model_bundle.pt`
  - `metrics.json`
  - `history.csv`
  - `training_curves.png`
  - `pred_vs_true_test.png`
  - `residual_hist_test.png`

**Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_training.py::test_run_training_job_supports_causal_gru_model -q
pytest tests/test_training.py -q
```

**Step 5: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: train and evaluate causal GRU models"
```

---

### Task 5: Adaptive Spectrum Layer

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add ASL shape and config tests:

```python
def test_adaptive_spectrum_layer_preserves_sequence_shape():
    layer = AdaptiveSpectrumLayer(input_dim=4, hidden_size=8, dropout=0.0, max_frequency_bins=5)
    x = torch.randn(3, 16, 4)

    y = layer(x)

    assert y.shape == x.shape
```

```python
def test_causal_gru_asl_regressor_forward_shape():
    model = CausalGRUASLRegressor(
        sequence_input_dim=4,
        current_input_dim=2,
        output_dim=6,
        gru_hidden_size=16,
        gru_num_layers=1,
        asl_hidden_size=8,
        asl_dropout=0.0,
        asl_max_frequency_bins=5,
        head_hidden_sizes=(12,),
    )

    out = model(torch.randn(5, 16, 4), torch.randn(5, 2))

    assert out.shape == (5, 6)
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_adaptive_spectrum_layer_preserves_sequence_shape \
       tests/test_training.py::test_causal_gru_asl_regressor_forward_shape -q
```

Expected: fail because ASL classes are not defined.

**Step 3: Implement**

Add:

```python
class AdaptiveSpectrumLayer(nn.Module):
    ...
```

ASL algorithm:
- input shape `[B, H, F]`
- `torch.fft.rfft(x, dim=1)`
- keep frequency bins up to `asl_max_frequency_bins` or all bins if `None`
- stack `abs`, `cos(angle)`, `sin(angle)` along the feature representation
- use a small learned MLP to produce a frequency gate
- multiply original complex spectrum by the gate
- pad omitted bins with zeros
- `torch.fft.irfft(..., n=H, dim=1)`
- add residual skip connection from input to reconstructed signal

Add:

```python
class CausalGRUASLRegressor(nn.Module):
    ...
```

This should apply ASL before the GRU and reuse the same head logic as `CausalGRURegressor`.

**Step 4: Verify GREEN**

Run focused tests.

**Step 5: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add adaptive spectrum sequence model"
```

---

### Task 6: CLI and Comparison Recipes

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`
- Modify: `scripts/run_baseline_comparison.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add tests that `run_baseline_comparison(...)` accepts:

```python
recipe_names=[
    "causal_gru_paper_no_accel_v2_phase_actuator_airdata",
    "causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata",
]
```

The summary must include:

```text
recipe_name
model_type
sequence_history_size
sequence_feature_mode
test_overall_r2
test_fx_b_rmse
```

**Step 2: Verify RED**

Run the focused comparison test.

Expected: fail with unknown recipes or unsupported CLI args.

**Step 3: Implement**

Add recipes:

```python
"causal_gru_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "causal_gru",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "sequence_history_size": 64,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
}
```

```python
"causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "causal_gru_asl",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "sequence_history_size": 64,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
    "asl_hidden_size": 128,
    "asl_dropout": 0.1,
    "asl_max_frequency_bins": None,
}
```

Add CLI options:

```text
--sequence-history-size
--sequence-feature-mode
--current-feature-mode
--gru-hidden-size
--gru-num-layers
--asl-hidden-size
--asl-dropout
--asl-max-frequency-bins
```

Keep existing defaults unchanged for MLP/PFNN.

**Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_training.py -q
python -m py_compile scripts/train_baseline_torch.py scripts/run_baseline_comparison.py
```

**Step 5: Commit**

```bash
git add src/system_identification/training.py scripts/train_baseline_torch.py scripts/run_baseline_comparison.py tests/test_training.py
git commit -m "feat: add causal sequence comparison recipes"
```

---

### Task 7: Leakage and Causality Config Checks

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add a test that the saved `training_config.json` explicitly records no dangerous history:

```python
def test_causal_gru_config_records_no_dangerous_history(tmp_path: Path):
    ...
    assert cfg["has_velocity_history"] is False
    assert cfg["has_angular_velocity_history"] is False
    assert cfg["has_alpha_beta_history"] is False
    assert cfg["has_acceleration_inputs"] is False
```

**Step 2: Verify RED**

Run the focused test.

Expected: fail because the audit flags are missing.

**Step 3: Implement**

Add config audit flags for all training jobs:

```text
has_acceleration_inputs
has_velocity_history
has_angular_velocity_history
has_alpha_beta_history
has_centered_window
uses_whole_log_split
```

For sequence models, compute history flags from `sequence_feature_columns`, not from current-only columns.

**Step 4: Verify GREEN**

Run focused test and `pytest tests/test_training.py -q`.

**Step 5: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "test: record sequence leakage audit flags"
```

---

### Task 8: Smoke Runs

**Files:**
- No tracked code changes expected.

**Step 1: Run tiny smoke comparison**

Run:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_causal_sequence_smoke \
  --recipes causal_gru_paper_no_accel_v2_phase_actuator_airdata causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata \
  --max-train-samples 512 \
  --max-val-samples 256 \
  --max-test-samples 256 \
  --sequence-history-size 16 \
  --hidden-sizes 32 \
  --batch-size 64 \
  --max-epochs 1 \
  --device cpu \
  --disable-amp
```

Expected:

```text
baseline_comparison_summary.csv
baseline_comparison_summary.json
baseline_comparison_summary.png
```

**Step 2: Inspect smoke config**

Read both `training_config.json` files and confirm:

```text
model_type is causal_gru or causal_gru_asl
sequence_feature_mode is phase_actuator_airdata
has_velocity_history is false
has_angular_velocity_history is false
has_alpha_beta_history is false
```

**Step 3: Fix smoke-only issues**

If shape, serialization, or plotting fails, add a focused regression test before fixing.

**Step 4: Commit fixes if needed**

```bash
git add <fixed files>
git commit -m "fix: stabilize causal sequence smoke run"
```

---

### Task 9: Full Baseline Comparison

**Files:**
- No tracked code changes expected.

**Step 1: Run full comparison**

Run:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_mlp_vs_causal_gru_asl \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --device auto
```

**Step 2: Compare headline metrics**

Read:

```text
baseline_comparison_summary.csv
```

Compare:

```text
test_overall_mae
test_overall_rmse
test_overall_r2
test_fx_b_r2 ... test_mz_b_r2
test_fx_b_rmse ... test_mz_b_rmse
best_epoch
best_val_loss
```

**Step 3: Run diagnostics on the best sequence candidate**

Run:

```bash
python scripts/run_training_diagnostics.py \
  --model-bundle artifacts/<comparison_run>/<best_recipe>/model_bundle.pt \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/<comparison_run>/<best_recipe>/diagnostics \
  --splits test
```

If `run_training_diagnostics.py` cannot evaluate sequence bundles, add support before reporting final results.

**Step 4: Interpret**

Report:
- whether GRU beats MLP on test overall metrics
- whether ASL beats plain GRU
- which targets improve or regress
- worst per-log cases
- worst airspeed/frequency/phase bins
- whether any result looks suspiciously high

Do not call the model better unless per-target and per-log behavior also improves.

---

### Task 10: Documentation Update

**Files:**
- Modify: `docs/plans/2026-05-06-leakage-resistant-baseline-protocol.md`
- Optionally create: `docs/results/<date>-causal-sequence-baseline-comparison.md`

**Step 1: Update protocol**

Add a section:

```text
Forward sequence candidates inspired by Sharvit et al. 2025
```

Include:
- `causal_gru_paper_no_accel_v2_phase_actuator_airdata`
- `causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata`
- why they are causal forward adaptations
- why velocity/omega histories are excluded from primary sequence history
- why centered windows remain diagnostic-only

**Step 2: Save result summary**

Create a short result note after full comparison:

```text
dataset
recipes
command
summary table
interpretation
artifact paths
known caveats
```

**Step 3: Commit**

```bash
git add docs/plans/2026-05-06-leakage-resistant-baseline-protocol.md docs/results/<date>-causal-sequence-baseline-comparison.md
git commit -m "docs: document causal sequence baseline protocol"
```

---

## Final Verification Checklist

Run before declaring the sequence work complete:

```bash
pytest tests/test_training.py -q
pytest -q
python -m py_compile scripts/train_baseline_torch.py scripts/run_baseline_comparison.py scripts/run_training_diagnostics.py
```

Confirm at least one smoke artifact exists:

```text
artifacts/YYYYMMDD_HHMMSS_causal_sequence_smoke/baseline_comparison_summary.csv
```

Confirm at least one full comparison artifact exists:

```text
artifacts/YYYYMMDD_HHMMSS_mlp_vs_causal_gru_asl/baseline_comparison_summary.csv
```

The final report must include per-target metrics, not just overall R2.
