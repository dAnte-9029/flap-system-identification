# Temporal Backbone Screening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run a staged, leakage-resistant experiment pipeline to decide whether deployable temporal backbones beyond the current causal GRU are worth full tuning.

**Architecture:** Reuse the existing whole-log split, no-acceleration feature protocol, causal sequence builder, model bundle format, and diagnostics. Add missing sequence backbones behind the same `(history_sequence, current_features) -> wrench` interface, then run quick screen, targeted sweep, and final full-data comparison with fixed promotion/rejection rules.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `src/system_identification/training.py`, existing `scripts/train_baseline_torch.py`, existing `scripts/run_baseline_comparison.py`, existing `scripts/run_training_diagnostics.py`.

---

## Non-Negotiable Protocol

Use this dataset split unless the user explicitly changes it:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
```

All candidate models must use:

```text
split_policy: whole_log
feature_set_name: paper_no_accel_v2
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
sequence_history_size: 64 for the first quick screen
no acceleration inputs
no centered window
no past wrench / no past target output
selection_metric: val_loss
headline metric: test_overall_rmse, test_overall_r2
secondary metrics: per-target R2/RMSE, per-log metrics, per-regime metrics
```

Reference models:

```text
mlp_paper_no_accel_v2
causal_gru_paper_no_accel_v2_phase_actuator_airdata
```

Candidate temporal backbones:

```text
causal_lstm_paper_no_accel_v2_phase_actuator_airdata
causal_tcn_paper_no_accel_v2_phase_actuator_airdata
causal_transformer_paper_no_accel_v2_phase_actuator_airdata
causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata
```

Keep CT SUBNET out of this screening plan. It is a separate research branch because the paper-faithful latent encoder would want past output, while our output is unmeasured aero/effective wrench.

## Decision Rules

Use `causal_gru_paper_no_accel_v2_phase_actuator_airdata` as the main reference.

Promote a model to full-data final comparison if any of these are true in the targeted sweep:

```text
test_overall_rmse <= 0.97 * causal_gru_test_overall_rmse
test_overall_r2 >= causal_gru_test_overall_r2 + 0.02
overall within 3% RMSE of causal GRU and improves at least two hard targets among fy_b, mx_b, mz_b by >= 0.03 R2
overall within 3% RMSE of causal GRU and improves the worst low-frequency or worst-log diagnostic by >= 5% RMSE
```

Keep as an ablation, but do not promote, if:

```text
0.97 * causal_gru_rmse < candidate_rmse <= 1.03 * causal_gru_rmse
and diagnostics show one clear niche improvement
```

Reject for now if:

```text
candidate_rmse > 1.05 * causal_gru_rmse
and it does not improve any hard target or worst-regime diagnostic
```

Mark as unstable if:

```text
train loss improves but validation loss degrades for most epochs
best_epoch <= 3 with no later recovery
validation/test ranking flips strongly between quick screen and sweep
```

---

### Task 1: Add Forward-Shape Tests for New Backbones

**Files:**
- Modify: `tests/test_training.py`
- Modify later: `src/system_identification/training.py:383-564`

**Step 1: Write failing tests**

Add focused model shape tests near the existing causal GRU tests:

```python
def test_causal_lstm_regressor_forward_shape_with_current_features():
    model = CausalLSTMRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )
    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)
    assert output.shape == (4, 6)


def test_causal_tcn_regressor_forward_shape_with_current_features():
    model = CausalTCNRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        channels=16,
        num_blocks=3,
        kernel_size=3,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )
    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)
    assert output.shape == (4, 6)


def test_causal_transformer_regressor_forward_shape_with_current_features():
    model = CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )
    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)
    assert output.shape == (4, 6)


def test_causal_tcn_gru_regressor_forward_shape_with_current_features():
    model = CausalTCNGRURegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        tcn_channels=16,
        tcn_num_blocks=2,
        tcn_kernel_size=3,
        gru_hidden_size=16,
        gru_num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )
    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)
    assert output.shape == (4, 6)
```

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_training.py::test_causal_lstm_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_tcn_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_transformer_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_tcn_gru_regressor_forward_shape_with_current_features -q
```

Expected: fail because the new classes are not imported/defined.

**Step 3: Commit after Task 2 passes**

Do not commit the failing tests alone unless the implementation is intentionally split.

---

### Task 2: Implement Deployable Sequence Backbones

**Files:**
- Modify: `src/system_identification/training.py:383-564`
- Test: `tests/test_training.py`

**Step 1: Add `CausalLSTMRegressor`**

Implementation rules:

```text
same forward signature as CausalGRURegressor
input: sequence [B, H, C], current_features [B, P] or None
use the final LSTM hidden state
concatenate current_features if present
feed through an MLP head
do not access targets or future samples
```

Use the same head style as `CausalGRURegressor`.

**Step 2: Add `CausalTCNRegressor`**

Implementation rules:

```text
sequence transpose: [B, H, C] -> [B, C, H]
use left-padded causal Conv1d blocks
dilation schedule: 1, 2, 4, ...
crop convolution output back to length H
use the last timestep representation
concatenate current_features if present
feed through an MLP head
```

Add a small private block:

```python
class _CausalConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation, dropout):
        ...

    def forward(self, x):
        ...
```

Use residual projection when input/output channels differ.

**Step 3: Add `CausalTransformerRegressor`**

Implementation rules:

```text
linear projection from sequence_input_dim to d_model
learned positional embedding for max_history_size
TransformerEncoder with causal attention mask
use final token representation
concatenate current_features if present
feed through an MLP head
```

Add validation:

```text
d_model % num_heads == 0
history length <= max_history_size
```

**Step 4: Add `CausalTCNGRURegressor`**

Implementation rules:

```text
TCN extracts local phase/actuator/airdata temporal features
GRU summarizes the TCN feature sequence
final GRU state + current_features -> MLP head
```

This is a second-stage candidate. It can be skipped in the first quick run if runtime is too high, but the code path should exist for later comparison.

**Step 5: Run shape tests**

Run:

```bash
pytest tests/test_training.py::test_causal_lstm_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_tcn_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_transformer_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_tcn_gru_regressor_forward_shape_with_current_features -q
```

Expected: pass.

---

### Task 3: Route New Backbones Through Sequence Training

**Files:**
- Modify: `src/system_identification/training.py:2008-2030`
- Modify: `src/system_identification/training.py:2180-2428`
- Modify: `src/system_identification/training.py:3060-3125`
- Test: `tests/test_training.py`

**Step 1: Write failing training-job tests**

Add one parameterized smoke test:

```python
@pytest.mark.parametrize(
    "model_type",
    ["causal_lstm", "causal_tcn", "causal_transformer", "causal_tcn_gru"],
)
def test_run_training_job_supports_temporal_sequence_model_types(tmp_path: Path, model_type: str):
    split_root = _write_tiny_split(tmp_path)
    output_dir = tmp_path / model_type

    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        feature_set_name="paper_no_accel_v2",
        model_type=model_type,
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        loss_type="huber",
        huber_delta=1.5,
        sequence_history_size=4,
        sequence_feature_mode="phase_actuator_airdata",
        current_feature_mode="remaining_current",
        device="cpu",
        use_amp=False,
    )

    assert Path(outputs["model_bundle_path"]).exists()
    cfg = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == model_type
    assert cfg["has_acceleration_inputs"] is False
    assert cfg["has_centered_window"] is False
```

Use the existing tiny split helper already present in `tests/test_training.py`.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_training.py::test_run_training_job_supports_temporal_sequence_model_types -q
```

Expected: fail because `_normalized_model_type` and sequence builder do not support the new model types.

**Step 3: Extend model-type routing**

Update:

```python
def _normalized_model_type(model_type: str | None) -> str:
    ...
    "causal_lstm",
    "causal_tcn",
    "causal_transformer",
    "causal_tcn_gru",


def _is_sequence_model_type(model_type: str | None) -> bool:
    return _normalized_model_type(model_type) in {
        "causal_gru",
        "causal_gru_asl",
        "causal_lstm",
        "causal_tcn",
        "causal_transformer",
        "causal_tcn_gru",
    }
```

**Step 4: Add sequence hyperparameters to `fit_torch_sequence_regressor(...)`**

Add parameters with conservative defaults:

```python
tcn_channels: int = 128
tcn_num_blocks: int = 4
tcn_kernel_size: int = 3
transformer_d_model: int = 64
transformer_num_layers: int = 1
transformer_num_heads: int = 4
transformer_dim_feedforward: int = 128
```

Keep `hidden_sizes[0]` as the default hidden/channel/d_model source when explicit values are not passed from recipe or CLI.

**Step 5: Build the correct sequence model**

Replace the current GRU-only branch with a small helper:

```python
def _build_sequence_regressor(...):
    if model_type == "causal_gru":
        ...
    if model_type == "causal_gru_asl":
        ...
    if model_type == "causal_lstm":
        ...
    if model_type == "causal_tcn":
        ...
    if model_type == "causal_transformer":
        ...
    if model_type == "causal_tcn_gru":
        ...
```

Save all new hyperparameters into the returned bundle and `training_config.json`.

**Step 6: Update bundle reload/prediction**

Ensure `_build_sequence_model_from_bundle(...)` reconstructs the correct class for all new model types.

**Step 7: Run focused tests**

Run:

```bash
pytest tests/test_training.py::test_run_training_job_supports_temporal_sequence_model_types -q
```

Expected: pass.

---

### Task 4: Add Recipes and CLI Arguments

**Files:**
- Modify: `src/system_identification/training.py:116-246`
- Modify: `src/system_identification/training.py:4035-4375`
- Modify: `scripts/train_baseline_torch.py:38-118`
- Modify: `scripts/run_baseline_comparison.py:33-111`
- Test: `tests/test_training.py`

**Step 1: Write failing recipe test**

Add:

```python
def test_run_baseline_comparison_supports_temporal_backbone_recipes(tmp_path: Path):
    split_root = _write_tiny_split(tmp_path)
    output_dir = tmp_path / "comparison"

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=output_dir,
        recipe_names=[
            "causal_lstm_paper_no_accel_v2_phase_actuator_airdata",
            "causal_tcn_paper_no_accel_v2_phase_actuator_airdata",
            "causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
            "causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata",
        ],
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        device="cpu",
        use_amp=False,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert set(summary["model_type"]) == {
        "causal_lstm",
        "causal_tcn",
        "causal_transformer",
        "causal_tcn_gru",
    }
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_training.py::test_run_baseline_comparison_supports_temporal_backbone_recipes -q
```

Expected: fail because recipes and/or CLI args do not exist.

**Step 3: Add recipes**

Add to `BASELINE_COMPARISON_RECIPES`:

```python
"causal_lstm_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "causal_lstm",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "window_mode": "single",
    "window_radius": 0,
    "window_feature_mode": "all",
    "sequence_history_size": 64,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
},
```

Add equivalent recipes for:

```text
causal_tcn
causal_transformer
causal_tcn_gru
```

Use conservative recipe defaults:

```text
tcn_channels: 128
tcn_num_blocks: 4
tcn_kernel_size: 3
transformer_d_model: 64
transformer_num_layers: 1
transformer_num_heads: 4
transformer_dim_feedforward: 128
```

**Step 4: Add CLI arguments**

Update `--model-type` choices in `scripts/train_baseline_torch.py`.

Add to both training scripts:

```text
--tcn-channels
--tcn-num-blocks
--tcn-kernel-size
--transformer-d-model
--transformer-num-layers
--transformer-num-heads
--transformer-dim-feedforward
```

Pass them into `run_training_job(...)` and `run_baseline_comparison(...)`.

**Step 5: Run focused tests and compile scripts**

Run:

```bash
pytest tests/test_training.py::test_run_baseline_comparison_supports_temporal_backbone_recipes -q
python -m py_compile scripts/train_baseline_torch.py scripts/run_baseline_comparison.py src/system_identification/training.py
```

Expected: pass.

**Step 6: Commit**

Run:

```bash
git add src/system_identification/training.py scripts/train_baseline_torch.py scripts/run_baseline_comparison.py tests/test_training.py
git commit -m "feat: add deployable temporal backbone recipes"
```

---

### Task 5: Add a Temporal Backbone Screening Runner

**Files:**
- Create: `scripts/run_temporal_backbone_screen.py`
- Test: `tests/test_training.py`

**Step 1: Write tests for screening-grid generation**

Add pure-function tests so the runner can be tested without launching training:

```python
def test_temporal_screen_quick_grid_contains_reference_and_candidates():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="quick")
    names = {config.recipe_name for config in configs}

    assert "mlp_paper_no_accel_v2" in names
    assert "causal_gru_paper_no_accel_v2_phase_actuator_airdata" in names
    assert "causal_lstm_paper_no_accel_v2_phase_actuator_airdata" in names
    assert "causal_tcn_paper_no_accel_v2_phase_actuator_airdata" in names
    assert "causal_transformer_paper_no_accel_v2_phase_actuator_airdata" in names
```

Add a decision-rule test:

```python
def test_classify_temporal_candidate_promotes_clear_rmse_win():
    from scripts.run_temporal_backbone_screen import classify_candidate

    decision = classify_candidate(
        candidate_rmse=0.96,
        reference_rmse=1.00,
        candidate_r2=0.72,
        reference_r2=0.70,
        hard_target_improvements=0,
        worst_regime_rmse_improvement=0.0,
    )

    assert decision == "promote"
```

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_quick_grid_contains_reference_and_candidates \
       tests/test_training.py::test_classify_temporal_candidate_promotes_clear_rmse_win -q
```

Expected: fail because the runner does not exist.

**Step 3: Implement the runner**

The script should expose:

```python
@dataclass(frozen=True)
class ScreenConfig:
    config_id: str
    stage: str
    recipe_name: str
    hidden_sizes: tuple[int, ...]
    sequence_history_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    dropout: float
    extra_args: dict[str, int | float | str | None]
```

Functions:

```python
def build_screen_configs(stage: str) -> list[ScreenConfig]:
    ...

def classify_candidate(...) -> str:
    ...

def main() -> None:
    ...
```

CLI:

```text
--split-root
--output-dir
--stage quick|sweep|final|all
--device
--batch-size
--num-workers
--max-train-samples
--max-val-samples
--max-test-samples
--dry-run
```

For each config, call `run_baseline_comparison(...)` with one recipe at a time and write:

```text
artifacts/<run>/runs/<config_id>/baseline_comparison_summary.csv
artifacts/<run>/temporal_backbone_screen_summary.csv
artifacts/<run>/temporal_backbone_screen_summary.json
```

**Step 4: Quick-stage grid**

Use this grid:

```text
mlp reference:
  hidden_sizes=(128, 128)

causal_gru reference:
  hidden_sizes=(128, 128)
  sequence_history_size=64

causal_lstm:
  hidden_sizes=(128, 128)
  sequence_history_size=64

causal_tcn:
  hidden_sizes=(128, 128)
  sequence_history_size=64
  tcn_channels=128
  tcn_num_blocks=4
  tcn_kernel_size=3

causal_transformer:
  hidden_sizes=(64, 128)
  sequence_history_size=64
  transformer_d_model=64
  transformer_num_layers=1
  transformer_num_heads=4
  transformer_dim_feedforward=128

causal_tcn_gru:
  hidden_sizes=(128, 128)
  sequence_history_size=64
  tcn_channels=128
  tcn_num_blocks=3
  tcn_kernel_size=3
```

Quick-stage common budget:

```text
max_epochs=12
early_stopping_patience=4
learning_rate=1e-3
weight_decay=1e-5
dropout=0.05
batch_size=512
max_train_samples=65536
max_val_samples=32768
max_test_samples=32768
```

**Step 5: Sweep-stage grid**

Only run this for candidates that pass quick screening. If the runner cannot auto-select candidates yet, pass `--recipes` manually or edit a small list at the top of the script.

Curated grids:

```text
causal_lstm:
  history: 32, 64, 128
  hidden: 64, 128
  num_layers: 1, 2
  choose at most 8 configs

causal_tcn:
  history: 32, 64, 128
  channels: 64, 128
  num_blocks: 3, 4, 5
  kernel_size: 3, 5
  choose at most 12 configs

causal_transformer:
  history: 64, 128
  d_model: 32, 64
  num_layers: 1, 2
  num_heads: 2, 4
  dropout: 0.05, 0.10
  choose at most 10 configs

causal_tcn_gru:
  history: 64, 128
  tcn_blocks: 2, 3
  gru_hidden: 64, 128
  kernel_size: 3
  choose at most 6 configs
```

Sweep-stage common budget:

```text
max_epochs=20
early_stopping_patience=5
batch_size=512
max_train_samples=131072
max_val_samples=65536
max_test_samples=65536
```

**Step 6: Final-stage grid**

Final stage should run:

```text
mlp reference
causal_gru reference
top 2-3 promoted candidate configs
```

Final-stage common budget:

```text
max_epochs=50
early_stopping_patience=8
batch_size=512
full train/val/test data
```

**Step 7: Run tests**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_quick_grid_contains_reference_and_candidates \
       tests/test_training.py::test_classify_temporal_candidate_promotes_clear_rmse_win -q
python -m py_compile scripts/run_temporal_backbone_screen.py
```

Expected: pass.

**Step 8: Commit**

Run:

```bash
git add scripts/run_temporal_backbone_screen.py tests/test_training.py
git commit -m "feat: add temporal backbone screening runner"
```

---

### Task 6: Run Quick Screen

**Files:**
- Read: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1`
- Create: `artifacts/20260507_temporal_backbone_quick`

**Step 1: Launch quick screen**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_temporal_backbone_quick \
  --stage quick \
  --batch-size 512 \
  --max-train-samples 65536 \
  --max-val-samples 32768 \
  --max-test-samples 32768 \
  --device cuda:0
```

If CUDA is unavailable, rerun with:

```bash
--device cpu
```

**Step 2: Inspect quick results**

Read:

```text
artifacts/20260507_temporal_backbone_quick/temporal_backbone_screen_summary.csv
```

Create a short note with:

```text
quick rank by test_overall_rmse
quick rank by test_overall_r2
per-target R2 for fy_b, mx_b, mz_b
which candidates pass to sweep
which candidates are rejected immediately
```

**Step 3: Commit only code/docs if changed**

Do not commit large artifacts unless the repository already tracks comparable artifacts. Prefer summarizing results in docs.

---

### Task 7: Run Targeted Sweep

**Files:**
- Read: `artifacts/20260507_temporal_backbone_quick/temporal_backbone_screen_summary.csv`
- Create: `artifacts/20260507_temporal_backbone_sweep`

**Step 1: Select candidates**

Apply the decision rules:

```text
promote to sweep if quick RMSE <= 1.05 * quick causal GRU RMSE
or quick hard-target R2 improves at least two of fy_b, mx_b, mz_b
or quick worst-regime/worst-log diagnostic suggests a niche improvement
```

If no candidate passes, still run one TCN sweep because TCN is the strongest domain-prior candidate for fixed-window periodic signals.

**Step 2: Launch sweep**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_temporal_backbone_sweep \
  --stage sweep \
  --batch-size 512 \
  --max-train-samples 131072 \
  --max-val-samples 65536 \
  --max-test-samples 65536 \
  --device cuda:0
```

**Step 3: Inspect sweep results**

Read:

```text
artifacts/20260507_temporal_backbone_sweep/temporal_backbone_screen_summary.csv
```

Select at most 3 final configs:

```text
best overall candidate
best hard-target candidate if different
best worst-regime candidate if different
```

---

### Task 8: Run Full-Data Final Comparison

**Files:**
- Create: `artifacts/20260507_temporal_backbone_final`

**Step 1: Launch final comparison**

If the screening runner supports final selected configs, run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_temporal_backbone_final \
  --stage final \
  --batch-size 512 \
  --device cuda:0
```

If final configs need manual launch, run `scripts/run_baseline_comparison.py` once per selected recipe/config so each run records the exact hyperparameters.

Reference final command pattern:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_temporal_backbone_final \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata <selected_candidate_recipe_1> <selected_candidate_recipe_2> \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --hidden-sizes 128,128 \
  --sequence-history-size 64 \
  --device cuda:0
```

**Step 2: Generate diagnostics for all final models**

For each final recipe:

```bash
python scripts/run_training_diagnostics.py \
  --model-bundle artifacts/20260507_temporal_backbone_final/<recipe>/model_bundle.pt \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_temporal_backbone_final/<recipe>/diagnostics \
  --splits test \
  --batch-size 8192 \
  --device cuda:0
```

**Step 3: Compare final metrics**

Extract:

```text
baseline_comparison_summary.csv
<recipe>/diagnostics/per_log_metrics.csv
<recipe>/diagnostics/per_regime_metrics.csv
```

Create final tables:

```text
overall MAE/RMSE/R2
per-target R2
per-target RMSE
worst log
worst regime bin
training cost proxy: best_epoch and total epochs
decision: default / ablation / reject
```

---

### Task 9: Write Result Report

**Files:**
- Create: `docs/results/2026-05-07-temporal-backbone-screening.md`

**Step 1: Write the report**

Required sections:

```markdown
# Temporal Backbone Screening

## Protocol
## Candidate Models
## Quick Screen
## Targeted Sweep
## Final Full-Data Comparison
## Per-Target Results
## Per-Log and Per-Regime Diagnostics
## Decision
## Artifacts
## Caveats
```

The decision section must answer:

```text
Should causal GRU remain the default?
Did TCN/LSTM/Transformer beat GRU overall?
Did any model improve fy_b, mx_b, mz_b enough to justify a hybrid/ensemble?
Did any model improve low-frequency or worst-log behavior?
Which model should be tuned next, if any?
```

**Step 2: Commit report**

Run:

```bash
git add docs/results/2026-05-07-temporal-backbone-screening.md
git commit -m "docs: summarize temporal backbone screening"
```

---

### Task 10: Final Verification

**Files:**
- Verify: `src/system_identification/training.py`
- Verify: `scripts/train_baseline_torch.py`
- Verify: `scripts/run_baseline_comparison.py`
- Verify: `scripts/run_temporal_backbone_screen.py`
- Verify: `tests/test_training.py`
- Verify: `docs/results/2026-05-07-temporal-backbone-screening.md`

**Step 1: Run focused tests**

Run:

```bash
pytest tests/test_training.py::test_causal_lstm_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_tcn_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_transformer_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_causal_tcn_gru_regressor_forward_shape_with_current_features \
       tests/test_training.py::test_run_training_job_supports_temporal_sequence_model_types \
       tests/test_training.py::test_run_baseline_comparison_supports_temporal_backbone_recipes \
       tests/test_training.py::test_temporal_screen_quick_grid_contains_reference_and_candidates \
       tests/test_training.py::test_classify_temporal_candidate_promotes_clear_rmse_win -q
```

Expected: pass.

**Step 2: Run full tests**

Run:

```bash
pytest -q
```

Expected: pass.

**Step 3: Compile scripts**

Run:

```bash
python -m py_compile \
  scripts/train_baseline_torch.py \
  scripts/run_baseline_comparison.py \
  scripts/run_training_diagnostics.py \
  scripts/run_temporal_backbone_screen.py \
  src/system_identification/training.py
```

Expected: pass.

**Step 4: Check formatting-sensitive diff**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors. Worktree may contain intended committed changes only.

**Step 5: Final commit if needed**

Run:

```bash
git add src/system_identification/training.py scripts/train_baseline_torch.py scripts/run_baseline_comparison.py scripts/run_temporal_backbone_screen.py tests/test_training.py docs/results/2026-05-07-temporal-backbone-screening.md
git commit -m "feat: screen deployable temporal backbones"
```

Skip this commit if the previous task commits already cover all changes cleanly.

