# Three-Stage SUBNET Wrench Identification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and evaluate three Beintema/SUBNET-inspired forward sequence models for flapping-wing wrench identification, and determine whether each stage improves over the existing MLP and causal GRU baselines.

**Architecture:** Keep the current leakage-resistant whole-log/no-acceleration protocol as the anchor. Add a shared causal subsection rollout data path, then implement: (1) multi-step subsection GRU, (2) discrete latent state-space SUBNET, and (3) continuous-time SUBNET-style Euler rollout with derivative normalization. Every model predicts current/near-future six-axis wrench from causal context plus known rollout inputs, and every comparison must report overall, per-target, per-log, and regime metrics.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `src/system_identification/training.py`, existing `scripts/train_baseline_torch.py`, existing `scripts/run_baseline_comparison.py`, existing `scripts/run_training_diagnostics.py`.

---

## Baseline Contract

Use this existing full comparison as the reference unless a newer baseline is intentionally rerun:

```text
artifacts/20260506_full_mlp_vs_causal_gru_v2/baseline_comparison_summary.csv
```

Current reference metrics:

```text
model        best_epoch  test_overall_mae  test_overall_rmse  test_overall_r2
MLP                   9          0.577540           1.195897         0.646652
Causal GRU            7          0.480028           1.007593         0.691769
```

Final claims must answer for each new stage:

```text
Did it beat MLP?
Did it beat causal GRU?
Which targets improved/regressed?
Which logs and regimes are still weak?
Does any result look suspiciously high or leakage-prone?
```

Primary metric:

```text
test_overall_rmse lower than causal GRU
```

Secondary metrics:

```text
test_overall_r2 higher than causal GRU
per-target R2 improves on at least 4/6 targets
no target has a large regression: R2 drop > 0.05 or RMSE increase > 10%
per-log metrics do not improve only by overfitting one log
```

Leakage guardrails:

```text
No acceleration inputs.
No target/wrench history as model input.
No centered windows.
No future target values as input.
No same-log train/val/test split.
Historical velocity/angular-velocity/alpha-beta remain excluded from the sequence/context branch by default.
Current or rollout-time non-acceleration kinematic features may be used only as current/output-side exogenous features, consistent with the existing causal GRU protocol.
```

---

## Shared Model Vocabulary

Use these names consistently:

```text
subsection_gru
subnet_discrete
ct_subnet_euler
```

Comparison recipes:

```text
subsection_gru_paper_no_accel_v2_phase_actuator_airdata
subnet_discrete_paper_no_accel_v2_phase_actuator_airdata
ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata
```

Common subsection notation:

```text
H = context/history size
T = rollout size
u_context = phase + actuator + airdata history before rollout
u_rollout = phase + actuator + airdata during rollout
x_now = non-history point/current features at each rollout step
y_rollout = fx_b, fy_b, fz_b, mx_b, my_b, mz_b during rollout
```

Default first credible settings:

```text
sequence_history_size: 64
rollout_size: 32
rollout_stride: 32
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
hidden_sizes: 128,128
latent_size: 16
dt_over_tau: 0.03 for first CT run, then sweep
```

Using `rollout_stride=rollout_size` avoids duplicated test targets in headline metrics. Later ablations may use overlapping training subsections, but the first result table should use the non-overlapping setting.

---

### Task 1: Shared Causal Rollout Subsection Builder

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add tests for:

```python
def prepare_causal_rollout_feature_target_frames(
    frame: pd.DataFrame,
    context_feature_columns: list[str],
    rollout_feature_columns: list[str],
    current_feature_columns: list[str],
    target_columns: list[str],
    *,
    history_size: int,
    rollout_size: int,
    rollout_stride: int | None = None,
    group_columns: tuple[str, ...] = ("log_id", "segment_id"),
    sort_column: str = "time_s",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    ...
```

Minimum tests:

```python
def test_prepare_causal_rollout_feature_target_frames_aligns_context_rollout_and_targets():
    frame = _synthetic_frame(n_rows=8, seed=1)
    frame["log_id"] = "a"
    frame["segment_id"] = 0
    frame["time_s"] = np.arange(8, dtype=float)
    frame["fx_b"] = frame["time_s"] * 10.0

    context, rollout, current, targets, meta = training_module.prepare_causal_rollout_feature_target_frames(
        frame,
        context_feature_columns=["phase_corrected_sin"],
        rollout_feature_columns=["phase_corrected_sin", "motor_cmd_0"],
        current_feature_columns=["velocity_b.x"],
        target_columns=["fx_b"],
        history_size=3,
        rollout_size=2,
        rollout_stride=2,
    )

    assert context.shape == (2, 3, 1)
    assert rollout.shape == (2, 2, 2)
    assert current.shape == (2, 2, 1)
    assert targets.shape == (2, 2, 1)
    np.testing.assert_allclose(meta["time_s"].to_numpy(), [3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(targets.reshape(-1), [30.0, 40.0, 50.0, 60.0])
```

```python
def test_prepare_causal_rollout_feature_target_frames_never_crosses_log_or_segment():
    frame = _synthetic_frame(n_rows=12, seed=2)
    frame["log_id"] = ["a"] * 6 + ["b"] * 6
    frame["segment_id"] = [0] * 3 + [1] * 3 + [0] * 6
    frame["time_s"] = list(reversed(range(6))) + list(reversed(range(6)))

    context, rollout, current, targets, meta = training_module.prepare_causal_rollout_feature_target_frames(
        frame,
        context_feature_columns=["phase_corrected_sin"],
        rollout_feature_columns=["phase_corrected_sin"],
        current_feature_columns=[],
        target_columns=["fx_b", "fz_b"],
        history_size=2,
        rollout_size=2,
        rollout_stride=1,
    )

    assert context.shape[0] == 4
    assert current.shape == (4, 2, 0)
    assert targets.shape == (4, 2, 2)
    assert not meta.duplicated(["log_id", "segment_id", "time_s"]).any()
```

**Step 2: Run tests to verify RED**

```bash
pytest tests/test_training.py::test_prepare_causal_rollout_feature_target_frames_aligns_context_rollout_and_targets \
       tests/test_training.py::test_prepare_causal_rollout_feature_target_frames_never_crosses_log_or_segment -q
```

Expected: fail because helper does not exist.

**Step 3: Implement**

Implementation requirements:

- call `_with_derived_columns(frame)` first
- validate `history_size >= 1`, `rollout_size >= 1`, `rollout_stride >= 1`
- default `rollout_stride = rollout_size`
- group by available `log_id`, `segment_id`
- sort each group by `time_s`
- context indices are `[start-H, ..., start-1]`
- rollout/target/current indices are `[start, ..., start+T-1]`
- never emit a subsection that crosses a group boundary
- return:

```text
context_features: [N, H, C]
rollout_features: [N, T, U]
current_features: [N, T, P], possibly P=0
target_sequences: [N, T, Y]
metadata: flattened rollout metadata with N*T rows
```

Use vectorized indexing or `sliding_window_view`; do not append one row at a time. The previous causal sequence optimization showed that row-by-row DataFrame appends make full runs impractical.

**Step 4: Run tests to verify GREEN**

```bash
pytest tests/test_training.py::test_prepare_causal_rollout_feature_target_frames_aligns_context_rollout_and_targets \
       tests/test_training.py::test_prepare_causal_rollout_feature_target_frames_never_crosses_log_or_segment -q
```

**Step 5: Add a full-split performance check**

Run:

```bash
python - <<'PY'
import time
from pathlib import Path
import pandas as pd
from system_identification.training import (
    DEFAULT_TARGET_COLUMNS,
    prepare_causal_rollout_feature_target_frames,
    resolve_current_feature_columns,
    resolve_feature_set_columns,
    resolve_sequence_feature_columns,
)
root = Path("dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1")
frame = pd.read_parquet(root / "train_samples.parquet")
base = resolve_feature_set_columns("paper_no_accel_v2")
seq = resolve_sequence_feature_columns(base, "phase_actuator_airdata")
cur = resolve_current_feature_columns(base, seq, "remaining_current")
t0 = time.perf_counter()
arrays = prepare_causal_rollout_feature_target_frames(
    frame, seq, seq, cur, DEFAULT_TARGET_COLUMNS,
    history_size=64, rollout_size=32, rollout_stride=32,
)
print("seconds", round(time.perf_counter() - t0, 3))
print([a.shape for a in arrays[:4]], arrays[4].shape)
PY
```

Expected: completes in seconds, not minutes.

**Step 6: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: build causal rollout subsections"
```

---

### Task 2: Shared Rollout Training and Evaluation Infrastructure

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing tests**

Add a tiny model-independent smoke test for the rollout data/evaluation path once `subsection_gru` exists as a placeholder or minimal model:

```python
def test_run_training_job_supports_rollout_model_config(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 1, "train_log"), ("val", 2, "val_log"), ("test", 3, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "run",
        feature_set_name="paper_no_accel_v2",
        model_type="subsection_gru",
        sequence_history_size=8,
        rollout_size=4,
        rollout_stride=4,
        sequence_feature_mode="phase_actuator_airdata",
        hidden_sizes=(16,),
        batch_size=8,
        max_epochs=1,
        device="cpu",
        use_amp=False,
    )

    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    assert cfg["model_type"] == "subsection_gru"
    assert cfg["rollout_size"] == 4
    assert cfg["rollout_stride"] == 4
    assert cfg["has_acceleration_inputs"] is False
    assert cfg["has_velocity_history"] is False
    assert metrics["test"]["sample_count"] > 0
```

**Step 2: Run test to verify RED**

```bash
pytest tests/test_training.py::test_run_training_job_supports_rollout_model_config -q
```

Expected: fail because `model_type="subsection_gru"` and rollout config are unsupported.

**Step 3: Implement common rollout path**

Add model-type helpers:

```python
ROLLOUT_MODEL_TYPES = {"subsection_gru", "subnet_discrete", "ct_subnet_euler"}
```

Add shared helpers:

```python
def _is_rollout_model_type(model_type: str | None) -> bool:
    ...

def _make_rollout_loader(...):
    ...

def _fit_rollout_feature_stats(...):
    ...

def _transform_rollout_features(...):
    ...

def fit_torch_rollout_regressor(...):
    ...

def evaluate_rollout_model_bundle(...):
    ...

def _rollout_arrays_for_bundle(...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    ...
```

Evaluation must flatten `[N, T, Y]` to `[N*T, Y]` before calling `_metrics_from_arrays(...)`.

Training config must save:

```text
model_type
sequence_history_size
rollout_size
rollout_stride
sequence_feature_mode
current_feature_mode
context_feature_columns
rollout_feature_columns
current_feature_columns
target_columns
rollout_sample_count_train
rollout_timestep_count_train
rollout_sample_count_val
rollout_timestep_count_val
has_acceleration_inputs
has_velocity_history
has_angular_velocity_history
has_alpha_beta_history
has_centered_window
uses_whole_log_split
```

**Step 4: Integrate dispatch**

Update:

```text
run_training_job(...)
predict_model_bundle(...)
evaluate_model_bundle(...)
_save_pred_vs_true_plot(...)
_save_residual_hist_plot(...)
run_baseline_comparison(...)
```

Rollout bundle predictions must align with rollout metadata. Do not compute regime diagnostics by filtering the frame before rollout construction; this breaks causal context. For rollout models, diagnostics should:

```text
1. build predictions on the full split
2. flatten predictions, targets, metadata
3. bin by metadata columns
```

**Step 5: Run tests to verify GREEN**

```bash
pytest tests/test_training.py::test_run_training_job_supports_rollout_model_config -q
pytest tests/test_training.py -q
```

**Step 6: Commit**

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add rollout training infrastructure"
```

---

### Task 3: Stage 1 Model - Multi-Step Subsection GRU

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`
- Modify: `scripts/run_baseline_comparison.py`
- Test: `tests/test_training.py`

**Step 1: Write failing model tests**

```python
def test_subsection_gru_regressor_forward_shape():
    model = training_module.SubsectionGRUWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    out = model(
        torch.randn(5, 8, 4),
        torch.randn(5, 6, 4),
        torch.randn(5, 6, 3),
    )

    assert out.shape == (5, 6, 6)
```

Also test no current features:

```python
def test_subsection_gru_regressor_forward_shape_without_current_features():
    model = training_module.SubsectionGRUWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=0,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    out = model(torch.randn(5, 8, 4), torch.randn(5, 6, 4), None)
    assert out.shape == (5, 6, 6)
```

**Step 2: Run tests to verify RED**

```bash
pytest tests/test_training.py::test_subsection_gru_regressor_forward_shape \
       tests/test_training.py::test_subsection_gru_regressor_forward_shape_without_current_features -q
```

Expected: fail because class is undefined.

**Step 3: Implement model**

Add:

```python
class SubsectionGRUWrenchRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        ...

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

Architecture:

```text
context_encoder = GRU(context_inputs) -> h0
rollout_decoder = GRU(rollout_inputs, h0) -> hidden sequence
head_input_k = [hidden_k, current_k]
head MLP -> wrench_hat_k
```

Do not feed previous true or predicted wrench into the decoder.

**Step 4: Add recipe and CLI**

Add recipe:

```python
"subsection_gru_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "subsection_gru",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "sequence_history_size": 64,
    "rollout_size": 32,
    "rollout_stride": 32,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
}
```

Add CLI arguments to both training scripts:

```text
--rollout-size
--rollout-stride
```

**Step 5: Run focused tests**

```bash
pytest tests/test_training.py::test_subsection_gru_regressor_forward_shape \
       tests/test_training.py::test_subsection_gru_regressor_forward_shape_without_current_features \
       tests/test_training.py::test_run_training_job_supports_rollout_model_config -q
```

**Step 6: Smoke run**

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_subsection_gru_smoke \
  --recipes subsection_gru_paper_no_accel_v2_phase_actuator_airdata \
  --max-train-samples 4096 \
  --max-val-samples 2048 \
  --max-test-samples 2048 \
  --sequence-history-size 16 \
  --rollout-size 8 \
  --rollout-stride 8 \
  --hidden-sizes 32 \
  --batch-size 64 \
  --max-epochs 1 \
  --device cpu \
  --disable-amp
```

Expected artifacts:

```text
baseline_comparison_summary.csv
subsection_gru_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
subsection_gru_paper_no_accel_v2_phase_actuator_airdata/training_config.json
```

**Step 7: Full run for Stage 1**

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_stage1_subsection_gru_vs_baselines \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata subsection_gru_paper_no_accel_v2_phase_actuator_airdata \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --hidden-sizes 128,128 \
  --sequence-history-size 64 \
  --rollout-size 32 \
  --rollout-stride 32 \
  --device cuda:0
```

**Step 8: Diagnose Stage 1**

Run diagnostics for the best baseline and Stage 1:

```bash
python scripts/run_training_diagnostics.py \
  --model-bundle artifacts/<stage1_run>/subsection_gru_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/<stage1_run>/subsection_gru_paper_no_accel_v2_phase_actuator_airdata/diagnostics \
  --splits test \
  --device cuda:0
```

**Step 9: Commit**

```bash
git add src/system_identification/training.py scripts/train_baseline_torch.py scripts/run_baseline_comparison.py tests/test_training.py
git commit -m "feat: add multi-step subsection GRU baseline"
```

---

### Task 4: Stage 2 Model - Discrete Latent SUBNET

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`
- Modify: `scripts/run_baseline_comparison.py`
- Test: `tests/test_training.py`

**Step 1: Write failing model tests**

```python
def test_discrete_subnet_wrench_regressor_forward_shape():
    model = training_module.DiscreteSUBNETWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        latent_size=10,
        hidden_sizes=(16,),
        dropout=0.0,
    )

    out = model(
        torch.randn(5, 8, 4),
        torch.randn(5, 6, 4),
        torch.randn(5, 6, 3),
    )

    assert out.shape == (5, 6, 6)
```

**Step 2: Run test to verify RED**

```bash
pytest tests/test_training.py::test_discrete_subnet_wrench_regressor_forward_shape -q
```

Expected: fail because class is undefined.

**Step 3: Implement model**

Add:

```python
class DiscreteSUBNETWrenchRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        latent_size: int = 16,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ):
        ...

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

Architecture:

```text
z0 = encoder_gru(context_inputs)
for k in 0..T-1:
    y_hat_k = output_net([z_k, rollout_input_k, current_k])
    dz_k = transition_net([z_k, rollout_input_k])
    z_{k+1} = z_k + dz_k
```

This is a discrete-time SUBNET-style model. It uses the SUBNET short-subsection loss and encoder initial state, but no continuous-time `dt/tau` scaling yet.

Add optional latent diagnostics to training history:

```text
train_latent_rms
val_latent_rms
train_delta_latent_rms
val_delta_latent_rms
```

If adding these metrics is too invasive, at minimum save them in `metrics.json` after train/val/test evaluation.

**Step 4: Add config and CLI**

Add `latent_size` argument:

```text
--latent-size
```

Add recipe:

```python
"subnet_discrete_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "subnet_discrete",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "sequence_history_size": 64,
    "rollout_size": 32,
    "rollout_stride": 32,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
    "latent_size": 16,
}
```

**Step 5: Smoke run**

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_subnet_discrete_smoke \
  --recipes subnet_discrete_paper_no_accel_v2_phase_actuator_airdata \
  --max-train-samples 4096 \
  --max-val-samples 2048 \
  --max-test-samples 2048 \
  --sequence-history-size 16 \
  --rollout-size 8 \
  --rollout-stride 8 \
  --latent-size 8 \
  --hidden-sizes 32 \
  --batch-size 64 \
  --max-epochs 1 \
  --device cpu \
  --disable-amp
```

**Step 6: Full run for Stage 2**

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_stage2_discrete_subnet_vs_baselines \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata subsection_gru_paper_no_accel_v2_phase_actuator_airdata subnet_discrete_paper_no_accel_v2_phase_actuator_airdata \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --hidden-sizes 128,128 \
  --sequence-history-size 64 \
  --rollout-size 32 \
  --rollout-stride 32 \
  --latent-size 16 \
  --device cuda:0
```

**Step 7: Debug if worse or unstable**

If `subnet_discrete` underperforms causal GRU:

```text
Check whether latent update magnitude explodes.
Try latent_size: 8, 16, 32.
Try rollout_size: 16 vs 32.
Try gradient clipping: 1.0.
Try transition residual scale: z_next = z + 0.1 * transition_net(...).
Try lower LR: 3e-4.
```

Do not run all ablations immediately. Run the smallest ablation that explains the failure.

**Step 8: Commit**

```bash
git add src/system_identification/training.py scripts/train_baseline_torch.py scripts/run_baseline_comparison.py tests/test_training.py
git commit -m "feat: add discrete SUBNET wrench model"
```

---

### Task 5: Stage 3 Model - Continuous-Time SUBNET Euler

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/train_baseline_torch.py`
- Modify: `scripts/run_baseline_comparison.py`
- Test: `tests/test_training.py`

**Step 1: Write failing model tests**

```python
def test_ct_subnet_euler_wrench_regressor_forward_shape_and_tau_config():
    model = training_module.ContinuousTimeSUBNETWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        latent_size=10,
        hidden_sizes=(16,),
        dropout=0.0,
        dt_over_tau=0.03,
        integrator="euler",
    )

    out = model(
        torch.randn(5, 8, 4),
        torch.randn(5, 6, 4),
        torch.randn(5, 6, 3),
    )

    assert out.shape == (5, 6, 6)
    assert model.dt_over_tau == pytest.approx(0.03)
```

```python
def test_ct_subnet_euler_rejects_bad_dt_over_tau():
    with pytest.raises(ValueError, match="dt_over_tau"):
        training_module.ContinuousTimeSUBNETWrenchRegressor(
            context_input_dim=4,
            rollout_input_dim=4,
            current_input_dim=0,
            output_dim=6,
            latent_size=10,
            hidden_sizes=(16,),
            dt_over_tau=0.0,
        )
```

**Step 2: Run tests to verify RED**

```bash
pytest tests/test_training.py::test_ct_subnet_euler_wrench_regressor_forward_shape_and_tau_config \
       tests/test_training.py::test_ct_subnet_euler_rejects_bad_dt_over_tau -q
```

Expected: fail because class is undefined.

**Step 3: Implement model**

Add:

```python
class ContinuousTimeSUBNETWrenchRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        latent_size: int = 16,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
        dt_over_tau: float = 0.03,
        integrator: str = "euler",
    ):
        ...
```

Euler rollout:

```text
z0 = encoder_gru(context_inputs)
for k in 0..T-1:
    y_hat_k = output_net([z_k, rollout_input_k, current_k])
    f_k = derivative_net([z_k, rollout_input_k])
    z_{k+1} = z_k + dt_over_tau * f_k
```

Save/record:

```text
dt_over_tau
integrator
latent_size
latent_rms
latent_derivative_rms
```

The derivative normalization check is important. A CT model with `latent_derivative_rms` orders of magnitude larger than `latent_rms` should be considered numerically suspect even if one aggregate metric improves.

**Step 4: Add config and CLI**

Add CLI arguments:

```text
--dt-over-tau
--ct-integrator
```

For this task, valid `--ct-integrator` can be only:

```text
euler
```

RK4 can be added later only after Euler is stable.

Add recipe:

```python
"ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata": {
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "ct_subnet_euler",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "sequence_history_size": 64,
    "rollout_size": 32,
    "rollout_stride": 32,
    "sequence_feature_mode": "phase_actuator_airdata",
    "current_feature_mode": "remaining_current",
    "latent_size": 16,
    "dt_over_tau": 0.03,
    "ct_integrator": "euler",
}
```

**Step 5: Smoke run**

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_ct_subnet_euler_smoke \
  --recipes ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata \
  --max-train-samples 4096 \
  --max-val-samples 2048 \
  --max-test-samples 2048 \
  --sequence-history-size 16 \
  --rollout-size 8 \
  --rollout-stride 8 \
  --latent-size 8 \
  --dt-over-tau 0.03 \
  --hidden-sizes 32 \
  --batch-size 64 \
  --max-epochs 1 \
  --device cpu \
  --disable-amp
```

**Step 6: Tau sweep**

Run a short controlled sweep before the full CT comparison:

```bash
for r in 0.01 0.03 0.1 0.3; do
  python scripts/run_baseline_comparison.py \
    --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
    --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_ct_subnet_tau_${r} \
    --recipes ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata \
    --max-train-samples 65536 \
    --max-val-samples 32768 \
    --max-test-samples 32768 \
    --sequence-history-size 64 \
    --rollout-size 32 \
    --rollout-stride 32 \
    --latent-size 16 \
    --dt-over-tau "$r" \
    --hidden-sizes 128,128 \
    --batch-size 512 \
    --max-epochs 12 \
    --early-stopping-patience 4 \
    --learning-rate 0.001 \
    --weight-decay 0.00001 \
    --device cuda:0
done
```

Select the `dt_over_tau` with the best validation RMSE, while checking latent derivative RMS is not pathological.

**Step 7: Full run for Stage 3**

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_stage3_ct_subnet_vs_baselines \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata subsection_gru_paper_no_accel_v2_phase_actuator_airdata subnet_discrete_paper_no_accel_v2_phase_actuator_airdata ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --hidden-sizes 128,128 \
  --sequence-history-size 64 \
  --rollout-size 32 \
  --rollout-stride 32 \
  --latent-size 16 \
  --dt-over-tau <best_tau_from_sweep> \
  --device cuda:0
```

**Step 8: Debug if unstable**

If loss becomes NaN:

```text
Disable AMP.
Set gradient clipping to 1.0.
Lower learning rate to 3e-4.
Try dt_over_tau = 0.01.
Add tanh output scaling to derivative_net.
Check input normalization stats for rollout features.
```

If CT model is stable but worse than discrete SUBNET:

```text
Try dt_over_tau sweep first.
Try rollout_size 16.
Try latent_size 32.
Only then consider RK4.
```

**Step 9: Commit**

```bash
git add src/system_identification/training.py scripts/train_baseline_torch.py scripts/run_baseline_comparison.py tests/test_training.py
git commit -m "feat: add continuous-time SUBNET wrench model"
```

---

### Task 6: Final Comparison and Diagnostics

**Files:**
- Modify: `docs/plans/2026-05-06-leakage-resistant-baseline-protocol.md`
- Create: `docs/results/2026-05-06-subnet-three-stage-comparison.md`

**Step 1: Run final comparison if not already done**

Use the final selected CT `dt_over_tau`:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/$(date +%Y%m%d_%H%M%S)_subnet_three_stage_final \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata subsection_gru_paper_no_accel_v2_phase_actuator_airdata subnet_discrete_paper_no_accel_v2_phase_actuator_airdata ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --hidden-sizes 128,128 \
  --sequence-history-size 64 \
  --rollout-size 32 \
  --rollout-stride 32 \
  --latent-size 16 \
  --dt-over-tau <best_tau_from_sweep> \
  --device cuda:0
```

**Step 2: Run diagnostics for all candidates**

```bash
for recipe in \
  mlp_paper_no_accel_v2 \
  causal_gru_paper_no_accel_v2_phase_actuator_airdata \
  subsection_gru_paper_no_accel_v2_phase_actuator_airdata \
  subnet_discrete_paper_no_accel_v2_phase_actuator_airdata \
  ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata; do
  python scripts/run_training_diagnostics.py \
    --model-bundle artifacts/<final_run>/${recipe}/model_bundle.pt \
    --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
    --output-dir artifacts/<final_run>/${recipe}/diagnostics \
    --splits test \
    --device cuda:0
done
```

Diagnostics must include:

```text
per_log_metrics.csv
per_regime_metrics.csv
```

For rollout models, regime diagnostics must be based on full-split predictions and flattened metadata, not by pre-filtering bins before constructing rollout windows.

**Step 3: Create final result note**

Create:

```text
docs/results/2026-05-06-subnet-three-stage-comparison.md
```

Required sections:

```text
Dataset and split
Compared recipes
Commands
Overall metrics table
Per-target R2 and RMSE table
Per-log worst cases
Worst regime bins
Leakage audit flags
Stage-by-stage decision:
  Stage 1 subsection_gru: better or not?
  Stage 2 subnet_discrete: better or not?
  Stage 3 ct_subnet_euler: better or not?
Recommended next default model
Known caveats
```

**Step 4: Update protocol doc**

Add a section to:

```text
docs/plans/2026-05-06-leakage-resistant-baseline-protocol.md
```

Title:

```text
SUBNET-inspired forward rollout candidates
```

Include:

```text
Why short subsections are used.
Why encoder initial latent state is used.
Why target history is excluded.
Why CT derivative normalization is swept.
How final comparison is judged.
```

**Step 5: Commit**

```bash
git add docs/plans/2026-05-06-leakage-resistant-baseline-protocol.md docs/results/2026-05-06-subnet-three-stage-comparison.md
git commit -m "docs: summarize SUBNET rollout comparison"
```

---

## Debugging Protocol

Use this order when something fails.

### Shape Mismatch

Check:

```text
context: [B, H, C]
rollout: [B, T, U]
current: [B, T, P]
targets: [B, T, Y]
predictions: [B, T, Y]
flattened targets/predictions: [B*T, Y]
metadata rows: B*T
```

Add a focused test before patching.

### Suspiciously High Metrics

Immediately inspect `training_config.json`:

```text
has_acceleration_inputs must be false
has_velocity_history must be false for sequence/context branch
has_angular_velocity_history must be false for sequence/context branch
has_alpha_beta_history must be false for sequence/context branch
has_centered_window must be false
uses_whole_log_split must be true
```

Also verify:

```text
target columns are never in context/rollout/current feature columns
rollout targets are not shifted into inputs
train/val/test logs are disjoint
```

### Slow Training

Check:

```bash
nvidia-smi
ps -p <pid> -o pid,etime,pcpu,pmem,rss,stat,cmd
```

If GPU idle and CPU high before first epoch:

```text
profile rollout builder
ensure vectorized windows are used
increase rollout_stride to rollout_size
avoid per-row DataFrame appends
```

If GPU active but slow:

```text
try batch_size 1024
enable AMP
reduce rollout_size from 32 to 16 for ablation
```

### NaN or Exploding CT State

Try in order:

```text
disable AMP
lower LR to 3e-4
gradient clipping = 1.0
dt_over_tau = 0.01
latent_size = 8
derivative output tanh scaling
rollout_size = 16
```

### Model Worse Than Causal GRU

Do not immediately discard it. First answer:

```text
Is it worse on all targets or only moments?
Is it worse on all logs or only one log?
Does it improve low-frequency or high-frequency bins?
Is the rollout horizon too long?
Is the latent dimension too small?
```

Then run one ablation at a time.

---

## Final Verification Checklist

Run before declaring the three-stage work complete:

```bash
pytest tests/test_training.py -q
pytest -q
python -m py_compile scripts/train_baseline_torch.py scripts/run_baseline_comparison.py scripts/run_training_diagnostics.py
git diff --check
```

Confirm final artifacts exist:

```text
artifacts/<final_run>/baseline_comparison_summary.csv
artifacts/<final_run>/subsection_gru_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
artifacts/<final_run>/subnet_discrete_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
artifacts/<final_run>/ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
```

Confirm final report exists:

```text
docs/results/2026-05-06-subnet-three-stage-comparison.md
```

Final answer must include:

```text
commit hashes
artifact paths
overall metric table
per-target table
stage-by-stage better/not-better verdict
known caveats
```

