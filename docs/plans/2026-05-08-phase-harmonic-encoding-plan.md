# Phase Harmonic Encoding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add wingbeat phase-aware multi-harmonic encoding and test whether it improves the current best causal Transformer over ordinary first-harmonic phase input.

**Architecture:** Extend the existing derived feature pipeline to produce harmonic phase columns for orders 2 and 3. Register new feature and sequence modes that include these columns, then run a validation-ranked ablation comparing no phase, raw phase, first-harmonic sin/cos, and multi-harmonic phase encoding. Use test metrics only after the best validation candidate is locked.

**Tech Stack:** Python, NumPy, pandas, PyTorch, pytest, existing `src/system_identification/training.py`, existing `scripts/run_temporal_backbone_screen.py`, existing whole-log split dataset.

---

## Context

Current accuracy-leading model:

```text
recipe: causal_transformer_paper_no_accel_v2_phase_actuator_airdata
history: 128
d_model: 64
layers: 2
heads: 4
dropout: 0.05
test RMSE: 0.892095
test R2: 0.736392
```

Current phase representation:

```text
phase_corrected_sin = sin(phi)
phase_corrected_cos = cos(phi)
```

Proposed representation:

```text
sin(phi),  cos(phi)
sin(2phi), cos(2phi)
sin(3phi), cos(3phi)
```

Paper framing:

```text
A wingbeat phase-aware harmonic encoding is introduced to provide the temporal backbone with explicit periodic information of the flapping cycle.
```

Fixed protocol:

```text
split_root: dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
split_policy: whole_log
no acceleration inputs
no centered windows
no past wrench / no past target output
rank sweep configs by validation metrics only
evaluate test only for locked final models
```

Primary metrics:

```text
val_overall_rmse
val_overall_r2
val_fx_b/fz_b/my_b R2 for control-dominant channels
val_mx_b/mz_b R2 for roll/yaw
val_fy_b R2 as low-priority but diagnostic channel
```

Final report must include:

```text
test_all
test_without_suspect
suspect_only
per-target R2
control-relevant interpretation
```

---

## Ablation Design

Run four phase representations with the same Transformer hyperparameters:

```text
1. no_phase
   Remove phase_corrected_rad, phase_corrected_sin, phase_corrected_cos,
   phase harmonic columns, wing_stroke_angle_rad from sequence history.

2. raw_phase_only
   Use phase_corrected_rad as the periodic phase feature.

3. sin_cos
   Current baseline: phase_corrected_sin, phase_corrected_cos.

4. harmonic_3
   phase_corrected_sin/cos plus phase_h2_sin/cos and phase_h3_sin/cos.
```

Use current best Transformer hyperparameters:

```text
history: 128
d_model: 64
layers: 2
heads: 4
dropout: 0.05
dim_feedforward: 128
```

Recommended stages:

```text
phase_harmonic_quick:
  max_train_samples=131072
  max_val_samples=65536
  skip_test_eval=true
  max_epochs=20
  patience=5

phase_harmonic_final:
  full train/val/test
  only locked best validation candidate plus baseline sin_cos repeat
  max_epochs=50
  patience=8
```

---

## Task 1: Add Multi-Harmonic Derived Columns

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing test**

Add near `test_prepare_feature_target_frames_adds_phase_encoding`:

```python
def test_prepare_feature_target_frames_adds_phase_harmonics():
    frame = _synthetic_frame(n_rows=8, seed=123)

    features, _ = prepare_feature_target_frames(
        frame,
        feature_columns=[
            "phase_corrected_h2_sin",
            "phase_corrected_h2_cos",
            "phase_corrected_h3_sin",
            "phase_corrected_h3_cos",
        ],
    )

    phase = frame["phase_corrected_rad"].to_numpy()
    np.testing.assert_allclose(features["phase_corrected_h2_sin"].to_numpy(), np.sin(2.0 * phase))
    np.testing.assert_allclose(features["phase_corrected_h2_cos"].to_numpy(), np.cos(2.0 * phase))
    np.testing.assert_allclose(features["phase_corrected_h3_sin"].to_numpy(), np.sin(3.0 * phase))
    np.testing.assert_allclose(features["phase_corrected_h3_cos"].to_numpy(), np.cos(3.0 * phase))
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_prepare_feature_target_frames_adds_phase_harmonics -q
```

Expected: fail because harmonic columns do not exist.

**Step 3: Implement derived columns**

In `_with_derived_columns`, after first-harmonic phase columns, add:

```python
for harmonic in (2, 3):
    derived[f"phase_corrected_h{harmonic}_sin"] = np.sin(float(harmonic) * phase)
    derived[f"phase_corrected_h{harmonic}_cos"] = np.cos(float(harmonic) * phase)
```

**Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_training.py::test_prepare_feature_target_frames_adds_phase_harmonics -q
```

Expected: pass.

**Step 5: Commit**

Run:

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add phase harmonic derived features"
```

---

## Task 2: Add Harmonic Feature Set And Sequence Modes

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing feature-set test**

Add near `test_paper_no_accel_v2_feature_set_excludes_label_derivative_inputs`:

```python
def test_paper_no_accel_v2_phase_harmonic_feature_set_adds_harmonics():
    feature_columns = resolve_feature_set_columns("paper_no_accel_v2_phase_harmonic")

    assert set(PAPER_NO_ACCEL_V2_FEATURE_COLUMNS).issubset(feature_columns)
    assert {
        "phase_corrected_h2_sin",
        "phase_corrected_h2_cos",
        "phase_corrected_h3_sin",
        "phase_corrected_h3_cos",
    }.issubset(feature_columns)
```

**Step 2: Write failing sequence-mode test**

Add near `test_resolve_sequence_feature_columns_defaults_to_leakage_resistant_history`:

```python
def test_resolve_sequence_feature_columns_supports_phase_harmonics():
    columns = resolve_feature_set_columns("paper_no_accel_v2_phase_harmonic")

    sequence_columns = training_module.resolve_sequence_feature_columns(columns, "phase_harmonic_actuator_airdata")

    assert "phase_corrected_sin" in sequence_columns
    assert "phase_corrected_h2_sin" in sequence_columns
    assert "phase_corrected_h3_cos" in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns
    assert "velocity_b.x" not in sequence_columns
```

**Step 3: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_paper_no_accel_v2_phase_harmonic_feature_set_adds_harmonics \
       tests/test_training.py::test_resolve_sequence_feature_columns_supports_phase_harmonics -q
```

Expected: fail because feature set and sequence mode do not exist.

**Step 4: Implement constants**

In `src/system_identification/training.py`, add:

```python
PHASE_HARMONIC_FEATURE_COLUMNS = [
    "phase_corrected_h2_sin",
    "phase_corrected_h2_cos",
    "phase_corrected_h3_sin",
    "phase_corrected_h3_cos",
]

PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS = PAPER_NO_ACCEL_V2_FEATURE_COLUMNS + PHASE_HARMONIC_FEATURE_COLUMNS
```

Register:

```python
DEFAULT_FEATURE_SETS["paper_no_accel_v2_phase_harmonic"] = PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS
```

Extend `SEQUENCE_FEATURE_MODE_COLUMNS`:

```python
"phase_harmonic": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + PHASE_HARMONIC_FEATURE_COLUMNS,
"phase_harmonic_actuator_airdata": (
    WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"]
    + PHASE_HARMONIC_FEATURE_COLUMNS
    + WINDOW_FEATURE_MODE_COLUMNS["airdata"]
),
```

**Step 5: Verify GREEN**

Run:

```bash
pytest tests/test_training.py::test_paper_no_accel_v2_phase_harmonic_feature_set_adds_harmonics \
       tests/test_training.py::test_resolve_sequence_feature_columns_supports_phase_harmonics -q
```

Expected: pass.

**Step 6: Commit**

Run:

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add phase harmonic feature modes"
```

---

## Task 3: Add Phase Harmonic Recipes

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing recipe test**

Add near `test_run_baseline_comparison_supports_temporal_backbone_recipes`:

```python
def test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 430, "train_log"), ("val", 431, "val_log"), ("test", 432, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "runs",
        recipe_names=["causal_transformer_paper_no_accel_v2_phase_harmonic_airdata"],
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
    assert summary.loc[0, "feature_set_name"] == "paper_no_accel_v2_phase_harmonic"
    assert summary.loc[0, "sequence_feature_mode"] == "phase_harmonic_actuator_airdata"
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe -q
```

Expected: fail because recipe does not exist.

**Step 3: Add recipe**

Add to `BASELINE_COMPARISON_RECIPES`:

```python
"causal_transformer_paper_no_accel_v2_phase_harmonic_airdata": {
    "feature_set_name": "paper_no_accel_v2_phase_harmonic",
    "model_type": "causal_transformer",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "window_mode": "single",
    "window_radius": 0,
    "window_feature_mode": "all",
    "sequence_history_size": 64,
    "sequence_feature_mode": "phase_harmonic_actuator_airdata",
    "current_feature_mode": "remaining_current",
    "transformer_d_model": 64,
    "transformer_num_layers": 1,
    "transformer_num_heads": 4,
    "transformer_dim_feedforward": 128,
},
```

**Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_training.py::test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe -q
```

Expected: pass.

**Step 5: Commit**

Run:

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add phase harmonic transformer recipe"
```

---

## Task 4: Add Phase Harmonic Screening Stage

**Files:**
- Modify: `scripts/run_temporal_backbone_screen.py`
- Test: `tests/test_training.py`

**Step 1: Write failing stage test**

Add near existing temporal screen tests:

```python
def test_temporal_screen_phase_harmonic_grid_has_four_ablation_configs():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="phase_harmonic")

    assert len(configs) == 4
    assert {config.stage for config in configs} == {"phase_harmonic"}
    assert {
        "phase_harmonic_no_phase",
        "phase_harmonic_raw_phase",
        "phase_harmonic_sin_cos",
        "phase_harmonic_harmonic3",
    } == {config.config_id for config in configs}
```

**Step 2: Verify RED**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_phase_harmonic_grid_has_four_ablation_configs -q
```

Expected: fail because stage does not exist.

**Step 3: Implement stage**

Add `_phase_harmonic_configs(final: bool = False)` to `scripts/run_temporal_backbone_screen.py`.

Config specs:

```text
phase_harmonic_no_phase:
  recipe_name: causal_transformer_paper_no_accel_v2_no_phase_airdata
  feature_set_name via recipe

phase_harmonic_raw_phase:
  recipe_name: causal_transformer_paper_no_accel_v2_raw_phase_airdata

phase_harmonic_sin_cos:
  recipe_name: causal_transformer_paper_no_accel_v2_phase_actuator_airdata

phase_harmonic_harmonic3:
  recipe_name: causal_transformer_paper_no_accel_v2_phase_harmonic_airdata
```

All configs:

```text
hidden_sizes=(64, 128)
sequence_history_size=128
dropout=0.05
transformer_d_model=64
transformer_num_layers=2
transformer_num_heads=4
transformer_dim_feedforward=128
max_epochs=20 for quick
patience=5 for quick
max_epochs=50 for final
patience=8 for final
```

Extend `build_screen_configs`, CLI choices, and `_stage_sample_defaults`:

```python
if stage in {"sweep", "tcn_gru_focused", "transformer_focused", "phase_harmonic"}:
    return 131072, 65536, 65536
```

Make `skip_test_eval` true for `phase_harmonic` unless `--include-test-eval` is passed:

```python
skip_test_eval = args.stage in {"transformer_focused", "phase_harmonic"} and not args.include_test_eval
```

**Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_phase_harmonic_grid_has_four_ablation_configs -q
```

Expected: pass.

**Step 5: Commit**

Run:

```bash
git add scripts/run_temporal_backbone_screen.py tests/test_training.py
git commit -m "feat: add phase harmonic screen stage"
```

---

## Task 5: Add Raw/No-Phase Transformer Recipes

**Files:**
- Modify: `src/system_identification/training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing feature mode tests**

Add:

```python
def test_resolve_sequence_feature_columns_supports_raw_phase_airdata():
    columns = resolve_feature_set_columns("paper_no_accel_v2")

    sequence_columns = training_module.resolve_sequence_feature_columns(columns, "raw_phase_actuator_airdata")

    assert "phase_corrected_rad" in sequence_columns
    assert "phase_corrected_sin" not in sequence_columns
    assert "phase_corrected_cos" not in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns


def test_resolve_sequence_feature_columns_supports_no_phase_actuator_airdata():
    columns = resolve_feature_set_columns("paper_no_accel_v2")

    sequence_columns = training_module.resolve_sequence_feature_columns(columns, "no_phase_actuator_airdata")

    assert "phase_corrected_rad" not in sequence_columns
    assert "phase_corrected_sin" not in sequence_columns
    assert "phase_corrected_cos" not in sequence_columns
    assert "wing_stroke_angle_rad" not in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns
```

**Step 2: Implement modes and recipes**

Add sequence modes:

```python
"raw_phase_actuator_airdata": [
    "phase_corrected_rad",
    "flap_frequency_hz",
    "cycle_flap_frequency_hz",
    "motor_cmd_0",
    "servo_left_elevon",
    "servo_right_elevon",
    "servo_rudder",
    "elevator_like",
    "aileron_like",
] + WINDOW_FEATURE_MODE_COLUMNS["airdata"],

"no_phase_actuator_airdata": [
    "flap_frequency_hz",
    "cycle_flap_frequency_hz",
    "motor_cmd_0",
    "servo_left_elevon",
    "servo_right_elevon",
    "servo_rudder",
    "elevator_like",
    "aileron_like",
] + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
```

Add recipes:

```python
"causal_transformer_paper_no_accel_v2_raw_phase_airdata": {...}
"causal_transformer_paper_no_accel_v2_no_phase_airdata": {...}
```

Both recipes use:

```text
feature_set_name: paper_no_accel_v2
model_type: causal_transformer
sequence_feature_mode: raw_phase_actuator_airdata or no_phase_actuator_airdata
current_feature_mode: remaining_current
```

**Step 3: Verify GREEN**

Run:

```bash
pytest tests/test_training.py::test_resolve_sequence_feature_columns_supports_raw_phase_airdata \
       tests/test_training.py::test_resolve_sequence_feature_columns_supports_no_phase_actuator_airdata \
       tests/test_training.py::test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe \
       tests/test_training.py::test_temporal_screen_phase_harmonic_grid_has_four_ablation_configs -q
```

Expected: pass.

**Step 4: Commit**

Run:

```bash
git add src/system_identification/training.py tests/test_training.py
git commit -m "feat: add phase ablation transformer recipes"
```

---

## Task 6: Run Validation-Only Phase Harmonic Screen

**Files:**
- Runtime output: `artifacts/20260508_phase_harmonic_screen/`

**Step 1: Dry run**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260508_phase_harmonic_screen \
  --stage phase_harmonic \
  --dry-run
```

Expected: four configs are written to `temporal_backbone_screen_summary.csv`.

**Step 2: Validation-only screen**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260508_phase_harmonic_screen \
  --stage phase_harmonic \
  --batch-size 512 \
  --device cuda:0 \
  --random-seed 42
```

Expected:

```text
skip_test_eval=true
no test_* columns in ranking summary
four validation rows complete
```

**Step 3: Inspect validation ranking**

Run:

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("artifacts/20260508_phase_harmonic_screen/temporal_backbone_screen_summary.csv")
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
print("test columns:", [c for c in summary.columns if c.startswith("test_")])
PY
```

Lock the best candidate by:

```text
primary: lowest val_overall_rmse
tie-breaker: higher mean(val_mx_b_r2, val_mz_b_r2)
must not degrade val_fx_b/fz_b/my_b materially
```

**Step 4: Commit code before final training**

Run:

```bash
pytest tests/test_training.py::test_prepare_feature_target_frames_adds_phase_harmonics \
       tests/test_training.py::test_paper_no_accel_v2_phase_harmonic_feature_set_adds_harmonics \
       tests/test_training.py::test_resolve_sequence_feature_columns_supports_phase_harmonics \
       tests/test_training.py::test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe \
       tests/test_training.py::test_temporal_screen_phase_harmonic_grid_has_four_ablation_configs -q
python -m py_compile src/system_identification/training.py scripts/run_temporal_backbone_screen.py scripts/run_baseline_comparison.py scripts/train_baseline_torch.py
git diff --check
git status --short --branch
```

Expected: tests and compile pass.

---

## Task 7: Run Locked Full Final

**Files:**
- Runtime output: `artifacts/20260508_phase_harmonic_final/`

**Step 1: Select locked configs**

Always include current baseline repeat:

```text
phase_harmonic_final_sin_cos
```

If `phase_harmonic_harmonic3` wins validation, run:

```text
phase_harmonic_final_harmonic3
```

If another ablation wins, run that winner plus `harmonic3` as a method candidate, but clearly report that harmonic3 did not win validation.

**Step 2: Full-data final command**

Run selected configs with test evaluation:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260508_phase_harmonic_final \
  --stage phase_harmonic_final \
  --config-ids phase_harmonic_final_sin_cos phase_harmonic_final_harmonic3 \
  --batch-size 512 \
  --device cuda:0 \
  --random-seed 42 \
  --include-test-eval
```

Adjust `--config-ids` based on locked validation outcome.

**Step 3: Inspect final results**

Run:

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("artifacts/20260508_phase_harmonic_final/temporal_backbone_screen_summary.csv")
cols = [
    "config_id",
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
print(summary[cols].sort_values("val_overall_rmse").to_string(index=False))
PY
```

**Step 4: Run suspect-log diagnostics on final model(s)**

For each final model bundle, run:

```bash
python scripts/run_lateral_diagnostics.py \
  --model-bundle artifacts/20260508_phase_harmonic_final/runs/<config_id>/<recipe_name>/model_bundle.pt \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --split test \
  --output-dir artifacts/20260508_phase_harmonic_final/diagnostics/<config_id> \
  --batch first \
  --batch-size 8192 \
  --device cuda:0
```

Then run:

```bash
python scripts/run_lateral_diagnostics.py \
  --model-bundle artifacts/20260508_phase_harmonic_final/runs/<config_id>/<recipe_name>/model_bundle.pt \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --split test \
  --output-dir artifacts/20260508_phase_harmonic_final/diagnostics/<config_id> \
  --batch second \
  --batch-size 8192 \
  --device cuda:0
```

Use `with_without_suspect_log_metrics.csv` to compare `test_all`, `test_without_suspect`, and `suspect_only`.

---

## Task 8: Write Result Report And Update Insights

**Files:**
- Create: `docs/results/2026-05-08-phase-harmonic-encoding.md`
- Modify: `docs/insights/research-narrative-notes.md`

**Report must include:**

```text
1. Method description: wingbeat phase-aware harmonic encoding
2. Validation-only screen ranking
3. Locked final selection rule
4. Final test results
5. With/without suspect log comparison
6. Per-target and control-relevant interpretation
7. Decision: promote / keep as ablation / reject
```

Suggested decision rules:

```text
promote:
  test overall RMSE improves by >= 1%
  OR roll/yaw mean R2 improves by >= 0.03 without degrading fx/fz/my by > 0.01

keep as ablation:
  validation improves but test gain is small
  OR improves fy_b/mx_b/mz_b but not overall

reject:
  validation and test both worse than sin_cos baseline
```

**Update insights:**

Add a short section under `docs/insights/research-narrative-notes.md`:

```text
Phase-aware harmonic encoding was tested as a low-risk periodic inductive bias.
Outcome: <promote/ablation/reject>.
Key result: <one sentence>.
```

**Final verification**

Run:

```bash
pytest tests/test_training.py::test_prepare_feature_target_frames_adds_phase_harmonics \
       tests/test_training.py::test_paper_no_accel_v2_phase_harmonic_feature_set_adds_harmonics \
       tests/test_training.py::test_resolve_sequence_feature_columns_supports_phase_harmonics \
       tests/test_training.py::test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe \
       tests/test_training.py::test_temporal_screen_phase_harmonic_grid_has_four_ablation_configs -q
python -m py_compile src/system_identification/training.py scripts/run_temporal_backbone_screen.py scripts/run_baseline_comparison.py scripts/train_baseline_torch.py scripts/run_lateral_diagnostics.py
git diff --check
git status --short --branch
```

**Commit**

Run:

```bash
git add src/system_identification/training.py scripts/run_temporal_backbone_screen.py tests/test_training.py docs/results/2026-05-08-phase-harmonic-encoding.md docs/insights/research-narrative-notes.md
git commit -m "feat: evaluate phase harmonic encoding"
```

---

## Expected Outcomes

Most likely:

```text
small improvement or neutral result
```

Why:

```text
The current model already has sin/cos phase and strong temporal history.
Harmonic features mainly help represent higher-order periodic structure.
The biggest possible gains should appear in phase-sensitive hard targets: mx_b, mz_b, fy_b.
```

Interpretation if it improves:

```text
The model benefits from explicit higher-order wingbeat periodic structure.
This supports using wingbeat phase-aware harmonic encoding as a method contribution.
```

Interpretation if neutral:

```text
The Transformer can already infer higher-order periodic components from sin/cos plus temporal history.
Keep harmonic encoding as an ablation, not the main method.
```

Interpretation if worse:

```text
Extra harmonic inputs may overfit or add redundant correlated features.
Reject for default model, but report as a tested periodic inductive bias.
```
