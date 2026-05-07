# Transformer Focused Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a validation-ranked focused sweep for the causal Transformer backbone and determine whether a tuned Transformer improves over `final_transformer_d64_l2_h4_hist128`.

**Architecture:** Extend the existing temporal screening runner with Transformer-only stages and a `skip_test_eval` path for sweep runs. The sampled sweep ranks configs only by validation metrics and does not evaluate the test split; test metrics are produced only for locked final models. Reuse the existing whole-log split, causal no-acceleration sequence protocol, model bundle format, diagnostics runner, and result-report style.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `scripts/run_temporal_backbone_screen.py`, existing `scripts/run_training_diagnostics.py`, existing `src/system_identification/training.py`.

---

## Context

Current accuracy-leading model:

```text
config_id: final_transformer_d64_l2_h4_hist128
recipe: causal_transformer_paper_no_accel_v2_phase_actuator_airdata
history: 128
d_model: 64
num_layers: 2
num_heads: 4
dim_feedforward: 128
dropout: 0.0
test_rmse: 0.904964
test_r2: 0.718291
```

Fixed leakage-resistant protocol:

```text
split_root: dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
split_policy: whole_log
feature_set_name: paper_no_accel_v2
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
no acceleration inputs
no centered windows
no past wrench / no past target output
checkpoint_selection_metric: scaled validation Huber loss
```

Selection protocol:

```text
Do not use test metrics for sweep ranking.
Do not use test diagnostics for sweep ranking.
Use validation metrics and validation diagnostics to choose final configs.
Use test metrics only after final configs are locked.
```

Primary validation ranking:

```text
1. val_overall_rmse
2. best_val_loss
3. val_overall_r2
```

Secondary validation ranking:

```text
hard_target_r2 = mean(val_fy_b_r2, val_mz_b_r2)
low_frequency_val_rmse in cycle_flap_frequency_hz (-0.001, 3.0]
batch=1 pure-forward latency for deployment tradeoff
```

---

## Sweep Design

Use a 12-config targeted sweep around the current best Transformer:

```text
1.  transformer_focused_hist96_d64_l2_h4_do0
2.  transformer_focused_hist128_d64_l2_h4_do0
3.  transformer_focused_hist160_d64_l2_h4_do0
4.  transformer_focused_hist192_d64_l2_h4_do0

5.  transformer_focused_hist128_d64_l1_h4_do0
6.  transformer_focused_hist128_d64_l3_h4_do0

7.  transformer_focused_hist128_d96_l2_h4_do0
8.  transformer_focused_hist160_d96_l2_h4_do0

9.  transformer_focused_hist128_d128_l2_h4_do0
10. transformer_focused_hist128_d64_l2_h2_do0
11. transformer_focused_hist128_d64_l2_h8_do0

12. transformer_focused_hist128_d64_l2_h4_do005
```

What the groups test:

```text
1-4: causal history length
5-6: encoder depth
7-9: model width/capacity
10-11: attention head count
12: light regularization
```

Full-data final should run at most 4 locked configs:

```text
1. best validation overall RMSE
2. best validation hard_target_r2
3. best validation low-frequency diagnostic
4. compact/fast config if within 2% validation RMSE of the best config
```

Deduplicate configs. If fewer than 3 unique configs remain, fill by next-best validation RMSE until there are 3.

---

### Task 1: Add Transformer Focused Stage Tests

**Files:**
- Modify: `tests/test_training.py`

**Step 1: Add failing grid test**

Add near the existing temporal screening runner tests:

```python
def test_temporal_screen_transformer_focused_grid_has_12_configs():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="transformer_focused")

    assert len(configs) == 12
    assert {config.recipe_name for config in configs} == {
        "causal_transformer_paper_no_accel_v2_phase_actuator_airdata"
    }
    assert len({config.config_id for config in configs}) == 12
    assert all(config.stage == "transformer_focused" for config in configs)
    assert {config.sequence_history_size for config in configs} == {96, 128, 160, 192}
    assert "transformer_focused_hist128_d64_l2_h4_do0" in {config.config_id for config in configs}
```

**Step 2: Add final-stage budget test**

```python
def test_temporal_screen_transformer_focused_final_grid_uses_full_budget():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="transformer_focused_final")

    assert len(configs) == 12
    assert all(config.max_epochs == 50 for config in configs)
    assert all(config.early_stopping_patience == 8 for config in configs)
    assert "transformer_focused_final_hist128_d64_l2_h4_do0" in {config.config_id for config in configs}
```

**Step 3: Run tests to verify RED**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_transformer_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_transformer_focused_final_grid_uses_full_budget -q
```

Expected: fail because `transformer_focused` is not a supported stage.

---

### Task 2: Implement Transformer Focused Stages

**Files:**
- Modify: `scripts/run_temporal_backbone_screen.py`
- Test: `tests/test_training.py`

**Step 1: Add config builder**

Add this helper near `_tcn_gru_focused_configs`:

```python
def _transformer_focused_configs(*, final: bool = False) -> list[ScreenConfig]:
    stage = "transformer_focused_final" if final else "transformer_focused"
    max_epochs = 50 if final else 20
    patience = 8 if final else 5
    specs = [
        (96, 64, 2, 4, 0.0),
        (128, 64, 2, 4, 0.0),
        (160, 64, 2, 4, 0.0),
        (192, 64, 2, 4, 0.0),
        (128, 64, 1, 4, 0.0),
        (128, 64, 3, 4, 0.0),
        (128, 96, 2, 4, 0.0),
        (160, 96, 2, 4, 0.0),
        (128, 128, 2, 4, 0.0),
        (128, 64, 2, 2, 0.0),
        (128, 64, 2, 8, 0.0),
        (128, 64, 2, 4, 0.05),
    ]
    configs: list[ScreenConfig] = []
    for history, d_model, layers, heads, dropout in specs:
        dropout_tag = "do0" if dropout == 0.0 else f"do{int(dropout * 1000):03d}"
        configs.append(
            _config(
                config_id=f"{stage}_hist{history}_d{d_model}_l{layers}_h{heads}_{dropout_tag}",
                stage=stage,
                recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
                hidden_sizes=(d_model, 128),
                sequence_history_size=history,
                max_epochs=max_epochs,
                early_stopping_patience=patience,
                dropout=dropout,
                extra_args={
                    "transformer_d_model": d_model,
                    "transformer_num_layers": layers,
                    "transformer_num_heads": heads,
                    "transformer_dim_feedforward": 2 * d_model,
                },
            )
        )
    return configs
```

**Step 2: Wire stage into `build_screen_configs`**

Add:

```python
if resolved_stage == "transformer_focused":
    return _transformer_focused_configs(final=False)
if resolved_stage == "transformer_focused_final":
    return _transformer_focused_configs(final=True)
```

Also include both stages in the `"all"` return list.

**Step 3: Add CLI choices**

Extend `--stage` choices:

```python
choices=[
    "quick",
    "sweep",
    "final",
    "tcn_gru_focused",
    "tcn_gru_focused_final",
    "transformer_focused",
    "transformer_focused_final",
    "all",
]
```

**Step 4: Add sampled-data default**

Modify `_stage_sample_defaults`:

```python
if stage in {"sweep", "tcn_gru_focused", "transformer_focused"}:
    return 131072, 65536, None
```

Do not use `max_test_samples=0` as a protocol mechanism. Test evaluation should be skipped explicitly by Task 3.

**Step 5: Run focused tests**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_transformer_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_transformer_focused_final_grid_uses_full_budget -q
```

Expected: pass.

**Step 6: Run existing temporal stage tests**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_tcn_gru_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_tcn_gru_focused_final_grid_uses_full_budget \
       tests/test_training.py::test_run_baseline_comparison_supports_temporal_backbone_recipes -q
```

Expected: pass.

**Step 7: Commit**

Run:

```bash
git add scripts/run_temporal_backbone_screen.py tests/test_training.py
git commit -m "feat: add Transformer focused sweep configs"
```

---

### Task 3: Add Explicit Test-Evaluation Skipping

**Files:**
- Modify: `src/system_identification/training.py`
- Modify: `scripts/run_baseline_comparison.py`
- Modify: `scripts/run_temporal_backbone_screen.py`
- Modify: `tests/test_training.py`

**Step 1: Add failing smoke test**

Add a small test near existing `run_baseline_comparison` tests:

```python
def test_run_baseline_comparison_can_skip_test_eval(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 210, "train_log"), ("val", 211, "val_log"), ("test", 212, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=["causal_transformer_paper_no_accel_v2_phase_actuator_airdata"],
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        device="cpu",
        random_seed=210,
        num_workers=0,
        use_amp=False,
        skip_test_eval=True,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert "val_overall_rmse" in summary.columns
    assert "test_overall_rmse" not in summary.columns
```

**Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_training.py::test_run_baseline_comparison_can_skip_test_eval -q
```

Expected: fail because `skip_test_eval` is not supported yet.

**Step 3: Add `skip_test_eval` to `run_training_job`**

In `src/system_identification/training.py`, add `skip_test_eval: bool = False` to `run_training_job`.

Change test split loading/evaluation from unconditional to conditional:

```python
test_frame = None if skip_test_eval else _load_split_frame(split_root, "test", max_test_samples, random_seed + 2)
```

Build metrics as:

```python
metrics = {
    "train": evaluate_model_bundle(bundle, train_frame, split_name="train", batch_size=batch_size, device=device),
    "val": evaluate_model_bundle(bundle, val_frame, split_name="val", batch_size=batch_size, device=device),
}
if test_frame is not None:
    metrics["test"] = evaluate_model_bundle(bundle, test_frame, split_name="test", batch_size=batch_size, device=device)
```

Only write test plots when `test_frame is not None`:

```python
if test_frame is not None:
    _save_pred_vs_true_plot(bundle, test_frame, pred_vs_true_test_path, batch_size=batch_size, device=device)
    _save_residual_hist_plot(bundle, test_frame, residual_hist_test_path, batch_size=batch_size, device=device)
```

Record the flag in `training_config.json`:

```python
"skip_test_eval": bool(skip_test_eval),
```

**Step 4: Thread `skip_test_eval` through comparison runners**

Add `skip_test_eval: bool = False` to:

```text
run_baseline_comparison(...)
_run_single_baseline_comparison(...)
scripts/run_baseline_comparison.py CLI
scripts/run_temporal_backbone_screen.py CLI
```

In `scripts/run_temporal_backbone_screen.py`, default `skip_test_eval=True` for `stage == "transformer_focused"` and `False` otherwise. Allow `--include-test-eval` to override for ad hoc debugging, but do not use it in this plan.

**Step 5: Run skip-test test**

Run:

```bash
pytest tests/test_training.py::test_run_baseline_comparison_can_skip_test_eval -q
```

Expected: pass.

**Step 6: Run focused stage tests**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_transformer_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_transformer_focused_final_grid_uses_full_budget \
       tests/test_training.py::test_run_baseline_comparison_can_skip_test_eval -q
```

Expected: pass.

**Step 7: Commit**

Run:

```bash
git add src/system_identification/training.py scripts/run_baseline_comparison.py scripts/run_temporal_backbone_screen.py tests/test_training.py
git commit -m "feat: support validation-only temporal sweeps"
```

---

### Task 4: Dry Run and Smoke Test

**Files:**
- Create: `artifacts/20260507_transformer_focused_dryrun`
- Create: `artifacts/20260507_transformer_focused_smoke`

**Step 1: Dry run**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_transformer_focused_dryrun \
  --stage transformer_focused \
  --dry-run
```

Expected:

```text
12 configs
only causal_transformer_paper_no_accel_v2_phase_actuator_airdata
includes transformer_focused_hist128_d64_l2_h4_do0
```

**Step 2: Smoke one config**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_transformer_focused_smoke \
  --stage transformer_focused \
  --config-ids transformer_focused_hist128_d64_l2_h4_do0 \
  --batch-size 512 \
  --max-train-samples 32768 \
  --max-val-samples 16384 \
  --device cuda:0
```

Expected: training completes, the summary has validation columns, and no `test_*` metric columns are present.

**Step 3: Inspect validation metrics only**

Run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/20260507_transformer_focused_smoke/temporal_backbone_screen_summary.csv")
print(df[["config_id", "val_overall_rmse", "val_overall_r2", "best_val_loss"]].to_string(index=False))
PY
```

---

### Task 5: Run 12-Config Transformer Focused Sweep

**Files:**
- Create: `artifacts/20260507_transformer_focused_sweep`

**Step 1: Launch sweep**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_transformer_focused_sweep \
  --stage transformer_focused \
  --batch-size 512 \
  --max-train-samples 131072 \
  --max-val-samples 65536 \
  --device cuda:0
```

Expected: no `test_*` metric columns are present in the sweep summary.

**Step 2: Rank by validation metrics**

Run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/20260507_transformer_focused_sweep/temporal_backbone_screen_summary.csv")
df["val_hard_target_r2"] = df[["val_fy_b_r2", "val_mz_b_r2"]].mean(axis=1)
cols = [
    "config_id",
    "sequence_history_size",
    "hidden_sizes",
    "dropout",
    "best_val_loss",
    "val_overall_rmse",
    "val_overall_r2",
    "val_fy_b_r2",
    "val_mz_b_r2",
    "val_hard_target_r2",
    "best_epoch",
]
print("Best validation RMSE:")
print(df[cols].sort_values("val_overall_rmse").head(8).to_string(index=False))
print("\nBest validation hard targets:")
print(df[cols].sort_values("val_hard_target_r2", ascending=False).head(8).to_string(index=False))
PY
```

Do not inspect or rank by `test_*` columns in this task.

---

### Task 6: Generate Validation Diagnostics for Sweep Candidates

**Files:**
- Read: `artifacts/20260507_transformer_focused_sweep/temporal_backbone_screen_summary.csv`
- Create: `artifacts/20260507_transformer_focused_sweep/runs/<config_id>/<recipe>/diagnostics_val`

**Step 1: Run diagnostics on validation split for all 12 configs**

Run:

```bash
python - <<'PY'
import pandas as pd
import subprocess
import sys

summary = pd.read_csv("artifacts/20260507_transformer_focused_sweep/temporal_backbone_screen_summary.csv")
for _, row in summary.iterrows():
    out = row["output_dir"]
    cmd = [
        sys.executable,
        "scripts/run_training_diagnostics.py",
        "--model-bundle",
        f"{out}/model_bundle.pt",
        "--split-root",
        "dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1",
        "--output-dir",
        f"{out}/diagnostics_val",
        "--splits",
        "val",
        "--batch-size",
        "8192",
        "--device",
        "cuda:0",
    ]
    print("running validation diagnostics", row["config_id"], flush=True)
    subprocess.run(cmd, check=True)
PY
```

**Step 2: Extract validation low-frequency rows**

Run:

```bash
python - <<'PY'
import pandas as pd

summary = pd.read_csv("artifacts/20260507_transformer_focused_sweep/temporal_backbone_screen_summary.csv")
rows = []
for _, row in summary.iterrows():
    reg = pd.read_csv(f"{row['output_dir']}/diagnostics_val/per_regime_metrics.csv")
    low = reg[
        (reg["group_column"] == "cycle_flap_frequency_hz")
        & (reg["group_value"] == "(-0.001, 3.0]")
    ].iloc[0]
    rows.append({
        "config_id": row["config_id"],
        "val_overall_rmse": row["val_overall_rmse"],
        "val_hard_target_r2": (row["val_fy_b_r2"] + row["val_mz_b_r2"]) / 2.0,
        "low_freq_val_r2": low["test_overall_r2"],
        "low_freq_val_rmse": low["test_overall_rmse"],
    })
df = pd.DataFrame(rows)
print(df.sort_values("low_freq_val_rmse").to_string(index=False))
PY
```

`run_training_diagnostics.py` names metric columns `test_*` even when `--splits val` is used. Treat these rows as validation diagnostics because the split argument is `val`.

**Step 3: Select locked final IDs by validation only**

Selection algorithm:

```text
selected = []
add best val_overall_rmse
add best val_hard_target_r2
add best low_freq_val_rmse
if baseline repeat transformer_focused_hist128_d64_l2_h4_do0 is within 2% of best val_overall_rmse, add it
if len(unique selected) < 3, add next-best val_overall_rmse until 3
cap at 4 configs
```

Map selected sweep IDs to final IDs by replacing:

```text
transformer_focused_ -> transformer_focused_final_
```

Write the locked final IDs into notes before running test evaluation.

---

### Task 7: Run Full-Data Final for Locked Transformer Configs

**Files:**
- Create: `artifacts/20260507_transformer_focused_final`

**Step 1: Launch final stage**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_transformer_focused_final \
  --stage transformer_focused_final \
  --config-ids <LOCKED_FINAL_CONFIG_IDS> \
  --batch-size 512 \
  --device cuda:0
```

Replace `<LOCKED_FINAL_CONFIG_IDS>` with the validation-selected final IDs from Task 5.

**Step 2: Summarize final validation and test metrics**

Run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/20260507_transformer_focused_final/temporal_backbone_screen_summary.csv")
df["val_hard_target_r2"] = df[["val_fy_b_r2", "val_mz_b_r2"]].mean(axis=1)
df["test_hard_target_r2"] = df[["test_fy_b_r2", "test_mz_b_r2"]].mean(axis=1)
cols = [
    "config_id",
    "best_val_loss",
    "val_overall_rmse",
    "val_overall_r2",
    "val_hard_target_r2",
    "test_overall_rmse",
    "test_overall_r2",
    "test_fy_b_r2",
    "test_mz_b_r2",
    "test_hard_target_r2",
    "best_epoch",
]
print(df[cols].sort_values("val_overall_rmse").to_string(index=False))
PY
```

At this point test metrics may be reported because final configs are locked.

---

### Task 8: Run Final Diagnostics and Latency Benchmark

**Files:**
- Create: `artifacts/20260507_transformer_focused_final/runs/<config_id>/<recipe>/diagnostics`
- Read: final `model_bundle.pt` files

**Step 1: Run diagnostics on final validation and test splits**

Run:

```bash
python - <<'PY'
import pandas as pd
import subprocess
import sys

summary = pd.read_csv("artifacts/20260507_transformer_focused_final/temporal_backbone_screen_summary.csv")
for _, row in summary.iterrows():
    out = row["output_dir"]
    cmd = [
        sys.executable,
        "scripts/run_training_diagnostics.py",
        "--model-bundle",
        f"{out}/model_bundle.pt",
        "--split-root",
        "dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1",
        "--output-dir",
        f"{out}/diagnostics",
        "--splits",
        "val",
        "test",
        "--batch-size",
        "8192",
        "--device",
        "cuda:0",
    ]
    print("running final diagnostics", row["config_id"], flush=True)
    subprocess.run(cmd, check=True)
PY
```

**Step 2: Benchmark pure-forward latency**

Use the same benchmark style from `docs/results/2026-05-07-tcn-gru-focused-sweep.md`:

```text
batch sizes: 1 and 512
device: cuda:0
inputs: random tensors with bundle sequence/current dimensions
measure: median ms per forward after warmup
```

Include at least:

```text
old final Transformer: artifacts/20260507_temporal_backbone_final/runs/final_transformer_d64_l2_h4_hist128/...
new locked Transformer finals
GRU reference
TCN+GRU compact reference
```

---

### Task 9: Write Result Report

**Files:**
- Create: `docs/results/2026-05-07-transformer-focused-sweep.md`

**Step 1: Write report**

Report structure:

```markdown
# Transformer Focused Sweep

## Protocol
- dataset split
- no acceleration inputs
- causal sequence inputs
- validation-only selection rule
- test-only-after-final rule

## Sweep Grid
- 12 configs and what each group tests

## Validation-Ranked Sweep Results
- rank by val_overall_rmse
- rank by val_hard_target_r2
- validation low-frequency diagnostics
- locked final configs and why

## Full-Data Final Results
- validation metrics
- test metrics after final lock
- compare to previous Transformer, GRU, TCN, TCN+GRU

## Per-Target Results
- test per-target R2/RMSE for locked finals

## Per-Log and Per-Regime Diagnostics
- validation and test diagnostics
- low-frequency bin
- worst log

## Inference Timing
- batch=1 and batch=512 pure-forward latency

## Decision
- promote/reject tuned Transformer
- recommended default
- remaining failure modes
```

**Step 2: Commit report**

Run:

```bash
git add docs/results/2026-05-07-transformer-focused-sweep.md
git commit -m "docs: summarize Transformer focused sweep"
```

---

### Task 10: Verification

**Files:**
- Verify: `scripts/run_temporal_backbone_screen.py`
- Verify: `tests/test_training.py`
- Verify: `docs/results/2026-05-07-transformer-focused-sweep.md`

**Step 1: Run focused tests**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_transformer_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_transformer_focused_final_grid_uses_full_budget \
       tests/test_training.py::test_run_baseline_comparison_can_skip_test_eval \
       tests/test_training.py::test_run_baseline_comparison_supports_temporal_backbone_recipes -q
```

Expected: pass.

**Step 2: Run full tests**

Run:

```bash
pytest -q
```

Expected: all tests pass.

**Step 3: Compile changed Python files**

Run:

```bash
python -m py_compile scripts/run_temporal_backbone_screen.py scripts/run_training_diagnostics.py scripts/run_baseline_comparison.py scripts/train_baseline_torch.py src/system_identification/training.py
```

Expected: exit code 0.

**Step 4: Check whitespace**

Run:

```bash
git diff --check
```

Expected: exit code 0.

**Step 5: Final commit if needed**

Run:

```bash
git status --short
git add scripts/run_temporal_backbone_screen.py tests/test_training.py docs/plans/2026-05-07-transformer-focused-sweep-plan.md docs/results/2026-05-07-transformer-focused-sweep.md docs/results/2026-05-07-tcn-gru-focused-sweep.md
git commit -m "feat: run Transformer focused sweep"
```

Only commit files that changed. Do not add large `artifacts/` unless the repository already tracks the specific file type.
