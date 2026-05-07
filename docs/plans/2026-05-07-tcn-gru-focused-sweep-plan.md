# TCN+GRU Focused Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a focused TCN+GRU hyperparameter sweep to determine whether the hybrid backbone has a stable advantage for hard lateral targets without losing too much overall accuracy.

**Architecture:** Extend the existing temporal screening runner with two TCN+GRU-only stages: a 12-config sampled-data focused sweep and a selected-config full-data final stage. Reuse the existing whole-log split, no-acceleration causal sequence protocol, model bundle format, diagnostics, and result-report style from the temporal backbone screening work.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, existing `scripts/run_temporal_backbone_screen.py`, existing `scripts/run_training_diagnostics.py`, existing `src/system_identification/training.py`.

---

## Scientific Question

The current final comparison found:

```text
Transformer hist128:
  best overall RMSE/R2

TCN+GRU hist128:
  not best overall
  best or near-best on hard lateral targets fy_b and mz_b
  much faster than Transformer on GPU forward
```

This plan tests whether TCN+GRU's hard-target advantage is stable under targeted hyperparameter changes, or whether the previous result was a single lucky configuration.

## Fixed Protocol

Use the existing leakage-resistant protocol:

```text
split_root: dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
feature_set_name: paper_no_accel_v2
model_type: causal_tcn_gru
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
no acceleration inputs
no centered windows
no past wrench / no target history
selection_metric: scaled validation Huber loss
```

Reference results to compare against:

```text
artifacts/20260507_temporal_backbone_final/temporal_backbone_screen_summary.csv
```

Key reference numbers:

```text
Transformer hist128: test_rmse 0.904964, test_r2 0.718291
TCN hist128:         test_rmse 0.934629, test_r2 0.710291
TCN+GRU hist128:     test_rmse 0.946922, test_r2 0.714331
GRU hist64:          test_rmse 1.007040, test_r2 0.709279
```

## Focused 12-Config Grid

Run exactly these 12 TCN+GRU configs in the sampled-data focused stage:

```text
1.  hist128, tcn64,  blocks2, kernel3, gru64
2.  hist128, tcn96,  blocks2, kernel3, gru64
3.  hist128, tcn128, blocks2, kernel3, gru64
4.  hist128, tcn64,  blocks3, kernel3, gru64
5.  hist128, tcn64,  blocks4, kernel3, gru64
6.  hist128, tcn64,  blocks2, kernel5, gru64
7.  hist96,  tcn64,  blocks2, kernel3, gru64
8.  hist160, tcn64,  blocks2, kernel3, gru64
9.  hist128, tcn64,  blocks2, kernel3, gru96
10. hist128, tcn64,  blocks2, kernel3, gru128
11. hist160, tcn96,  blocks3, kernel3, gru96
12. hist160, tcn128, blocks4, kernel3, gru128
```

What each group tests:

```text
1: current compact baseline
2-3: more TCN channel capacity
4-5: deeper/dilated TCN context
6: wider local temporal kernel
7-8: shorter/longer causal history
9-10: larger GRU memory
11-12: larger combined models
```

## Promotion Rules

Promote at most 4 configs to full-data final:

```text
1. best overall RMSE
2. best hard-target score = mean(test_fy_b_r2, test_mz_b_r2)
3. best low-frequency diagnostic in cycle_flap_frequency_hz (-0.001, 3.0]
4. optional compact/fast config if it is within 5% RMSE of best focused config
```

Deduplicate configs. If fewer than 3 unique configs remain, fill by next-best overall RMSE until there are 3.

Strong keep:

```text
full-data RMSE <= 1.03 * Transformer RMSE
and both fy_b and mz_b beat Transformer
```

Keep as hard-target ablation:

```text
full-data RMSE <= 1.05 * Transformer RMSE
and either fy_b or mz_b beats Transformer
```

Reject:

```text
full-data RMSE > 1.08 * Transformer RMSE
or hard-target advantage disappears
```

---

### Task 1: Add Focused TCN+GRU Config Tests

**Files:**
- Modify: `tests/test_training.py`
- Modify later: `scripts/run_temporal_backbone_screen.py`

**Step 1: Write failing tests**

Add tests near the existing temporal screening runner tests:

```python
def test_temporal_screen_tcn_gru_focused_grid_has_12_configs():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="tcn_gru_focused")

    assert len(configs) == 12
    assert {config.recipe_name for config in configs} == {
        "causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata"
    }
    assert len({config.config_id for config in configs}) == 12
    assert all(config.stage == "tcn_gru_focused" for config in configs)
    assert {config.sequence_history_size for config in configs} == {96, 128, 160}
```

Add a full-stage budget test:

```python
def test_temporal_screen_tcn_gru_focused_final_grid_uses_full_budget():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="tcn_gru_focused_final")

    assert len(configs) == 12
    assert all(config.max_epochs == 50 for config in configs)
    assert all(config.early_stopping_patience == 8 for config in configs)
    assert all(config.dropout == 0.0 for config in configs)
```

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_tcn_gru_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_tcn_gru_focused_final_grid_uses_full_budget -q
```

Expected: fail because `tcn_gru_focused` stages do not exist yet.

---

### Task 2: Add TCN+GRU Focused Stages to the Runner

**Files:**
- Modify: `scripts/run_temporal_backbone_screen.py`
- Test: `tests/test_training.py`

**Step 1: Add a helper for focused configs**

Add:

```python
def _tcn_gru_focused_configs(*, final: bool = False) -> list[ScreenConfig]:
    stage = "tcn_gru_focused_final" if final else "tcn_gru_focused"
    max_epochs = 50 if final else 20
    patience = 8 if final else 5
    dropout = 0.0
    specs = [
        (128, 64, 2, 3, 64),
        (128, 96, 2, 3, 64),
        (128, 128, 2, 3, 64),
        (128, 64, 3, 3, 64),
        (128, 64, 4, 3, 64),
        (128, 64, 2, 5, 64),
        (96, 64, 2, 3, 64),
        (160, 64, 2, 3, 64),
        (128, 64, 2, 3, 96),
        (128, 64, 2, 3, 128),
        (160, 96, 3, 3, 96),
        (160, 128, 4, 3, 128),
    ]
    configs = []
    for history, channels, blocks, kernel, gru_hidden in specs:
        configs.append(
            _config(
                config_id=f"{stage}_hist{history}_c{channels}_b{blocks}_k{kernel}_gru{gru_hidden}",
                stage=stage,
                recipe_name="causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata",
                hidden_sizes=(gru_hidden, gru_hidden),
                sequence_history_size=history,
                max_epochs=max_epochs,
                early_stopping_patience=patience,
                dropout=dropout,
                extra_args={
                    "tcn_channels": channels,
                    "tcn_num_blocks": blocks,
                    "tcn_kernel_size": kernel,
                },
            )
        )
    return configs
```

**Step 2: Add stage dispatch**

Update `build_screen_configs(...)`:

```python
if resolved_stage == "tcn_gru_focused":
    return _tcn_gru_focused_configs(final=False)
if resolved_stage == "tcn_gru_focused_final":
    return _tcn_gru_focused_configs(final=True)
```

Update CLI `--stage` choices:

```text
quick
sweep
final
tcn_gru_focused
tcn_gru_focused_final
all
```

Update `_stage_sample_defaults(...)`:

```python
if stage == "tcn_gru_focused":
    return 131072, 65536, 65536
if stage == "tcn_gru_focused_final":
    return None, None, None
```

**Step 3: Run focused tests**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_tcn_gru_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_tcn_gru_focused_final_grid_uses_full_budget -q
```

Expected: pass.

**Step 4: Compile runner**

Run:

```bash
python -m py_compile scripts/run_temporal_backbone_screen.py
```

Expected: pass.

**Step 5: Commit**

Run:

```bash
git add scripts/run_temporal_backbone_screen.py tests/test_training.py
git commit -m "feat: add TCN-GRU focused sweep configs"
```

---

### Task 3: Dry-Run and Smoke-Test the Focused Stage

**Files:**
- Read: `scripts/run_temporal_backbone_screen.py`
- Create: `artifacts/20260507_tcn_gru_focused_dryrun`

**Step 1: Dry-run config generation**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_tcn_gru_focused_dryrun \
  --stage tcn_gru_focused \
  --dry-run
```

Expected:

```text
artifacts/20260507_tcn_gru_focused_dryrun/temporal_backbone_screen_summary.csv
```

Verify:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/20260507_tcn_gru_focused_dryrun/temporal_backbone_screen_summary.csv")
print(len(df))
print(df[["config_id", "sequence_history_size", "hidden_sizes", "extra_args"]].to_string(index=False))
PY
```

Expected: `12` rows.

**Step 2: One-config smoke run**

Run only the compact baseline config:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_tcn_gru_focused_smoke \
  --stage tcn_gru_focused \
  --config-ids tcn_gru_focused_hist128_c64_b2_k3_gru64 \
  --batch-size 512 \
  --max-train-samples 32768 \
  --max-val-samples 16384 \
  --max-test-samples 16384 \
  --device cuda:0
```

Expected: one model trains and writes a summary row.

**Step 3: Inspect smoke output**

Run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/20260507_tcn_gru_focused_smoke/temporal_backbone_screen_summary.csv")
print(df[["config_id", "model_type", "test_overall_rmse", "test_fy_b_r2", "test_mz_b_r2"]].to_string(index=False))
PY
```

Expected: finite metrics.

---

### Task 4: Run the 12-Config Focused Sweep

**Files:**
- Read: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1`
- Create: `artifacts/20260507_tcn_gru_focused_sweep`

**Step 1: Launch focused sweep**

Run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_tcn_gru_focused_sweep \
  --stage tcn_gru_focused \
  --batch-size 512 \
  --max-train-samples 131072 \
  --max-val-samples 65536 \
  --max-test-samples 65536 \
  --device cuda:0
```

**Step 2: Rank by overall and hard targets**

Run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/20260507_tcn_gru_focused_sweep/temporal_backbone_screen_summary.csv")
df["hard_target_r2"] = df[["test_fy_b_r2", "test_mz_b_r2"]].mean(axis=1)
cols = [
    "config_id",
    "sequence_history_size",
    "hidden_sizes",
    "test_overall_rmse",
    "test_overall_r2",
    "test_fy_b_r2",
    "test_mz_b_r2",
    "hard_target_r2",
    "best_epoch",
]
print("Best overall:")
print(df[cols].sort_values("test_overall_rmse").head(6).to_string(index=False))
print("\nBest hard targets:")
print(df[cols].sort_values("hard_target_r2", ascending=False).head(6).to_string(index=False))
PY
```

**Step 3: Record initial candidates**

Create a short note in the eventual result report with:

```text
best overall config
best fy_b config
best mz_b config
best hard_target_r2 config
configs within 3% of best focused RMSE
```

---

### Task 5: Generate Diagnostics for Focused Sweep Candidates

**Files:**
- Read: `artifacts/20260507_tcn_gru_focused_sweep/temporal_backbone_screen_summary.csv`
- Create: `artifacts/20260507_tcn_gru_focused_sweep/runs/<config_id>/<recipe>/diagnostics`

**Step 1: Generate diagnostics for all 12 sweep configs**

Run:

```bash
python - <<'PY'
import pandas as pd
import subprocess
import sys

summary = pd.read_csv("artifacts/20260507_tcn_gru_focused_sweep/temporal_backbone_screen_summary.csv")
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
        "test",
        "--batch-size",
        "8192",
        "--device",
        "cuda:0",
    ]
    print("running diagnostics", row["config_id"], flush=True)
    subprocess.run(cmd, check=True)
PY
```

**Step 2: Extract low-frequency regime rows**

Run:

```bash
python - <<'PY'
import pandas as pd

summary = pd.read_csv("artifacts/20260507_tcn_gru_focused_sweep/temporal_backbone_screen_summary.csv")
rows = []
for _, row in summary.iterrows():
    reg = pd.read_csv(f"{row['output_dir']}/diagnostics/per_regime_metrics.csv")
    low = reg[
        (reg["regime_column"] == "cycle_flap_frequency_hz")
        & (reg["bin_label"] == "(-0.001, 3.0]")
    ].iloc[0]
    rows.append({
        "config_id": row["config_id"],
        "test_overall_rmse": row["test_overall_rmse"],
        "hard_target_r2": (row["test_fy_b_r2"] + row["test_mz_b_r2"]) / 2.0,
        "low_freq_r2": low["test_overall_r2"],
        "low_freq_rmse": low["test_overall_rmse"],
    })
df = pd.DataFrame(rows)
print(df.sort_values("low_freq_rmse").to_string(index=False))
PY
```

**Step 3: Select final config IDs**

Selection algorithm:

```text
selected = []
add best test_overall_rmse
add best hard_target_r2
add best low_freq_rmse
if len(unique selected) < 3, add next-best test_overall_rmse until 3
if compact config is within 5% of best RMSE, add it as optional 4th
```

Write selected IDs into the report draft before running full-data final.

---

### Task 6: Run Full-Data Final for Selected TCN+GRU Configs

**Files:**
- Create: `artifacts/20260507_tcn_gru_focused_final`

**Step 1: Run selected configs on full data**

Replace `<selected_config_ids>` with the IDs selected in Task 5:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_tcn_gru_focused_final \
  --stage tcn_gru_focused_final \
  --config-ids <selected_config_id_1> <selected_config_id_2> <selected_config_id_3> \
  --batch-size 512 \
  --device cuda:0
```

If a 4th compact config was selected:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_tcn_gru_focused_final \
  --stage tcn_gru_focused_final \
  --config-ids <selected_config_id_1> <selected_config_id_2> <selected_config_id_3> <selected_config_id_4> \
  --batch-size 512 \
  --device cuda:0
```

**Step 2: Generate diagnostics for final selected configs**

Run:

```bash
python - <<'PY'
import pandas as pd
import subprocess
import sys

summary = pd.read_csv("artifacts/20260507_tcn_gru_focused_final/temporal_backbone_screen_summary.csv")
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

---

### Task 7: Benchmark Inference Latency for Final Selected Configs

**Files:**
- Create or modify only if needed: `scripts/benchmark_model_forward.py`
- Otherwise run inline benchmark command

**Step 1: Decide whether to create a reusable benchmark script**

If this benchmark will be reused, create:

```text
scripts/benchmark_model_forward.py
```

It should accept:

```text
--model-bundles
--device
--batches 1,512,8192
--repeats
--output-csv
```

If one-off is enough, use the inline benchmark pattern from the previous temporal backbone inference timing.

**Step 2: Run benchmark**

At minimum benchmark:

```text
current Transformer hist128
current TCN hist128
current GRU hist64
selected TCN+GRU full-data configs
```

Report:

```text
ms_per_batch
us_per_sample
samples_per_s
```

Use both:

```text
batch=1 for realtime point inference
batch=512 for offline/mini-batch throughput
```

---

### Task 8: Write Focused Sweep Result Report

**Files:**
- Create: `docs/results/2026-05-07-tcn-gru-focused-sweep.md`

**Step 1: Write the report**

Required sections:

```markdown
# TCN+GRU Focused Sweep

## Protocol
## Focused Grid
## Focused Sweep Results
## Final Selected Configs
## Full-Data Final Results
## Per-Target Results
## Per-Log and Per-Regime Diagnostics
## Inference Timing
## Decision
## Artifacts
## Caveats
```

The decision section must answer:

```text
Did any TCN+GRU config beat the previous TCN+GRU final?
Did any TCN+GRU config approach or beat Transformer overall?
Is fy_b/mz_b advantage stable?
Is low-frequency behavior better or worse than GRU/Transformer?
Should TCN+GRU become:
  default model
  hard-target ablation
  split-head candidate
  rejected for now
```

**Step 2: Include reference table**

Include the old reference results:

```text
Transformer hist128: test_rmse 0.904964, test_r2 0.718291
TCN hist128:         test_rmse 0.934629, test_r2 0.710291
TCN+GRU hist128:     test_rmse 0.946922, test_r2 0.714331
GRU hist64:          test_rmse 1.007040, test_r2 0.709279
```

**Step 3: Commit report**

Run:

```bash
git add docs/results/2026-05-07-tcn-gru-focused-sweep.md
git commit -m "docs: summarize TCN-GRU focused sweep"
```

---

### Task 9: Final Verification

**Files:**
- Verify: `scripts/run_temporal_backbone_screen.py`
- Verify: `scripts/run_training_diagnostics.py`
- Verify: `src/system_identification/training.py`
- Verify: `tests/test_training.py`
- Verify: `docs/results/2026-05-07-tcn-gru-focused-sweep.md`

**Step 1: Run focused tests**

Run:

```bash
pytest tests/test_training.py::test_temporal_screen_tcn_gru_focused_grid_has_12_configs \
       tests/test_training.py::test_temporal_screen_tcn_gru_focused_final_grid_uses_full_budget \
       tests/test_training.py::test_run_baseline_comparison_supports_temporal_backbone_recipes -q
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
  scripts/run_temporal_backbone_screen.py \
  scripts/run_training_diagnostics.py \
  scripts/run_baseline_comparison.py \
  scripts/train_baseline_torch.py \
  src/system_identification/training.py
```

Expected: pass.

**Step 4: Check diff**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors. Worktree should contain only intended changes before the final commit.

**Step 5: Final commit if needed**

Run:

```bash
git add scripts/run_temporal_backbone_screen.py tests/test_training.py docs/results/2026-05-07-tcn-gru-focused-sweep.md
git commit -m "feat: run TCN-GRU focused sweep"
```

Skip this commit if earlier commits already cover all changes cleanly.

