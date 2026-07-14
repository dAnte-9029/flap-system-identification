# Prior TCN Sample Efficiency and Capacity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the DeLaurier calibrated prior helps TCN force prediction under limited training data and compact model capacity.

**Architecture:** Extend `scripts/run_prior_vs_tcn_comparison.py` with two experiment modes instead of creating a separate training stack. Both modes reuse the same calibrated prior roots, whole-log outer folds, TCN residual and pure TCN training functions, common-row evaluation, condition-binned metrics, residual spectra, and diagnostic plots. The only controlled variables are the development-data fraction for Experiment A and the TCN capacity configuration for Experiment B.

**Tech Stack:** Python, pandas, numpy, PyTorch through `system_identification.training.fit_torch_sequence_regressor`, existing prior alignment helpers, pytest, matplotlib.

---

### Task 1: Add Data Subsampling Helpers

**Files:**
- Modify: `scripts/run_prior_vs_tcn_comparison.py`
- Modify: `tests/test_prior_vs_tcn_comparison.py`

**Step 1: Write failing tests for deterministic group subsampling**

Add tests:

```python
def test_select_train_fraction_keeps_whole_segments_and_is_deterministic():
    frame = pd.DataFrame(
        {
            "log_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "segment_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "time_s": np.arange(8, dtype=float),
        }
    )
    first = select_training_fraction(frame, fraction=0.5, seed=7)
    second = select_training_fraction(frame, fraction=0.5, seed=7)

    assert first.equals(second)
    assert set(first["log_id"]).issubset({"a", "b", "c", "d"})
    assert first.groupby("log_id").size().nunique() == 1
    assert 0 < first["log_id"].nunique() < frame["log_id"].nunique()
```

Add test for `fraction=1.0` returning the full frame in original order.

**Step 2: Verify tests fail**

Run:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh && conda activate flap-train-gpu
pytest tests/test_prior_vs_tcn_comparison.py::test_select_train_fraction_keeps_whole_segments_and_is_deterministic -q
```

Expected: FAIL because `select_training_fraction` does not exist.

**Step 3: Implement `select_training_fraction`**

Add:

```python
def select_training_fraction(frame: pd.DataFrame, *, fraction: float, seed: int) -> pd.DataFrame:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction >= 1.0:
        return frame.copy().reset_index(drop=True)
    group_columns = ["log_id", "segment_id"]
    groups = frame.loc[:, group_columns].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(groups))
    target_rows = max(1, int(round(len(frame) * float(fraction))))
    selected_keys = []
    selected_rows = 0
    for index in order:
        key = groups.iloc[index]
        mask = (frame["log_id"].astype(str) == str(key["log_id"])) & (
            pd.to_numeric(frame["segment_id"], errors="raise").astype("int64") == int(key["segment_id"])
        )
        selected_keys.append((str(key["log_id"]), int(key["segment_id"])))
        selected_rows += int(mask.sum())
        if selected_rows >= target_rows:
            break
    selected = pd.DataFrame(selected_keys, columns=group_columns)
    keyed = frame.copy()
    keyed["log_id"] = keyed["log_id"].astype(str)
    keyed["segment_id"] = pd.to_numeric(keyed["segment_id"], errors="raise").astype("int64")
    out = keyed.merge(selected, on=group_columns, how="inner", sort=False)
    return out.sort_values(["log_id", "segment_id", "time_s"], kind="mergesort").reset_index(drop=True)
```

**Step 4: Run tests**

Run:

```bash
pytest tests/test_prior_vs_tcn_comparison.py -q
```

Expected: all tests pass.

---

### Task 2: Add Experiment Configuration Grid

**Files:**
- Modify: `scripts/run_prior_vs_tcn_comparison.py`
- Modify: `tests/test_prior_vs_tcn_comparison.py`

**Step 1: Write failing tests for experiment config expansion**

Add tests:

```python
def test_expand_sample_efficiency_configs_uses_requested_fractions():
    configs = expand_experiment_configs(
        experiment="sample_efficiency",
        train_fractions=(0.1, 0.2),
        capacity_presets=("base",),
    )
    assert [config["train_fraction"] for config in configs] == [0.1, 0.2]
    assert {config["capacity_preset"] for config in configs} == {"base"}
```

```python
def test_expand_capacity_configs_uses_full_train_fraction():
    configs = expand_experiment_configs(
        experiment="capacity",
        train_fractions=(1.0,),
        capacity_presets=("tiny", "small", "base"),
    )
    assert [config["capacity_preset"] for config in configs] == ["tiny", "small", "base"]
    assert {config["train_fraction"] for config in configs} == {1.0}
```

**Step 2: Verify tests fail**

Run:

```bash
pytest tests/test_prior_vs_tcn_comparison.py::test_expand_sample_efficiency_configs_uses_requested_fractions -q
```

Expected: FAIL because `expand_experiment_configs` does not exist.

**Step 3: Implement preset definitions**

Add:

```python
CAPACITY_PRESETS = {
    "tiny": {"sequence_history_size": 16, "tcn_channels": 16, "tcn_num_blocks": 1, "hidden_sizes": (32,)},
    "small": {"sequence_history_size": 32, "tcn_channels": 32, "tcn_num_blocks": 2, "hidden_sizes": (64,)},
    "base": {"sequence_history_size": 64, "tcn_channels": 128, "tcn_num_blocks": 4, "hidden_sizes": (128, 128)},
}
```

Add:

```python
def expand_experiment_configs(
    *,
    experiment: str,
    train_fractions: tuple[float, ...],
    capacity_presets: tuple[str, ...],
) -> list[dict[str, object]]:
    configs = []
    for fraction in train_fractions:
        for preset in capacity_presets:
            if preset not in CAPACITY_PRESETS:
                raise ValueError(f"unknown capacity preset: {preset}")
            configs.append(
                {
                    "experiment": experiment,
                    "config_id": f"{experiment}__frac_{str(fraction).replace('.', 'p')}__{preset}",
                    "train_fraction": float(fraction),
                    "capacity_preset": preset,
                    **CAPACITY_PRESETS[preset],
                }
            )
    return configs
```

**Step 4: Add CLI arguments**

Add:

```python
parser.add_argument("--experiment", default="baseline", choices=["baseline", "sample_efficiency", "capacity", "sample_efficiency_capacity"])
parser.add_argument("--train-fractions", type=_parse_float_tuple, default=(1.0,))
parser.add_argument("--capacity-presets", type=_parse_string_tuple, default=("base",))
```

For the planned runs:

- Experiment A uses `--experiment sample_efficiency --train-fractions 0.1,0.2,0.4,0.6,1.0 --capacity-presets base`
- Experiment B uses `--experiment capacity --train-fractions 1.0 --capacity-presets tiny,small,base`

**Step 5: Run tests**

Run:

```bash
pytest tests/test_prior_vs_tcn_comparison.py -q
python -m py_compile scripts/run_prior_vs_tcn_comparison.py tests/test_prior_vs_tcn_comparison.py
```

Expected: pass.

---

### Task 3: Refactor Fold Evaluation to Accept a Config

**Files:**
- Modify: `scripts/run_prior_vs_tcn_comparison.py`

**Step 1: Add a config argument to `_evaluate_fold`**

Change:

```python
def _evaluate_fold(..., args: argparse.Namespace) -> ...
```

to:

```python
def _evaluate_fold(..., args: argparse.Namespace, config: dict[str, object]) -> ...
```

**Step 2: Apply train fraction only to TCN training frames**

Inside `_evaluate_fold`, after `frames = _load_frames(...)`, create:

```python
train_fraction = float(config["train_fraction"])
tcn_train_frame = select_training_fraction(
    frames["train"],
    fraction=train_fraction,
    seed=int(args.random_seed) + 10000 * fold + int(round(train_fraction * 1000)),
)
```

Use `tcn_train_frame` for both:

- TCN residual training
- Pure TCN training

Keep `frames["val"]` unchanged for early stopping. Keep test unchanged.

**Important:** Do not subsample validation or test. Do not change prior roots. Do not change gain-bias fitting unless the run explicitly includes gain-bias baselines. For A/B, the key comparison is `TCN residual` vs `Pure TCN`.

**Step 3: Apply capacity preset to TCN training**

Before `_train_tcn`, resolve:

```python
tcn_args = copy.copy(args)
tcn_args.sequence_history_size = int(config["sequence_history_size"])
tcn_args.tcn_channels = int(config["tcn_channels"])
tcn_args.tcn_num_blocks = int(config["tcn_num_blocks"])
tcn_args.hidden_sizes = tuple(config["hidden_sizes"])
```

Use `tcn_args` for training and prediction so common test rows use the matching history length.

**Step 4: Store config metadata**

Add these fields to each metric row and manifest:

```python
"experiment": config["experiment"],
"config_id": config["config_id"],
"train_fraction": config["train_fraction"],
"capacity_preset": config["capacity_preset"],
"sequence_history_size": config["sequence_history_size"],
"tcn_channels": config["tcn_channels"],
"tcn_num_blocks": config["tcn_num_blocks"],
```

**Step 5: Preserve common-row fairness**

For each config, common test rows are defined by that config's TCN history. Do not compare tiny/base using different row sets without recording `n_samples`.

---

### Task 4: Update Run Loop and Outputs

**Files:**
- Modify: `scripts/run_prior_vs_tcn_comparison.py`

**Step 1: Expand configs in `run`**

At start of `run(args)`:

```python
configs = expand_experiment_configs(
    experiment=args.experiment,
    train_fractions=tuple(args.train_fractions),
    capacity_presets=tuple(args.capacity_presets),
)
```

For each `config` and each `fold`, write outputs under:

```text
artifacts/<root>/<config_id>/fold_<k>/
```

**Step 2: Aggregate across config and fold**

Write:

- `overall_metrics_by_fold.csv`
- `overall_metrics_summary.csv`
- `condition_binned_metrics_by_fold.csv`
- `condition_binned_metrics_summary.csv`
- `residual_spectrum_by_fold.csv`
- `residual_spectrum_summary.csv`

Group summaries by:

```python
["experiment", "config_id", "train_fraction", "capacity_preset", "model"]
```

**Step 3: Add compact comparison tables**

Write:

- `sample_efficiency_tcn_comparison.csv`
- `capacity_tcn_comparison.csv`

Each should keep only:

```text
config_id, train_fraction, capacity_preset, model,
rmse_fx_b_mean, rmse_fz_b_mean, rmse_force_norm_mean,
mae_fx_b_mean, mae_fz_b_mean, r2_fx_b_mean, r2_fz_b_mean
```

Then add a delta table:

```text
delta_rmse_force_norm = Pure TCN - TCN residual
```

Positive delta means the prior residual model is better.

---

### Task 5: Run a Smoke Test

**Files:**
- Output: `artifacts/20260707_prior_tcn_ab_smoke/`

**Step 1: Run one fold, one epoch**

Run:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh && conda activate flap-train-gpu
python scripts/run_prior_vs_tcn_comparison.py \
  --experiment sample_efficiency \
  --train-fractions 0.1,1.0 \
  --capacity-presets tiny \
  --folds 0 \
  --max-epochs 1 \
  --early-stopping-patience 1 \
  --output-root artifacts/20260707_prior_tcn_ab_smoke \
  --force
```

Expected:

- two config directories
- each config has `fold_0/tcn_residual/model_bundle.pt`
- each config has `fold_0/pure_tcn/model_bundle.pt`
- summary tables include both `TCN residual` and `Pure TCN`

**Step 2: Verify smoke artifacts**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("artifacts/20260707_prior_tcn_ab_smoke")
print(pd.read_csv(root / "overall_metrics_summary.csv")[["config_id", "model", "rmse_force_norm_mean"]])
assert (root / "overall_metrics_summary.csv").exists()
assert (root / "sample_efficiency_tcn_comparison.csv").exists()
PY
```

Expected: exits 0.

---

### Task 6: Run Experiment A: Sample Efficiency

**Files:**
- Output: `artifacts/20260707_prior_tcn_sample_efficiency/`

**Step 1: Run full sample-efficiency sweep**

Run:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh && conda activate flap-train-gpu
python scripts/run_prior_vs_tcn_comparison.py \
  --experiment sample_efficiency \
  --train-fractions 0.1,0.2,0.4,0.6,1.0 \
  --capacity-presets base \
  --output-root artifacts/20260707_prior_tcn_sample_efficiency \
  --force
```

**Step 2: Verify expected training count**

Expected number of TCN trainings:

```text
5 fractions x 6 folds x 2 models = 60 TCN trainings
```

**Step 3: Generate the interpretation table**

Read:

```text
artifacts/20260707_prior_tcn_sample_efficiency/sample_efficiency_tcn_delta.csv
```

Key interpretation:

- `delta_rmse_force_norm > 0`: prior + residual TCN is better.
- Prior value is strongest if positive at 10%, 20%, or 40%, even if it disappears at 100%.

---

### Task 7: Run Experiment B: Capacity Sweep

**Files:**
- Output: `artifacts/20260707_prior_tcn_capacity/`

**Step 1: Run full capacity sweep**

Run:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh && conda activate flap-train-gpu
python scripts/run_prior_vs_tcn_comparison.py \
  --experiment capacity \
  --train-fractions 1.0 \
  --capacity-presets tiny,small,base \
  --output-root artifacts/20260707_prior_tcn_capacity \
  --force
```

**Step 2: Verify expected training count**

Expected number of TCN trainings:

```text
3 capacity settings x 6 folds x 2 models = 36 TCN trainings
```

**Step 3: Generate the interpretation table**

Read:

```text
artifacts/20260707_prior_tcn_capacity/capacity_tcn_delta.csv
```

Key interpretation:

- If prior helps `tiny` and `small` but not `base`, the prior reduces model capacity requirements.
- If pure TCN wins in all capacity settings, the current prior does not provide measurable predictive value for sequence models under these inputs.

---

### Task 8: Final Verification and Report

**Files:**
- Read: `artifacts/20260707_prior_tcn_sample_efficiency/*.csv`
- Read: `artifacts/20260707_prior_tcn_capacity/*.csv`

**Step 1: Run tests**

Run:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh && conda activate flap-train-gpu
pytest tests/test_prior_vs_tcn_comparison.py -q
python -m py_compile scripts/run_prior_vs_tcn_comparison.py tests/test_prior_vs_tcn_comparison.py
```

Expected: pass.

**Step 2: Verify output completeness**

Run:

```bash
python - <<'PY'
from pathlib import Path
for root in [
    Path("artifacts/20260707_prior_tcn_sample_efficiency"),
    Path("artifacts/20260707_prior_tcn_capacity"),
]:
    assert (root / "overall_metrics_summary.csv").exists(), root
    assert (root / "condition_binned_metrics_summary.csv").exists(), root
    assert (root / "manifest.json").exists(), root
print("ok")
PY
```

Expected: `ok`.

**Step 3: Report**

Report:

- Main A table: data fraction vs RMSE norm for `TCN residual` and `Pure TCN`.
- Main B table: capacity preset vs RMSE norm for `TCN residual` and `Pure TCN`.
- Delta table where positive means prior helps.
- Whether the evidence supports sample efficiency, compact-model benefit, both, or neither.
- Any caveats about shared common-row history length and whether alpha/phase/stroke residual structure changes.
