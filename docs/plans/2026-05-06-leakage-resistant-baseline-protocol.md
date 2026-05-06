# Leakage-Resistant Baseline Protocol

## Purpose

This protocol fixes the main baseline used for flapping-wing effective-wrench regression. It is meant to prevent two common sources of optimistic results:

- label-construction leakage from linear acceleration and angular acceleration inputs
- same-log temporal/regime leakage from splitting one flight log across train, validation, and test

Cycle-block and full-feature runs remain useful sanity checks, but they are not the primary research baseline.

## Primary Baseline

Use this as the default reported baseline:

```text
Split policy: whole-log
Backbone: MLP
Feature set: paper_no_accel_v2
Forbidden inputs:
  vehicle_local_position.ax
  vehicle_local_position.ay
  vehicle_local_position.az
  vehicle_angular_velocity.xyz_derivative[0]
  vehicle_angular_velocity.xyz_derivative[1]
  vehicle_angular_velocity.xyz_derivative[2]
Loss: Huber in scaled target space
Huber delta: 1.5
Optimizer: AdamW
Selection: best validation loss
Primary metrics: per-target MAE, RMSE, and R2
```

The canonical parquet files may retain acceleration channels for label construction and auditing. The model input feature set must exclude them for the primary baseline.

## Required Diagnostics

Every serious baseline run should be followed by:

```text
per-log evaluation
binned evaluation by airspeed
binned evaluation by flap frequency
binned evaluation by corrected phase
```

The repository calls this binned evaluation or per-regime diagnostics. It answers where the model fails rather than only whether the aggregate test score is high.

Run:

```bash
python scripts/run_training_diagnostics.py \
  --model-bundle artifacts/<run>/model_bundle.pt \
  --split-root dataset/<whole_log_split> \
  --output-dir artifacts/<run>/diagnostics \
  --splits test
```

Outputs:

```text
per_log_metrics.csv
per_regime_metrics.csv
diagnostics_config.json
```

## Backbone Comparison

After the primary MLP baseline is stable, compare backbones under the same whole-log protocol:

```text
mlp_paper_no_accel_v2
split_axis_mlp_paper_no_accel_v2
mlp_paper_pfnn_10
pfnn_paper_pfnn_10
mlp_paper_no_accel_v2_causal_phase_actuator
causal_gru_paper_no_accel_v2_phase_actuator_airdata
causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata
```

`split_axis_mlp_paper_no_accel_v2` trains two independent MLPs under the same feature and split protocol:

```text
longitudinal targets: fx_b, fz_b, my_b
lateral targets: fy_b, mx_b, mz_b
```

The comparison summary merges their disjoint per-target metrics into one row so it can be compared directly against the single six-output MLP.

## Forward Sequence Candidates Inspired by Sharvit et al. 2025

The Sharvit et al. 2025 model is an inverse mapping model, but the useful forward-system-ID idea is to treat flapping flight as a causal, periodic, multivariate time-series problem.

The forward adaptation uses:

```text
past/current inputs -> current wrench
sequence history: phase + actuator + airdata
current features: remaining non-history point features
targets: fx_b, fy_b, fz_b, mx_b, my_b, mz_b
```

Primary sequence candidates:

```text
causal_gru_paper_no_accel_v2_phase_actuator_airdata
causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata
```

These are causal forward models, not inverse reproductions. They must not use desired future forces, future labels, or centered windows.

For the primary sequence history, velocity, angular velocity, `alpha_rad`, and `beta_rad` are excluded. They may be kept as current-time point features, but their histories can reintroduce finite-difference shortcuts similar to acceleration leakage. The saved `training_config.json` should therefore be checked for:

```text
has_velocity_history: false
has_angular_velocity_history: false
has_alpha_beta_history: false
has_acceleration_inputs: false
```

Run:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/<whole_log_split> \
  --output-dir artifacts/<comparison_run> \
  --device cuda
```

Outputs:

```text
baseline_protocol.json
baseline_comparison_summary.csv
baseline_comparison_summary.json
baseline_comparison_summary.png
```

## Interpretation Rules

Use test metrics only for final comparison after selecting the model by validation loss.

Report per-target metrics first. Overall R2 is only a compact summary and can hide weak axes such as `fy_b`, `mx_b`, or `mz_b`.

Treat full-feature runs that include acceleration channels as pipeline sanity checks, not aerodynamic surrogate evidence.

Do feature ablations only after the training recipe is fixed. Otherwise a measured gain may come from optimizer, loss, split, or random seed changes instead of the feature group being tested.
