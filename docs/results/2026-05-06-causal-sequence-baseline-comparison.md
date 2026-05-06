# Causal Sequence Baseline Comparison

## Dataset

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
split_policy: whole_log
```

## Compared Recipes

```text
mlp_paper_no_accel_v2
causal_gru_paper_no_accel_v2_phase_actuator_airdata
causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata
```

The sequence models use causal history only:

```text
sequence_history_size: 64
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
```

The saved configs for both sequence runs record:

```text
has_velocity_history: false
has_angular_velocity_history: false
has_alpha_beta_history: false
has_acceleration_inputs: false
```

## Command

This was a full-dataset, one-epoch engineering comparison to verify the new sequence pipeline end to end:

```bash
MPLCONFIGDIR=/tmp/mpl-mlp-vs-causal-gru-asl-20260506_165350 \
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260506_165350_mlp_vs_causal_gru_asl \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata \
  --batch-size 4096 \
  --max-epochs 1 \
  --early-stopping-patience 1 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --hidden-sizes 32 \
  --sequence-history-size 64 \
  --device cpu \
  --disable-amp
```

Longer full-dataset GRU runs with larger hidden sizes were attempted first, but were too slow for interactive execution in the current environment. Treat this result as a pipeline and early-training comparison, not a converged architecture ranking.

## Summary

```text
recipe                                                model_type      test_overall_mae  test_overall_rmse  test_overall_r2
mlp_paper_no_accel_v2                                mlp                    1.605591           3.346804         0.219649
causal_gru_paper_no_accel_v2_phase_actuator_airdata  causal_gru             0.870709           1.749518         0.454103
causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata causal_gru_asl       0.867524           1.753057         0.443421
```

Per-target R2:

```text
recipe                                                fx_b      fy_b      fz_b      mx_b      my_b      mz_b
mlp_paper_no_accel_v2                                0.456290  0.077481  0.401859  0.160814  0.199674  0.021779
causal_gru_paper_no_accel_v2_phase_actuator_airdata  0.784148  0.090092  0.871775  0.191711  0.649464  0.137428
causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata 0.780886 0.089768 0.871852 0.185858 0.615590 0.116573
```

## Diagnostics

Diagnostics were run for the plain causal GRU:

```text
artifacts/20260506_165350_mlp_vs_causal_gru_asl/causal_gru_paper_no_accel_v2_phase_actuator_airdata/diagnostics/per_log_metrics.csv
artifacts/20260506_165350_mlp_vs_causal_gru_asl/causal_gru_paper_no_accel_v2_phase_actuator_airdata/diagnostics/per_regime_metrics.csv
```

Worst test logs by overall R2:

```text
log_id                          samples  overall_r2  overall_rmse
log_4_2026-4-14-12-30-12          14710    0.447333      1.710448
log_34_2026-4-16-19-13-30         13998    0.449778      1.711825
log_18_2026-4-15-12-56-08         14689    0.453576      1.764880
log_4_2026-4-12-17-43-30          18105    0.460658      1.796584
```

Worst test regimes by overall R2:

```text
regime_column                         bin_label        samples  overall_r2  overall_rmse
phase_corrected_rad                   (3.142, 4.712]     14037   -0.878647      5.606836
phase_corrected_rad                   (1.571, 3.142]     12758   -0.521805      2.500754
phase_corrected_rad                   (-0.001, 1.571]    18373   -0.486132      4.666437
phase_corrected_rad                   (4.712, 6.283]     13488    0.135712      1.962136
airspeed_validated.true_airspeed_m_s  (-0.001, 6.0]       3186    0.284571      3.326000
```

## Interpretation

The causal GRU sequence path is worth keeping. Even after only one epoch, it outperformed the one-epoch MLP on overall test R2 and on most force/moment targets.

ASL did not improve over plain GRU in this short run. This is not enough to reject ASL, but it means ASL should be treated as an ablation candidate rather than the new default.

The phase-binned diagnostics are poor despite good aggregate force-axis gains. That means the model still has strong phase-dependent failure modes, and future sequence experiments should inspect phase-conditioned residuals before claiming a better aerodynamic surrogate.

## Artifacts

```text
artifacts/20260506_165350_mlp_vs_causal_gru_asl/baseline_comparison_summary.csv
artifacts/20260506_165350_mlp_vs_causal_gru_asl/baseline_comparison_summary.json
artifacts/20260506_165350_mlp_vs_causal_gru_asl/baseline_comparison_summary.png
```

## Caveats

This was not a converged training run. The sequence implementation is now functional and diagnostically covered, but the final architecture decision requires longer GPU training with a controlled compute budget.
