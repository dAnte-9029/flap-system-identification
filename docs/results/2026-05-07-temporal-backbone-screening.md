# Temporal Backbone Screening

## Protocol

Dataset split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
split_policy: whole_log
```

All temporal candidates use:

```text
feature_set_name: paper_no_accel_v2
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
no acceleration inputs
no centered windows
no past wrench / no past target output
selection_metric: scaled validation Huber loss
```

## Candidate Models

References:

```text
mlp_paper_no_accel_v2
causal_gru_paper_no_accel_v2_phase_actuator_airdata
```

Screened temporal backbones:

```text
causal_lstm_paper_no_accel_v2_phase_actuator_airdata
causal_tcn_paper_no_accel_v2_phase_actuator_airdata
causal_transformer_paper_no_accel_v2_phase_actuator_airdata
causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata
```

## Commands

Quick screen:

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

Targeted sweep:

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

Final full-data comparison:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_temporal_backbone_final \
  --stage final \
  --batch-size 512 \
  --device cuda:0
```

## Quick Screen

```text
config_id                              model_type          test_rmse  test_r2
quick_transformer_d64_l1_h4_hist64     causal_transformer  1.172853   0.649120
quick_tcn_c128_b4_k3_hist64            causal_tcn          1.204047   0.627386
quick_tcn_gru_c128_b3_k3_h128_hist64   causal_tcn_gru      1.211778   0.615793
quick_lstm_h128x2_hist64               causal_lstm         1.231808   0.625280
quick_gru_h128x2_hist64                causal_gru          1.232096   0.624514
quick_mlp_h128x2                       mlp                 1.289307   0.626428
```

Quick result: no candidate was rejected. Transformer and TCN were the strongest first-pass candidates.

## Targeted Sweep

Best per model type:

```text
config_id                              model_type          history  best_epoch  test_rmse  test_r2
sweep_transformer_d64_l2_h4_hist128    causal_transformer  128      17          1.025834   0.687012
sweep_tcn_gru_c64_b2_h64_hist128       causal_tcn_gru      128      14          1.122923   0.637433
sweep_tcn_c128_b4_k3_hist128           causal_tcn          128       9          1.145721   0.648889
sweep_lstm_h128_l1_hist128             causal_lstm         128      12          1.195940   0.634450
sweep_gru_h128_hist64                  causal_gru           64      13          1.244931   0.632776
```

Promoted to final:

```text
sweep_transformer_d64_l2_h4_hist128
sweep_tcn_c128_b4_k3_hist128
sweep_tcn_gru_c64_b2_h64_hist128
```

LSTM was not promoted because it did not beat the stronger temporal candidates and did not show a clear hard-target advantage.

## Final Full-Data Comparison

```text
config_id                              model_type          history  samples  best_epoch  test_mae  test_rmse  test_r2
final_transformer_d64_l2_h4_hist128    causal_transformer  128      60478    17          0.461937  0.904964   0.718291
final_tcn_c128_b4_k3_hist128           causal_tcn          128      60478    12          0.477940  0.934629   0.710291
final_tcn_gru_c64_b2_h64_hist128       causal_tcn_gru      128      60478     9          0.484697  0.946922   0.714331
final_gru_h128_hist64                  causal_gru           64      61502    15          0.482974  1.007040   0.709279
final_mlp_h128x2                       mlp                   -      62510    20          0.574047  1.198693   0.652377
```

The final Transformer reduced overall RMSE by about 10.1% relative to the causal GRU:

```text
(1.007040 - 0.904964) / 1.007040 = 10.14%
```

## Per-Target Results

Per-target R2:

```text
model_type          fx_b    fy_b    fz_b    mx_b    my_b    mz_b
mlp                 0.9190  0.2476  0.9426  0.4768  0.8437  0.4847
causal_gru          0.9388  0.3455  0.9634  0.5583  0.8943  0.5553
causal_transformer  0.9692  0.3195  0.9678  0.5706  0.9093  0.5733
causal_tcn          0.9667  0.3219  0.9647  0.5573  0.8992  0.5520
causal_tcn_gru      0.9622  0.3362  0.9640  0.5573  0.8888  0.5776
```

Per-target RMSE:

```text
model_type          fx_b      fy_b      fz_b      mx_b      my_b      mz_b
mlp                 1.420171  1.283642  2.226332  0.002407  0.001710  0.000369
causal_gru          1.226997  1.189067  1.779149  0.002213  0.001406  0.000342
causal_transformer  0.847246  1.190933  1.666614  0.002175  0.001300  0.000333
causal_tcn          0.881681  1.188883  1.746532  0.002208  0.001371  0.000341
causal_tcn_gru      0.939236  1.176274  1.764702  0.002208  0.001439  0.000331
```

The Transformer win is mainly from `fx_b`, `fz_b`, `mx_b`, `my_b`, and `mz_b`. It does not improve `fy_b`; TCN+GRU is best among final models for `fy_b` and `mz_b`, but worse on `fx_b`, `fz_b`, and `my_b`.

## Per-Log and Per-Regime Diagnostics

Worst log:

```text
model_type          worst_log                    test_r2   test_rmse
mlp                 log_4_2026-4-12-17-43-30     0.513706  1.251973
causal_gru          log_4_2026-4-12-17-43-30     0.545949  1.237152
causal_transformer  log_4_2026-4-12-17-43-30     0.532526  1.105417
causal_tcn          log_4_2026-4-12-17-43-30     0.543987  1.119722
causal_tcn_gru      log_4_2026-4-12-17-43-30     0.549395  1.125236
```

Worst regime:

```text
model_type          worst_regime              bin             test_r2   test_rmse
mlp                 cycle_flap_frequency_hz   (-0.001, 3.0]  0.079440  1.353678
causal_gru          cycle_flap_frequency_hz   (-0.001, 3.0]  0.508076  0.729493
causal_transformer  cycle_flap_frequency_hz   (-0.001, 3.0]  0.465746  1.096782
causal_tcn          cycle_flap_frequency_hz   (-0.001, 3.0]  0.436086  0.890904
causal_tcn_gru      cycle_flap_frequency_hz   (-0.001, 3.0]  0.428566  0.792530
```

The worst regime remains the low-flap-frequency bin for all models. The Transformer wins overall and improves worst-log RMSE, but it does not improve the low-frequency worst-regime diagnostic relative to the GRU. This should be treated as a remaining weakness.

## Decision

Default recommendation:

```text
Use causal_transformer_paper_no_accel_v2_phase_actuator_airdata with history 128 as the new accuracy-leading backbone candidate.
Keep causal_gru_paper_no_accel_v2_phase_actuator_airdata as the simple, cheaper reference baseline.
```

Model decisions:

```text
causal_transformer:
  Promote. Best full-data RMSE/R2 and strongest overall per-target profile.

causal_tcn:
  Keep as a strong ablation. Faster/simpler than Transformer, beats GRU overall, but does not beat Transformer.

causal_tcn_gru:
  Keep as hard-target ablation. Best final fy_b and mz_b among promoted candidates, but not best overall.

causal_lstm:
  Reject for now. It did not beat the stronger candidates in sweep.

causal_gru:
  Keep as reference baseline. It remains competitive and has better low-frequency worst-regime R2 than the promoted candidates.
```

Next useful work:

```text
1. Tune the promoted Transformer lightly: dropout 0.0/0.05/0.1, history 96/128, d_model 64/96.
2. Add a low-frequency-focused diagnostic or loss weighting experiment, because the new models do not solve that regime.
3. Consider a small ensemble or split-head variant only if fy_b/mz_b remain important enough to justify TCN+GRU's niche advantage.
```

## Artifacts

```text
artifacts/20260507_temporal_backbone_quick/temporal_backbone_screen_summary.csv
artifacts/20260507_temporal_backbone_sweep/temporal_backbone_screen_summary.csv
artifacts/20260507_temporal_backbone_final/temporal_backbone_screen_summary.csv
artifacts/20260507_temporal_backbone_final/runs/<config_id>/<recipe>/model_bundle.pt
artifacts/20260507_temporal_backbone_final/runs/<config_id>/<recipe>/diagnostics/per_log_metrics.csv
artifacts/20260507_temporal_backbone_final/runs/<config_id>/<recipe>/diagnostics/per_regime_metrics.csv
```

## Caveats

The final Transformer/TCN/TCN+GRU use `history=128`, while the reference GRU uses `history=64`. This intentionally tests whether longer causal history helps, but the aligned test sample count is slightly lower for `history=128`.

The comparison is still leakage-resistant because all temporal windows are causal and the split is whole-log. No past wrench or target output is used as an input.

