# Transformer Focused Sweep

## Protocol

Dataset split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
split_policy: whole_log
```

All runs use the leakage-resistant temporal protocol:

```text
feature_set_name: paper_no_accel_v2
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
no acceleration inputs
no centered windows
no past wrench / no past target output
```

Selection rule:

```text
Sweep ranking used validation metrics only.
The transformer_focused sampled sweep ran with skip_test_eval=true and produced no test_* metric columns.
Test metrics were evaluated only for locked full-data final configs.
```

## Sweep Grid

The 12 sampled-data configs tested:

```text
history: 96 / 128 / 160 / 192
depth: 1 / 2 / 3 encoder layers around d64
width: d_model 64 / 96 / 128
heads: 2 / 4 / 8 around d64
dropout: 0.0 / 0.05
```

## Validation-Ranked Sweep Results

Best validation RMSE:

```text
config_id                                      val RMSE  val R2    fy R2    mz R2    hard avg  best_epoch
transformer_focused_hist128_d96_l2_h4_do0      1.042563  0.622295  0.212546 0.347018 0.279782  16
transformer_focused_hist128_d64_l2_h8_do0      1.047266  0.624399  0.250332 0.349266 0.299799  17
transformer_focused_hist96_d64_l2_h4_do0       1.047597  0.617925  0.202604 0.369297 0.285951  12
transformer_focused_hist128_d64_l3_h4_do0      1.049407  0.623065  0.225093 0.352040 0.288566  14
transformer_focused_hist128_d64_l2_h4_do0      1.051815  0.620729  0.220296 0.363595 0.291945  14
transformer_focused_hist128_d64_l2_h2_do0      1.056712  0.623477  0.226002 0.374210 0.300106  14
transformer_focused_hist128_d64_l2_h4_do050    1.058631  0.635574  0.258561 0.362421 0.310491  17
```

Best validation hard-target score:

```text
config_id                                      val RMSE  val R2    hard avg
transformer_focused_hist128_d64_l2_h4_do050    1.058631  0.635574  0.310491
transformer_focused_hist128_d64_l1_h4_do0      1.128798  0.621473  0.304501
transformer_focused_hist128_d128_l2_h4_do0     1.065285  0.628725  0.300419
transformer_focused_hist128_d64_l2_h2_do0      1.056712  0.623477  0.300106
transformer_focused_hist128_d64_l2_h8_do0      1.047266  0.624399  0.299799
```

Validation low-frequency diagnostics:

```text
config_id                                      low-freq val R2  low-freq val RMSE
transformer_focused_hist160_d96_l2_h4_do0      0.474369         0.765937
transformer_focused_hist128_d64_l2_h4_do050    0.510880         0.767728
transformer_focused_hist96_d64_l2_h4_do0       0.462242         0.778468
transformer_focused_hist192_d64_l2_h4_do0      0.459587         0.788855
transformer_focused_hist128_d96_l2_h4_do0      0.422292         0.792255
```

Locked final configs selected by validation only:

```text
transformer_focused_final_hist128_d96_l2_h4_do0
  best validation overall RMSE

transformer_focused_final_hist128_d64_l2_h4_do050
  best validation hard-target score

transformer_focused_final_hist160_d96_l2_h4_do0
  best validation low-frequency RMSE

transformer_focused_final_hist128_d64_l2_h4_do0
  baseline repeat within 2% of best validation RMSE
```

## Full-Data Final Results

```text
config_id                                          val RMSE  val R2    test MAE test RMSE test R2   fy R2    mz R2    hard avg best_epoch
transformer_focused_final_hist128_d64_l2_h4_do050   0.935145 0.680652  0.457400 0.892095  0.736392 0.343493 0.605771 0.474632 14
transformer_focused_final_hist128_d64_l2_h4_do0     0.940080 0.663394  0.461937 0.904964  0.718291 0.319529 0.573306 0.446417 17
transformer_focused_final_hist160_d96_l2_h4_do0     0.957252 0.658803  0.465453 0.905295  0.708037 0.300049 0.568130 0.434089  8
transformer_focused_final_hist128_d96_l2_h4_do0     0.976130 0.653298  0.483299 0.940875  0.709479 0.309321 0.570788 0.440054  6
```

The tuned winner is:

```text
transformer_focused_final_hist128_d64_l2_h4_do050
history=128
d_model=64
layers=2
heads=4
dropout=0.05
```

It improves over the previous Transformer baseline:

```text
old Transformer: RMSE 0.904964, R2 0.718291
new Transformer: RMSE 0.892095, R2 0.736392
RMSE improvement: 1.42%
```

It also improves hard targets:

```text
old Transformer hard avg: 0.446417
new Transformer hard avg: 0.474632
```

## Comparison To Previous Backbones

```text
model/config                                      test RMSE  test R2   fy R2    mz R2
new Transformer d64 l2 h4 dropout0.05             0.892095  0.736392  0.343493 0.605771
old Transformer d64 l2 h4 dropout0.0              0.904964  0.718291  0.319529 0.573306
TCN hist128                                       0.934629  0.710291  0.321869 0.552049
TCN+GRU compact hist128                           0.946922  0.714331  0.336177 0.577578
GRU hist64                                        1.007040  0.709279  0.345528 0.555318
MLP                                               1.198693  0.652377  0.247556 0.484696
```

The new Transformer is now the best overall model and also beats the previous TCN+GRU on `mz_b`. The GRU still has a slightly higher `fy_b R2`, but it loses clearly on overall RMSE and `mz_b`.

## Per-Target Final Results

Per-target R2 for the locked final configs:

```text
config_id                                          fx_b     fy_b     fz_b     mx_b     my_b     mz_b
transformer_focused_final_hist128_d64_l2_h4_do050  0.968832 0.343493 0.968979 0.620899 0.910379 0.605771
transformer_focused_final_hist128_d64_l2_h4_do0    0.969224 0.319529 0.967846 0.570562 0.909280 0.573306
transformer_focused_final_hist160_d96_l2_h4_do0    0.967695 0.300049 0.968657 0.545439 0.898253 0.568130
transformer_focused_final_hist128_d96_l2_h4_do0    0.959074 0.309321 0.966229 0.553177 0.898285 0.570788
```

The main gain from dropout 0.05 is not just global smoothing; it improves `fy_b`, `mz_b`, and `mx_b` together while preserving `fx_b`, `fz_b`, and `my_b`.

## Diagnostics

Worst test log:

```text
config_id                                          worst_test_log                R2       RMSE
transformer_focused_final_hist128_d64_l2_h4_do050  log_4_2026-4-12-17-43-30     0.595584 1.049277
transformer_focused_final_hist128_d64_l2_h4_do0    log_4_2026-4-12-17-43-30     0.532526 1.105417
transformer_focused_final_hist160_d96_l2_h4_do0    log_4_2026-4-12-17-43-30     0.523398 1.063580
transformer_focused_final_hist128_d96_l2_h4_do0    log_4_2026-4-12-17-43-30     0.558791 1.086811
```

Low-frequency and worst-regime diagnostics:

```text
config_id                                          val low R2 val low RMSE test low R2 test low RMSE
transformer_focused_final_hist128_d64_l2_h4_do050  0.533269   0.647833     0.435379    0.887971
transformer_focused_final_hist128_d64_l2_h4_do0    0.484071   0.744996     0.465746    1.096782
transformer_focused_final_hist160_d96_l2_h4_do0    0.485189   0.724264     0.443682    0.725861
transformer_focused_final_hist128_d96_l2_h4_do0    0.493742   0.746279     0.439059    0.876127
```

The worst test regime remains:

```text
cycle_flap_frequency_hz (-0.001, 3.0]
```

The dropout winner substantially improves low-frequency RMSE versus the old Transformer but has lower low-frequency R2. The `hist160 d96` candidate has the best test low-frequency RMSE but loses too much overall and on hard targets.

## Inference Timing

Pure CUDA forward timing on NVIDIA GeForce RTX 4090. Timing excludes data loading, feature construction, and standardization.

```text
model                                  batch history params  ms/forward us/sample
GRU hist64                             1     64       77830   0.128030   128.030266
GRU hist64                             512   64       77830   0.652591     1.274593
Old Transformer d64 l2 h4 hist128      1     128     112774   1.081075  1081.074613
Old Transformer d64 l2 h4 hist128      512   128     112774   7.546253    14.738775
TCN+GRU compact hist128                1     128      53196   0.215441   215.441117
TCN+GRU compact hist128                512   128      53196   0.988176     1.930031
Transformer final d64 l2 h4 do0        1     128     112774   0.793656   793.655654
Transformer final d64 l2 h4 do0        512   128     112774   4.087790     7.983964
Transformer final d64 l2 h4 do005      1     128     112774   0.293877   293.877077
Transformer final d64 l2 h4 do005      512   128     112774   3.628602     7.087113
Transformer final d96 l2 h4 hist128    1     128     216582   0.282942   282.942123
Transformer final d96 l2 h4 hist128    512   128     216582   4.819911     9.413888
Transformer final d96 l2 h4 hist160    1     160     216582   0.334489   334.488790
Transformer final d96 l2 h4 hist160    512   160     216582   7.782036    15.199289
```

The timing numbers vary across runs, so they should be treated as same-session comparisons rather than absolute deployment guarantees. The tuned dropout Transformer is still slower than TCN+GRU and GRU, but it is much stronger in accuracy.

## Decision

Promote:

```text
causal_transformer_paper_no_accel_v2_phase_actuator_airdata
history=128
d_model=64
num_layers=2
num_heads=4
dim_feedforward=128
dropout=0.05
```

Rationale:

```text
1. Best final validation RMSE/R2 among locked configs.
2. Best final test RMSE/R2.
3. Improves both hard targets versus old Transformer.
4. Improves worst-log RMSE and R2.
5. Does not require a larger Transformer.
```

Do not promote the larger d96 variants yet. The sampled sweep made d96 look promising, but full-data final did not confirm a useful overall gain.

Next useful work:

```text
1. Repeat the promoted dropout Transformer with 2-3 random seeds to check stability.
2. Sweep dropout 0.025 / 0.05 / 0.075 around the winner before changing architecture again.
3. Keep low-frequency diagnostics as a gating metric; the low-frequency bin remains the worst regime.
4. If deployment speed becomes important, compare this model against TCN+GRU under an end-to-end preprocessing + inference benchmark.
```

## Artifacts

```text
artifacts/20260507_transformer_focused_dryrun/
artifacts/20260507_transformer_focused_smoke/
artifacts/20260507_transformer_focused_sweep/temporal_backbone_screen_summary.csv
artifacts/20260507_transformer_focused_final/temporal_backbone_screen_summary.csv
artifacts/20260507_transformer_focused_final/runs/<config_id>/<recipe>/model_bundle.pt
artifacts/20260507_transformer_focused_final/runs/<config_id>/<recipe>/diagnostics/per_log_metrics.csv
artifacts/20260507_transformer_focused_final/runs/<config_id>/<recipe>/diagnostics/per_regime_metrics.csv
```
