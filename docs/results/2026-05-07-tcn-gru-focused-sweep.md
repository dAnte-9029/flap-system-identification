# TCN+GRU Focused Sweep

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
selection_metric: scaled validation Huber loss
```

Protocol note:

```text
The tables below include test metrics because this was an exploratory completed run.
For future sweeps, config selection should use validation metrics only.
The test split should be used only after final configs are locked.
```

The focused sweep only varies the causal TCN+GRU backbone:

```text
tcn_channels: 64 / 96 / 128
tcn_num_blocks: 2 / 3 / 4
tcn_kernel_size: 3 / 5
gru_hidden_size: 64 / 96 / 128
history_size: 96 / 128 / 160
```

## Commands

Dry run:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_tcn_gru_focused_dryrun \
  --stage tcn_gru_focused \
  --dry-run
```

Focused sampled-data sweep:

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

Full-data final for selected candidates:

```bash
python scripts/run_temporal_backbone_screen.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_tcn_gru_focused_final \
  --stage tcn_gru_focused_final \
  --config-ids \
    tcn_gru_focused_final_hist128_c96_b2_k3_gru64 \
    tcn_gru_focused_final_hist128_c64_b4_k3_gru64 \
    tcn_gru_focused_final_hist160_c96_b3_k3_gru96 \
    tcn_gru_focused_final_hist128_c64_b2_k3_gru64 \
  --batch-size 512 \
  --device cuda:0
```

## Focused Sweep

Sampled-data results:

```text
config_id                                          RMSE      R2       fy_b R2  mz_b R2  hard avg  best_epoch
tcn_gru_focused_hist128_c96_b2_k3_gru64            1.143309  0.630218 0.190314 0.446703 0.318509  18
tcn_gru_focused_hist128_c64_b2_k3_gru128           1.145826  0.635386 0.215053 0.460314 0.337683   5
tcn_gru_focused_hist128_c128_b2_k3_gru64           1.147537  0.635879 0.221621 0.445509 0.333565   6
tcn_gru_focused_hist160_c64_b2_k3_gru64            1.147775  0.616320 0.187324 0.388343 0.287834  15
tcn_gru_focused_hist96_c64_b2_k3_gru64             1.157643  0.630152 0.201665 0.448455 0.325060   9
tcn_gru_focused_hist160_c96_b3_k3_gru96            1.161220  0.618729 0.156986 0.416987 0.286986  11
tcn_gru_focused_hist128_c64_b4_k3_gru64            1.163539  0.643168 0.213471 0.475658 0.344564  15
tcn_gru_focused_hist128_c64_b2_k3_gru96            1.170753  0.618503 0.174936 0.407206 0.291071   8
tcn_gru_focused_hist128_c64_b2_k3_gru64            1.172196  0.631198 0.210429 0.439458 0.324944  17
tcn_gru_focused_hist128_c64_b2_k5_gru64            1.174343  0.619227 0.194551 0.407874 0.301212  16
tcn_gru_focused_hist128_c64_b3_k3_gru64            1.178049  0.630099 0.200116 0.439099 0.319608  12
tcn_gru_focused_hist160_c128_b4_k3_gru128          1.220657  0.623275 0.188967 0.442351 0.315659   6
```

Selected for full-data final in this exploratory run:

```text
tcn_gru_focused_hist128_c96_b2_k3_gru64
  best sampled-data overall RMSE

tcn_gru_focused_hist128_c64_b4_k3_gru64
  best sampled-data hard-target average, using (fy_b R2 + mz_b R2) / 2

tcn_gru_focused_hist160_c96_b3_k3_gru96
  best sampled-data low-frequency diagnostic

tcn_gru_focused_hist128_c64_b2_k3_gru64
  compact reference from the earlier temporal-backbone final comparison
```

## Full-Data Final

```text
config_id                                           MAE      RMSE     R2       fx_b R2  fy_b R2  fz_b R2  mx_b R2  my_b R2  mz_b R2  hard avg  best_epoch
tcn_gru_focused_final_hist128_c64_b2_k3_gru64       0.484697 0.946922 0.714331 0.962178 0.336177 0.963950 0.557262 0.888844 0.577578 0.456877   9
tcn_gru_focused_final_hist160_c96_b3_k3_gru96       0.491552 0.962351 0.680689 0.964225 0.273797 0.962823 0.476461 0.879546 0.527284 0.400541   5
tcn_gru_focused_final_hist128_c64_b4_k3_gru64       0.502776 0.982580 0.684127 0.960629 0.270099 0.961183 0.498082 0.886480 0.528290 0.399195   8
tcn_gru_focused_final_hist128_c96_b2_k3_gru64       0.504114 0.992244 0.670616 0.961293 0.239991 0.960405 0.476161 0.875129 0.510718 0.375355   8
```

The compact `hist128 c64 b2 k3 gru64` model remains the best TCN+GRU after full-data training. The sampled-data winner did not transfer to the full-data final.

For paper-grade follow-up experiments, this selection should be repeated with validation-only ranking. Test metrics should be reported for locked final models, not used to choose them.

Compared with the earlier final Transformer:

```text
final_transformer_d64_l2_h4_hist128:
  RMSE 0.904964
  R2   0.718291
  fy_b R2 0.319529
  mz_b R2 0.573306

best TCN+GRU focused final:
  RMSE 0.946922
  R2   0.714331
  fy_b R2 0.336177
  mz_b R2 0.577578
```

The best TCN+GRU is about 4.6% worse than the Transformer in overall RMSE:

```text
(0.946922 - 0.904964) / 0.904964 = 4.64%
```

It still slightly beats the Transformer on `fy_b` and `mz_b`, but not enough to become the default accuracy model.

## Diagnostics

Worst log:

```text
config_id                                           worst_log                    R2       RMSE
tcn_gru_focused_final_hist128_c64_b2_k3_gru64       log_4_2026-4-12-17-43-30     0.549395 1.125236
tcn_gru_focused_final_hist128_c96_b2_k3_gru64       log_4_2026-4-12-17-43-30     0.446091 1.180728
tcn_gru_focused_final_hist128_c64_b4_k3_gru64       log_4_2026-4-12-17-43-30     0.502299 1.142825
tcn_gru_focused_final_hist160_c96_b3_k3_gru96       log_4_2026-4-12-17-43-30     0.479215 1.126087
```

Low-frequency and worst-regime diagnostics:

```text
config_id                                           low_freq R2  low_freq RMSE  worst_regime              bin             worst R2  worst RMSE
tcn_gru_focused_final_hist128_c64_b2_k3_gru64       0.428566     0.792530       cycle_flap_frequency_hz   (-0.001, 3.0]  0.428566  0.792530
tcn_gru_focused_final_hist128_c96_b2_k3_gru64       0.402172     0.784344       cycle_flap_frequency_hz   (-0.001, 3.0]  0.402172  0.784344
tcn_gru_focused_final_hist128_c64_b4_k3_gru64       0.376195     0.968365       cycle_flap_frequency_hz   (-0.001, 3.0]  0.376195  0.968365
tcn_gru_focused_final_hist160_c96_b3_k3_gru96       0.438917     0.755163       cycle_flap_frequency_hz   (-0.001, 3.0]  0.438917  0.755163
```

The `hist160 c96 b3 gru96` candidate is best within this focused final on the low-frequency bin, but it loses too much overall accuracy. It also does not beat the earlier causal GRU low-frequency R2 of `0.508076`.

## Inference Timing

Pure CUDA forward timing on NVIDIA GeForce RTX 4090. Timing excludes data loading, feature construction, and standardization.

```text
model                                  batch  history  params  ms/forward  us/sample
GRU hist64                             1      64        77830   0.066291    66.290910
GRU hist64                             512    64        77830   0.321023     0.626997
Transformer hist128                    1      128      112774   0.276420   276.420176
Transformer hist128                    512    128      112774   3.425141     6.689729
TCN hist128                            1      128      178694   0.182171   182.170807
TCN hist128                            512    128      178694   1.339997     2.617182
TCN+GRU compact hist128 c64 b2 g64     1      128       53196   0.159814   159.814063
TCN+GRU compact hist128 c64 b2 g64     512    128       53196   0.480676     0.938820
TCN+GRU c96 b2 g64                     1      128       79532   0.169984   169.984399
TCN+GRU c96 b2 g64                     512    128       79532   0.652599     1.274607
TCN+GRU c64 b4 g64                     1      128       77900   0.223524   223.523697
TCN+GRU c64 b4 g64                     512    128       77900   0.702943     1.372935
TCN+GRU hist160 c96 b3 g96             1      160      141324   0.201979   201.978557
TCN+GRU hist160 c96 b3 g96             512    160      141324   1.635410     3.194161
```

The compact TCN+GRU is materially faster than the Transformer and only moderately slower than the pure GRU. It is still slower than the pure GRU for single-sample deployment.

## Decision

This focused sweep did not produce a better TCN+GRU than the existing compact candidate.

Default recommendation:

```text
Use causal_transformer_paper_no_accel_v2_phase_actuator_airdata, history 128, as the accuracy-leading default.
Keep causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata, history 128, c64 b2 k3 gru64, as a hard-target ablation.
Do not promote larger TCN+GRU variants.
```

Practical interpretation:

```text
1. Increasing TCN width, TCN depth, GRU width, or history length did not improve the full-data final.
2. The sampled-data ranking was not reliable enough to select a larger TCN+GRU as final default.
3. TCN+GRU remains useful because it is faster than Transformer and slightly stronger on fy_b/mz_b.
4. The low-frequency regime remains unresolved. Larger TCN+GRU variants shift the tradeoff but do not solve it.
```

Next useful work:

```text
1. If continuing TCN+GRU, explore split-head or per-target loss weighting instead of simply scaling capacity.
2. If optimizing the default model, tune Transformer lightly because it is still the overall winner.
3. Keep low-frequency diagnostics as a gating metric, since global RMSE alone hides that failure mode.
```

## Artifacts

```text
artifacts/20260507_tcn_gru_focused_dryrun/
artifacts/20260507_tcn_gru_focused_smoke/
artifacts/20260507_tcn_gru_focused_sweep/temporal_backbone_screen_summary.csv
artifacts/20260507_tcn_gru_focused_final/temporal_backbone_screen_summary.csv
artifacts/20260507_tcn_gru_focused_final/runs/<config_id>/<recipe>/model_bundle.pt
artifacts/20260507_tcn_gru_focused_final/runs/<config_id>/<recipe>/diagnostics/per_log_metrics.csv
artifacts/20260507_tcn_gru_focused_final/runs/<config_id>/<recipe>/diagnostics/per_regime_metrics.csv
```
