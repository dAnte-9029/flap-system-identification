# DeLaurier Residual NN GPU Results

Date: 2026-05-13

## Purpose

This run repeats the first DeLaurier-residual neural-network experiment on GPU. The residual target is

```text
residual = log-derived effective wrench - physically calibrated DeLaurier prior
```

The final prediction is

```text
combined = DeLaurier prior + residual NN prediction
```

The physics prior was exported from IsaacLab and the residual learning was performed in this repository.

## Root Cause Of The CPU Run

The previous smoke result was not caused by missing CUDA support. The `flap-train-gpu` environment reports CUDA support, and the machine has two RTX 4090 GPUs. The old MLP artifact recorded:

```text
artifacts/20260513_delaurier_residual_nn_v1/residual_mlp_smoke/training_config.json
device: cpu
resolved_device_type: cpu
```

So the issue was the explicit run configuration. The GPU rerun used `CUDA_VISIBLE_DEVICES=0` and `--device cuda`.

## Data And Prior

Residual split:

```text
dataset/delaurier_residual_physical_v1
```

Source effective-wrench split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
```

Prior predictions:

```text
artifacts/delaurier_physical_prior_v1
prior_name: delaurier_physical_calibrated_v1
```

Row counts:

```text
train: 308702
val:    79587
test:   60671
```

## GPU Runs

### Residual MLP

Output:

```text
artifacts/20260513_delaurier_residual_nn_v1/residual_mlp_gpu
artifacts/20260513_delaurier_residual_nn_v1/residual_mlp_gpu_combined_eval
```

Configuration:

```text
model_type: mlp
feature_set: paper_no_accel_v2
hidden_sizes: 128,128
loss: Huber(delta=1.5)
max_epochs: 20
best_epoch: 14
device: cuda
resolved_device_type: cuda
use_amp: true
```

### Residual Transformer

Output:

```text
artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu
artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_combined_eval
```

Configuration:

```text
model_type: causal_transformer
feature_set: paper_no_accel_v2
sequence_history_size: 128
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
d_model: 64
layers: 2
heads: 4
dropout: 0.05
loss: Huber(delta=1.5)
lr_scheduler: warmup_cosine
gradient_clip_norm: 1.0
ema_decay: 0.999
max_epochs: 50
best_epoch: 38
device: cuda
resolved_device_type: cuda
use_amp: true
```

The Transformer aligned test set has fewer rows (`58639`) than the pointwise MLP (`60671`) because the causal sequence model drops samples that do not have enough preceding history within each log/segment.

## Held-Out Test Summary

Overall metrics are computed on each model's aligned held-out test rows. For the Transformer, the prior-only numbers are recomputed on the same sequence-aligned subset.

| Method | Test rows | Prior RMSE | Combined RMSE | Prior MAE | Combined MAE |
|---|---:|---:|---:|---:|---:|
| Physical DeLaurier prior + residual MLP | 60671 | 5.970957 | 2.480561 | 2.962202 | 1.285911 |
| Physical DeLaurier prior + residual Transformer | 58639 | 5.965060 | 0.852140 | 2.960602 | 0.437877 |

## Per-Target Test Metrics

### Residual MLP

| Target | Prior RMSE | Combined RMSE | Prior R2 | Combined R2 |
|---|---:|---:|---:|---:|
| fx_b | 5.420306 | 1.349492 | -0.258099 | 0.922016 |
| fy_b | 3.523583 | 2.514538 | -4.942126 | -2.026141 |
| fz_b | 13.105000 | 5.362723 | -0.980166 | 0.668412 |
| mx_b | 0.071418 | 0.006611 | -461.109517 | -2.959132 |
| my_b | 0.601734 | 0.110672 | -19347.292284 | -653.494481 |
| mz_b | 0.101904 | 0.063178 | -39635.358722 | -15234.092566 |

### Residual Transformer

| Target | Prior RMSE | Combined RMSE | Prior R2 | Combined R2 |
|---|---:|---:|---:|---:|
| fx_b | 5.416479 | 0.769239 | -0.259721 | 0.974592 |
| fy_b | 3.521280 | 1.181126 | -4.936699 | 0.332062 |
| fz_b | 13.091176 | 1.539424 | -0.984088 | 0.972564 |
| mx_b | 0.071568 | 0.004276 | -464.362433 | -0.661440 |
| my_b | 0.599690 | 0.014390 | -19229.937121 | -10.073546 |
| mz_b | 0.101846 | 0.004522 | -39893.353076 | -77.654784 |

## Interpretation Notes

The residual formulation removes the unfair framing of comparing a data-trained effective-wrench model directly against a hand-built DeLaurier model. The DeLaurier model remains the physics prior, and the NN is evaluated as a correction term on held-out logs.

The GPU MLP reproduces the previous CPU smoke result within small numerical differences, so the earlier result should be treated as valid numerically but replaced by the GPU artifact for consistency and speed.

The residual Transformer is the stronger first residual model. It reduces the combined overall test RMSE from about `5.97` for the physical prior to `0.85` on its aligned test subset.

Moment-channel R2 can remain negative even when RMSE is small, because the held-out moment labels have very small variance relative to the DeLaurier prior error. For these channels, MAE/RMSE and trajectory-level consequences are more interpretable than R2 alone.

## Phase-Locked Residual Analysis

A first residual-interpretability check was run on the residual Transformer aligned test predictions:

```text
artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_phase_analysis
```

The analysis bins the true DeLaurier residual, predicted residual, and remaining residual by corrected wingbeat phase. The key outputs are:

```text
phase_binned_residuals.csv
phase_residual_summary.csv
phase_residual_medians.png
phase_residual_medians.pdf
```

| Target | True residual bias | True residual RMSE | Remaining RMSE | RMSE reduction | Phase R2 of true residual | True phase peak-to-peak | Remaining phase peak-to-peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| fx_b | 1.548 | 5.416 | 0.769 | 0.858 | 0.817 | 15.103 | 0.428 |
| fy_b | -0.221 | 3.521 | 1.181 | 0.665 | 0.020 | 2.181 | 0.860 |
| fz_b | 1.257 | 13.091 | 1.539 | 0.882 | 0.509 | 28.793 | 0.769 |
| mx_b | 0.0147 | 0.0716 | 0.0043 | 0.940 | -0.014 | 0.0280 | 0.0019 |
| my_b | 0.533 | 0.600 | 0.0144 | 0.976 | 0.052 | 0.2266 | 0.0031 |
| mz_b | 0.0137 | 0.1018 | 0.0045 | 0.956 | 0.004 | 0.0297 | 0.0016 |

The most interpretable phase-locked model discrepancy is in the longitudinal force channels. For `fx_b`, phase-bin medians explain about `0.817` of the true residual variance, and the phase-conditioned median residual has a peak-to-peak amplitude of about `15.1`. For `fz_b`, phase-bin medians explain about `0.509` of the residual variance, with a peak-to-peak amplitude of about `28.8`. In both channels, the residual Transformer closely follows the phase-conditioned residual pattern, reducing the remaining phase peak-to-peak amplitude to less than `0.8`.

The moment channels should be interpreted differently. `my_b` has a large positive DeLaurier residual bias (`0.533`) and a low phase R2 (`0.052`), so its dominant correction is closer to a bias/scale correction than a strongly phase-locked wingbeat shape. `mx_b` and `mz_b` have very small absolute amplitudes, so R2 values are unstable; their RMSE and peak-to-peak reductions are more useful than their R2 values.

This supports a bounded paper claim: the calibrated DeLaurier prior leaves a systematic wingbeat-phase-dependent discrepancy in the main force channels, especially `fx_b` and `fz_b`. The residual Transformer is not only reducing aggregate RMSE; it is also removing a repeatable phase-conditioned model mismatch. This does not uniquely identify the missing physics as one mechanism, because the residual can include unsteady aerodynamics, wing/actuator nonidealities, parameter mismatch, and label-side reconstruction error.

## Flight-Condition Residual Analysis

A second residual-interpretability check bins the aligned residual Transformer test predictions by flight-condition variables:

```text
artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_condition_analysis
```

The analysis uses five quantile bins per condition variable and reports DeLaurier residual RMSE versus remaining RMSE after the residual Transformer. The key outputs are:

```text
condition_residual_bins.csv
condition_residual_summary.csv
condition_residual_rmse_key_targets.png
condition_residual_rmse_key_targets.pdf
```

For the main force and pitching-moment channels:

| Condition | Target | True RMSE min | True RMSE max | Max/min | Worst bin median | Worst-bin true RMSE | Worst-bin remaining RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| true airspeed | fx_b | 4.686 | 6.575 | 1.40 | 6.51 | 6.575 | 0.844 |
| true airspeed | fz_b | 12.854 | 13.982 | 1.09 | 6.51 | 13.982 | 1.600 |
| true airspeed | my_b | 0.527 | 0.659 | 1.25 | 9.76 | 0.659 | 0.013 |
| dynamic pressure | fx_b | 4.669 | 6.570 | 1.41 | 23.96 | 6.570 | 0.846 |
| dynamic pressure | fz_b | 12.943 | 13.987 | 1.08 | 23.96 | 13.987 | 1.600 |
| dynamic pressure | my_b | 0.528 | 0.658 | 1.25 | 53.89 | 0.658 | 0.013 |
| angle of attack | fx_b | 3.233 | 7.145 | 2.21 | 0.230 | 7.145 | 0.880 |
| angle of attack | fz_b | 11.818 | 14.809 | 1.25 | 0.447 | 14.809 | 1.578 |
| angle of attack | my_b | 0.542 | 0.644 | 1.19 | 0.391 | 0.644 | 0.012 |
| flap frequency | fx_b | 4.127 | 6.735 | 1.63 | 4.92 | 6.735 | 1.084 |
| flap frequency | fz_b | 10.140 | 15.466 | 1.53 | 4.92 | 15.466 | 1.965 |
| flap frequency | my_b | 0.532 | 0.663 | 1.24 | 4.62 | 0.663 | 0.017 |

The strongest condition dependence in this held-out test set appears in `fx_b` versus angle of attack: the DeLaurier residual RMSE changes by a factor of `2.21` across the tested angle-of-attack bins, with the largest `fx_b` residual at the lowest angle-of-attack bin. The `fz_b` residual shows the opposite trend with angle of attack, increasing toward the highest angle-of-attack bin. Flap frequency is also important: both `fx_b` and `fz_b` residual RMSE increase toward the highest flap-frequency bin, reaching `6.735` and `15.466`, respectively.

Airspeed and dynamic pressure show similar trends because they are strongly related in this dataset. The lowest true-airspeed bin includes low/uncertain airspeed values, so dynamic pressure is the cleaner interpretation of that trend. Within the tested safe flight envelope, the DeLaurier prior is least accurate in low-dynamic-pressure/low-airspeed portions for `fx_b`, high-flap-frequency portions for both `fx_b` and `fz_b`, and high-angle-of-attack portions for `fz_b`.

The residual Transformer reduces the residual RMSE in every reported bin. The reduction remains large even in the worst bins, for example `fx_b` at the lowest angle-of-attack bin is reduced from `7.145` to `0.880`, and `fz_b` at the highest flap-frequency bin is reduced from `15.466` to `1.965`. This supports an applicability-boundary claim: even inside the safe flight envelope used for the experiments, the calibrated quasi-steady DeLaurier prior has condition-dependent mismatch, while the residual model acts as a real-flight correction over those regimes.

## Frequency-Domain Residual Analysis

A third residual-interpretability check decomposes the aligned test residuals into diagnostic frequency bands:

```text
artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_frequency_analysis
```

The analysis computes one-sided FFT energy for each log/segment after subtracting the segment mean. Therefore, this frequency analysis describes oscillatory residual content; constant bias terms are handled by the phase and condition analyses above. Frequency bins are assigned to non-overlapping diagnostic bands in this order: `0-1 Hz`, `1-3 Hz`, wingbeat fundamental `1f`, second harmonic `2f`, third harmonic `3f`, and `8-25 Hz` broadband excluding the structured bands. The key outputs are:

```text
frequency_residual_energy.csv
frequency_residual_summary.csv
frequency_residual_energy_key_targets.png
frequency_residual_energy_key_targets.pdf
```

Summary by target:

| Target | Dominant residual component | Dominant true energy fraction | Dominant remaining / true band energy | Dominant band reduction | Structured true energy fraction | Broadband high true energy fraction |
|---|---|---:|---:|---:|---:|---:|
| fx_b | 1f | 0.559 | 0.0020 | 0.998 | 0.870 | 0.068 |
| fy_b | 1f | 0.526 | 0.0045 | 0.996 | 0.837 | 0.059 |
| fz_b | 1f | 0.884 | 0.0012 | 0.999 | 0.938 | 0.027 |
| mx_b | 0-1 Hz | 0.824 | 0.0013 | 0.999 | 0.975 | 0.007 |
| my_b | 1f | 0.506 | 0.0011 | 0.999 | 0.960 | 0.009 |
| mz_b | 1f | 0.508 | 0.0007 | 0.999 | 0.958 | 0.016 |

For the main force channels, the residual left by DeLaurier is strongly organized around wingbeat frequency. In `fx_b`, about `55.9%` of the demeaned residual energy is at the wingbeat fundamental and another `26.9%` is at the second harmonic; only about `6.8%` is in the high-frequency broadband band. In `fz_b`, the fundamental is even more dominant, carrying about `88.4%` of the demeaned residual energy.

The residual Transformer mostly removes these repeatable wingbeat-synchronous components. In the dominant `1f` band, the remaining energy is about `0.20%` of the original band energy for `fx_b`, `0.12%` for `fz_b`, and `0.11%` for `my_b`. The broadband high-frequency component is reduced less strongly than the structured wingbeat bands, which is expected if that band contains more sensor/reconstruction noise and less repeatable aerodynamics.

This frequency result supports the same interpretation as the phase analysis from a different view: the residual model is not merely fitting an arbitrary offset. It removes a systematic wingbeat-synchronous discrepancy left by the calibrated DeLaurier prior. The frequency analysis alone does not identify the missing physical mechanism, but it narrows the discrepancy to repeatable components tied to the flapping cycle and its harmonics.

## Date-Stratified 5-Fold Result

The selected paper-facing model was rerun as a five-fold whole-log cross-validation experiment:

```text
dataset/delaurier_residual_physical_v1_log_kfold5_date_stratified_v1
artifacts/20260513_delaurier_residual_nn_kfold5_date_stratified_v1
```

The folds are approximately date-stratified and sample-balanced at the log level. The date `2026-4-12` has only three logs, so it cannot appear in the test split of all five folds. The split manifest and each fold's `date_coverage.csv` record this limitation explicitly. Each fold uses one log fold as test, the next fold as validation, and the remaining folds as training data. The residual Transformer configuration is the same as the single-split run above.

Overall held-out test results:

| Fold | Aligned rows | Prior RMSE | Prior + residual Transformer RMSE | Prior MAE | Prior + residual Transformer MAE |
|---|---:|---:|---:|---:|---:|
| fold_00 | 91668 | 5.925 | 0.824 | 2.938 | 0.426 |
| fold_01 | 75853 | 5.838 | 0.784 | 2.899 | 0.402 |
| fold_02 | 88388 | 5.844 | 0.830 | 2.912 | 0.426 |
| fold_03 | 86872 | 5.913 | 0.858 | 2.931 | 0.441 |
| fold_04 | 91924 | 5.896 | 0.841 | 2.913 | 0.437 |
| mean ± std | 86941 ± 6562 | 5.883 ± 0.040 | 0.827 ± 0.027 | 2.918 ± 0.016 | 0.426 ± 0.015 |

Per-target five-fold summary:

| Target | Prior RMSE | Prior + residual Transformer RMSE | Mean RMSE reduction | Prior MAE | Prior + residual Transformer MAE |
|---|---:|---:|---:|---:|---:|
| fx_b | 5.379 ± 0.091 | 0.777 ± 0.044 | 0.855 | 3.826 | 0.592 |
| fy_b | 3.443 ± 0.062 | 1.042 ± 0.034 | 0.697 | 2.728 | 0.777 |
| fz_b | 12.904 ± 0.099 | 1.554 ± 0.056 | 0.880 | 10.295 | 1.176 |
| mx_b | 0.0658 ± 0.0029 | 0.0035 ± 0.0004 | 0.946 | 0.0467 | 0.0025 |
| my_b | 0.5907 ± 0.0233 | 0.0113 ± 0.0015 | 0.981 | 0.5327 | 0.0078 |
| mz_b | 0.1020 ± 0.0022 | 0.0041 ± 0.0005 | 0.960 | 0.0826 | 0.0028 |

This cross-validation result is the strongest aggregate evidence for the residual formulation. Across date-stratified held-out log folds, the calibrated DeLaurier prior is stable at about `5.88` overall RMSE, while the residual-corrected model is stable at about `0.83` overall RMSE. The low fold-to-fold standard deviation indicates that the improvement is not concentrated in a single original test split.
