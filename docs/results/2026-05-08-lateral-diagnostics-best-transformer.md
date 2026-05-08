# Lateral Diagnostics for Best Transformer

Date: 2026-05-08

Model bundle:

```text
artifacts/20260507_transformer_focused_final/runs/transformer_focused_final_hist128_d64_l2_h4_do050/causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt
```

Split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1/test_samples.parquet
```

Diagnostics output:

```text
artifacts/20260507_lateral_diagnostics_best_transformer
```

## Summary

The lateral targets are genuinely harder than the longitudinal/reference targets, but the test result is strongly affected by one suspect log:

```text
log_4_2026-4-12-17-43-30
```

With all four test logs, lateral mean R2 is 0.5234. Excluding the suspect log, lateral mean R2 rises to 0.6270. The largest single improvement is `fy_b`, which rises from 0.3435 to 0.4944.

## Target Scale

| target | RMSE/std | R2 |
| --- | ---: | ---: |
| fy_b | 0.8103 | 0.3435 |
| mz_b | 0.6279 | 0.6058 |
| mx_b | 0.6157 | 0.6209 |
| my_b | 0.2994 | 0.9104 |
| fx_b | 0.1765 | 0.9688 |
| fz_b | 0.1761 | 0.9690 |

The main lateral bottleneck is `fy_b`. `mx_b` and `mz_b` are worse than `my_b`, but they are not as pathological as `fy_b`.

## Suspect Log

| case | samples | logs | lateral mean R2 | fy_b R2 | mx_b R2 | mz_b R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 60478 | 4 | 0.5234 | 0.3435 | 0.6209 | 0.6058 |
| without_suspect | 42757 | 3 | 0.6270 | 0.4944 | 0.7144 | 0.6722 |
| suspect_only | 17721 | 1 | 0.2834 | -0.0184 | 0.4452 | 0.4235 |

The suspect log should not be silently removed from the dataset yet. It should be marked and reported separately because it may represent a real hard regime or a data-quality issue. Earlier inspection showed that this log has a high positive-rudder fraction, strong negative `elevon_diff`, and about 40% missing `true_airspeed` values in the raw test parquet.

Recommended handling:

1. Keep the log in the canonical split for now.
2. Report both all-test and without-suspect metrics.
3. Inspect the source airspeed/merge path for this log before deciding whether to filter it.

## Regime Findings

The worst Batch 1 regime bins were:

| regime | bin | samples | lateral mean R2 | fy_b RMSE |
| --- | --- | ---: | ---: | ---: |
| phase_corrected_rad | [0.0, 1.571) | 18723 | 0.5691 | 1.2994 |
| servo_rudder | [0.05, 1.0) | 24986 | 0.3939 | 1.2904 |
| elevon_diff | [-1.0, -0.05) | 24940 | 0.4715 | 1.2742 |
| phase_corrected_rad | [4.712, 6.283) | 11804 | 0.4534 | 1.2726 |

These bins should not be interpreted as independent causes. The positive-rudder and negative-`elevon_diff` bins are heavily confounded with the suspect log. After removing that log, their `fy_b` performance improves substantially, so the dominant explanation is suspect-log/OOD composition rather than a clean universal failure of those control regimes.

## Phase/Lag Diagnostics

Batch 2 found only one meaningful lag case:

| log_id | target | best lag | zero-lag corr | best corr | zero-lag RMSE | best RMSE | RMSE improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| log_4_2026-4-12-17-43-30 | fy_b | 1 | 0.2909 | 0.3308 | 1.4603 | 1.4251 | 0.0352 |

Positive lag means the prediction lags the target and would need to be shifted earlier. The effect is small: it slightly improves the suspect log `fy_b`, but it does not explain the lateral degradation overall. Other target/log pairs prefer zero lag.

## Residual Correlations

The largest absolute residual correlations are weak:

| target | feature | corr | abs corr |
| --- | --- | ---: | ---: |
| mx_b | phase_corrected_cos | 0.0864 | 0.0864 |
| mz_b | vehicle_angular_velocity.xyz[1] | -0.0845 | 0.0845 |
| fy_b | vehicle_angular_velocity.xyz[0] | 0.0824 | 0.0824 |
| mz_b | vehicle_angular_velocity.xyz[2] | 0.0580 | 0.0580 |
| fy_b | phase_corrected_sin | -0.0557 | 0.0557 |

This does not support a simple story that one missing scalar feature is causing the error. The residuals are not strongly linearly explained by rudder, elevon, phase, airspeed, or angular velocity alone.

## Interpretation

Current evidence points to:

1. A real `fy_b` bottleneck.
2. One suspect log strongly depressing test metrics.
3. Some regime/OOD interaction, especially positive rudder and negative `elevon_diff`, but much of that is confounded with the suspect log.
4. No strong global phase lag; only suspect-log `fy_b` has a small 1-sample lag improvement.
5. No strong single-feature residual correlation.

The next practical step is not to immediately delete the suspect log. Instead, keep all-test metrics, add without-suspect metrics to model comparisons, and inspect the source airspeed/label construction for `log_4_2026-4-12-17-43-30`.
