# sg0p03 Residual Analysis and Correction

## Scope

This run uses the `sg0p03` clean-label split as the current effective-wrench candidate:

```text
dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1
```

The sequence was:

1. Build pre-correction aligned residual inputs using `sg0p03` labels and `artifacts/delaurier_physical_prior_v1`.
2. Run phase, frequency, and condition residual diagnostics before correction.
3. Train the phase-structured B+C correction model.
4. Build post-correction aligned residual inputs and rerun the same residual diagnostics.

## Artifacts

Pre-correction residual analysis:

```text
artifacts/20260528_sg0p03_pre_correction_residual_analysis
```

Correction model:

```text
artifacts/20260528_sg0p03_phase_structured_wrench_correction_v1
```

Post-correction residual analysis:

```text
artifacts/20260528_sg0p03_post_correction_residual_analysis
```

Resource snapshot:

```text
artifacts/20260528_sg0p03_residual_then_correction_resources.json
```

## Selected Correction Model

Validation-selected force model:

```text
family = phase_structured_plus_rates_controls
variant = affine
alpha = 1.0
```

Validation-selected moment model:

```text
form = force_arm_plus_free
feature_family = phase_structured_plus_rates_controls
alpha = 1.0
```

This is the same high-level B+C design selected on the raw-label run, but trained against the `sg0p03` labels.

## Test Metrics

Compared with the previous raw-label B+C result:

| label set | target | RMSE | MAE | bias | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| raw old | `fx_b` | 1.632 | 1.259 | 0.0216 | 0.886 |
| `sg0p03` | `fx_b` | 1.478 | 1.093 | 0.0380 | 0.901 |
| raw old | `fy_b` | 1.338 | 0.980 | -0.0482 | 0.143 |
| `sg0p03` | `fy_b` | 0.933 | 0.673 | -0.0538 | 0.277 |
| raw old | `fz_b` | 2.992 | 2.292 | -0.105 | 0.897 |
| `sg0p03` | `fz_b` | 2.733 | 2.064 | -0.120 | 0.911 |
| raw old | `force_mean` | 2.114 | 1.510 | -0.0439 | 0.642 |
| `sg0p03` | `force_mean` | 1.873 | 1.277 | -0.0452 | 0.696 |
| raw old | `mx_b` | 0.00267 | 0.00201 | -0.000123 | 0.352 |
| `sg0p03` | `mx_b` | 0.00255 | 0.00187 | 0.000018 | 0.230 |
| raw old | `my_b` | 0.00205 | 0.00161 | 0.000085 | 0.775 |
| `sg0p03` | `my_b` | 0.00189 | 0.00147 | 0.000098 | 0.801 |
| raw old | `mz_b` | 0.000525 | 0.000400 | -0.000009 | -0.051 |
| `sg0p03` | `mz_b` | 0.000468 | 0.000358 | -0.000008 | 0.009 |
| raw old | `moment_mean` | 0.00197 | 0.00134 | -0.000016 | 0.359 |
| `sg0p03` | `moment_mean` | 0.00185 | 0.00123 | 0.000036 | 0.347 |

The main improvement is in force channels. `fy_b` improves the most, consistent with the earlier diagnosis that raw lateral-force labels contain substantial low-SNR content.

## Residual Structure Before and After Correction

Held-out test residual RMSE:

| target | prior residual RMSE | post-correction residual RMSE | reduction |
| --- | ---: | ---: | ---: |
| `fx_b` | 5.267 | 1.478 | 0.719 |
| `fy_b` | 3.388 | 0.933 | 0.724 |
| `fz_b` | 12.869 | 2.733 | 0.788 |
| `mx_b` | 0.0713 | 0.00255 | 0.964 |
| `my_b` | 0.602 | 0.00189 | 0.997 |
| `mz_b` | 0.1019 | 0.000468 | 0.995 |

Phase-binned residual peak-to-peak reduction:

| target | pre-correction phase p2p | post-correction phase p2p | reduction |
| --- | ---: | ---: | ---: |
| `fx_b` | 14.55 | 3.16 | 0.783 |
| `fy_b` | 1.82 | 0.90 | 0.509 |
| `fz_b` | 28.17 | 4.82 | 0.829 |
| `mx_b` | 0.0268 | 0.00326 | 0.878 |
| `my_b` | 0.2267 | 0.00263 | 0.988 |
| `mz_b` | 0.0285 | 0.00064 | 0.978 |

Frequency diagnostics show that the uncorrected DeLaurier residual is dominated by flapping-structured components:

| target | dominant pre-correction component | dominant true-energy fraction | remaining fraction after correction |
| --- | --- | ---: | ---: |
| `fx_b` | flap main | 0.538 | 0.021 |
| `fy_b` | flap main | 0.527 | 0.009 |
| `fz_b` | flap main | 0.834 | 0.015 |
| `my_b` | flap main | 0.539 | 0.000011 |

Condition-binned diagnostics also show broad residual reduction across airspeed, dynamic pressure, angle of attack, and flapping frequency bins. For example, mean RMSE reduction over bins is approximately:

| condition | `fx_b` | `fy_b` | `fz_b` | `my_b` |
| --- | ---: | ---: | ---: | ---: |
| airspeed | 0.722 | 0.726 | 0.787 | 0.997 |
| alpha | 0.698 | 0.725 | 0.787 | 0.997 |
| flap frequency | 0.720 | 0.725 | 0.786 | 0.997 |
| dynamic pressure | 0.722 | 0.726 | 0.787 | 0.997 |

## Interpretation

The `sg0p03` labels preserve enough wingbeat structure for the residual diagnostics to remain meaningful, while improving both replay consistency and correction performance. The correction model removes most of the structured DeLaurier discrepancy, especially in `fx_b` and `fz_b`, and it improves the difficult `fy_b` channel relative to the raw-label run.

The moment numbers improve slightly in absolute error, but the moment story should still be written conservatively. The model is effectively fitting low-magnitude effective rotational wrench targets, and the previous rotational replay diagnostics have not fully closed.

Recommended paper use:

```text
Use sg0p03 as the main effective-wrench label reconstruction variant.
Report raw-label or sg0p06 sensitivity separately if space allows.
Use residual diagnostics to support structured simulator-prior mismatch, not closed-loop simulator validation.
```

