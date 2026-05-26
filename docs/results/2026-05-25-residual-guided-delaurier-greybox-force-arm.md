# Residual-Guided DeLaurier Grey-Box Force-Arm Results

## Summary

This experiment tested a three-stage path:

```text
DeLaurier prior
-> force-only recalibration
-> phase/frequency/AoA grey-box force correction
-> dynamic equivalent-arm moment head
```

The strongest result is in force prediction. A low-dimensional grey-box affine correction using phase, frequency, AoA, airspeed, dynamic pressure, and interactions reduced held-out force error substantially relative to the exported DeLaurier prior. The moment result is more mixed: dynamic arm models improve strongly over fixed-arm assumptions, but the best low-dimensional structured moment head is close to a direct low-dimensional moment baseline and still does not reach the previously trained temporal neural models.

Main interpretation:

```text
The residual-guided force correction is promising.
The force-arm structure is useful as a diagnostic and modeling prior.
A low-dimensional dynamic arm alone is not sufficient for high-quality moment prediction.
```

Artifacts:

```text
artifacts/20260525_delaurier_force_recalibration_v1
artifacts/20260525_delaurier_greybox_force_correction_v1
artifacts/20260525_dynamic_arm_moment_head_v1
```

## Stage 1: Force-Only Recalibration

Stage 1 fitted train-only affine wrappers on top of exported DeLaurier force predictions. This is not a true IsaacLab internal DeLaurier parameter re-export; it is an effective force-channel recalibration of the already exported prior.

Test force-mean results:

| model | force mean RMSE | force mean R2 | comment |
|---|---:|---:|---|
| current DeLaurier prior | 7.350 | -2.060 | exported calibrated prior |
| per-channel affine | 5.178 | 0.0066 | best Stage 1 variant |
| weighted shared gain+bias | 5.193 | -0.0039 | similar but slightly worse |

Per-channel test RMSE for the best Stage 1 variant:

| target | prior RMSE | affine RMSE | reduction |
|---|---:|---:|---:|
| fx_b | 5.420 | 4.790 | 11.6% |
| fy_b | 3.524 | 1.446 | 59.0% |
| fz_b | 13.105 | 9.298 | 29.0% |

Interpretation:

```text
Stage 1 removes large global scale/bias mismatch, but R2 remains near zero.
This suggests that the remaining force residual is structured rather than just a constant calibration error.
```

## Stage 2: Phase/Frequency/AoA Grey-Box Force Correction

Stage 2 trained low-dimensional residual-guided force corrections using the exported DeLaurier force plus features derived from wingbeat phase, flap frequency, AoA proxy, true airspeed, dynamic pressure, and interactions. The selected model was an affine correction chosen by validation force-mean RMSE.

Test force results:

| model | target | RMSE | R2 |
|---|---|---:|---:|
| prior | force_mean | 8.437 | -2.060 |
| grey-box affine | fx_b | 1.817 | 0.859 |
| grey-box affine | fy_b | 1.399 | 0.063 |
| grey-box affine | fz_b | 3.257 | 0.878 |
| grey-box affine | force_mean | 2.300 | 0.600 |

Interpretation:

```text
The residual is strongly correctable with phase/regime features.
This supports the idea that DeLaurier mismatch is not random; it is organized by wingbeat and flight condition.
```

This should still be described conservatively:

```text
residual-guided effective force correction
```

not:

```text
physical re-identification of DeLaurier internal aerodynamic coefficients
```

## Stage 3: Dynamic Equivalent-Arm Moment Head

Stage 3 trained ridge moment heads on the Stage 2 low-dimensional feature frame and compared fixed-arm, dynamic-arm, dynamic-arm plus free-moment, and direct moment baselines. Models were fit on train, selected by validation `moment_mean` RMSE, and reported on test. Force sources included diagnostic ground-truth effective force, raw DeLaurier prior force, and Stage 2 corrected force.

The selected model was `dynamic_arm_plus_free_linear` with diagnostic ground-truth effective force (`alpha=1000`), giving test `moment_mean` RMSE `0.002147` and mean channel R2 `0.00984`.

Test moment-mean ranking:

| model | force source | moment mean RMSE | moment mean R2 |
|---|---|---:|---:|
| dynamic arm + free | true force | 0.002147 | 0.0098 |
| dynamic arm only | true force | 0.002277 | -0.0486 |
| dynamic arm + free | Stage 2 corrected force | 0.002277 | 0.2718 |
| direct moment linear | features only | 0.002291 | 0.3104 |
| dynamic arm only | Stage 2 corrected force | 0.002342 | 0.1935 |
| dynamic arm + free | prior force | 0.002529 | -0.2593 |
| fixed arm | true force | 0.003072 | -0.0187 |
| fixed arm | Stage 2 corrected force | 0.003081 | -0.0400 |
| fixed arm | prior force | 0.003162 | -0.0007 |

Selected model channel metrics:

| target | RMSE | R2 |
|---|---:|---:|
| mx_b | 0.002937 | 0.219 |
| my_b | 0.002166 | 0.749 |
| mz_b | 0.000713 | -0.938 |

For the selected diagnostic ground-truth force model on test, `|r_hat|` median/p90/p99 were:

```text
0.000741 / 0.001494 / 0.001979 m
```

The free-moment term was moderate for the selected ground-truth-force model:

```text
tau_free_fraction_of_predicted = 0.092
tau_free_fraction_of_arm_plus_tau = 0.109
```

For Stage 2 corrected force, dynamic arm plus free moment improved clearly over fixed arm and prior-force variants, but it did not beat the direct low-dimensional moment baseline by R2.

Interpretation:

```text
The force-arm structure is much better than a fixed-arm assumption.
Using corrected force improves dynamic-arm moment modeling versus raw prior force.
However, low-dimensional dynamic-arm moment modeling is not yet strong enough to replace direct temporal moment modeling.
```

## Paper Takeaway

The strongest paper claim from this batch is not that a full force-arm moment model is solved. The stronger and safer claim is:

> Residual diagnostics reveal that the DeLaurier force prior has systematic phase- and regime-dependent mismatch. A low-dimensional grey-box correction using wingbeat phase, flap frequency, AoA, and airspeed-related features substantially improves held-out force prediction. Because fixed-arm moment reconstruction remains weak, moment channels should be modeled as part of a full effective wrench, with dynamic force-arm structure treated as a useful inductive bias rather than a complete replacement for direct temporal moment prediction.

Recommended next step:

```text
Use the Stage 2 corrected force as a force prior in a temporal model.
Keep dynamic arm + free moment as a structured head candidate, but compare it against the best Transformer/Head-FiLM moment models.
```
