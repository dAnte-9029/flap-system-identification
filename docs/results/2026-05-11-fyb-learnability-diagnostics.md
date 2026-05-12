# fy_b Learnability Diagnostics

Date: 2026-05-11

## Setup

- Model: Head FiLM Transformer no-suspect candidate
- Aligned predictions:
  - `artifacts/20260511_head_film_prediction_curves_all_splits/val/aligned_val_predictions.parquet`
  - `artifacts/20260511_head_film_prediction_curves_all_splits/test/aligned_test_predictions.parquet`
- Diagnostics output: `artifacts/20260511_fyb_learnability_diagnostics/`
- Script: `scripts/diagnose_fyb_learnability.py`

## Label Source

`fy_b` is not directly measured. It is part of the effective body-frame force label derived from inertial acceleration:

```text
acc_n = vehicle_local_position.ax/ay/az
specific_acc_n = acc_n - gravity_n
force_b = mass_kg * R_nb^T * specific_acc_n
fy_b = force_b[:, 1]
```

See `src/system_identification/pipeline.py`, `_compute_effective_wrench_labels`, lines 341-381.

This means `fy_b` can contain acceleration-estimation noise, frame-rotation errors, synchronization errors, and boundary artifacts. It is a derived target, not a direct aerodynamic force sensor.

## Frequency-Domain Result

Band-limited R2 shows that `fy_b` is partially learnable:

| split | raw R2 | 0-1 Hz R2 | 1-3 Hz R2 | flap-main R2 | 8-25 Hz R2 |
|---|---:|---:|---:|---:|---:|
| val | 0.359 | 0.775 | 0.501 | 0.414 | 0.480 |
| test | 0.515 | 0.755 | 0.625 | 0.848 | 0.577 |

Interpretation:

- The low-frequency component is well learned.
- The test split also shows strong tracking around the flapping fundamental frequency.
- The high-frequency band contains a large share of target variance and is only moderately captured.
- Therefore the low raw `fy_b` R2 is not simply because the model fails completely; it is partly because raw `fy_b` contains difficult high-frequency content.

PSD plots:

- `artifacts/20260511_fyb_learnability_diagnostics/val_fy_b_psd.png`
- `artifacts/20260511_fyb_learnability_diagnostics/test_fy_b_psd.png`

The PSD peaks are near the flapping frequency and harmonics. The target has strong energy near about 8.5 Hz and 22 Hz; the prediction follows the main structure but with lower amplitude at high-frequency peaks. The updated PSD plots include the residual spectrum.

Residual PSD peaks are mostly at higher frequencies:

| split | strongest residual PSD regions |
|---|---|
| val | around 26.6-27.9 Hz, plus a smaller peak near 4.3 Hz |
| test | around 18 Hz, 21.7-22.7 Hz, 26.6-28.3 Hz, and 9 Hz |

This suggests the model captures part of the wingbeat-periodic structure, while the remaining error contains broader high-frequency content.

## Filtered R2

Low-pass filtering both target and prediction increases R2 substantially:

| split | raw R2 | lowpass 1 Hz | lowpass 3 Hz | lowpass 5 Hz | lowpass 8 Hz | lowpass 12 Hz |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.359 | 0.775 | 0.739 | 0.595 | 0.618 | 0.617 |
| test | 0.515 | 0.755 | 0.736 | 0.783 | 0.788 | 0.764 |

This supports the view that the model learns the smoother / structured part of `fy_b` better than the raw signal score suggests.

Median filtering gives smaller gains than FFT low-pass:

| split | rolling median 0.05 s | 0.10 s | 0.20 s |
|---|---:|---:|---:|
| val | 0.311 | 0.366 | 0.476 |
| test | 0.495 | 0.516 | 0.572 |

So the problem is not only isolated one-sample spikes. There is broader high-frequency/oscillatory content.

## Spike Capture

For the largest `fy_b` magnitudes, the model usually predicts the correct sign but underestimates amplitude:

| split | top abs quantile | true abs mean | pred abs mean | amplitude ratio | sign agreement | R2 |
|---|---:|---:|---:|---:|---:|---:|
| val | 90% | 2.96 | 1.38 | 0.47 | 0.889 | 0.555 |
| val | 95% | 3.59 | 1.63 | 0.45 | 0.915 | 0.558 |
| val | 99% | 5.28 | 2.43 | 0.46 | 0.953 | 0.444 |
| test | 90% | 3.27 | 1.59 | 0.49 | 0.964 | 0.634 |
| test | 95% | 3.99 | 1.83 | 0.46 | 0.981 | 0.597 |
| test | 99% | 5.90 | 2.31 | 0.39 | 0.991 | 0.328 |

Interpretation:

- The spikes are not pure random noise; sign/timing is often correct.
- The amplitude is systematically compressed.
- This is consistent with a model trained with regression loss on noisy/high-variance peaks: it learns the conditional mean and avoids chasing extreme amplitudes.

## Phase-Locked High-Frequency Structure

To test whether high-frequency `fy_b` is phase-locked rather than random, `HPF(fy_b)` was averaged by wingbeat phase using 36 bins.

| split | true HPF phase-average peak-to-peak | pred HPF phase-average peak-to-peak |
|---|---:|---:|
| val | 2.17 | 1.97 |
| test | 2.70 | 1.91 |

The high-pass component does not vanish after phase averaging. The strongest repeated phase regions occur near:

```text
phase bin centers around 0.44 rad, 1.31-1.48 rad, and 5.32-5.50 rad
```

Interpretation:

- There is phase-locked high-frequency structure in `fy_b`.
- The prediction also shows phase-locked structure, but with compressed amplitude.
- Therefore it is too strong to say "all high-frequency fy_b is noise."

## High-Frequency Correlation With Inputs / States

Simple HPF correlation with available signals is weak:

| split | strongest correlations with HPF(true fy_b) |
|---|---|
| val | phase sin `0.143`, yaw rate `0.119`, roll rate `0.110`, lateral acceleration `0.075` |
| test | roll rate `0.162`, phase sin `0.159`, phase cos `0.113`, left elevon `0.104`, yaw rate `0.092` |

These are not zero, but they are not strong enough to explain the high-frequency component alone. This is consistent with a mixture of:

```text
phase-locked flapping-induced side-force content
+ weakly observed lateral-directional dynamics
+ broadband low-confidence transients / label noise
```

## Current Conclusion

`fy_b` is not hopeless, but the raw target is much less clean than `fx_b/fz_b/my_b`.

Best current interpretation:

```text
The model learns the structured low-frequency and wingbeat-periodic part of fy_b, but underestimates high-frequency and large-amplitude side-force excursions. The remaining error is likely a mixture of real lateral-directional dynamics not fully observed by the current inputs and derivative/acceleration noise in the force label.
```

This points more toward **label/noise-aware treatment** than simply making the backbone larger.

The strongest defensible wording is:

> High-frequency components of `fy_b` include both phase-locked flapping-induced aerodynamic content and broadband low-confidence transients. The current model captures the structured periodic component but compresses large-amplitude excursions and leaves a high-frequency residual.

## Suggested Next Steps

1. Report both raw `fy_b` and filtered/band-limited `fy_b` diagnostics.
2. Try label-side smoothing or robust target construction for `fy_b`, not only model-side changes.
3. Consider target weighting or asymmetric loss only after deciding whether large `fy_b` spikes are physically meaningful.
4. For control-oriented reporting, emphasize `fy_b` as a difficult/noisy side-force channel rather than the core control channel.
5. Run a separate target-generation sensitivity experiment with multiple acceleration/derivative smoothing settings.
