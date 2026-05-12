# fy_b Target Generation Sensitivity

Date: 2026-05-11

## Setup

- Split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Metadata: `metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Script: `scripts/diagnose_target_generation_sensitivity.py`
- Output: `artifacts/20260511_fyb_target_generation_sensitivity/`

This checks whether the `fy_b` label changes strongly when the force label is recomputed from different acceleration estimates.

The baseline `fy_b` is the existing parquet label. The variants are:

```text
raw_recomputed: recompute force_b from existing vehicle_local_position.ax/ay/az
sg_0p06_p2: recompute acceleration from smoothed velocity, Savitzky-Golay window 0.06 s
sg_0p12_p2: recompute acceleration from smoothed velocity, Savitzky-Golay window 0.12 s
sg_0p20_p2: recompute acceleration from smoothed velocity, Savitzky-Golay window 0.20 s
sg_0p30_p2: recompute acceleration from smoothed velocity, Savitzky-Golay window 0.30 s
```

Important: this is not model-side low-pass filtering. It is label-side target reconstruction sensitivity.

## Results

| split | variant | corr with raw | RMSE vs raw | raw std | variant std | p99 abs raw | p99 abs variant | top 1% overlap | highpass energy >8 Hz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | raw_recomputed | 1.000 | 0.000 | 1.385 | 1.385 | 4.273 | 4.273 | 1.000 | 0.816 |
| val | sg_0p06_p2 | 0.468 | 1.224 | 1.385 | 0.669 | 4.273 | 1.504 | 0.117 | 0.248 |
| val | sg_0p12_p2 | 0.384 | 1.279 | 1.385 | 0.509 | 4.273 | 1.135 | 0.126 | 0.037 |
| val | sg_0p20_p2 | 0.251 | 1.345 | 1.385 | 0.444 | 4.273 | 1.060 | 0.118 | 0.034 |
| val | sg_0p30_p2 | 0.199 | 1.367 | 1.385 | 0.420 | 4.273 | 1.103 | 0.118 | 0.007 |
| test | raw_recomputed | 1.000 | 0.000 | 1.484 | 1.484 | 4.804 | 4.804 | 1.000 | 0.823 |
| test | sg_0p06_p2 | 0.491 | 1.294 | 1.484 | 0.677 | 4.804 | 1.658 | 0.118 | 0.291 |
| test | sg_0p12_p2 | 0.413 | 1.357 | 1.484 | 0.493 | 4.804 | 1.099 | 0.111 | 0.054 |
| test | sg_0p20_p2 | 0.249 | 1.439 | 1.484 | 0.416 | 4.804 | 0.953 | 0.113 | 0.029 |
| test | sg_0p30_p2 | 0.177 | 1.471 | 1.484 | 0.401 | 4.804 | 0.979 | 0.125 | 0.007 |

Full CSV:

```text
artifacts/20260511_fyb_target_generation_sensitivity/fy_b_target_generation_sensitivity.csv
```

## Interpretation

The `raw_recomputed` result exactly matches the existing parquet label. That is a useful sanity check: the diagnostic is using the same effective wrench formula as the preprocessing pipeline.

The smoothed-derivative variants change `fy_b` substantially:

- Correlation with the raw target drops to about `0.47-0.49` even with a short `0.06 s` smoothing window.
- The 99th-percentile absolute `fy_b` drops from about `4.3-4.8 N` to about `1.5-1.7 N` for `0.06 s`, and to about `1.0-1.1 N` for `0.12-0.30 s`.
- Only about `11-13%` of the raw top-1% large-`fy_b` samples remain top-1% samples after smoothing.
- Raw `fy_b` has about `82%` of its spectral energy above `8 Hz`. With `0.12 s` smoothing this drops to about `4-5%`.

This means the large raw `fy_b` excursions are highly sensitive to acceleration/derivative construction. That is a strong warning sign for label confidence.

## Current Conclusion

This does not prove that all high-frequency `fy_b` is fake. The previous phase-binned and PSD diagnostics showed some phase-locked flapping structure. But this sensitivity test says the broad high-frequency energy and largest side-force spikes are not robust to reasonable label-side smoothing choices.

The defensible interpretation is:

> The raw `fy_b` target contains a structured, phase-locked flapping component, but its largest broadband excursions are low-confidence because they are strongly affected by acceleration/derivative target construction.

For paper/reporting, avoid saying simply "the model cannot predict side force." A better statement is:

> The lateral force channel is the least reliable target: it contains useful wingbeat-periodic structure, but raw effective-force reconstruction introduces high-frequency, low-confidence transients that dominate the unfiltered R2.

## Practical Next Step

Before changing the model again, decide whether the training target should remain raw `fy_b` or use a label-side smoothing/robust target for the side-force channel.

For control-oriented use, a smoothed or band-limited `fy_b` target may be more meaningful than forcing the neural network to fit raw broadband spikes.
