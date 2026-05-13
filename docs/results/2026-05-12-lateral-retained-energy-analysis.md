# Lateral Retained Energy Analysis

Date: 2026-05-12

## Setup

- Raw split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Smoothed splits:
  - `lateral_sg0p06`: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_lateral_sg0p06_p2`
  - `lateral_sg0p12`: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_lateral_sg0p12_p2`
  - `lateral_sg0p20`: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_lateral_sg0p20_p2`
- Script: `scripts/analyze_lateral_retained_energy.py`
- Artifact directory: `artifacts/20260512_lateral_retained_energy_analysis/`

The analysis computes de-meaned one-sided FFT energy per `log_id/segment_id`, then aggregates by sample-count weighting. Dynamic bands use the median per-segment flap frequency:

```text
0-1 Hz
1-3 Hz
flap-main: f0 +/- 0.75 Hz
2f harmonic: 2f0 +/- 0.75 Hz
3f harmonic: 3f0 +/- 0.75 Hz
broadband high: 8-25 Hz excluding flap-main, 2f, and 3f windows
```

Manifest-derived edge trimming was used for the smoothed labels:

```text
lateral_sg0p06: 0.03 s
lateral_sg0p12: 0.06 s
lateral_sg0p20: 0.10 s
```

## Test Split Retained Energy Ratio

| target | band | lateral_sg0p06 | lateral_sg0p12 | lateral_sg0p20 |
|---|---|---:|---:|---:|
| fy_b | 0-1 Hz | 0.967 | 0.942 | 0.903 |
| fy_b | 1-3 Hz | 0.978 | 0.876 | 0.709 |
| fy_b | flap-main | 0.753 | 0.361 | 0.056 |
| fy_b | 2f harmonic | 0.528 | 0.064 | 0.022 |
| fy_b | 3f harmonic | 0.204 | 0.008 | 0.003 |
| fy_b | broadband high | 0.063 | 0.006 | 0.002 |
| mx_b | 0-1 Hz | 0.579 | 0.563 | 0.513 |
| mx_b | 1-3 Hz | 0.887 | 0.755 | 0.535 |
| mx_b | flap-main | 0.852 | 0.553 | 0.185 |
| mx_b | 2f harmonic | 0.524 | 0.052 | 0.007 |
| mx_b | 3f harmonic | 0.205 | 0.006 | 0.001 |
| mx_b | broadband high | 0.075 | 0.005 | 0.001 |
| mz_b | 0-1 Hz | 0.962 | 0.941 | 0.910 |
| mz_b | 1-3 Hz | 0.918 | 0.842 | 0.706 |
| mz_b | flap-main | 0.853 | 0.549 | 0.185 |
| mz_b | 2f harmonic | 0.507 | 0.048 | 0.010 |
| mz_b | 3f harmonic | 0.225 | 0.003 | 0.005 |
| mz_b | broadband high | 0.111 | 0.004 | 0.002 |

## sg0p06 Energy Fractions on Test

| target | band | raw fraction | smoothed fraction | retained ratio | removed ratio |
|---|---|---:|---:|---:|---:|
| fy_b | 0-1 Hz | 0.040 | 0.201 | 0.967 | 0.033 |
| fy_b | 1-3 Hz | 0.026 | 0.131 | 0.978 | 0.022 |
| fy_b | flap-main | 0.048 | 0.188 | 0.753 | 0.247 |
| fy_b | 2f harmonic | 0.087 | 0.236 | 0.528 | 0.472 |
| fy_b | 3f harmonic | 0.017 | 0.018 | 0.204 | 0.796 |
| fy_b | broadband high | 0.257 | 0.084 | 0.063 | 0.937 |
| mx_b | 0-1 Hz | 0.000 | 0.001 | 0.579 | 0.421 |
| mx_b | 1-3 Hz | 0.005 | 0.022 | 0.887 | 0.113 |
| mx_b | flap-main | 0.117 | 0.486 | 0.852 | 0.148 |
| mx_b | 2f harmonic | 0.088 | 0.223 | 0.524 | 0.476 |
| mx_b | 3f harmonic | 0.015 | 0.015 | 0.205 | 0.795 |
| mx_b | broadband high | 0.240 | 0.088 | 0.075 | 0.925 |
| mz_b | 0-1 Hz | 0.008 | 0.036 | 0.962 | 0.038 |
| mz_b | 1-3 Hz | 0.015 | 0.062 | 0.918 | 0.082 |
| mz_b | flap-main | 0.047 | 0.187 | 0.853 | 0.147 |
| mz_b | 2f harmonic | 0.086 | 0.203 | 0.507 | 0.493 |
| mz_b | 3f harmonic | 0.128 | 0.133 | 0.225 | 0.775 |
| mz_b | broadband high | 0.515 | 0.266 | 0.111 | 0.889 |

## Interpretation

`lateral_sg0p06` is the useful compromise. It preserves most low-frequency energy and most flap-main energy, keeps about half of the 2f harmonic, attenuates 3f strongly, and removes most non-structured 8-25 Hz broadband energy. On the test split, the broadband high-frequency removed ratios are `0.937` for `fy_b`, `0.925` for `mx_b`, and `0.889` for `mz_b`.

The stronger smoothing variants remove broadband high-frequency energy more aggressively, but they also remove much more structured content. On the test split, `lateral_sg0p12` retains only `0.361` of the `fy_b` flap-main band and `0.064` of the `fy_b` 2f band; `lateral_sg0p20` retains only `0.056` and `0.022`, respectively. This explains why sg0p12/sg0p20 are less attractive as control-oriented targets despite stronger noise suppression.

The `mx_b` 0-1 Hz ratio should not be over-interpreted because the raw 0-1 Hz energy fraction is nearly zero. The meaningful `mx_b` structure is concentrated around the flap-main and harmonic bands, where sg0p06 retains the same qualitative pattern as `fy_b` and `mz_b`.

## Files

```text
artifacts/20260512_lateral_retained_energy_analysis/lateral_retained_energy_summary.csv
artifacts/20260512_lateral_retained_energy_analysis/lateral_retained_energy_segments.csv
artifacts/20260512_lateral_retained_energy_analysis/lateral_retained_energy_config.json
artifacts/20260512_lateral_retained_energy_analysis/lateral_retained_energy_report.md
artifacts/20260512_lateral_retained_energy_analysis/lateral_retained_energy_ratio.png
```

## Verification

The retained-energy tests and the previously skipped torch-dependent frequency-resolved tests were re-run in the `flap-train-gpu` conda environment:

```text
python 3.11.14
torch 2.7.0+cu128
cuda_available True
```

Command:

```bash
conda run -n flap-train-gpu pytest tests/test_lateral_retained_energy.py tests/test_fyb_learnability_diagnostics.py tests/test_frequency_resolved_backbone_comparison.py -q
```

Result:

```text
8 passed in 1.12s
```

The full retained-energy analysis was also re-run under `flap-train-gpu` into `artifacts/20260512_lateral_retained_energy_analysis_flap_train_gpu_check/`. Its `lateral_retained_energy_summary.csv` is byte-identical to the primary artifact summary.
