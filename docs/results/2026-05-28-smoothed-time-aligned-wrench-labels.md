# Smoothed Time-Aligned Wrench Labels: Tasks 1-6 Report

## Scope

This report covers implementation-plan Tasks 1-6 only: reusable signal preprocessing, split rewriting with smoothed inverse-dynamics derivatives, train-only lag diagnostics, optional input filtering hooks, six-axis raw-vs-clean label diagnostics, and construction of one candidate clean split.

It does not yet rerun the B+C correction model or translational replay checks.

## Inputs and Outputs

- Raw split: `dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1`
- Metadata: `metadata/aircraft/flapper_01/aircraft_metadata.yaml`
- Candidate clean split: `dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1`
- Artifact root: `artifacts/20260528_smoothed_time_aligned_wrench_labels_v1`
- Label-quality diagnostics: `artifacts/20260528_smoothed_time_aligned_wrench_labels_v1/label_quality`

The candidate split uses Savitzky-Golay derivatives with `window_s=0.12` and `polyorder=2`, then recomputes the effective wrench from the smoothed translational and angular accelerations. Input filtering was implemented but not enabled for this first candidate. Lag diagnostics were enabled, but the conservative selection policy kept zero lag for all inspected signals.

## Implemented Components

- `src/system_identification/signal_preprocessing.py`
  - Groupwise Savitzky-Golay derivative
  - Groupwise cubic-spline derivative
  - Groupwise low-pass filtering
  - Groupwise time shifting
  - Label-quality helper metrics

- `src/system_identification/pipeline.py`
  - Added `compute_kinematic_derivatives(...)`
  - Kept `compute_smoothed_kinematic_derivatives(...)` as a compatibility wrapper

- `scripts/build_time_aligned_smoothed_label_split.py`
  - Rewrites train/val/test parquet splits with recomputed labels
  - Supports train-only lag diagnostics
  - Supports optional input filtering hooks
  - Writes a dataset manifest

- `scripts/diagnose_wrench_label_preprocessing.py`
  - Compares raw and clean six-axis wrench labels
  - Writes split-level and log-level metrics
  - Writes a Markdown diagnostic report

## Candidate Split Manifest

- Sample counts:
  - train: `308702`
  - val: `79587`
  - test: `60671`
- Label valid ratio:
  - train: `1.0`
  - val: `1.0`
  - test: `1.0`
- Selected lag values:
  - `phase_raw_rad`: `0.0 s`
  - `phase_raw_unwrapped_rad`: `0.0 s`
  - `flap_frequency_hz`: `0.0 s`
  - `airspeed_validated.true_airspeed_m_s`: `0.0 s`
  - `servo_left_elevon`: `0.0 s`
  - `servo_right_elevon`: `0.0 s`
  - `servo_rudder`: `0.0 s`

## Key Raw-vs-Clean Diagnostics

Test split, selected columns:

| channel | corr clean/raw | RMSE clean vs raw | p99 raw | p99 clean | p99 jump raw | p99 jump clean | high-pass frac raw 8 Hz | high-pass frac clean 8 Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fx_b` | 0.854 | 2.779 | 16.987 | 9.595 | 6.881 | 1.584 | 0.333 | 0.032 |
| `fy_b` | 0.337 | 1.362 | 4.501 | 1.066 | 6.794 | 0.280 | 0.844 | 0.069 |
| `fz_b` | 0.945 | 3.702 | 27.981 | 19.679 | 11.622 | 3.353 | 0.077 | 0.002 |
| `mx_b` | 0.477 | 0.00296 | 0.01021 | 0.00281 | 0.01421 | 0.00090 | 0.771 | 0.039 |
| `my_b` | 0.768 | 0.00323 | 0.00994 | 0.00354 | 0.00783 | 0.00124 | 0.593 | 0.150 |
| `mz_b` | 0.389 | 0.00047 | 0.00148 | 0.00040 | 0.00160 | 0.00014 | 0.847 | 0.085 |

## Interpretation

The first candidate does what it is supposed to do mechanically: it strongly reduces sample-to-sample jumps, p99 magnitudes, and high-frequency energy in all six channels. This is consistent with the intended "smooth state first, then reconstruct wrench" pipeline.

The same diagnostics also show that `window_s=0.12` may be too aggressive for some channels. The low clean/raw correlation for `fy_b`, `mx_b`, and `mz_b` means this split should not yet be treated as the final paper dataset. It is a useful candidate for downstream testing, but it needs comparison against shorter windows such as `0.06 s`, plus B+C correction and translational replay checks.

The force channels behave differently: `fz_b` remains highly correlated with the raw label while losing high-frequency artifacts, `fx_b` remains moderately correlated, and `fy_b` changes substantially. This matches prior observations that lateral force is the most fragile channel.

Moment channels remain lower-confidence. The smoothing reduces derivative-driven spikes strongly, but because rotational oracle replay is still unresolved, moment improvements should be interpreted as label-conditioning diagnostics rather than validated rotational dynamics.

## Verification

Commands run:

```bash
pytest tests/test_signal_preprocessing.py tests/test_wrench_label_lag_sweep.py tests/test_wrench_label_input_filtering.py tests/test_wrench_label_preprocessing_diagnostics.py tests/test_build_lateral_smoothed_label_split.py tests/test_target_generation_sensitivity.py -q
```

Result:

```text
14 passed
```

Compile check:

```bash
python -m py_compile src/system_identification/signal_preprocessing.py scripts/build_time_aligned_smoothed_label_split.py scripts/diagnose_wrench_label_preprocessing.py
```

Result: passed with no output.

Build command:

```bash
python scripts/build_time_aligned_smoothed_label_split.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --output dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1 \
  --artifact-dir artifacts/20260528_smoothed_time_aligned_wrench_labels_v1 \
  --derivative-method savgol \
  --window-s 0.12 \
  --polyorder 2 \
  --enable-lag-sweep
```

Diagnostic command:

```bash
python scripts/diagnose_wrench_label_preprocessing.py \
  --raw-split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 \
  --clean-split-root dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_v1 \
  --output-dir artifacts/20260528_smoothed_time_aligned_wrench_labels_v1/label_quality
```

## Next Decision

Before using this split for the paper, run Tasks 7-8:

1. Rerun B+C correction on the candidate split.
2. Rerun translational replay sanity checks.
3. Build at least one shorter-window candidate, preferably `window_s=0.06`, to check whether `fy_b` and moment channels retain more raw-label structure while still suppressing obvious derivative spikes.
