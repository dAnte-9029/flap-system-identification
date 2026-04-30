# Dataset log admission tracker

This file is the current source of truth for reviewed flight logs and their dataset admission status.

It is intentionally stricter than "recording exists":

- `accepted_formal_wing_phase`: can enter the formal dataset with logged `wing_phase`
- `accepted_formal_encoder_fallback`: can enter a fallback-only dataset after forcing `encoder_count` phase reconstruction
- `debug_only_failed_or_short`: recording chain is useful for debugging, but the flight is failed or too short for the formal dataset
- `rejected_missing_phase_chain`: cannot be used because `encoder_count` and/or `wing_phase` are missing
- `rejected_bad_logged_wing_phase`: cannot be used with logged `wing_phase` because the topic itself is structurally wrong

## Formal dataset: accepted with logged `wing_phase`

These logs are already packaged in
`dataset/canonical_v0.2_seed_labels_2026_4_12`.

| Log path | Branch | Status | Notes |
| --- | --- | --- | --- |
| `/home/zn/QgcLogs/2026.4.12/log_0_2026-4-12-16-23-16.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_0_2026-4-12-17-07-40.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_1_2026-4-12-16-33-22.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_1_2026-4-12-17-18-22.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_2_2026-4-12-17-25-40.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_3_2026-4-12-10-43-46.ulg` | `air` | `accepted_formal_wing_phase` | corrected replacement for older 09:08 log; lower `cycle_valid_ratio`, keep with care |
| `/home/zn/QgcLogs/2026.4.12/log_3_2026-4-12-17-34-16.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_4_2026-4-12-11-10-26.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_4_2026-4-12-17-43-30.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.12/log_5_2026-4-12-17-51-44.ulg` | `air` | `accepted_formal_wing_phase` | logged `wing_phase` usable |

## Formal dataset: accepted with logged `wing_phase` and packaged as FUSION cohorts

These logs are reviewed, accepted, and now packaged into these dataset folders:

- `dataset/canonical_v0.2_seed_labels_2026_4_14_3algorithms_weakwing`
- `dataset/canonical_v0.2_seed_labels_2026_4_14_3algorithms`
- `dataset/canonical_v0.2_seed_labels_2026_4_15`

| Log path | Branch | Status | Notes |
| --- | --- | --- | --- |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms/log_0_2026-4-14-10-21-28.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms/log_3_2026-4-14-10-52-38.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms-weakwing/log_0_2026-4-14-11-50-14.ulg` | `FUSION` | `accepted_formal_wing_phase` | weakwing batch; long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms-weakwing/log_1_2026-4-14-12-01-42.ulg` | `FUSION` | `accepted_formal_wing_phase` | weakwing batch; long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms-weakwing/log_2_2026-4-14-12-10-52.ulg` | `FUSION` | `accepted_formal_wing_phase` | weakwing batch; long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms-weakwing/log_3_2026-4-14-12-19-22.ulg` | `FUSION` | `accepted_formal_wing_phase` | weakwing batch; long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms-weakwing/log_4_2026-4-14-12-30-12.ulg` | `FUSION` | `accepted_formal_wing_phase` | weakwing batch; long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_5_2026-4-15-10-30-38.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_6_2026-4-15-10-48-46.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_7_2026-4-15-10-54-58.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_8_2026-4-15-11-23-44.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_9_2026-4-15-11-30-52.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_10_2026-4-15-11-37-24.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_12_2026-4-15-11-57-08.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_13_2026-4-15-12-04-42.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_15_2026-4-15-12-11-28.ulg` | `FUSION` | `accepted_formal_wing_phase` | shortest accepted in this batch; still above the current long-flight cutoff |
| `/home/zn/QgcLogs/2026.4.15/log_17_2026-4-15-12-47-22.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_18_2026-4-15-12-56-08.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_19_2026-4-15-13-02-32.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_20_2026-4-15-13-12-34.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_21_2026-4-15-13-20-12.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |
| `/home/zn/QgcLogs/2026.4.15/log_22_2026-4-15-13-30-06(1).ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; raw `wing_phase` usable |

## Reviewed 2026.4.16 logs: accepted with logged `wing_phase`, not yet packaged

These logs have now been scanned into:

- `dataset/scan_2026_4_16.csv`
- `dataset/scan_2026_4_16_wing_phase_raw.csv`
- `dataset/scan_2026_4_16_usable_samples.csv`

The current recommendation is:

- accept the long nominal flights below as formal dataset candidates
- keep the very short failed flights as debug-only
- keep `log_33_2026-4-16-18-57-22.ulg` out of the formal dataset for now to stay consistent with the current minimum usable-flight bar

| Log path | Branch | Status | Notes |
| --- | --- | --- | --- |
| `/home/zn/QgcLogs/2026.4.16/1/log_24_2026-4-16-09-49-52.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 165.25` |
| `/home/zn/QgcLogs/2026.4.16/1/log_25_2026-4-16-09-57-28.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 158.54` |
| `/home/zn/QgcLogs/2026.4.16/1/log_26_2026-4-16-10-09-34.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 151.43` |
| `/home/zn/QgcLogs/2026.4.16/1/log_27_2026-4-16-10-17-56.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 137.78` |
| `/home/zn/QgcLogs/2026.4.16/1/log_28_2026-4-16-10-25-32.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 138.66` |
| `/home/zn/QgcLogs/2026.4.16/1/log_30_2026-4-16-10-36-20.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 155.54` |
| `/home/zn/QgcLogs/2026.4.16/2/log_31_2026-4-16-18-41-26.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 150.57` |
| `/home/zn/QgcLogs/2026.4.16/2/log_32_2026-4-16-18-53-24.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 142.82` |
| `/home/zn/QgcLogs/2026.4.16/2/log_34_2026-4-16-19-13-30.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 143.44` |
| `/home/zn/QgcLogs/2026.4.16/2/log_35_2026-4-16-19-19-22.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 142.31` |
| `/home/zn/QgcLogs/2026.4.16/2/log_36_2026-4-16-19-24-20.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 144.44` |
| `/home/zn/QgcLogs/2026.4.16/2/log_37_2026-4-16-19-29-42.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 149.06` |
| `/home/zn/QgcLogs/2026.4.16/2/log_38_2026-4-16-19-37-00.ulg` | `FUSION` | `accepted_formal_wing_phase` | long nominal flight; `same_timestamp_resets = 0`; `usable_duration_s = 140.39` |
| `/home/zn/QgcLogs/2026.4.16/1/log_23_2026-4-16-09-41-18.ulg` | `FUSION` | `debug_only_failed_or_short` | very short failed segment; `active_duration_s = 0.0`; `phase_valid_ratio_raw = 0.0` |
| `/home/zn/QgcLogs/2026.4.16/1/log_29_2026-4-16-10-31-50.ulg` | `FUSION` | `debug_only_failed_or_short` | very short failed segment; `active_duration_s = 0.0`; `phase_valid_ratio_raw = 0.0` |
| `/home/zn/QgcLogs/2026.4.16/2/log_33_2026-4-16-18-57-22.ulg` | `FUSION` | `debug_only_failed_or_short` | border case: one usable active run exists (`usable_duration_s = 101.23`), but overall `label_valid_ratio = 0.747` and usable flight duration is below the current formal cutoff |

## Formal dataset: accepted only with forced `encoder_count` fallback

These logs are already packaged in
`dataset/canonical_v0.2_seed_labels_2026_4_13_air_encoder_fallback`.

Reason: the logged `wing_phase` topic exists, but its recorded values are all `NaN` with `phase_valid = 0`.

| Log path | Branch | Status | Notes |
| --- | --- | --- | --- |
| `/home/zn/QgcLogs/2026.4.13/air/log_0_2026-4-13-09-34-42.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_0_2026-4-13-09-52-12.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_1_2026-4-13-10-15-00.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_2_2026-4-13-10-20-38.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_4_2026-4-13-10-49-22.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_5_2026-4-13-10-58-32.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_6_2026-4-13-11-06-38.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_7_2026-4-13-11-35-50.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |
| `/home/zn/QgcLogs/2026.4.13/air/log_8_2026-4-13-11-43-38.ulg` | `air` | `accepted_formal_encoder_fallback` | forced `encoder_count_fallback` |

## Rejected or debug-only reviewed logs

| Log path | Branch | Status | Notes |
| --- | --- | --- | --- |
| `/home/zn/QgcLogs/2026.4.12/log_0_2026-4-12-09-08-10.ulg` | `air` | `rejected_bad_logged_wing_phase` | old same-timestamp reset bug in `wing_phase` (`same_timestamp_resets = 261`) |
| `/home/zn/QgcLogs/2026.4.13/air/log_3_2026-4-13-10-22-54.ulg` | `air` | `debug_only_failed_or_short` | too short; `samples = 4625`, fallback `cycle_valid_ratio = 0.248` |
| `/home/zn/QgcLogs/2026.4.13/log_0_2026-4-13-12-16-54.ulg` | `FUSION` | `rejected_missing_phase_chain` | no `encoder_count`, no `wing_phase`; has `rpm` and `flap_frequency` only |
| `/home/zn/QgcLogs/2026.4.13/log_1_2026-4-13-12-22-36.ulg` | `FUSION` | `rejected_missing_phase_chain` | no `encoder_count`, no `wing_phase`; short log |
| `/home/zn/QgcLogs/2026.4.13/log_2_2026-4-13-18-06-24.ulg` | `FUSION` | `rejected_missing_phase_chain` | no `encoder_count`, no `wing_phase`; has `rpm` and `flap_frequency` only |
| `/home/zn/QgcLogs/2026.4.14/log_0_2026-4-14-09-10-36.ulg` | `FUSION` | `rejected_missing_phase_chain` | no `encoder_count`, no `wing_phase` |
| `/home/zn/QgcLogs/2026.4.14/log_1_2026-4-14-09-25-34.ulg` | `FUSION` | `debug_only_failed_or_short` | logging chain partially restored; `encoder_count` and `wing_phase` present, but only `964` samples and raw `phase_valid_ratio = 0.665` |
| `/home/zn/QgcLogs/2026.4.14/log_0_2026-4-14-09-33-52.ulg` | `FUSION` | `debug_only_failed_or_short` | recording chain basically correct; failed short flight; raw `cycle_valid_ratio = 0.096` |
| `/home/zn/QgcLogs/2026.4.14/log_1_2026-4-14-09-37-58.ulg` | `FUSION` | `debug_only_failed_or_short` | recording chain basically correct; failed short flight; raw `cycle_valid_ratio = 0.111` |
| `/home/zn/QgcLogs/2026.4.14/log_2_2026-4-14-10-00-08.ulg` | `FUSION` | `debug_only_failed_or_short` | best of the failed 2026-04-14 batch; raw `wing_phase_valid_ratio = 1.0`, raw `cycle_valid_ratio = 0.407`, still not a formal dataset flight |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms/log_1_2026-4-14-10-47-06.ulg` | `FUSION` | `debug_only_failed_or_short` | short failed segment; `raw_sample_count = 6565`, raw `phase_valid_ratio = 0.0` |
| `/home/zn/QgcLogs/2026.4.14/3Algorithms/log_2_2026-4-14-10-48-16.ulg` | `FUSION` | `debug_only_failed_or_short` | short failed segment; `raw_sample_count = 3050`, raw `phase_valid_ratio = 0.0` |
| `/home/zn/QgcLogs/2026.4.15/log_11_2026-4-15-11-52-40.ulg` | `FUSION` | `debug_only_failed_or_short` | short failed segment; `raw_sample_count = 3713`, `core_overlap_s = 37.1` |
| `/home/zn/QgcLogs/2026.4.15/log_14_2026-4-15-12-09-12.ulg` | `FUSION` | `debug_only_failed_or_short` | very short failed segment; `raw_sample_count = 982`, `core_overlap_s = 9.8` |
| `/home/zn/QgcLogs/2026.4.15/log_16_2026-4-15-12-43-32.ulg` | `FUSION` | `debug_only_failed_or_short` | very short failed segment; `raw_sample_count = 671`, `core_overlap_s = 6.7` |

## Current `wing_phase` guidance

For the current effective wrench regression task, treat `hall_event` as a
recommended phase-quality enhancement, not as a hard admission requirement.

The practical rule is:

- `wing_phase` is already sufficient for v1 training if it is monotonic within a
  cycle, wraps stably, and no longer shows the old same-timestamp reset bug
- `hall_event` remains valuable for later higher-confidence offline phase
  alignment, cycle-boundary auditing, and future phase-heavy models
- lack of `hall_event` alone should not exclude an otherwise long, nominal, and
  numerically stable log from the current v1 dataset

This is why the accepted `2026.4.16` FUSION logs can still be used even though
that batch recorded `wing_phase` and `encoder_count` but not `hall_event`.

## High-quality subset for v1 training

To remove the weaker tail from the merged seed dataset, a stricter high-quality
subset is now packaged in:

- `dataset/canonical_v0.2_seed_labels_hq_v1`
- `dataset/canonical_v0.2_training_ready_split_hq_v1`

The current hq selection policy is:

- `active_duration_s >= 120`
- `usable_active_ratio >= 0.97`
- `active_phase_valid_ratio >= 0.994`

This keeps 29 logs and drops the weaker 16 logs from the broader accepted pool.
It is intentionally stricter than the normal admission bar and is meant to be
the default split for the next round of baseline training.

The quality scan and curated membership live in:

- `dataset/canonical_v0.2_seed_labels_hq_v1/quality_scan.csv`
- `dataset/canonical_v0.2_seed_labels_hq_v1/accepted_logs.json`
- `dataset/canonical_v0.2_seed_labels_hq_v1/excluded_logs.json`
- `dataset/canonical_v0.2_training_ready_split_hq_v1/dataset_manifest.json`

For training-ready splits, there is now an additional row-level flight-window
trim on top of the log-level hq filter:

- keep only rows that still satisfy the normal valid-row mask
- also require `vehicle_land_detected.landed == 0` when that signal exists

This removes the launch-preload / hand-throw idle rows and the landed tail rows
even when the flapping cycle tracker still marks them as active. The current
airborne-trimmed split is:

- `dataset/canonical_v0.2_training_ready_split_hq_v2_airborne`

## Current FUSION branch acceptance rule

For future `FUSION` logs, treat a log as a formal dataset candidate only if all of the following are true:

- `encoder_count` is present
- `wing_phase` is present
- `wing_phase` contains actual numeric phase values, not only topic headers
- `phase_valid_ratio` is high, ideally close to `1.0`
- flight duration is long enough for training, not just a failed short segment
- `cycle_valid_ratio` is nontrivial on the raw or fallback path
- overall flight is nominal enough to represent the target operating regime

The reviewed `FUSION` logs under `2026.4.14/3Algorithms-weakwing`,
`2026.4.14/3Algorithms`, and most of `2026.4.15` now meet this bar and can be
used as formal dataset candidates with logged `wing_phase`.

## Reference files

- `dataset/canonical_v0.2_seed_labels_2026_4_12/dataset_manifest.json`
- `dataset/canonical_v0.2_seed_labels_2026_4_13_air_encoder_fallback/dataset_manifest.json`
- `dataset/canonical_v0.2_seed_labels_2026_4_14_3algorithms_weakwing/dataset_manifest.json`
- `dataset/canonical_v0.2_seed_labels_2026_4_14_3algorithms/dataset_manifest.json`
- `dataset/canonical_v0.2_seed_labels_2026_4_15/dataset_manifest.json`
- `dataset/canonical_v0.2_seed_labels_hq_v1/dataset_manifest.json`
- `dataset/canonical_v0.2_training_ready_split_hq_v1/dataset_manifest.json`
- `dataset/scan_2026_4_13_air.csv`
- `dataset/scan_2026_4_13_air_encoder_fallback.csv`
- `dataset/scan_2026_4_14_latest3.csv`
- `dataset/scan_2026_4_14_3algorithms_weakwing.csv`
- `dataset/scan_2026_4_14_3algorithms.csv`
- `dataset/scan_2026_4_15.csv`
- `dataset/scan_2026_4_16.csv`
- `dataset/scan_2026_4_16_wing_phase_raw.csv`
- `dataset/scan_2026_4_16_usable_samples.csv`
- `dataset/high_quality_log_quality_scan_2026_04_16.csv`
