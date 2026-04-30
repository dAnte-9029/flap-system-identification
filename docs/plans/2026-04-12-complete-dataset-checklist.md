# Complete dataset checklist for flapping flight-data system identification

_Checklist for turning current PX4 ULog data into a trainable supervised dataset for effective external force and moment estimation_

---

## 🎯 Goal

Build a reproducible dataset pipeline that converts PX4 flapping-aircraft flight logs into canonical samples with:

- stable input features
- corrected wing-phase features
- valid rigid-body effective wrench labels
- quality masks
- train/validation/test splits with full provenance

## 📋 Current status

| Area | Status | Notes |
| --- | --- | --- |
| Normal large logs | In progress | `2026.4.9`, `2026.4.10`, and `2026.4.12` already contain usable candidate flights |
| Canonical parquet export | Working | [`scripts/ulg_to_canonical_parquet.py`](../../scripts/ulg_to_canonical_parquet.py) can already write `samples.parquet`, `segments.parquet`, and preprocessing reports |
| Logged `wing_phase` topic | Partially working | New `2026.4.12` logs contain `wing_phase`, but `phase_unwrapped_rad` still needs downstream correction |
| Corrected phase in pipeline | Missing | Current pipeline still reconstructs phase mainly from `encoder_count` |
| Effective wrench labels | Blocked | [`metadata/aircraft/flapper_01/aircraft_metadata.yaml`](../../metadata/aircraft/flapper_01/aircraft_metadata.yaml) still has placeholder `mass_kg` and `inertia_b_kg_m2` |
| Dataset QC masks | Partial | `label_valid` exists, but phase/cycle/QC masks are not complete yet |
| Train/val/test packaging | Missing | No final split manifest yet |

## 🔄 Completion flow

```mermaid
flowchart LR
    accTitle: Dataset completion flow
    accDescr: This diagram shows the sequence from raw ULog auditing through canonical conversion, phase correction, metadata completion, label generation, quality control, and final dataset packaging.

    audit[🔍 Audit raw ULogs]
    canonical[📦 Build canonical samples]
    phase[🔄 Correct wing phase and assign cycle IDs]
    metadata[📋 Complete aircraft metadata]
    labels[🏷️ Generate force and moment labels]
    qc[🛡️ Apply sample and cycle quality control]
    package[📦 Package train val test dataset]

    audit --> canonical --> phase --> metadata --> labels --> qc --> package

    classDef primary fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef warning fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class audit,canonical,phase,metadata,labels primary
    class qc warning
    class package success
```

## 📥 Raw log intake checklist

- [ ] Keep the reviewed-log status table up to date in [`docs/plans/2026-04-14-dataset-log-admission-tracker.md`](./2026-04-14-dataset-log-admission-tracker.md)
- [ ] Define the candidate flight set for the first supervised dataset version
- [ ] Keep only normal, sufficiently long flights in the main dataset
- [ ] Exclude failed flights, aborted launches, and very short logs from the main training pool
- [ ] Keep failed or abnormal flights in a separate folder or manifest for later robustness studies
- [ ] Record one row per input log with date, file name, duration, size, and inclusion decision
- [ ] Confirm each accepted log contains the minimum required topics:
  - `vehicle_local_position`
  - `vehicle_attitude`
  - `vehicle_angular_velocity`
  - `vehicle_acceleration`
  - `actuator_motors`
  - `actuator_servos`
  - `encoder_count`
  - `flap_frequency`
  - `rpm`
  - `wing_phase` when available
- [ ] Record logger configuration and firmware provenance for each accepted log

### Acceptance criteria

- [ ] Every accepted log has a clear keep/drop decision in a manifest
- [ ] Every accepted log passes the minimum-topic check
- [ ] The final candidate set excludes known failed flights

## 📦 Canonical sample generation checklist

- [ ] Convert every accepted `.ulg` file through [`scripts/ulg_to_canonical_parquet.py`](../../scripts/ulg_to_canonical_parquet.py)
- [ ] Keep the canonical output structure consistent with [`docs/2026-03-23-ulg-to-canonical-parquet-preprocessing-spec.md`](../2026-03-23-ulg-to-canonical-parquet-preprocessing-spec.md)
- [ ] Verify `samples.parquet`, `segments.parquet`, `source_manifest.json`, and `preprocessing_report.json` are written for each log
- [ ] Lock the target sample rate and resampling behavior for the dataset version
- [ ] Confirm topic freshness rules are applied consistently across logs
- [ ] Confirm `pipeline_version` is recorded in the manifest

### Acceptance criteria

- [ ] All accepted logs convert without crashing
- [ ] Output schema is consistent across accepted logs
- [ ] Per-log preprocessing reports are present and readable

## 🔄 Wing-phase and cycle checklist

- [ ] Read `wing_phase` preferentially when it exists
- [ ] Validate `wing_phase.phase_unwrapped_rad` before trusting it directly
- [ ] Fall back to `encoder_count`-based reconstruction when `wing_phase` is absent or inconsistent
- [ ] Add cycle segmentation based on phase reset detection
- [ ] Add `cycle_id` to canonical samples
- [ ] Add `phase_corrected_rad` to canonical samples
- [ ] Add `phase_corrected_unwrapped_rad` to canonical samples
- [ ] Add `cycle_valid` to canonical samples
- [ ] Add cycle-level statistics:
  - cycle duration
  - mean flap frequency
  - phase start
  - phase max
  - monotonicity check
- [ ] Mark incomplete startup and shutdown cycles as invalid
- [ ] Keep both raw and corrected phase columns for auditability
- [ ] For future logs, add `hall_event` logging to improve cycle boundary reconstruction

### Recommended canonical fields

| Field | Meaning |
| --- | --- |
| `phase_source` | `wing_phase` or `encoder_count_fallback` |
| `phase_raw_rad` | raw wrapped phase from the source topic |
| `phase_raw_unwrapped_rad` | raw unwrapped phase from the source topic |
| `phase_corrected_rad` | per-cycle corrected phase in `[0, 2π)` |
| `phase_corrected_unwrapped_rad` | corrected continuously increasing phase |
| `cycle_id` | integer cycle index |
| `cycle_valid` | cycle-level validity mask |
| `cycle_duration_s` | duration of the current cycle |
| `cycle_flap_frequency_hz` | mean flap frequency of the current cycle |

### Acceptance criteria

- [ ] Every active flapping sample belongs to a cycle or is explicitly marked invalid
- [ ] Corrected phase stays in `[0, 2π)` for valid samples
- [ ] Corrected phase is monotonic within each valid cycle
- [ ] Raw and corrected phase are both retained in the dataset

## 📋 Aircraft metadata checklist

- [ ] Fill `mass_properties.mass_kg.value` in [`metadata/aircraft/flapper_01/aircraft_metadata.yaml`](../../metadata/aircraft/flapper_01/aircraft_metadata.yaml)
- [ ] Fill `mass_properties.cg_b_m.value`
- [ ] Fill `mass_properties.inertia_b_kg_m2.value`
- [ ] Fill `flapping_drive.encoder_to_drive_sign`
- [ ] Fill `flapping_drive.drive_phase_zero_offset_rad`
- [ ] Update `log_coverage.has_logged_wing_phase_topic`
- [ ] Update `log_coverage.current_good_log_notes`
- [ ] Fill `px4_firmware.commit`
- [ ] Fill control-surface sign conventions:
  - `actuators.left_elevon.positive_deflection`
  - `actuators.right_elevon.positive_deflection`
  - `actuators.rudder.positive_deflection`

### Acceptance criteria

- [ ] Metadata no longer reports placeholder mass for force labels
- [ ] Metadata no longer reports placeholder inertia for moment labels
- [ ] Metadata warnings no longer block absolute drive-phase reconstruction

## 🏷️ Label definition and generation checklist

- [ ] Freeze the exact label meaning for dataset v0.1
- [ ] Confirm whether the target is:
  - rigid-body effective external wrench
  - or pure aerodynamic wrench
- [ ] Keep the current v0.1 definition aligned with [`docs/plans/2026-03-23-effective-wrench-labels-implementation-plan.md`](./2026-03-23-effective-wrench-labels-implementation-plan.md)
- [ ] Regenerate `fx_b`, `fy_b`, `fz_b`, `mx_b`, `my_b`, `mz_b` after metadata is complete
- [ ] Verify `label_valid` is no longer all false
- [ ] Confirm label sign conventions match FRD body axes
- [ ] Confirm the label reference point is the center of gravity

### Acceptance criteria

- [ ] Accepted logs contain finite labels for a nontrivial fraction of samples
- [ ] `label_valid` is high on nominal flight segments
- [ ] Label columns are not all `NaN`

## 🛡️ Quality control checklist

- [ ] Add sample-level validity masks for:
  - phase validity
  - cycle validity
  - state estimate validity
  - airspeed validity
  - GNSS or relative heading validity where relevant
  - actuator saturation
  - dropout proximity
  - estimator reset proximity
- [ ] Add segment-level validity summaries
- [ ] Add cycle-level validity summaries
- [ ] Define thresholds for discarding low-quality samples, cycles, and segments
- [ ] Keep masks in the dataset rather than silently deleting everything upstream

### Recommended masks

| Mask | Purpose |
| --- | --- |
| `phase_valid` | Raw phase source is usable |
| `cycle_valid` | Current cycle passes geometric and timing checks |
| `flap_active` | Vehicle is actively flapping |
| `state_valid` | Position, velocity, and attitude estimates are usable |
| `airspeed_valid` | Airspeed source is fresh and valid |
| `dropout_nearby` | Sample is near a logger dropout |
| `estimator_reset_nearby` | Sample is near an EKF reset |
| `label_valid` | Label columns are finite and usable |

### Acceptance criteria

- [ ] Every training sample can be filtered with explicit masks
- [ ] QC logic is deterministic and documented
- [ ] Bad cycles and bad segments are traceable rather than silently lost

## 📚 Dataset packaging checklist

- [ ] Group outputs by dataset version under `dataset/`
- [ ] Write a top-level manifest for the dataset version
- [ ] Split data into train, validation, and test by log or flight, not by randomly mixed samples
- [ ] Prevent leakage by keeping samples from the same flight in the same split
- [ ] Record split membership per log
- [ ] Record the generation command and config used for the dataset version
- [ ] Keep source file hashes in the manifest
- [ ] Keep preprocessing warnings in the manifest

### Acceptance criteria

- [ ] Split boundaries are reproducible
- [ ] No flight appears in multiple splits
- [ ] Every packaged sample can be traced back to a source log

## ✅ Definition of done for the first complete dataset

- [ ] All accepted logs convert into canonical parquet successfully
- [ ] `phase_corrected_rad` and `cycle_id` are present in canonical samples
- [ ] `mass_kg`, `cg_b_m`, and `inertia_b_kg_m2` are filled in metadata
- [ ] `fx_b...mz_b` contain finite values on nominal flight samples
- [ ] `label_valid` is meaningful and nonzero on accepted logs
- [ ] Sample-, cycle-, and segment-level masks are present
- [ ] Train/validation/test manifests are written
- [ ] Dataset version, source logs, and firmware provenance are recorded

## 🚧 Immediate blockers

1. `mass_kg` is still missing, so force labels remain invalid.
2. `inertia_b_kg_m2` is still missing, so moment labels remain invalid.
3. `wing_phase.phase_unwrapped_rad` still needs downstream correction before direct use.
4. The current pipeline does not yet emit formal `phase_corrected_rad` and `cycle_id` columns.

## ▶️ Recommended next order of work

1. Add `phase_corrected_rad`, `phase_corrected_unwrapped_rad`, and `cycle_id` to the canonical pipeline.
2. Fill `mass_kg`, `cg_b_m`, `inertia_b_kg_m2`, `encoder_to_drive_sign`, and `drive_phase_zero_offset_rad`.
3. Regenerate canonical parquet for the accepted normal logs.
4. Verify `label_valid` and QC masks on the regenerated dataset.
5. Freeze the first train/validation/test split and write the dataset manifest.
