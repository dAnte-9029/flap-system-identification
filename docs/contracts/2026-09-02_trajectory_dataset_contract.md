# Step 1: Trajectory Dataset Contract and Builder

Contract version: `trajectory_dataset_v1`

Source audit: `docs/audits/2026-09-02_august_ulg_data_audit.md`

Frozen cohort: Step 0 `F5 + C4`

Default materialization: train and validation only

## 1. Learning problem

The dataset fixes the supervised problem as

\[
x(t_0),\;u(t_0:t_0+T) \longrightarrow x(t_0:t_0+T).
\]

For v1, the nominal sample rate is 50 Hz, the prediction horizon is 2.0 s, and the window stride is 0.2 s. A window therefore contains 101 state samples, including both endpoints, and 100 controls. The control at the final state timestamp is not an input to that window.

No history before `t0` is part of the v1 benchmark (`history_steps = 0`). The initial angular velocity, relative flap phase, and flap frequency make the otherwise hidden short-time dynamics explicit enough for this first contract. A later benchmark that adds history must use a new manifest and may not silently change the v1 window index.

## 2. Sample and window artifacts

Each materialized partition contains:

- `samples_<partition>.parquet`: source-timestamped state, actuator, flap-state, optional air-data, and validity fields, including invalid rows for auditability;
- `windows_<partition>.parquet`: immutable window start/end keys that select only a single log and a single contiguous valid segment;
- `manifest.json`: source, split, frame, signal-role, alignment, and builder provenance;
- `quality_summary.json`: row, duration, segment, window, air-data coverage, and horizon-jitter statistics.

`window_id`, `log_id`, `segment_id`, `sample_in_log`, `sample_in_segment`, and `timestamp_us` preserve sample identity and time order. A consumer joins a window to the sample table by `log_id + segment_id + sample_in_segment`; it must not select rows by global DataFrame position.

## 3. State at the initial instant

The mandatory rigid-body state `x(t0)` is:

| Stored fields | Meaning | Frame / unit |
| --- | --- | --- |
| `position_ned_m_{x,y,z}` | PX4 local-position estimate | local NED, m; z positive down |
| `velocity_ned_m_s_{x,y,z}` | PX4 local-velocity estimate | local NED, m/s; vz positive down |
| `attitude_q_{w,x,y,z}` | unit quaternion | rotates body FRD vectors into local NED |
| `angular_velocity_body_rad_s_{x,y,z}` | body angular velocity | body FRD, rad/s |

The mandatory initial flap state is:

- `relative_flap_phase_rad` and its sine/cosine encoding;
- `flap_frequency_hz`.

The stored PX4 local position is estimator-origin dependent. Translation-invariant models and metrics should use `p(t) - p(t0)`, computed inside each indexed window. The raw position remains in the sample table so the transformation is reversible and the exact trajectory can be reconstructed.

The following causal values may be read at `t0` only as an explicitly declared enriched variant:

- `true_airspeed_m_s`;
- `wind_ned_m_s_n`, `wind_ned_m_s_e`.

They are estimator/fusion outputs rather than independent ground truth. They are excluded from the mandatory v1 input, guarded by `valid_airdata`, and may not be used to define the core sample set or fit statistics outside train.

## 4. Future-known controls

Only these four post-allocation actuator outputs form `u(t0:t0+T)`:

| Field | ULog source | Meaning |
| --- | --- | --- |
| `control_flap_motor_normalized` | `actuator_motors.control[0]` | normalized main flapping-drive command |
| `control_left_elevon_normalized` | `actuator_servos.control[0]` | normalized left elevon command |
| `control_right_elevon_normalized` | `actuator_servos.control[1]` | normalized right elevon command |
| `control_rudder_normalized` | `actuator_servos.control[2]` | normalized rudder command |

The channel interpretation is frozen only for the exact Step 0 `F5 + C4` hardware, firmware, and allocation cohort. The logged controls are a prescribed future control tape for conditional trajectory prediction. They do not turn this dataset into an autonomous open-loop simulator, because the flight controller originally generated them in closed loop.

## 5. Targets and prohibited future information

The inclusive target sequence from `t0` through `tT` contains the rigid-body state and relative flap state. Position loss should use displacement from `t0`; quaternion loss must respect the `q` and `-q` equivalence.

Beyond `t0`, none of the following may be supplied as a model input:

- realized position, velocity, attitude, or angular velocity;
- relative or absolute flap phase and realized flap frequency;
- true airspeed or estimated wind;
- navigation mode, estimator-reset counters, validity flags, maneuver labels, GPS/RTK, or any other quality/evaluation signal;
- a control at `tT` or any signal recorded after `tT`.

`nav_state`, reset boundaries, air-data validity, and phase validity are quality/segmentation metadata only. Step 0 maneuver labels are audit proxies and are not supervision targets.

## 6. Causal time alignment

`vehicle_local_position` is the native master time axis; v1 does not synthesize a uniform interpolation grid. Its `timestamp_sample` is used when valid, otherwise `timestamp` is used, matching Step 0. Other topics are aligned by past-only zero-order hold, so an observation or command from after the reference timestamp cannot enter that row.

Freshness limits are:

| Signal | Maximum age |
| --- | ---: |
| attitude, angular velocity, actuator outputs | 0.05 s |
| encoder count, flap frequency | 0.10 s |
| true airspeed | 0.25 s |
| wind | 0.35 s |
| arming, land, failsafe, navigation state | 1.5 s |

The nominal 50 Hz duration is stored as `horizon_s`; every window also stores `observed_horizon_s` from its actual endpoint timestamps. The representative build had at most 0.0111 s absolute 2 s horizon error.

## 7. Validity and boundary rules

The row masks are deliberately separate:

- `valid_state`: finite valid PX4 position/velocity, fresh body rate, and a fresh quaternion whose norm is within `1e-3` of one;
- `valid_control`: all four commands are finite and fresh;
- `valid_phase`: fresh cumulative encoder count and a finite reported flap frequency in `(0.5, 20) Hz`;
- `valid_airdata`: fresh nonnegative TAS no greater than 30 m/s plus fresh horizontal wind;
- `valid_airborne_safe`: armed, airborne, fresh status, and no failsafe;
- `valid_core`: state, control, phase, and airborne/safe gates, excluding reset and mode-transition rows.

`valid_airdata` is not part of `valid_core`. Invalid rows remain in the sample Parquet with `segment_id = -1`.

`exclusion_reason` records the failed gate or gates for every invalid row. `reset_boundary`, `mode_boundary`, and `gap_boundary` keep the three discontinuity causes separately inspectable.

A new segment starts after any invalid row, nonmonotonic timestamp, gap over 0.05 s, navigation-mode change, local-position reset-counter change, attitude quaternion reset, or encoder total-count reversal. Windows are generated only within one `log_id + segment_id`, so they cannot cross a log, invalid interval, dropout, mode boundary, or estimator reset.

## 8. Split contract and sealed data

The builder reads log lists from the Step 0 machine summary rather than discovering a newer directory or choosing files by modification time. The exact primary cohort is firmware group `F5` and structural-configuration group `C4`.

| Partition | Date | Logs | v1 policy |
| --- | --- | ---: | --- |
| train | 2026-08-19 | 6 | materialized; only source permitted for future normalization/fitting |
| validation | 2026-08-20 | 5 | materialized; selection/evaluation only |
| sealed test | 2026-08-26 | 3 | frozen in manifest, not opened by the default Step 1 build |
| OOD holdout | mixed, non-primary configuration | 12 | not part of v1 materialization |

The validator rejects both repeated log paths and a calendar date shared by train, validation, or sealed test. Sample-level or window-level random splitting is prohibited. The builder does not fit or apply normalization.

## 9. Relative-phase limitation and forward compatibility

All usable August phase is reconstructed from cumulative encoder count as

\[
\phi_{rel}(t)=\operatorname{wrap}_{[0,2\pi)}\left[
\frac{2\pi\,(c(t)-c_{origin})}{4096\,r_{FLAP}}
\right],
\]

where `c_origin` is the first finite count in that log and `r_FLAP` is the audited per-log `FLAP_RATIO`. This preserves within-log phase evolution but assigns an arbitrary zero independently to every log. Therefore phase zero is not comparable across logs and must not be interpreted as a shared wing pose, Hall crossing, upstroke onset, or absolute mechanical phase.

The sample schema already reserves `absolute_flap_phase_rad`, its sine/cosine fields, and `hall_reference_valid`. For the August build they are respectively NaN and false. A future Hall-referenced dataset can populate these fields while retaining the relative fields; promotion to cross-log absolute-phase modeling still requires a new versioned contract and validation of sign, gearing, zero offset, and Hall association.

## 10. Reproduction

Use only the repository environment:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python \
  scripts/build_trajectory_dataset.py \
  --output-root dataset/trajectory_v1_august_f5_c4 \
  --summary-output docs/audits/results/2026-09-02_trajectory_dataset_build_summary.json
```

The command refuses to overwrite a nonempty output directory. `sealed_test` must be named explicitly in `--partitions` and is intentionally absent from the command above.

## 11. Representative build result

The checked Step 1 build produced:

| Partition | Source rows | Valid rows | Valid duration | Segments | 2 s windows | Core rows with valid air data |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 93,684 | 44,708 | 893.478 s | 32 | 4,214 | 94.775% |
| validation | 59,401 | 41,163 | 822.774 s | 23 | 3,920 | 98.445% |

The small reduction from Step 0 model-ready duration comes from the Step 1 phase/frequency gate and the explicit reset, mode-change, and gap boundaries. No sealed-test samples or statistics were materialized.
