# Quadratic-2 spanwise dynamic-twist sweep: read-only audit

Date: 2026-07-28
Audit branch: `exp/quadratic2-twist-sweep`
Required base commit: `59d90e62c9468aaafab36d1c7393ff5cfe06e64d`

## Audit disposition

The Git, environment, registry, artifact, split, and mechanical-phase preconditions
are reproducible. The baseline full test suite passes (`503 passed in 92.53 s`).
The registered train and validation dataset/prior files exist and match their
declared SHA-256 hashes. No test Parquet was opened or hashed during this audit.

The experiment initially stopped before implementation because of a force-axis
contradiction:

- the repository metadata, canonical dataset, active prior, correction-ready
  artifact, and longitudinal correction contract all use body FRD, so `+Fx` is
  forward and `+Fz` is **down**;
- the experiment request requires `Fx: body forward` and `Fz: body up`;
- the requested `Fz minimum` metric matches the existing FRD convention because
  upward aerodynamic force is negative `fz_b`. Converting both data and model to
  body-up would require `Fz_up = -fz_b`, in which case the physically corresponding
  lift extremum is a **maximum**, not a minimum.

The user resolved this on 2026-07-28: the experiment must retain canonical body
FRD (`+Fx` forward, `+Fz` down), and `Fz minimum` remains the upward-lift
extremum. Every output must state this sign convention.

## 1. Repository and artifact baseline

Startup checks:

```text
initial branch: feat/static-correction-model-selection
initial local HEAD: b3c565b6d2b075f3269f6b34863f913a6ee6b8c9
required base: 59d90e62c9468aaafab36d1c7393ff5cfe06e64d
origin/feat/static-correction-model-selection after fetch:
  59d90e62c9468aaafab36d1c7393ff5cfe06e64d
experiment branch:
  exp/quadratic2-twist-sweep at 59d90e62c9468aaafab36d1c7393ff5cfe06e64d
```

The local source branch contained one unpushed descendant,
`b3c565b6d2b075f3269f6b34863f913a6ee6b8c9` (`feat(physics): add zero-wind
DeLaurier diagnostic`). It is deliberately excluded from the experiment branch.

The active registry resolutions are:

| Contract | Resolved identity | Lifecycle | Path |
|---|---|---|---|
| canonical dataset | `canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3` | active | `dataset/canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3` |
| DeLaurier prior | `delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4` | active | `artifacts/20260721_delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4` |

Verified hashes, without accessing the sealed test Parquet:

| Artifact | SHA-256 |
|---|---|
| dataset manifest | `aa12aa66f762390ab1a356b94916694f5ed9689af670f313544aeb57a250cc07` |
| train samples | `fe50d4a6c609be7c27cabc604a2587d78e8be21c34af61592af5c6e33f2de70e` |
| validation samples | `839e02a3600f3aa1d12d5d933f10872022f424481fff754614bc798ca5d6b93e` |
| active prior manifest | `c86b34ea10328207b1867b117d44656eacf54b751e883a1ee99de0656695200c` |
| train prior predictions | `8df283f2a877124979545c5b255fd9d5799b4edf11cd0c657a9f6b19801669fb` |
| validation prior predictions | `3f253f36c41bddc537cc377d7003c141058f2316a5876f4b16dd174ea848002c` |

The active prior records physics source commit
`3b5d4ec1d28f1384cf042402992ad7ea59995f49`, attached flow, 80 strips,
attitude-aware ground-minus-EKF-wind airflow, constant-frequency-step phase
acceleration, and zero tip-twist amplitude.

## 2. Current dynamic-twist implementation and callers

The canonical implementation is
`src/system_identification/physics/delaurier/dynamic_twist.py`:

- `map_canonical_phase_to_delaurier`
- `compute_delaurier_dynamic_twist`
- `DeLaurierTwistKinematics`

Compatibility exports exist at:

- `src/system_identification/physics/delaurier_dynamic_twist.py`;
- `src/system_identification/physics/delaurier/__init__.py`;
- `src/system_identification/physics/__init__.py`.

The production caller is
`src/system_identification/physics/baselines/wing_only.py::_chunk_result`.
The public segment API is
`evaluate_wing_only_delaurier_segment`. The train/validation prior exporter in
`src/system_identification/physics/priors/export.py::_evaluate_partition` calls
that API once per `(log_id, segment_id)`. Other current callers are the
wing-wrench theta analysis and component-attribution analysis; tests exercise
both the canonical and compatibility import paths.

The current linear-span formula uses nonnegative one-wing strip centers:

```text
eta_i = x_mid_i / R
delta_theta_i = -A_tip eta_i sin(phi_D)
delta_theta_dot_i = -A_tip eta_i cos(phi_D) phi_dot
delta_theta_ddot_i =
    A_tip eta_i [sin(phi_D) phi_dot^2 - cos(phi_D) phi_ddot]
theta_i = theta_bar + delta_theta_i
```

`R` is the theoretical semi-span, not the outer strip center. If no explicit
semi-span is supplied it is reconstructed as
`max(strip_center + 0.5 strip_width)`. The active authoritative prior sets
`A_tip=0`, so its dynamic-twist contract is
`disabled_zero_tip_amplitude`; the legacy linear implementation still supports
nonzero amplitudes and is covered by the frozen IsaacLab fixture.

## 3. Stroke, theta, theta-dot, and theta-double-dot generation

In `src/system_identification/physics/baselines/wing_only.py::_chunk_result`:

```text
phi_D = wrap(phi_mech - pi/2)
omega = 2 pi flap_frequency_hz
phi_ddot = 0                         # constant_frequency_step mode
q = A_stroke cos(phi_D)
q_dot = -A_stroke sin(phi_D) omega
q_ddot = -A_stroke [
    cos(phi_D) omega^2 + sin(phi_D) phi_ddot
]
h_i = -q x_mid_i
h_dot_i = -q_dot x_mid_i
h_ddot_i = -q_ddot x_mid_i
theta_bar = airflow_incidence + mean_pitch_offset
```

The twist function then adds its analytic `delta_theta`,
`delta_theta_dot`, and `delta_theta_ddot`. No formal derivative uses numerical
differencing. The optional experimental phase-acceleration path applies
`numpy.gradient` to phase rate, but the active prior and requested baseline use
`constant_frequency_step`, so `phi_ddot=0`.

## 4. Mechanical-phase mapping and checkpoints

The metadata contract is:

```text
q = 30 deg sin(phi_mech)
phi_mech = 0: neutral wing, starting upstroke
```

The frozen DeLaurier implementation uses cosine stroke, so the explicit mapping
is:

```text
phi_D = wrap(phi_mech - pi/2)
cos(phi_D) = sin(phi_mech)
```

The checked values are:

| Mechanical phase | Internal DeLaurier phase | Stroke angle | Direction |
|---:|---:|---:|---|
| 0 deg | 270 deg | 0 deg | upstroke starts |
| 90 deg | 0 deg | +30 deg | upper reversal |
| 180 deg | 90 deg | 0 deg | downstroke, maximum speed magnitude |
| 270 deg | 180 deg | -30 deg | lower reversal |

This matches the user-specified mechanical-phase checkpoints. All reported
experimental phases must remain in `mechanical_phase_rad`; `phase_D` is an
internal diagnostic only.

One important implementation detail is that the current legacy dynamic-twist
formula receives `phi_D`, not `phi_mech`. A new `psi_theta` described relative
to mechanical phase must therefore either be evaluated in mechanical phase or
be converted explicitly. Treating `psi_theta=0` as both a mechanical-phase and
an internal-phase offset would introduce an unintended 90-degree shift.

## 5. Left/right span and rotation contracts

The strip solver builds one wing with positive root-to-tip coordinates
`x_mid in (0,R)`. The same strip loads and the same twist field are then mapped
to both wings. Thus the current implementation does not pass a negative right
span coordinate into the twist formula.

`_wing_polar_transforms_frd` applies:

- left commanded roll: `left_fixed_roll + q`;
- right commanded roll: `right_fixed_roll - q`;
- a reflected Wang-to-right-link map;
- one FLU-to-FRD polar-vector conversion `diag(1,-1,-1)`.

For moments, `transform_wrench` detects the negative determinant of the right
reflection and applies the axial-vector determinant factor. The final symmetric
relations are tested: total `Fy`, `Mx`, and `Mz` cancel for symmetric geometry,
while total `Fx`, `Fz`, and `My` remain.

A Quadratic-2 profile must consequently use `eta=abs(y)/R` at its public
kinematics boundary or continue passing the shared positive one-wing geometry.
It must never allow signed right-wing span to reverse physical washout.

## 6. Strip force and integrated body-force path

Only the twist generator may branch. Both profiles must feed:

1. `compute_delaurier_strip_loads`;
2. `integrate_delaurier_strip_wrench`;
3. `_wing_polar_transforms_frd`;
4. `transform_wrench`;
5. left/right force summation.

The strip model consumes `h`, `h_dot`, `h_ddot`, `theta`, `theta_dot`, and
`theta_ddot`. Attached-flow components are `dN_c`, `dN_a`, `dT_s`,
`dD_camber`, and `dD_f`; the active prior has separation disabled. The active
prior also records `alpha0=0 deg`, `eta_s=0.65`, `cd_f=0.028`, stall bounds
`[-12,12] deg`, and `c_mac=0`.

Final prior forces are two-wing forces in body FRD at the IMU origin. Moments
are translated from each wing root to the measured aircraft CG. This audit
does not modify IsaacLab and does not rely on its local worktree.

## 7. Airflow reconstruction and current comparison run

The main airflow mode is `attitude_ground_wind_3d`:

```text
V_air_NED = V_ground_NED - [wind_north, wind_east, 0]
V_air_body_FRD = R_body_to_NED^T V_air_NED
alpha = atan2(w_body_down, u_body_forward)
U_used = max(u_body_forward, 0.5 m/s)
```

The quaternion is PX4 `wxyz`, body FRD to NED. Vertical wind is explicitly
zero because the canonical input contains only horizontal EKF wind.

The existing airflow-comparison output is:

```text
outputs/delaurier_airflow_comparison/20260727T111450Z_59d90e6
```

It uses 72 mechanical-phase bins, the active current-wind prior, the
train/validation-only correction-ready artifact
`longitudinal_mean_wb_ratio8_20260721T140238Z_09b4bb6`, and diagnostic
zero-wind prior
`delaurier_zero_wind_ratio8_20260727T111450Z_59d90e6`. Its manifest states
`test_partition_loaded=false` and `selection_performed=false`.

That run was produced from dirty changes on commit `59d90e6`; the implementation
was subsequently committed only in local descendant `b3c565b`. Therefore its
figures and manifests are useful audit evidence, but its CLI is not present at
the required experiment base. Any zero-wind support added here must be reviewed
as new experiment infrastructure, not silently treated as base behavior.

The existing comparison gives the following unsmoothed 72-bin macro extrema:

| Partition | Curve | Fx full-cycle max | Fx max in [180,270] | Fz minimum (FRD) |
|---|---|---:|---:|---:|
| train | data | 212.5 deg | 212.5 deg | 177.5 deg |
| train | current EKF wind | 177.5 deg | 182.5 deg | 162.5 deg |
| validation | data | 217.5 deg | 217.5 deg | 182.5 deg |
| validation | current EKF wind | 177.5 deg | 182.5 deg | 167.5 deg |
| train | zero wind | 177.5 deg | 182.5 deg | 162.5 deg |
| validation | zero wind | 177.5 deg | 182.5 deg | 167.5 deg |

The difference between 177.5 and 182.5 degrees is the declared metric window:
177.5 degrees is the full-cycle maximum bin center, while the primary search
window is closed at 180 degrees and therefore starts with the 182.5-degree bin.

## 8. Phase-binned data and split contract

The canonical dataset contains immutable whole-log assignments:

| Partition | Rows | Logs | Experiment access |
|---|---:|---:|---|
| train | 308,702 | 20 | allowed |
| validation | 79,587 | 5 | allowed for reporting and shortlist selection |
| test | 60,671 | 4 | sealed and forbidden |

The correction-ready waveform table has 382,297 accepted train/validation rows
across 15,360 complete cycles. It preserves `log_id + timestamp_us`, partition,
mechanical `phase_rad`, original label/prior forces, cycle means, and zero-mean
waveforms. Its manifest explicitly excludes test and records no missing or
duplicate alignment keys.

The current formal phase curves used by the airflow comparison:

1. use the correction-ready accepted train/validation rows;
2. bin `phase_rad mod 2pi` into 72 left-closed bins;
3. compute the mean within each `(partition, log_id, phase_bin)`;
4. compute an equal-log macro mean across logs.

This prevents long logs from dominating the displayed train or validation
curve. The Quadratic-2 experiment should reuse this identity, accepted-cycle,
binning, and equal-log aggregation contract. No test path may be accepted by
the experiment CLI.

## 9. Baseline reproduction commands

The active prior can be independently rematerialized without reading test:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python \
  scripts/materialize_authoritative_delaurier_prior.py \
  --dataset-root dataset/canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3 \
  --output-root /tmp/quadratic2_baseline_prior \
  --prior-id delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4 \
  --aircraft-metadata metadata/aircraft/flapper_01/aircraft_metadata.yaml \
  --wing-geometry metadata/aircraft/flapper_01/wing_geometry_isaaclab_3b5d4ec.csv \
  --partitions train validation \
  --chunk-size 4096
```

The existing 72-bin baseline evidence can be inspected without test access:

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python - <<'PY'
import numpy as np
import pandas as pd

table = pd.read_csv(
    "outputs/delaurier_airflow_comparison/"
    "20260727T111450Z_59d90e6/phase_curves.csv"
)
for partition in ("train", "validation"):
    curve = table.loc[
        (table.partition == partition)
        & (table.component == "fx")
        & (table.model == "current_ekf_wind")
    ]
    peak = curve.loc[curve["mean"].idxmax()]
    print(partition, np.degrees(peak.phase_rad), peak["mean"])
PY
```

Expected full-cycle Fx peak phase: `177.5 deg` for both train and validation.
The formal sweep baseline stage must additionally regenerate its own manifest,
metrics, and figure and compare them numerically with the active prior before
any parameter sweep begins.

## 10. Resolved force convention

The confirmed convention is canonical body FRD. The experiment retains `fx_b`
and `fz_b` unchanged, labels every figure and table as `+Fx forward, +Fz down`,
and uses `Fz minimum` for the upward-lift extremum. No body-up presentation
transform is applied.
