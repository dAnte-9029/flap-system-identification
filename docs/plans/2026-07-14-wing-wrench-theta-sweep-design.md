# Wing-Only DeLaurier Wrench Theta-Sweep Design Record

Date: 2026-07-14
Repository baseline: `main` at `609b1835981af30685cbfcdfdb4306bd13d8a4ea`
Frozen IsaacLab source: `dAnte-9029/IsaacLab` `flapping_rl` at `3b5d4ec1d28f1384cf042402992ad7ea59995f49`

## Comparison contract

The analysis compares a **wing-only DeLaurier baseline** with the **total reconstructed effective-wrench label**. The label may contain wing, tail, fuselage, disturbance, left/right asymmetry, and reconstruction-error contributions. It is therefore not an isolated wing ground truth.

## Audit answers

1. **Existing force-only prior.** The aerodynamic equations are not implemented in this repository. `scripts/flapping_px4/export_delaurier_prior_predictions.py` in the frozen IsaacLab repository evaluates `compute_aero_wrench_delaurier1993`, maps the two-wing force into FRD, and writes keyed Parquet files. This repository consumes those files through `scripts/build_delaurier_residual_split.py`, `scripts/materialize_keyed_prior_for_split.py`, calibration scripts, and `scripts/diagnose_delaurier_prior_conventions.py`. Existing exported moments are explicit zero placeholders.
2. **Same and different physics.** Both paths use the same chord-distribution CSV, 80 uniform strips, `h=-q y`, logged instantaneous frequency, true airspeed, logged density, and the DeLaurier strip-load equations. The old exporter uses `q=A sin(phi)`, a historical full-span-uniform `qd`-scaled twist proxy, an aggregate force mapping, float32, and zero moment. The frozen model uses `q=Gamma cos(phi_D)`, prescribed linear-span dynamic twist, attached strip components, strip-integrated force-arm and free-couple moments, polar/axial left-right mapping, and wing-root-to-COM translation.
3. **Canonical inputs.** Direct columns are `mechanical_phase_rad`, `flap_frequency_hz`, `airspeed_validated.true_airspeed_m_s`, `vehicle_air_data.rho`, `airspeed_validated.pitch_filtered`, `phase_valid`, `cycle_valid`, `label_valid`, `log_id`, `segment_id`, `time_s`, `timestamp_us`, and the six label columns `fx_b` through `mz_b`. The first implementation uses these same-row columns and does not construct a second timeline.
4. **Phase, frequency, airspeed, and rho.** Mechanical phase is the logged `wing_phase` when available, otherwise the metadata-calibrated encoder phase. Canonical motion is `q=A sin(phi_C)`, with phase zero at neutral wing starting upstroke. The frozen cosine phase is mapped explicitly as `phi_D = wrap(phi_C - pi/2)`, so `Gamma cos(phi_D)=A sin(phi_C)` and phase-rate sign is unchanged. Main-mode phase rate is `2 pi flap_frequency_hz` and phase acceleration is zero. Airspeed is raw `airspeed_validated.true_airspeed_m_s`, clipped only by the frozen 0.5 m/s low-speed guard. Density is the same-row logged `vehicle_air_data.rho`. Axis incidence follows the existing offline prior contract, `airspeed_validated.pitch_filtered`; the current selected split contains this field but it is degenerate, so the manifest must record that vertical airflow and sideslip are not used.
5. **Moment-label reference.** `pipeline._compute_effective_wrench_labels` computes `I alpha + omega x I omega`; metadata defines it as effective external moment about the whole-aircraft CG. The baseline must also end about that CG.
6. **Frames.** Canonical input/output is body FRD (`+x` forward, `+y` right, `+z` down). The DeLaurier/Wang strip frame is right-handed with `+x` root-to-tip, `+y` surface-normal, and `+z` chordwise toward the leading edge. Canonical FRD airflow is already the DeLaurier section convention and must not receive an additional FLU flip. Isaac URDF/link geometry is FLU and is converted once through `diag(1,-1,-1)` before final FRD output. Force is a polar vector; moment is an axial vector, including determinant parity for the mirrored right-wing Wang-to-link map.
7. **Origins and geometry.** Canonical body origin is the IMU origin. Metadata gives real-aircraft CG relative to IMU as `[-0.12154, 0.00541, -0.04298] m`. Frozen Isaac geometry uses `base_link` as its body origin, wing-root joint positions `[0,+/-0.056,0] m` in FLU, and measured COM `[-0.12154,0.00541,-0.01298] m` relative to `base_link`. Metadata states the IMU is `[0,0,0.030] m` in the aircraft construction frame, so the offline FRD wing roots relative to IMU are left `[0,-0.056,-0.030] m` and right `[0,+0.056,-0.030] m`. Moments are first integrated about each wing-root pitching-axis origin and then translated with `M_G=M_O+(p_O-p_G) cross F` to the real metadata CG. Geometry and origin sources are kept separately in the run manifest rather than silently mixed.
8. **Reuse versus addition.** Reuse metadata loading, canonical phase semantics, complete-segment grouping from `signal_preprocessing.iter_groups`, gap splitting from `plot_prediction_curves` after promoting it to a public plotting helper, and the keyed long-table convention. Add isolated pure NumPy modules for geometry/airflow/twist/strip wrench, an IsaacLab-frozen wing baseline adapter, and analysis utilities for windows, metrics, cycle means, phase bins, and plots. Existing exported-prior code and artifacts remain untouched.

## Post-audit airflow correction

The initial compatibility run above exposed that `airspeed_validated.pitch_filtered` is degenerate in the selected split and that scalar true airspeed suppresses the real body-vertical airflow. The primary analysis was therefore rerun without changing the frozen strip physics. For every canonical row it now reconstructs

\[
v_{air}^{NED}=v_{ground}^{NED}-v_{wind}^{NED},\qquad
v_{air}^{body}=R_{body\rightarrow NED}^{T}v_{air}^{NED},
\]

using the real PX4 `wxyz` attitude quaternion, NED ground velocity, and logged north/east wind. Vertical wind is unavailable and explicitly zero. The frozen model receives `U=max(u_b,0.5 m/s)` and `theta_a=atan2(w_b,u_b)`. Body lateral airflow and sideslip are saved diagnostically but are not inputs to the frozen two-dimensional section equations. The old scalar mode remains isolated as `legacy_scalar_true_airspeed` and is used only for direct comparison figures.

## Label bandwidth audit

The selected real-data split `dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1` reconstructs the labels from Savitzky-Golay-smoothed kinematic derivatives (`0.03 s`, polynomial order 3). It does not apply a post-hoc filter to the final wrench. The split separately filters airspeed at 5 Hz and flap frequency at 12 Hz, but the existing offline prior manifest uses the unfiltered canonical columns. Because there is no single linear operator that maps a raw aerodynamic baseline to these inverse-dynamics labels, the main artifact is `baseline_raw`; no `baseline_label_bandwidth` claim will be made. Per-wingbeat cycle means provide the lower-bandwidth comparison.

## Implementation decisions

- Keep the new six-axis baseline opt-in and independent of IsaacLab/Isaac Sim/PhysX.
- Vendor the frozen wing geometry with source commit and hash.
- Use float64 for offline evaluation and a frozen numerical fixture; preserve float32 fixture tolerance when comparing with the source implementation.
- Evaluate every complete `(log_id, segment_id)` before window cropping.
- Default theta sweep is `0, 5, 10, 15 deg`; it is sensitivity analysis only.
- Store aggregate strip-component sums by default; make full per-strip debug output optional.
- Automatic windows are deterministic, require valid labels/phase/airspeed, avoid gaps and short edge cycles, and rank 4 s candidates on a 2 s grid without inspecting residual error. Low/high airspeed windows are restricted to the better half of stability scores; mild pitch/turn windows are nearest their channel's 75th-percentile RMS.
- `docs/PROJECT_STATE.md` is absent at audit time. The implementation will create only the minimal requested project-state entry rather than infer prior project state.
