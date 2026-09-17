# Step 4 phase / input contract audit

Only the explicit historical August F5/C4 dataset and its preserved train/validation flights are used. This is not a canonical-dataset migration. Source hashes and the complete Step 3 result hashes are recorded in the new result manifest. No sealed-test ULog or sample artifact is loaded.

## Exact phase path

1. ULog `encoder_count.total_count` is cumulative motor encoder count; `position_raw` is also logged. August has no usable Hall-based mechanical zero. Logged `wing_phase.phase_rad` and absolute-phase placeholders cannot provide a valid common zero.
2. `data/trajectory_dataset.py::relative_phase_from_total_count` computes `(count - first_finite_aligned_count) * 2*pi / (4096 * audited_FLAP_RATIO)`. The origin is the first finite count **after alignment to the log's local-position reference**, not a new origin at each segment. `build_log_samples` uses past-only zero-order hold (encoder freshness 0.10 s). The stored `relative_flap_phase_unwrapped_rad` preserves continuous within-log phase; `relative_flap_phase_rad` is its wrapped coordinate. `wing_phase.flap_frequency_hz` is aligned separately, with the existing `flap_frequency.frequency_hz` fallback.
3. `training/trajectory_main_v1.py::assemble_history_trajectory_windows` chooses `phase_anchor = selected.iloc[-1].relative_flap_phase_rad`, i.e. the current window t0. All history samples subtract this same anchor before sin/cos encoding. Their last phase pair is always (0,1).
4. `models/trajectory_main_v1.py::_state_features` subtracts `phase_anchor_rad` again when forming each rollout step's feature. Both the legacy Main V2 forward and the stateful adapter initialize the anchor to t0 phase. The GRU sees phase elapsed since that rollout's t0, not stored encoder phase.
5. `models/main_v2_simulator.py::step` integrates frequency with actual dt and phase with trapezoidal frequency: `phi_next = remainder(phi + 2*pi*(f+f_next)/2*dt, 2*pi)`. The anchor remains fixed across steps and pause/resume. This retains internal recursion consistency but does not restore physical phase position.

Consequences: the network phase is **per-window / rollout-t0 relative**, while the stored phase is **per-log relative**. Two windows at the same within-log encoder phase need not have equal network phase features if their anchors differ. Two t0 phase features are always equal even when the wings are at different cycle positions. Across logs, even equal stored phase does not establish equal mechanical phase. Physical-state histories may nevertheless indirectly reveal cycle position; re-anchoring does not prove that every representation of phase information has disappeared.

## Timing and control sensitivity

The base model is history-only (`use_controls=False`). At a step, the derivative depends on the old drive/tail proxies. The current command updates these proxies at the end of that step, so the derivative's immediate command Jacobian is exactly zero by construction. The audit also measures responses one and five steps later, applying the same small command perturbation continuously and retaining the native operating-point dt. Those responses include recurrent state changes; they are not an unknown true-aircraft Jacobian.

Tail proxies are dimensionless filtered post-allocation commands, not actual servo positions. Structural output masks route symmetric tail to ax/az/q-dot, differential tail to p-dot, and rudder to ay/r-dot; the drive residual initially routes to frequency-dot only. Initial zero cross-axis sensitivities follow these masks. Small nonzero masked finite differences can be float32 cancellation.

## Units and corrected Step 3 interpretation

The approximately 3.30 rad/s² prediction std in Step 3 is computed over **recursive free-run** points. The teacher-state one-step vector std is approximately 12 rad/s² on validation, versus approximately 30 rad/s² for native state differences. Step 4 records both contexts without changing Step 3 files.

Linear derivatives in this audit are `d(v_NED)/dt`, not accelerometer specific force. Angular derivatives are body-FRD `d(omega)/dt`. Zero-phase Butterworth filters are **offline target diagnosis only**. No filtered future-dependent state is passed into Main V2 or the diagnostic model inputs.

## Telemetry interpretation

The audit opens only the 11 manifest-authorized train/validation ULogs. Per-topic availability, native publication rate/gap, and past-aligned core-row freshness/range are in the CSV files. Topics are not treated as sensors simply because they exist.

- `airspeed_quality_input.input_source=1` denotes differential-pressure input in the available firmware message reference; `airspeed_validated.airspeed_source=1` denotes sensor 1. Wind is estimator output, not a direct wind measurement. Pitot-derived speed can contain flapping contamination; a confidence field fixed at 1 does not certify accuracy.
- `actuator_servos` and `actuator_outputs` are commanded outputs/PWM-like outputs, not measured angles. No `servo_feedback`, `esc_report`, or `esc_status` appears in these 11 logs.
- `battery_status.current_a` is pack current, not motor torque/current telemetry. `voltage_v` is pack voltage. `rpm_raw` and `rpm_estimate` are encoder-derived motor speed channels, not independent aerodynamic state.
- `vehicle_angular_velocity.xyz_derivative` is a published sensor-derived derivative and not independent physical ground truth. Logged `vehicle_acceleration` is only about 1 Hz, so it cannot independently validate the observed 4–13 Hz rigid-body harmonic components.

Message references were inspected read-only at `/home/zn/PX4-Autopilot/msg/versioned/AirspeedValidated.msg` and `/home/zn/PX4-Autopilot/ral_phase4_1_delivery/untracked/msg/AirspeedQualityInput.msg`. These local references explain field names/enums; they do not certify that the current firmware tree exactly matches every historical flight binary. Actual logged field values remain primary evidence.
