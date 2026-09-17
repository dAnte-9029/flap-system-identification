# Main V2 stateful simulator / validation benchmark contract

Scope: frozen August Main V2 checkpoint SHA256
`0ca42b8a3cca07f507b4a52a910daf130bd98da31d590669bf6c47b6ac3031ff`.
No training, architecture change, split change or sealed-test access. The existing
`ActuatorAwareTrajectoryModel.forward` remains unchanged as the parity reference.

## State and initialization

`MainV2Simulator.step(state, command, dt)` returns `(next_state, diagnostics)`.
With fixed model weights, buffers, drive/tail time constants, state + command + dt
uniquely determine the next state (YES). The state is batched and contains:

- NED position/velocity, body-to-NED wxyz quaternion, FRD body rates;
- flapping frequency in Hz and log-relative phase in radians;
- 64-component GRU hidden state;
- scalar normalized drive proxy, three-component normalized tail proxy
  (symmetric, differential, rudder);
- fixed phase anchor in radians.

No redundant body velocity/gravity/phase sine/cosine, actuator output, clock,
or derivative is stored. They are derived or externally supplied. The training
regularization cache is not dynamic memory and does not enter this interface.
The adapter holds a reference to the existing model; no weight copying or change.

`reset(**warm_inputs)` uses the exact existing masked history encoder and filters.
`reset(state=restored_state)` copies and validates all state tensors without
re-encoding history, normalizing quaternion, or changing phase anchor. Serialize
`state.snapshot()` with `torch.save`, load with `weights_only=True`, then use
`SimulatorState.from_snapshot(..., device=...)`. Snapshots must be paired with the
same model weights and configuration; they do not contain model weights.

Warm26 is the primary benchmark. Warm13 and warm5 shorten only the available past.
Cold-single encodes t0 once, matching existing one-point-history behavior;
cold-zero uses zero GRU hidden and current-command steady proxies;
cold-repeated26 encodes identical t0 features 26 times, explicitly assuming a
stationary prehistory. These three expose sensitivity to unknown initial memory,
not three newly trained models. Their proxy initialization is identical, so the
comparison isolates GRU initialization within the cold group.

## Frozen evaluation design

Use `dataset/trajectory_v1_august_f5_c4` explicitly, as required for the historical
checkpoint. Resolve its split from its existing manifest and verify the actual
sample log IDs. Record hashes of manifest, train/validation samples, checkpoint
and executed sources. This manifest predates artifact hashes, so recorded hashes
freeze this run; they are not a claim that older unrecorded hashes were verified.
Train samples supply descriptive envelope statistics only. Validation has five
flights; no test file is opened, even for range statistics.

Within each valid `(log_id, segment_id)`, start at index 25 and advance by 50
samples (about one second). Require all 26 history states and 251 rollout states,
contiguous sample indices and positive native gaps <= 0.05 s. Never interpolate
or cross a segment. Use the same eligible origins for every initialization and
horizon. Report exclusions for short segments and each flight's selected origins.
The 5-second requirement follows the existing nominal step-count contract:
250 native steps, with actual elapsed duration reported rather than forced to
exactly 5.000 s. Report actual dt and history-duration distributions. Preserve the
legacy 0.02 s history-filter dt; changing it would change the initialization.

Horizon steps: 10/25/50/100/150/250, labeled 0.2/0.5/1/2/3/5 s. Future commands are
post-allocation motor/left/right/rudder values, excluding the terminal sample.
Future truth is available solely for offline scoring/diagnostics after rollout.

Metrics: horizontal/vertical/3D position, per-axis and vector velocity, geodesic
attitude, per-axis and vector body rates, frequency, circular phase. Report
mean/median/p90/p95/max and RMSE, plus equal-log RMSE and per-flight tables.
RMSE is over endpoint error (not quaternion components); axis error distributions
use absolute errors, whose squares give axis RMSE. Include finite failed paths;
nonfinite metric exclusions have explicit finite counts and failure counts.
Overlapping starts are correlated and are not independent flight replications.

A frozen constant-twist baseline (same t0, tape and dt) provides context. It does
not establish absolute acceptable flight error or a trusted counterfactual model.

## Gates and finite-but-wrong diagnostics

Before full evaluation, define these independent gates:

1. Numeric: all persistent state and derivative outputs finite; quaternion norm
   deviation <= 1e-4.
2. Observed support: speed, body-rate norm, frequency, acceleration norm and
   angular-acceleration norm stay within train min/max. Report train/validation
   min/max/p1/p99 and excursions outside p1/p99 separately. The identical support
   test on real validation trajectories reports how much genuine held-out flight
   already lies outside training support. These are conservative support gates,
   **not certified physical safety limits**. Acceleration is native-step velocity
   difference/dt, with no new smoothing; its noisy extrema limit interpretation.
3. Clipping: any pre-clamp drive/residual/derivative/frequency clipping is a failure
   signal; count affected steps, clipped scalar components and affected rollouts.
   The built-in frequency clamp is not the observed envelope.

`failed` is the union, cumulative to each horizon. Continue finite failed paths to
measure deterioration, retaining first-failure time. A completed run exits 2 if
any primary warm26 path fails by 5 s; save all reports before this exit. The exit
is an explicit benchmark FAIL, not an execution error. p1/p99 excursions alone do
not trip the min/max gate. A pass still does not prove RL safety.

Paired final-one-second state variation and 0-to-5-second drift expose damping
and abnormal equilibria. Diagnostic flags, not certified causes:

- angular damping suspect: predicted last-second body-rate vector standard
  deviation < 0.5 × truth, with truth variation > 0.001 rad/s;
- unsupported equilibrium suspect: both velocity and rate variation < 0.25 × truth
  and final state outside train support;
- energy-growth suspect: among the lowest third of validation command total
  variation, speed-squared or rate-squared rises more than truth and >75% of its
  increments are positive. Keep translational and rotational proxies separate;
  they are not aircraft energy without mass/inertia and aerodynamic accounting.

These relative thresholds are declared heuristics, not tuned on results. Report
continuous ratios and growth statistics as primary evidence. Low command
variation does not imply zero physical power. Offline continuous state bins use
train tertiles of vertical velocity, |body yaw rate|, |roll|, pitch, motor command
and tail-command norm at t0. Do not name these bins maneuvers. Future command
variation tertiles are descriptive validation strata only, never model inputs.

## Readiness reporting rubric

The requested percentages are explicitly evidence-checklist scores, not calibrated
probabilities of safe RL use. Structural score: ten equal items (sufficient state,
warm reset, explicit reset, serialization, preserved anchor, pure step, legacy
parity, resume parity, label-leakage test, documented fixed-model contract).

Fidelity score: ten equal evidence items: all 5 s paths numeric-pass; all within
train support; no clipping; lower equal-log RMSE than constant twist at **every**
horizon for velocity, attitude, body rate, and frequency (four items); no flagged
finite-but-wrong suspects; an externally specified absolute accuracy acceptance
criterion met; validated counterfactual action response. The last two have not
been supplied/proven and cannot be awarded by this benchmark. This rubric cannot
by itself authorize RL. Show individual items so the percentage is auditable.
