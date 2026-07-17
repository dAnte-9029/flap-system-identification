# DeLaurier prior authority and legacy policy

Date: 2026-07-17

## Authoritative prior

New longitudinal-force analysis resolves its default prior through
`configs/physics/delaurier_prior_registry.yaml`. The current active contract is
`delaurier_attitude_aware_3b5d4ec_trainval_v1`, frozen from IsaacLab commit
`3b5d4ec1d28f1384cf042402992ad7ea59995f49`.

Its force output is wing-only DeLaurier force expressed in canonical body FRD.
The airflow input is reconstructed from NED ground velocity, horizontal wind,
and the logged body-to-NED attitude quaternion. The current frozen candidate is
attached flow, with separation disabled and prescribed dynamic twist disabled
(zero tip amplitude). Complete train and validation keyed predictions are
required before EDA or correction-structure diagnostics may run.

`latest` means the registry entry marked `active`, not the directory with the
largest date and not the prior referenced by the most recent downstream model.
Promoting a future prior therefore requires a new immutable artifact and an
explicit registry update.

## Legacy prior and pipeline

`delaurier_separation_20260604_v1` and the downstream
`artifacts/20260623_final_effective_force_fx_fz_v1` pipeline are legacy. They
remain available for historical reproduction, but they are not valid defaults
for new analysis. The legacy exporter used scalar true airspeed plus a pitch
proxy, a different phase/stroke reconstruction, separation, and the historical
`qd`-scaled twist proxy. Its recorded IsaacLab source worktree was dirty.

Legacy use must be explicit via `--allow-legacy-prior` and must be recorded in
the run manifest. Missing authoritative data is an error and never triggers an
automatic legacy fallback.

## Test isolation

The July-14 theta-sweep output under
`outputs/wing_wrench_theta_sweep/20260714_test_partition_attitude_airflow_frozen_3b5d4ec`
is a diagnostic over selected test windows. It establishes implementation and
frame provenance but is not a training-ready prior bank. New EDA uses only the
materialized train and validation predictions and does not load test labels.
