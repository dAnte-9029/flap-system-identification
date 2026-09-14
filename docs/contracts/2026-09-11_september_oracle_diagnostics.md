# Offline attitude/rate oracle diagnostic

Authorized to execute the four-arm diagnostic on the frozen history_26 checkpoint.
This is NOT a new forecasting model, training experiment, RL simulator or promotion.
No production model/training changes. Diagnostic rollout lives only in evaluation.

## Frozen inputs

Use source artifacts from september_history_lengths_20260911, registered September
v2 dataset, exact 3,966 train 2s and 6,669 validation 5s windows. History remains
26 real samples ending at t0; use saved normalization and control tapes, actual dt,
GPU 1, batch 256. Verify source hashes and exact window identities. Use common
initial state. All future truth channels except the explicitly selected oracle
are metric targets only. No sealed September 8 reads and no fitting.

## Four arms and intervention timing

- free_run: identical to production Main V1; no oracle inputs accepted. Preflight
  checks exact tensors against production, and saved errors within rtol=1e-6,
  atol=1e-4 (small batch-size numeric variation). Full run verifies all saved
  position/velocity/attitude/rate errors before other arms in each partition.
- oracle_attitude_rotation: use q_true[k] only to rotate predicted body acceleration
  to NED. q and omega state remain predicted. Subsequent velocity, network inputs
  and hidden state may change indirectly; this is not a fixed-derivative replay.
- oracle_attitude_feedback: q_true[k] is used in current state features and rotation;
  q_true[k+1] replaces integrated q for next-state recurrent features and stored q.
  omega, body acceleration, frequency, phase and translational state remain learned.
  Stored q is forced; its error is NOT model performance.
- oracle_rate: omega_true[k] used as current omega in features; omega_true[k+1]
  replaces predicted next omega before midpoint quaternion integration and recurrent
  update. Predicted angular acceleration is discarded for omega evolution. q is
  integrated from these rates and is never replaced by true attitude. This uses
  future endpoint omega, explicitly diagnostic. Stored omega error is forced and
  NOT model performance. Logging/attitude-estimator inconsistency may still limit q.

No interventions reset position or velocity. No group combines attitude and rate
oracle inputs. The distinction between direct intervention and indirect feedback
must be retained in interpretation; contrasts are not additive causal percentages.

## Evaluation and outputs

Train 1/2s, validation continuous 1/2/3/5s. Save all predicted states and all 50Hz
error traces with window identity; fixed endpoint tables plus per-log/equal-log
curves. Main errors: logged position, velocity, unforced attitude and angular rate.
Auxiliary position target is true p[0] plus trapezoid integral of true NED velocity;
it helps distinguish estimator output consistency from rollout error and is NOT
an alternative official trajectory label. Save it as integrated_position_error_m.

Forced q/omega metrics are NaN in summary curves, with explicit forced flags;
raw error arrays/endpoints retain values and flags for audit only. Compare position
>10m fraction across arms; do not compare the old position-or-attitude threshold
when one arm forces attitude. Nonfinite predictions are counted, not dropped.

This diagnostic can identify influential feedback pathways, not prove a feasible
learned improvement. Oracle benefit is neither an achievable bound nor evidence
of RL-ready action response. Keep validation-conditioned future modeling proposals
separate from completed evidence; do not remove difficult logs.

## Run contract

Entry: scripts/run_september_oracle_diagnostics.py. --preflight creates no experiment
output. Default output: artifacts/september_oracle_diagnostics_20260911, with sibling
.run.log and .launch.json. Refuse existing output, no automatic retries. Background
execution without interactive monitoring; failed runs save traceback and status.
No completion claim until summary/status and metric artifacts exist.
