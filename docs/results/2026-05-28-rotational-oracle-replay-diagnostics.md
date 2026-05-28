# Rotational Oracle Replay Diagnostics

Date: 2026-05-28

This report covers a narrowed diagnostic deliverable for the held-out oracle rotational replay failure described in `docs/results/2026-05-27-oracle-short-horizon-replay.md`. The exact full sensitivity command was attempted, but it exceeded about 10 minutes without completing and was stopped. The generated results therefore cover only the first three gates: kinematic attitude closure, moment-label closure, and forward omega replay.

These results do not compare prior/corrected six-degree-of-freedom models and do not validate closed-loop simulation.

## Commands and Status

| command | status | evidence |
| --- | --- | --- |
| `pytest tests/test_rotational_replay_diagnostics.py -q` before implementation | fail as expected | initial red failure: `ModuleNotFoundError: No module named 'scripts.diagnose_rotational_replay_oracle'` |
| `python -m py_compile scripts/diagnose_rotational_replay_oracle.py` after implementation | pass | exit code 0 |
| `pytest tests/test_rotational_replay_diagnostics.py tests/test_short_horizon_replay.py -q` after implementation | pass | `20 passed in 1.64s` |
| exact full diagnostic command from the plan | fail to complete within cutoff | stopped after about 11 minutes; full lag/smoothing/inertia/reference/spike sensitivity sweep was not completed |
| scoped first-three-gates diagnostic run | pass | wrote scoped artifacts under `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/` |

The attempted full command was:

```bash
python scripts/diagnose_rotational_replay_oracle.py --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml --output-root artifacts/20260528_rotational_oracle_replay_diagnostics_v1 --split test --horizons 0.10,0.25,0.50,1.00,2.00 --stride-s 0.25 --lags-s -0.100,-0.080,-0.060,-0.040,-0.020,0.000,0.020,0.040,0.060,0.080,0.100 --overwrite
```

## Artifact Evidence

Scoped artifacts:

- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/attitude_kinematic_closure.csv`
- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/attitude_kinematic_closure_summary.csv`
- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/moment_label_closure.csv`
- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/omega_replay_window_metrics.csv`
- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/omega_replay_summary.csv`
- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/diagnostic_decision.json`
- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/config.json`
- `artifacts/20260528_rotational_oracle_replay_diagnostics_v1/README.md`

`diagnostic_decision.json` records `full_sensitivity_sweep_completed: false` and lists the skipped diagnostics: time lag, smoothing/noise, inertia/fitted inertia, reference-point/CG transform, and spike/outlier robustness.

## Gate Results

| gate | result | key evidence |
| --- | --- | --- |
| Kinematic attitude closure | pass | nominal `right_multiply_rate_plus` is best at 0.25 s, median attitude error 0.365 deg, p90 0.764 deg |
| Moment-label closure | pass | moment-norm RMSE `4.75e-20` N m and alpha RMSE `5.88e-15` rad/s^2 across 60,671 samples |
| Forward omega replay | fail | nominal logged-gyro Euler 0.25 s median body-rate error 0.428 rad/s, p90 0.929 rad/s; best completed variant, trapezoid alpha, still has 0.255 rad/s median error |

The attitude gate indicates the quaternion convention used here is consistent with the logged body-rate kinematics over short windows. The moment-label closure gate indicates that the stored moment labels are algebraically consistent with the logged angular acceleration columns and metadata inertia. The forward omega replay gate still fails, so the rotational oracle mismatch is not explained by a basic quaternion multiplication/sign error or by moment-label algebra mismatch.

## Ranked Likely Causes

Based only on the completed gates:

1. Forward rotational dynamics mismatch after kinematic and label-algebra closure. Logged rates integrated from oracle moments still do not close at the body-rate level.
2. Integration method contributes but is not sufficient. Trapezoid alpha improves the 0.25 s median body-rate error from 0.428 rad/s to 0.255 rad/s, but this remains too large for a clean oracle replay sanity check.
3. Unresolved timing/noise/inertia/reference/spike causes remain plausible. They cannot be ranked from this run because the full sensitivity sweep timed out and was not completed.

## Claim Boundary

Full six-degree-of-freedom prior/corrected replay remains blocked. The completed diagnostics support only this narrower statement: attitude kinematics and moment-label algebra are internally consistent, but forward rotational replay remains unresolved. This is not closed-loop simulator validation and should not be used as a six-degree-of-freedom replay validation claim.

The paper-safe boundary is:

```text
The replay diagnostics indicate that attitude kinematics and moment-label algebra close under the logged conventions, but forward rotational oracle replay remains unresolved. We therefore do not use six-degree-of-freedom replay as a validation claim in this version; replay evidence is restricted to translational/local force consistency unless follow-on rotational sensitivity diagnostics pass.
```

## Recommended Next Step

Run scoped lag, smoothing, inertia, reference-point, and spike diagnostics separately, preferably as smaller commands or cached/batched jobs. Do not run or interpret prior/corrected six-degree-of-freedom replay comparison until those sensitivity diagnostics explain the remaining forward omega replay error.
