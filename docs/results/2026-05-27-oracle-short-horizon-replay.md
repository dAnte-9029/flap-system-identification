# Oracle Short-Horizon Replay Sanity Check

Date: 2026-05-27

This report covers the first-stage oracle-only sanity check for open-loop, log-seeded short-horizon replay. The replay uses held-out effective-wrench labels as the applied wrench and does not claim closed-loop simulator validation.

## Command

```bash
python scripts/evaluate_short_horizon_replay.py --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 --metadata-path metadata/aircraft/flapper_01/aircraft_metadata.yaml --output-root artifacts/20260527_short_horizon_replay_v1/oracle_sanity --split test --modes oracle_teacher_forced,coupled_oracle --horizons 0.10,0.25,0.50,1.00,2.00 --stride-s 0.25 --overwrite
```

## Data Volume

- Split: `test`
- Modes: `oracle_teacher_forced`, `coupled_oracle`
- Logs: 4
- Window metric rows: 23,880
- Artifact root: `artifacts/20260527_short_horizon_replay_v1/oracle_sanity`

## Horizon Summary

Median / p90 errors by horizon:

| mode | horizon_s | n_windows | pos_med_m | pos_p90_m | vel_med_m_s | vel_p90_m_s | att_med_deg | att_p90_deg | rate_med_rad_s | rate_p90_rad_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_teacher_forced | 0.10 | 2432 | 0.022 | 0.042 | 0.169 | 0.311 | 2.70 | 4.95 | 0.534 | 1.064 |
| oracle_teacher_forced | 0.25 | 2422 | 0.053 | 0.099 | 0.064 | 0.157 | 5.32 | 9.99 | 0.428 | 0.929 |
| oracle_teacher_forced | 0.50 | 2406 | 0.105 | 0.204 | 0.102 | 0.229 | 10.70 | 20.70 | 0.536 | 1.050 |
| oracle_teacher_forced | 1.00 | 2372 | 0.216 | 0.422 | 0.145 | 0.299 | 22.12 | 42.61 | 0.577 | 1.146 |
| oracle_teacher_forced | 2.00 | 2308 | 0.482 | 0.928 | 0.174 | 0.343 | 46.38 | 88.74 | 0.609 | 1.210 |
| coupled_oracle | 0.10 | 2432 | 0.022 | 0.042 | 0.174 | 0.313 | 2.70 | 4.97 | 0.534 | 1.066 |
| coupled_oracle | 0.25 | 2422 | 0.054 | 0.101 | 0.142 | 0.256 | 5.32 | 10.03 | 0.432 | 0.924 |
| coupled_oracle | 0.50 | 2406 | 0.136 | 0.246 | 0.466 | 0.895 | 10.77 | 20.75 | 0.539 | 1.063 |
| coupled_oracle | 1.00 | 2372 | 0.628 | 1.190 | 1.777 | 3.488 | 22.28 | 43.05 | 0.587 | 1.167 |
| coupled_oracle | 2.00 | 2308 | 4.737 | 8.995 | 7.300 | 13.344 | 47.60 | 91.39 | 0.635 | 1.234 |

## Per-Log Notes

Teacher-forced translational errors are consistent across the four held-out logs. At 0.25 s, median velocity error ranges from 0.051 to 0.081 m/s and median position error ranges from 0.046 to 0.062 m. At 2.00 s, teacher-forced median position error remains below 0.56 m for every log, but attitude median error ranges from 40.75 to 51.51 deg.

Coupled oracle drift grows much faster in translation after 0.50 s. At 2.00 s, coupled median velocity error is 6.24 to 8.34 m/s by log, and coupled median position error is 4.09 to 5.58 m.

## Conclusion

Conditional pass for translational oracle sanity at 0.10-0.25 s; fail as a full six-DoF oracle sanity check because rotational replay drift is too large even in teacher-forced mode.

The short-horizon translational numbers indicate that the force labels, mass, gravity sign, and body-to-NED quaternion convention are at least broadly self-consistent for diagnostic reset windows. The rotational channel does not provide a clean oracle baseline: median attitude error is already 2.70 deg at 0.10 s and 5.32 deg at 0.25 s, with body-rate median error around 0.43-0.53 rad/s. The most likely issue sources are moment-label time alignment, inertia or CG metadata, angular-acceleration label noise, or the simple explicit integration scheme. A quaternion sign convention failure is less likely because the translational teacher-forced channel uses the same body-to-NED convention and remains plausible at short horizons.

Based on this result, prior/corrected model comparison should not be interpreted as validated six-DoF replay until the rotational oracle mismatch is diagnosed. If model comparison proceeds later, translational claims should be separated from rotational claims and framed strictly as open-loop, log-seeded local replay.
