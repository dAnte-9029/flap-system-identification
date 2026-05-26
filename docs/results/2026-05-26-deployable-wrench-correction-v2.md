# Deployable Wrench Correction v2

This run evaluates a deployable v2 wrench correction on the held-out log split. Selection used validation RMSE only. No deployable metric or prediction used `true_force`; the v1 dynamic-arm artifact's `true_force` rows are excluded from all deployable comparisons below.

## Run

```bash
python scripts/train_deployable_wrench_correction_v2.py --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 --prior-root artifacts/delaurier_physical_prior_v1 --force-v1-root artifacts/20260525_delaurier_greybox_force_correction_v1 --moment-v1-root artifacts/20260525_dynamic_arm_moment_head_v1 --output-root artifacts/20260526_deployable_wrench_correction_v2 --alphas 0,0.001,0.01,0.1,1,10,100,1000 --overwrite
```

The selected force v2 model is `affine`, feature group `base+rates+controls+lateral+interactions`, alpha `1`. The selected moment v2 model is `force_arm`, feature group `base+rates+controls+lateral+interactions`, alpha `1000`.

## Test Force Metrics

| target | prior RMSE | force v1 RMSE | force v2 RMSE |
| --- | ---: | ---: | ---: |
| fx_b | 5.420306 | 1.817299 | 1.632014 |
| fy_b | 3.523583 | 1.398994 | 1.337994 |
| fz_b | 13.105000 | 3.257462 | 2.992153 |
| force_mean | 8.436752 | 2.300059 | 2.113975 |

Adding the residual-attribution variables is associated with a further `fy_b` improvement over force v1: test RMSE decreases from `1.398994` to `1.337994`. The improvement is not driven by one held-out log; v2 improves over force v1 on `fy_b` in all 4 test logs.

## Test Moment Metrics

| target | prior RMSE | v1 corrected-force best RMSE | v1 features-only RMSE | v2 direct_residual RMSE | v2 force_arm RMSE | v2 hybrid RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mx_b | 0.071418 | 0.003097 | 0.003086 | 0.023137 | 0.002764 | 0.059724 |
| my_b | 0.601734 | 0.002388 | 0.002447 | 0.224164 | 0.002240 | 0.294321 |
| mz_b | 0.101904 | 0.000515 | 0.000482 | 0.087919 | 0.000551 | 0.090169 |
| moment_mean | 0.354762 | 0.002277 | 0.002291 | 0.139676 | 0.002078 | 0.181040 |

Each v2 model-form column reports the validation-selected candidate within that form, then evaluates it on test. The deployable `force_arm` v2 candidate improves `mx_b`, `my_b`, and moment mean relative to the corrected-force v1 dynamic-arm baseline. It does not improve `mz_b`: the v1 features-only direct moment row remains best for `mz_b` (`0.000482` vs v2 `0.000551`). The v2 `direct_residual` and `hybrid` forms are negative results on this split and should not be carried forward as selected deployable candidates.

## Per-Log Stability

| log_id | n | fy v1 RMSE | fy v2 RMSE | mx prior RMSE | mx v2 RMSE | mz prior RMSE | mz v2 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| log_18_2026-4-15-12-56-08 | 14202 | 1.502038 | 1.419655 | 0.052522 | 0.002635 | 0.097475 | 0.000527 |
| log_34_2026-4-16-19-13-30 | 14074 | 1.127175 | 1.098101 | 0.046409 | 0.002125 | 0.107355 | 0.000414 |
| log_4_2026-4-12-17-43-30 | 18184 | 1.451491 | 1.418846 | 0.097896 | 0.003125 | 0.093470 | 0.000610 |
| log_4_2026-4-14-12-30-12 | 14211 | 1.465267 | 1.363353 | 0.067993 | 0.002951 | 0.110659 | 0.000609 |

The `fy_b` improvement over force v1 is per-log stable across all held-out logs. For moments, v2 improves `mx_b` and `mz_b` over the physical prior in all held-out logs, but that is a lower bar than the deployable v1 moment heads. The v2 `force_arm` result is consistent with the residual-attribution variables motivating deployable rate/control/lateral interactions, but it does not prove a unique aerodynamic cause.

## Carry-Forward

Carry forward the validation-selected deployable candidate: force v2 `affine` plus moment v2 `force_arm`, both with `base+rates+controls+lateral+interactions`. The strongest positive evidence is the stable `fy_b` reduction and the improved deployable moment mean. The main limitation is `mz_b`, where the selected v2 moment head is worse than the v1 features-only deployable baseline; this should be reported as an unresolved residual channel rather than hidden by moment mean.

Artifact anchors:

- `artifacts/20260526_deployable_wrench_correction_v2/model_config.json`
- `artifacts/20260526_deployable_wrench_correction_v2/force_metrics_by_split.csv`
- `artifacts/20260526_deployable_wrench_correction_v2/moment_metrics_by_split.csv`
- `artifacts/20260526_deployable_wrench_correction_v2/per_log_metrics.csv`
- `artifacts/20260526_deployable_wrench_correction_v2/prediction_parquets/test_predictions.parquet`
- `artifacts/20260526_deployable_wrench_correction_v2/inference_model_state.json`
