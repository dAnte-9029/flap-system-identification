# Phase-Structured Wrench Correction

This run evaluates the clean B+C experiment for held-out log effective-wrench prediction: a phase-structured force correction around the DeLaurier-like prior, followed by a wrench-consistent force-arm plus free-torque moment correction. Model selection used validation `force_mean` and `moment_mean` RMSE only; test metrics below are final reporting.

## Command

```bash
python scripts/train_phase_structured_wrench_correction.py --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 --prior-root artifacts/delaurier_physical_prior_v1 --v2-reference-root artifacts/20260526_deployable_wrench_correction_v2 --output-root artifacts/20260527_phase_structured_wrench_correction_v1 --alphas 0,0.001,0.01,0.1,1,10,100,1000 --overwrite
```

The selected force candidate is `phase_structured_plus_rates_controls`, variant `affine`, alpha `1`. Its validation `force_mean` RMSE is `2.093750`; its test `force_mean` RMSE is `2.113975`.

The selected moment candidate is `force_arm_plus_free`, feature family `phase_structured_plus_rates_controls`, alpha `1`. Its validation `moment_mean` RMSE is `0.001841`; its test `moment_mean` RMSE is `0.001969`.

## Force Candidate Table

Each row is the validation-selected configuration within that force family, evaluated on test.

| family | variant | alpha | val force_mean RMSE | test force_mean RMSE | test R2 | selected |
|:--|:--|--:|--:|--:|--:|:--|
| prior | prior | 0 | 8.358770 | 8.436752 | -2.060130 | no |
| slow_only | affine | 0 | 5.964420 | 5.972360 | 0.048621 | no |
| phase_only | affine | 0 | 5.452290 | 5.566450 | -1.175410 | no |
| phase_structured | affine | 0 | 2.265190 | 2.299394 | 0.599901 | no |
| phase_structured_plus_rates_controls | affine | 1 | 2.093750 | 2.113975 | 0.641971 | yes |

Selected force test channel metrics:

| target | RMSE | MAE | bias | R2 |
|:--|--:|--:|--:|--:|
| fx_b | 1.632014 | 1.258690 | 0.021587 | 0.885945 |
| fy_b | 1.337994 | 0.979895 | -0.048183 | 0.143197 |
| fz_b | 2.992153 | 2.292140 | -0.105071 | 0.896773 |
| force_mean | 2.113975 | 1.510240 | -0.043889 | 0.641971 |

## Moment Candidate Table

Each row is the validation-selected configuration within that moment form, evaluated on test.

| form | feature family | alpha | val moment_mean RMSE | test moment_mean RMSE | test R2 | selected |
|:--|:--|--:|--:|--:|--:|:--|
| prior_moment | prior | 0 | 0.347439 | 0.354762 | -19814.6 | no |
| direct_residual | phase_structured_plus_rates_controls | 100 | 0.132378 | 0.139676 | -10756.3 | no |
| force_arm_only | phase_structured_plus_rates_controls | 100 | 0.002035 | 0.002182 | 0.111964 | no |
| force_arm_plus_free | phase_structured_plus_rates_controls | 1 | 0.001841 | 0.001969 | 0.358908 | yes |
| hybrid_prior_arm_free | phase_structured_plus_rates_controls | 10 | 0.131189 | 0.138914 | -10730.1 | no |

Selected moment test channel metrics:

| target | RMSE | MAE | bias | R2 |
|:--|--:|--:|--:|--:|
| mx_b | 0.002674 | 0.002007 | -0.000123 | 0.352339 |
| my_b | 0.002050 | 0.001613 | 0.000085 | 0.775462 |
| mz_b | 0.000525 | 0.000400 | -0.000009 | -0.051077 |
| moment_mean | 0.001969 | 0.001340 | -0.000016 | 0.358908 |

## Per-Log Stability

| log_id | n | force_mean RMSE | moment_mean RMSE |
|:--|--:|--:|--:|
| log_4_2026-4-12-17-43-30 | 18184 | 2.12731 | 0.002145 |
| log_4_2026-4-14-12-30-12 | 14211 | 1.98292 | 0.002044 |
| log_18_2026-4-15-12-56-08 | 14202 | 2.25987 | 0.001903 |
| log_34_2026-4-16-19-13-30 | 14074 | 2.07221 | 0.001700 |

The selected force candidate is stable at the log level in the sense that held-out force RMSE remains in a narrow range across the four test logs. The selected moment candidate similarly stays near `0.002` moment-mean RMSE across logs.

## Arm and Free-Torque Summary

For the selected test predictions, the effective arm magnitude has mean `0.000558` and 95th percentile `0.001231`. The free-torque norm divided by corrected moment norm has mean `1.205029` and 95th percentile `3.042548`.

These are effective wrench parameters for prediction. The learned arm is not an identified center of pressure, and the large free-torque fraction indicates that the selected moment correction is not explained by an arm-only moment model.

## Limitations

The `fy_b` channel remains a limitation: selected test RMSE is low relative to the prior, but `fy_b` test R2 is only `0.143197`, much weaker than `fx_b` and `fz_b`.

The `mz_b` moment channel remains inconclusive: selected test RMSE is small in absolute units, but test R2 is negative (`-0.051077`). The moment mean is therefore not sufficient by itself to claim all moment channels are explained.

The results demonstrate held-out log effective-wrench prediction, not simulator rollout validation. Residual attribution motivates deployable phase, rate, control, and lateral features, but it does not prove aerodynamic causality.

## Artifact Anchors

- `artifacts/20260527_phase_structured_wrench_correction_v1/model_config.json`
- `artifacts/20260527_phase_structured_wrench_correction_v1/inference_model_state.json`
- `artifacts/20260527_phase_structured_wrench_correction_v1/force_metrics_by_split.csv`
- `artifacts/20260527_phase_structured_wrench_correction_v1/moment_metrics_by_split.csv`
- `artifacts/20260527_phase_structured_wrench_correction_v1/per_log_metrics.csv`
- `artifacts/20260527_phase_structured_wrench_correction_v1/prediction_parquets/test_predictions.parquet`
