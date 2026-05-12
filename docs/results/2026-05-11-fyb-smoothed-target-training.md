# fy_b Smoothed Target Training

Date: 2026-05-11

## Setup

- Base split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Derived splits:
  - `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_fyb_sg0p06_p2`
  - `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_fyb_sg0p12_p2`
  - `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_fyb_sg0p20_p2`
- Split builder: `scripts/build_fyb_smoothed_label_split.py`
- Training output: `artifacts/20260511_fyb_smooth_target_training_gpu/`
- Model: Head-FiLM causal Transformer, same config as the current no-suspect final model.

Only `fy_b` was replaced. The other targets were kept unchanged:

```text
fx_b, fz_b, mx_b, my_b, mz_b: unchanged
fy_b: recomputed from smoothed velocity-derived linear acceleration
```

## Same-Target Evaluation

Each model is evaluated against the same target definition it was trained on.

| variant | split | overall R2 | fy_b R2 | fy_b RMSE | fx_b R2 | fz_b R2 | mx_b R2 | my_b R2 | mz_b R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_existing | val | 0.693 | 0.359 | 1.065 | 0.969 | 0.965 | 0.515 | 0.908 | 0.441 |
| raw_existing | test | 0.800 | 0.515 | 1.004 | 0.976 | 0.974 | 0.705 | 0.943 | 0.688 |
| fyb_sg0p06 | val | 0.734 | 0.540 | 0.385 | 0.966 | 0.965 | 0.552 | 0.901 | 0.482 |
| fyb_sg0p06 | test | 0.839 | 0.773 | 0.284 | 0.975 | 0.972 | 0.697 | 0.938 | 0.680 |
| fyb_sg0p12 | val | 0.712 | 0.503 | 0.276 | 0.964 | 0.962 | 0.497 | 0.897 | 0.448 |
| fyb_sg0p12 | test | 0.833 | 0.741 | 0.202 | 0.975 | 0.971 | 0.703 | 0.941 | 0.668 |
| fyb_sg0p20 | val | 0.709 | 0.498 | 0.239 | 0.964 | 0.960 | 0.477 | 0.892 | 0.461 |
| fyb_sg0p20 | test | 0.808 | 0.606 | 0.202 | 0.973 | 0.970 | 0.683 | 0.937 | 0.678 |

Full table:

```text
artifacts/20260511_fyb_smooth_target_training_gpu/summary_metrics.csv
```

## Cross-Target Evaluation

Each smoothed-target model was also evaluated back against the original raw target. This checks whether the improvement is a real raw-`fy_b` improvement or mostly a change in target definition.

| model | eval target | val fy_b R2 | test fy_b R2 |
|---|---|---:|---:|
| raw_existing | raw | 0.359 | 0.515 |
| fyb_sg0p06 | smoothed own target | 0.540 | 0.773 |
| fyb_sg0p06 | raw target | 0.120 | 0.158 |
| fyb_sg0p12 | smoothed own target | 0.503 | 0.741 |
| fyb_sg0p12 | raw target | 0.071 | 0.102 |
| fyb_sg0p20 | smoothed own target | 0.498 | 0.606 |
| fyb_sg0p20 | raw target | -0.000 | 0.005 |

Full table:

```text
artifacts/20260511_fyb_smooth_target_training_gpu/cross_target_eval_summary.csv
```

## Interpretation

Training on smoothed `fy_b` works if the target is explicitly redefined as a smoothed/control-oriented side-force target.

It does not improve raw `fy_b` prediction. In fact, when smoothed-target models are evaluated against the original raw `fy_b`, `fy_b` R2 drops sharply. This confirms that the gain is mostly from removing the broadband high-frequency component from the target, not from the model learning those raw spikes better.

Among the tested windows, `0.06 s` is the best compromise:

- It gives the best same-target `fy_b` R2: val `0.540`, test `0.773`.
- It improves overall same-target R2: val `0.734`, test `0.839`.
- It is less aggressive than `0.12 s` and `0.20 s`; larger windows make the target easier/smoother but lose more raw-target correspondence.

## Recommendation

Do not simply use a very large smoothing window to erase all high-frequency content.

Use `fyb_sg0p06_p2` as the first candidate if the project wants a control-oriented smoothed side-force target. Keep the raw `fy_b` model and metrics as a separate diagnostic baseline.

For paper/reporting, phrase this as a target-definition choice:

> Because the lateral-force label is reconstructed from acceleration-based inverse dynamics and is sensitive to broadband high-frequency transients, we additionally evaluate a smoothed `fy_b` target for control-oriented aerodynamic modeling. This improves predictability of the low-confidence side-force channel, but it should not be interpreted as improved reconstruction of the original raw high-frequency `fy_b`.
