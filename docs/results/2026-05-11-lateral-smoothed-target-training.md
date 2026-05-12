# Lateral Smoothed Target Training

Date: 2026-05-11

## Setup

- Base split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Derived splits:
  - `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_lateral_sg0p06_p2`
  - `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_lateral_sg0p12_p2`
  - `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log_lateral_sg0p20_p2`
- Split builder: `scripts/build_lateral_smoothed_label_split.py`
- Training output: `artifacts/20260511_lateral_smooth_target_training_gpu/`
- Model: Head-FiLM causal Transformer, same config as the current no-suspect final model.

Only the lateral-directional channels were replaced:

```text
fx_b, fz_b, my_b: unchanged
fy_b: recomputed from smoothed velocity-derived linear acceleration
mx_b, mz_b: recomputed from smoothed angular-velocity derivatives
```

## Target Sensitivity Check

The derived `val` splits preserve the unchanged channels exactly:

```text
max abs diff for fx_b/fz_b/my_b: 0.0
lateral_label_valid ratio: 1.0
```

The lateral channels are strongly affected by smoothing:

| variant | fy corr/raw | mx corr/raw | mz corr/raw |
|---|---:|---:|---:|
| lateral_sg0p06 | 0.468 | 0.514 | 0.653 |
| lateral_sg0p12 | 0.384 | 0.433 | 0.441 |
| lateral_sg0p20 | 0.251 | 0.290 | 0.287 |

This confirms that `mx_b/mz_b`, like `fy_b`, contain high-frequency components that are sensitive to derivative construction.

## Same-Target Evaluation

Each model is evaluated against the same target definition it was trained on.

| variant | split | overall R2 | fy_b R2 | mx_b R2 | mz_b R2 | fx_b R2 | fz_b R2 | my_b R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw_existing | val | 0.693 | 0.359 | 0.515 | 0.441 | 0.969 | 0.965 | 0.908 |
| raw_existing | test | 0.800 | 0.515 | 0.705 | 0.688 | 0.976 | 0.974 | 0.943 |
| lateral_sg0p06 | val | 0.785 | 0.562 | 0.677 | 0.651 | 0.964 | 0.960 | 0.897 |
| lateral_sg0p06 | test | 0.863 | 0.774 | 0.788 | 0.738 | 0.973 | 0.969 | 0.936 |
| lateral_sg0p12 | val | 0.751 | 0.512 | 0.624 | 0.568 | 0.963 | 0.955 | 0.884 |
| lateral_sg0p12 | test | 0.836 | 0.712 | 0.786 | 0.661 | 0.970 | 0.964 | 0.923 |
| lateral_sg0p20 | val | 0.738 | 0.472 | 0.644 | 0.521 | 0.960 | 0.957 | 0.876 |
| lateral_sg0p20 | test | 0.820 | 0.621 | 0.799 | 0.646 | 0.969 | 0.962 | 0.923 |

Full table:

```text
artifacts/20260511_lateral_smooth_target_training_gpu/summary_metrics.csv
```

## Cross-Target Evaluation

Each smoothed-target model was also evaluated back against the original raw target.

| model | eval target | test overall R2 | test fy_b R2 | test mx_b R2 | test mz_b R2 |
|---|---|---:|---:|---:|---:|
| raw_existing | raw | 0.800 | 0.515 | 0.705 | 0.688 |
| lateral_sg0p06 | smoothed own target | 0.863 | 0.774 | 0.788 | 0.738 |
| lateral_sg0p06 | raw target | 0.589 | 0.157 | 0.194 | 0.304 |
| lateral_sg0p12 | smoothed own target | 0.836 | 0.712 | 0.786 | 0.661 |
| lateral_sg0p12 | raw target | 0.533 | 0.102 | 0.163 | 0.075 |
| lateral_sg0p20 | smoothed own target | 0.820 | 0.621 | 0.799 | 0.646 |
| lateral_sg0p20 | raw target | 0.498 | 0.006 | 0.080 | 0.050 |

Full table:

```text
artifacts/20260511_lateral_smooth_target_training_gpu/cross_target_eval_summary.csv
```

## Interpretation

Smoothing all lateral-directional targets produces a strong control-oriented target:

- `lateral_sg0p06` improves same-target test overall R2 from `0.800` to `0.863`.
- `fy_b`, `mx_b`, and `mz_b` all improve on their smoothed definitions.
- The longitudinal/pitch channels remain high and mostly unchanged, because their labels were not rewritten.

However, the cross-target evaluation shows that this is not improved reconstruction of the original raw lateral high-frequency target. When evaluated against raw targets, the lateral-smoothed models lose most raw `fy_b/mx_b/mz_b` R2.

The strongest practical candidate is `lateral_sg0p06_p2`. It is the least aggressive smoothing setting and gives the best same-target overall and lateral performance.

## Recommendation

Use two clearly separated result sets:

```text
Raw effective wrench model:
  Use for strict reconstruction claims.

Control-oriented smoothed lateral target model:
  Use for downstream control-oriented modeling where broadband derivative-sensitive lateral transients are not the main objective.
```

For paper/reporting:

> We additionally evaluate a control-oriented lateral-directional target in which `fy_b`, `mx_b`, and `mz_b` are reconstructed from smoothed kinematic derivatives. This improves the predictability of the lateral-directional channels, but these metrics are reported separately from raw effective-wrench reconstruction because the smoothed target intentionally suppresses derivative-sensitive broadband transients.
