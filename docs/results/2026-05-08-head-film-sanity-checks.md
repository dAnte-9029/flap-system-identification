# Head FiLM Sanity Checks

Date: 2026-05-08

## Setup

- Model: `artifacts/20260508_phase_film_final_no_suspect/runs/phase_film_final_head/causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt`
- Split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Script: `scripts/run_model_sanity_checks.py`
- Output: `artifacts/20260508_head_film_sanity_checks/`

## What Was Checked

1. **Whole-log split check**
   - `train`, `val`, `test` all contain `log_id`.
   - No overlapping `log_id` between train/val/test.

2. **Input leakage scan**
   - No acceleration-like inputs were found.
   - No target columns (`fx_b`, `fy_b`, `fz_b`, `mx_b`, `my_b`, `mz_b`) were found in model inputs.
   - No suspicious `target` / `label` / `wrench` named input columns were found.

3. **Negative control**
   - Keep model predictions fixed, randomly permute test targets.
   - Result: overall R2 drops to `-0.8043`.
   - Interpretation: the metric/evaluation path is not trivially producing high R2 regardless of target alignment.

4. **Naive/simple baselines**
   - Train-target mean baseline: overall R2 `-0.0002`.
   - Simple linear physics-feature baseline: overall R2 `0.3905`.
   - Interpretation: the task is not solved by a constant mean, but some longitudinal structure is learnable from simple airdata/phase/control features.

## Key Results

| Method | Overall R2 | fx R2 | fy R2 | fz R2 | mx R2 | my R2 | mz R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Head FiLM Transformer | 0.8003 | 0.9764 | 0.5153 | 0.9735 | 0.7051 | 0.9433 | 0.6882 |
| Train-target mean | -0.0002 | -0.0005 | -0.0006 | -0.0000 | -0.0000 | -0.0000 | -0.0000 |
| Simple linear physics features | 0.3905 | 0.7492 | 0.1090 | 0.8623 | 0.2122 | 0.3233 | 0.0869 |
| Predictions vs permuted targets | -0.8043 | -0.9457 | -0.4485 | -0.9620 | -0.8201 | -0.9473 | -0.7023 |

## Per-log Test Stability

The no-suspect test split contains three logs after sequence-window alignment:

| log_id | samples | overall R2 | fx R2 | fy R2 | fz R2 | mx R2 | my R2 | mz R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `log_18_2026-4-15-12-56-08` | 14497 | 0.7668 | 0.9777 | 0.4299 | 0.9694 | 0.6690 | 0.9364 | 0.6181 |
| `log_34_2026-4-16-19-13-30` | 13742 | 0.7959 | 0.9773 | 0.4808 | 0.9822 | 0.6676 | 0.9572 | 0.7107 |
| `log_4_2026-4-14-12-30-12` | 14518 | 0.8263 | 0.9742 | 0.5935 | 0.9690 | 0.7518 | 0.9370 | 0.7322 |

## Interpretation

Current high longitudinal R2 is more defensible after these checks:

- The evaluation uses a whole-log split, not mixed samples from the same log.
- The model bundle does not include acceleration or direct target columns.
- When target alignment is intentionally broken, R2 becomes strongly negative.
- A simple linear baseline already captures part of `fx_b` and `fz_b`, so high longitudinal R2 is physically plausible rather than automatically suspicious.

But this is not the end of leakage validation:

- The check does not prove there is no preprocessing-level leakage upstream of the parquet files.
- The simple linear baseline being strong for `fx_b`/`fz_b` means final claims should emphasize **generalization across held-out logs**, not just absolute R2.
- For paper/reporting, keep `test` reserved for final confirmation and rank sweeps by validation metrics.

## Next Sanity Checks

Recommended follow-up:

1. **Shuffled-label retrain**
   - Train the same model with shuffled train targets.
   - Expected result: validation/test R2 near zero or negative.

2. **No-airdata ablation**
   - Remove airspeed / dynamic pressure / density features.
   - Expected result: longitudinal performance should drop clearly if the model is using physically meaningful aerodynamic inputs.

3. **Time/order robustness**
   - Evaluate whether performance changes if sequence order is disrupted.
   - Expected result: temporal models should degrade if temporal context matters.
