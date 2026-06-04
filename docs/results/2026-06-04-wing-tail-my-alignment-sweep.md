# Wing-Tail Pitch-Moment Alignment Sweep

## Purpose

This diagnostic tests whether the measured effective pitch moment `my_b` can be explained better by a physically anchored wing moment prior plus a swept simple tail prior.  The target remains a whole-vehicle effective moment, so the result is not an isolated measurement of wing or tail loads.

The model form is:

```tex
\hat M_y^{eff}
=
a_w M_y^{wing,fixedAC}
+ a_t M_y^{tail}(\theta_t)
+ b .
```

The wing moment uses the dense corrected `fx_b/fz_b` force result and an area-weighted quarter-chord wing application point.  Tail parameters are swept broadly, then `a_w`, `a_t`, and `b` are fitted on the train split only. Selection uses validation RMSE.

## Artifact

```text
/home/zn/flap-system-identification/artifacts/20260604_wing_tail_my_alignment_sweep_v1
```

Command:

```bash
conda run -n env_isaaclab python scripts/diagnose_wing_tail_my_alignment.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --force-correction-root artifacts/20260604_fx_fz_structured_correction_best_separation_prior_v1/prediction_parquets \
  --output-root artifacts/20260604_wing_tail_my_alignment_sweep_v1 \
  --search-stage quick \
  --search-stride 20 \
  --top-k-full 20 \
  --max-random-cases 1200
```

## Baselines on Test Split

```text
bias_only_train_mean             RMSE 0.6276 N m, R2 ~ 0
wing_fixed_ac_raw                RMSE 1.6441 N m, R2 -5.862, corr 0.306
wing_fixed_ac_train_gain_bias    RMSE 0.5978 N m, R2 0.0928, corr 0.306
```

The raw fixed-AC wing moment is not directly usable because the force application point and effective-moment target are not consistent enough. A train-only gain/bias makes it a weak but usable diagnostic baseline.

## Selected Wing+Tail Case

Validation selected case `100`:

```text
fixed_horizontal_effectiveness       2.0
elevon_effectiveness                 1.5
horizontal_tail_q_scale              1.0
horizontal_tail_incidence_bias_deg   20 deg
elevon_alpha_limit_deg               45 deg
elevon_max_deg                       60 deg
elevon_sign                         -1
swap_elevons                         true
```

Train-fitted joint coefficients:

```text
a_w    0.14384
a_t    0.07063
b     -0.09237 N m
```

Full-split selected metrics:

```text
train: RMSE 0.5998 N m, R2 0.0805, corr 0.2837
val:   RMSE 0.5634 N m, R2 0.0979, corr 0.3133
test:  RMSE 0.5965 N m, R2 0.0968, corr 0.3123
```

## Interpretation

The best swept wing-plus-tail model improves test RMSE only slightly over the wing fixed-AC gain/bias baseline:

```text
0.5978 -> 0.5965 N m
```

The correlation also improves only slightly:

```text
0.3060 -> 0.3123
```

This means the broad tail sweep does not reveal a strong tail-prior explanation of `my_b` under the current simple tail model and current effective-moment label. The selected tail parameters are not absurd, but the fitted tail coefficient is small (`a_t = 0.0706`), so most of the selected prediction is still a weak global alignment of the wing fixed-AC moment plus bias.

## Paper-Safe Conclusion

This result should be used as a diagnostic limitation rather than a main contribution. A safe statement is:

```text
For pitch moment, a diagnostic wing-plus-tail prior alignment produced only marginal improvement over a gain-adjusted fixed-wing-arm baseline. This suggests that the effective pitch-moment label is not well explained by a fixed quarter-chord wing moment and the current simple tail prior alone. We therefore treat moment channels as secondary diagnostics and focus the main quantitative correction result on `fx_b` and `fz_b`.
```

Useful figure outputs:

```text
figures/test_my_timeseries_selected.png
figures/test_my_scatter_selected.png
figures/parameter_sensitivity_top_candidates.png
```
