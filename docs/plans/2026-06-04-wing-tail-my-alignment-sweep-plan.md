# Wing-Tail Pitch-Moment Alignment Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose whether measured effective pitch moment `my_b` can be better explained by a physically anchored wing moment arm plus a broadly swept tail prior, without claiming isolated component load measurement.

**Architecture:** Fix the wing aerodynamic-center prior from a literature/code-supported quarter-chord equivalent point, then fit only small global alignment terms for the wing contribution. Sweep tail geometry and aerodynamic parameters over a wider but still physically interpretable range, evaluate all parameter choices on train/val/test whole-log splits, and select by validation metrics before reporting test results.

**Tech Stack:** Python, pandas/parquet, NumPy/SciPy or scikit-learn, matplotlib, existing IsaacLab `TailAeroModel`, existing dense `fx/fz` correction artifact, existing measured-massprops ratio-8 split.

---

## Context and Claim Boundary

The measured target is whole-vehicle effective pitch moment:

```tex
M_y^{eff} = M_y^{wing} + M_y^{tail} + M_y^{body} + M_y^{mechanism} + M_y^{estimator/label} .
```

Therefore, this experiment must not fit `my_b` using wing force alone and then call the fitted arm an aerodynamic center. The intended decomposition is diagnostic:

```tex
\hat M_y^{eff}
=
a_w M_y^{wing,fixedAC}
+ a_t M_y^{tail}(\theta_t)
+ b
```

where `fixedAC` is set from an area-weighted quarter-chord wing point, and `theta_t` are tail-prior parameters. This tests whether the combined wing-plus-tail prior has useful structure for `my_b`; it does not identify true isolated wing or tail loads.

Evidence anchors already available:

- Dense corrected longitudinal force artifact:
  `/home/zn/flap-system-identification/artifacts/20260604_fx_fz_structured_correction_best_separation_prior_v1`
- Dataset split:
  `/home/zn/flap-system-identification/dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1`
- Existing tail model:
  `/home/zn/IsaacLab/source/flapping_bot/flapping_bot/physics/tail_aero.py`
- Existing quarter-chord helper:
  `/home/zn/IsaacLab/source/flapping_bot/flapping_bot/physics/wing_equivalent_ac.py`
- Existing wing geometry CSV:
  `/home/zn/IsaacLab/outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv`
- Previous diagnostic artifact:
  `/home/zn/flap-system-identification/artifacts/20260604_wing_ac_fixed_vs_fitted_my_diagnostic_v1`

## Task 1: Add a Minimal Wing-AC Helper Test

**Files:**
- Create: `/home/zn/flap-system-identification/tests/test_wing_tail_my_alignment.py`
- Reference: `/home/zn/IsaacLab/source/flapping_bot/flapping_bot/physics/wing_equivalent_ac.py`

**Step 1: Write tests for the fixed wing AC convention**

Test that the helper used by the diagnostic computes pitch moment with the FRD convention:

```python
def test_pitch_moment_from_arm_and_force_frd():
    arm = np.array([[0.12, 0.0, 0.05]])
    force = np.array([[3.0, 0.0, -8.0]])
    my = pitch_moment_from_force(arm, force)
    assert np.allclose(my, 0.05 * 3.0 - 0.12 * (-8.0))
```

Also test that the fixed AC arm returns finite `r_x` and `r_z` when given a small frame containing `wing_stroke_angle_rad`.

**Step 2: Run the test and verify it fails**

Run:

```bash
cd /home/zn/flap-system-identification
pytest tests/test_wing_tail_my_alignment.py -q
```

Expected: FAIL because the diagnostic script/helper does not exist yet.

## Task 2: Implement the Diagnostic Script

**Files:**
- Create: `/home/zn/flap-system-identification/scripts/diagnose_wing_tail_my_alignment.py`
- Test: `/home/zn/flap-system-identification/tests/test_wing_tail_my_alignment.py`

**Step 1: Add reusable helper functions**

Implement:

```python
pitch_moment_from_force(arm_frd_m, force_frd_n) -> np.ndarray
load_area_weighted_quarter_chord_link_point(csv_path) -> dict
fixed_wing_ac_arm_frd(samples, cg_frd_m, wing_geom_csv, urdf_joint_params) -> np.ndarray
metric_row(y_true, y_pred) -> dict
fit_train_gain_bias(train_y, train_x) -> tuple[float, float]
```

Use:

```tex
M_y = r_z F_x - r_x F_z
```

for FRD body axes.

**Step 2: Use fixed wing AC as the primary prior**

Compute:

```tex
M_y^{wing,fixedAC} = r_z^{fixedAC}(t) F_x^{dense}(t) - r_x^{fixedAC}(t) F_z^{dense}(t)
```

Use dense corrected force columns:

```text
force_v2_fx_b
force_v2_fz_b
```

from:

```text
artifacts/20260604_fx_fz_structured_correction_best_separation_prior_v1/prediction_parquets/{split}_selected_predictions.parquet
```

**Step 3: Add optional bounded wing alignment variants**

Do not let wing freely explain `my_b`. Fit only these train-only variants:

```tex
\hat M_y = a_w M_y^{wing,fixedAC} + b
```

and optionally:

```tex
\hat M_y = M_y^{wing,fixedAC + \Delta r} + b
```

with bounded:

```text
delta_rx_m in [-0.05, 0.05]
delta_rz_m in [-0.04, 0.04]
```

This range is deliberately larger than a tight uncertainty band but prevents the wing arm from becoming a meaningless whole-vehicle fitted parameter.

**Step 4: Run the tests**

Run:

```bash
cd /home/zn/flap-system-identification
pytest tests/test_wing_tail_my_alignment.py -q
```

Expected: PASS.

## Task 3: Implement Broad Tail Sweep

**Files:**
- Modify: `/home/zn/flap-system-identification/scripts/diagnose_wing_tail_my_alignment.py`
- Reference: `/home/zn/flap-system-identification/scripts/sweep_tail_geometry_alignment.py`
- Reference: `/home/zn/flap-system-identification/scripts/run_component_prior_ablation.py`

**Step 1: Sweep tail geometric/reference parameters**

Use measured full-aircraft CG as fixed. Sweep tail aerodynamic-center/reference adjustments:

```text
horizontal_x_offset_m: -0.12,-0.08,-0.04,0,0.04,0.08,0.12
elevon_x_offset_m:    -0.12,-0.08,-0.04,0,0.04,0.08,0.12
vertical_x_offset_m:  -0.10,-0.05,0,0.05,0.10
elevon_y_scale:       0.5,0.75,1.0,1.25,1.5
vertical_z_scale:     0.5,0.75,1.0,1.25,1.5
```

Reason: previous sweep was too narrow to discover whether the model is mainly missing an effective reference point. These values are broad enough to detect useful trends but still interpretable as prior-shaping, not arbitrary fitting.

**Step 2: Sweep tail aerodynamic/effectiveness parameters**

Use broad but finite ranges:

```text
fixed_horizontal_effectiveness: 0.0,0.25,0.5,1.0,1.5,2.0,3.0
elevon_effectiveness:           0.0,0.5,1.0,1.5,2.0,3.0,4.0
horizontal_tail_q_scale:        0.25,0.5,1.0,1.5,2.0,3.0
horizontal_tail_incidence_bias_deg: -20,-15,-10,-5,0,5,10,15,20
elevon_alpha_limit_deg:         15,20,25,35,45
elevon_max_deg:                 25,35,41,50,60
rudder_max_deg:                 15,25,35,45
```

Reason: if the current tail prior is under-scaled or has incidence/command mapping mismatch, narrow sweeps will falsely conclude that tail has no explanatory value.

**Step 3: Keep signs/swap as diagnostic options, not automatic changes**

Include optional cases:

```text
elevon_sign: -1,+1
rudder_sign: -1,+1
swap_elevons: false,true
```

Reason: prior sign diagnostics were weak, so sign changes should be evaluated but not committed to model convention unless validation/test behavior is consistently better and physically explainable.

## Task 4: Joint Wing+Tail Train-Only Alignment

**Files:**
- Modify: `/home/zn/flap-system-identification/scripts/diagnose_wing_tail_my_alignment.py`

For each tail parameter candidate, compute:

```tex
\hat M_y^{eff}
=
a_w M_y^{wing,fixedAC}
+ a_t M_y^{tail}(\theta_t)
+ b .
```

Fit `a_w`, `a_t`, and `b` on train only using ridge or ordinary least squares. Start with OLS; if coefficients become unstable, add ridge alpha grid:

```text
alpha: 0, 0.01, 0.1, 1, 10
```

Also save raw component diagnostics:

```text
my_b vs wing_fixed_ac
my_b vs tail_my
my_b vs wing_fixed_ac + tail_my
my_b vs fitted a_w*wing + a_t*tail + b
```

Add coefficient sanity columns:

```text
a_w
a_t
b_Nm
tail_parameter_case_id
```

Selection rule:

1. Primary: validation `my_b` RMSE.
2. Tie-breaker: validation `corr`.
3. Sanity filter: prefer cases with `abs(a_w) <= 3`, `abs(a_t) <= 10`, and test/train RMSE ratio not exploding.

Do not discard useful candidates solely because `a_t` is larger than 1; a large tail scale may indicate the simple model is underestimating dynamic pressure or control effectiveness.

## Task 5: Add Efficient Search Schedule

**Files:**
- Modify: `/home/zn/flap-system-identification/scripts/diagnose_wing_tail_my_alignment.py`

Avoid a full Cartesian explosion at first. Use a staged schedule:

1. **Stage A:** nominal tail plus wing fixed AC baseline.
2. **Stage B:** sweep aerodynamic/effectiveness parameters with nominal geometry.
3. **Stage C:** take top 30 Stage B candidates and sweep geometry/reference parameters.
4. **Stage D:** take top 30 Stage C candidates and add sign/swap options.
5. **Stage E:** rerun top 10 candidates on train/val/test and save full prediction parquet.

Reason: the full range is intentionally wide; staged search finds useful regions without wasting time on millions of combinations.

## Task 6: Generate Figures and Tables

**Files:**
- Modify: `/home/zn/flap-system-identification/scripts/diagnose_wing_tail_my_alignment.py`

Save artifact under:

```text
/home/zn/flap-system-identification/artifacts/20260604_wing_tail_my_alignment_sweep_v1
```

Required outputs:

```text
manifest.json
wing_tail_my_alignment_metrics.csv
wing_tail_my_alignment_selection.csv
top_val_candidates.csv
top_test_candidates.csv
{split}_selected_wing_tail_my_predictions.parquet
figures/test_my_timeseries_selected.png
figures/test_my_scatter_selected.png
figures/parameter_sensitivity_top_candidates.png
README.md
```

Figures:

1. `my_b` label vs fixed wing AC vs selected wing+tail prediction on a representative held-out log.
2. Scatter plot: label `my_b` vs selected prediction.
3. Bar/table plot comparing:
   - zero/bias baseline
   - fixed wing AC only
   - tail only
   - fixed wing AC + nominal tail
   - fixed wing AC + swept tail
   - train-fitted wing+tail gain/bias

## Task 7: Verification Commands

**Files:**
- Test: `/home/zn/flap-system-identification/tests/test_wing_tail_my_alignment.py`

Run:

```bash
cd /home/zn/flap-system-identification
pytest tests/test_wing_tail_my_alignment.py -q
python scripts/diagnose_wing_tail_my_alignment.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --force-correction-root artifacts/20260604_fx_fz_structured_correction_best_separation_prior_v1/prediction_parquets \
  --output-root artifacts/20260604_wing_tail_my_alignment_sweep_v1 \
  --search-stage full
```

Expected:

- Tests pass.
- Artifact directory contains the files listed in Task 6.
- `manifest.json` records train-only fitting and validation-based selection.
- `top_val_candidates.csv` and `top_test_candidates.csv` show whether tail sweep improves `my_b` metrics over fixed wing AC alone.

## Task 8: Paper-Safe Interpretation

**Files:**
- Create or update: `/home/zn/flap-system-identification/docs/results/2026-06-04-wing-tail-my-alignment-sweep.md`
- Optional paper note: `/home/zn/paper/AeroConf_effective_aero/research_notes/20260604_wing_tail_my_alignment_sweep_note.md`

Write results with this boundary:

```text
The sweep evaluates whether a fixed quarter-chord wing moment prior and a simple tail prior contain useful structure for the effective pitch moment. It does not identify isolated tail loads or a measured wing center of pressure.
```

If metrics improve:

```text
The selected wing-plus-tail prior explains additional held-out pitch-moment structure relative to either component alone, suggesting that moment prediction should be treated as a coupled whole-vehicle correction rather than a wing-only or tail-only attribution.
```

If metrics do not improve:

```text
The weak validation/test correlation indicates that the current effective pitch-moment label is dominated by effects not represented by the fixed wing AC and simple tail prior, including mechanism reaction, flexible-wing/tail coupling, inertial-label sensitivity, or estimator artifacts.
```

## Commit Point

After implementation and verification:

```bash
cd /home/zn/flap-system-identification
git add scripts/diagnose_wing_tail_my_alignment.py tests/test_wing_tail_my_alignment.py docs/plans/2026-06-04-wing-tail-my-alignment-sweep-plan.md docs/results/2026-06-04-wing-tail-my-alignment-sweep.md
git commit -m "diagnose wing-tail pitch moment alignment"
```

Do not commit large artifact parquet files unless the user explicitly asks for result artifacts to be versioned.
