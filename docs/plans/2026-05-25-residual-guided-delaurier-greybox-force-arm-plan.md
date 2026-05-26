# Residual-Guided DeLaurier Grey-Box Force-Arm Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and evaluate a residual-guided grey-box modeling path that first improves DeLaurier force prediction, then uses force plus a state-dependent equivalent arm to model moment.

**Architecture:** Treat DeLaurier as a structured physics prior, not as a black-box baseline to discard. Use residual diagnostics to identify phase-, frequency-, and angle-of-attack-dependent mismatch, then implement increasingly flexible correction models: global force recalibration, low-dimensional grey-box force modulation, and a structured moment head of the form `M = r(x) x F + tau_free(x)`.

**Tech Stack:** Python 3.11, pandas/parquet, NumPy/SciPy/scikit-learn or PyTorch where useful, existing `system_identification.training` utilities, IsaacLab DeLaurier export utilities, pytest, matplotlib.

---

## Context

### Primary dataset

Use the current DeLaurier residual dataset lineage unless a better locked split is discovered during implementation:

```text
dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
artifacts/delaurier_physical_prior_v1
```

The prior manifest currently points to:

```text
prior_name: delaurier_physical_calibrated_v1
split_root: dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
```

### Existing relevant files

In this repository:

```text
scripts/build_delaurier_residual_split.py
scripts/evaluate_delaurier_residual_model.py
scripts/analyze_delaurier_residual_phase.py
scripts/analyze_delaurier_residual_frequency.py
scripts/analyze_delaurier_residual_conditions.py
scripts/train_baseline_torch.py
src/system_identification/training.py
docs/results/2026-05-13-delaurier-residual-nn-gpu.md
```

In IsaacLab:

```text
/home/zn/IsaacLab/scripts/flapping_px4/export_delaurier_prior_predictions.py
/home/zn/IsaacLab/scripts/flapping_px4/run_delaurier_physical_calibration.py
/home/zn/IsaacLab/source/flapping_bot/flapping_bot/analysis/delaurier_gain_calibration.py
/home/zn/IsaacLab/source/flapping_bot/flapping_bot/physics/qsm_delaurier1993.py
```

### Important modeling distinction

DeLaurier internal aerodynamic parameters include:

```text
alpha0_rad
eta_s
cd_cf
alpha_stall_min_rad
alpha_stall_max_rad
xi
c_mac
nu
cd_f
```

Do not prioritize reinterpreting these from flight logs in the first pass. The first-pass correction should mostly operate on effective inputs, wrapper parameters, and low-dimensional force-term modulation:

```text
phase_delay_s
delaurier_theta_w_deg / effective AoA bias
twist_eta_max_deg
wing_normal_force_scale
wing_chordwise_force_scale
condition-dependent force gains
```

This is a guideline, not a hard rule. If implementation evidence suggests a targeted internal parameter experiment is clearly useful, add it as a clearly labeled ablation rather than mixing it into the main result.

### Key intuition

The force-moment arm diagnostic showed:

```text
ground-truth F_eff with per-sample r(t) reconstructs about 94% of moment energy
fixed global r explains only about 5% of held-out moment variance
```

Interpretation:

```text
Moment is not reliably determined by net force through a fixed arm.
But if force is predicted well, a dynamic equivalent arm plus small free-moment term is a plausible structured moment model.
```

---

## Stage 1: Force-Only DeLaurier Recalibration

### Objective

Establish a force-only recalibrated DeLaurier baseline. The purpose is to answer:

```text
Can the current DeLaurier prior improve on force channels if calibration is targeted only at fx_b, fy_b, fz_b?
```

This should separate force-prior quality from moment-head quality.

### Recommended scope

Start with current exported DeLaurier predictions and existing calibration code. If needed, create a local script that can rerun or wrap IsaacLab prediction without moving large implementation into this repository.

Compare:

```text
A0: current calibrated DeLaurier prior
A1: force-only global recalibration
A2: force-only global recalibration with longitudinal-weighted objective
```

Let the implementer decide whether to use random search, scipy optimization, least-squares over wrapper gains, or a small differentiable optimizer, as long as train-only fitting and held-out evaluation are preserved.

### Candidate parameters

Primary:

```text
wing_normal_force_scale
wing_chordwise_force_scale
delaurier_theta_w_deg
twist_eta_max_deg
phase_delay_s
```

Optional if useful:

```text
induced_drag_efficiency
fuselage_drag_cda
tail_lift_scale
```

Keep the force objective explicit:

```text
loss_force = normalized MSE over fx_b, fy_b, fz_b
```

Also try a control-weighted variant if cheap:

```text
fx_b: high
fz_b: high
fy_b: lower
```

### Deliverables

Create scripts as needed, for example:

```text
scripts/run_delaurier_force_recalibration.py
scripts/evaluate_delaurier_force_prior.py
tests/test_delaurier_force_recalibration.py
```

Suggested output root:

```text
artifacts/20260525_delaurier_force_recalibration_v1
```

Required outputs:

```text
parameters.csv
metrics_by_split.csv
force_metrics_summary.csv
residual_by_phase.csv
residual_by_condition.csv
README.md
```

### Evaluation

Report train/val/test:

```text
fx_b RMSE/R2
fy_b RMSE/R2
fz_b RMSE/R2
force mean RMSE/R2
force weighted score if used
```

Also report whether residual structure is reduced:

```text
phase-binned residual amplitude
flap-frequency-bin residual RMSE
AoA-bin residual RMSE
flap-main residual energy
```

### Interpretation to preserve

If global recalibration helps only modestly, that is still useful. It means the mismatch is not purely a global scale/bias issue, motivating Stage 2.

---

## Stage 2: Phase/Frequency/AoA Grey-Box Force Modulation

### Objective

Use residual structure to correct selected DeLaurier force terms without turning the method into an unconstrained black box.

Main question:

```text
Can low-dimensional phase-, frequency-, and AoA-dependent modulation improve DeLaurier force prediction and reduce structured residuals?
```

### Modeling idea

Start from DeLaurier force prediction:

```text
F_D = [fx_D, fy_D, fz_D]
```

Learn a correction such as:

```text
F_hat = G(z) * F_D + b(z)
```

where:

```text
z = phase encoding, flap frequency, AoA, airspeed or dynamic pressure
```

The exact implementation can vary. Favor simple interpretable models first:

```text
harmonic phase features:
sin(phi), cos(phi), sin(2phi), cos(2phi)

condition features:
AoA, flap_frequency, true airspeed, dynamic pressure

possible interactions:
AoA * sin(phi)
AoA * cos(phi)
flap_frequency * sin(phi)
flap_frequency * cos(phi)
```

Candidate correction families:

```text
B0: additive linear residual model, Delta F = Phi(z) beta
B1: multiplicative gain model, F_hat = (1 + g(z)) * F_D
B2: affine force model, F_hat = A(z) F_D + b(z)
B3: small MLP gain model with bounded output
```

Let the implementer choose which of these is most practical after inspecting available columns and prior export format.

### Important design freedom

The correction does not have to mimic a particular DeLaurier internal coefficient exactly. It should be described as:

```text
residual-guided effective correction to DeLaurier force components
```

If a correction can be mapped clearly to a wrapper quantity such as effective AoA bias, phase delay, or normal/chordwise force scale, document that mapping. If not, keep it as an empirical grey-box correction.

### Deliverables

Create scripts as needed, for example:

```text
scripts/train_delaurier_greybox_force_correction.py
scripts/evaluate_delaurier_greybox_force_correction.py
tests/test_delaurier_greybox_force_correction.py
```

Suggested output root:

```text
artifacts/20260525_delaurier_greybox_force_correction_v1
```

Required outputs:

```text
model_config.json
feature_columns.json
coefficients_or_model.pt
metrics_by_split.csv
condition_bin_comparison.csv
phase_residual_comparison.csv
frequency_residual_comparison.csv
prediction_parquets/
README.md
```

### Evaluation

Compare against Stage 1 and current prior:

```text
current calibrated DeLaurier
force-only global recalibration
grey-box force correction
direct NN force baseline if already available
```

Report:

```text
force RMSE/R2 by split
per-channel fx/fy/fz metrics
phase-binned residual reduction
AoA-bin worst-case reduction
flap-frequency-bin worst-case reduction
flap-main residual energy reduction
```

### Interpretation to preserve

Positive result:

```text
Residual mismatch is structured and can be reduced by phase/regime-conditioned grey-box correction.
```

Negative or weak result:

```text
Residual structure is visible diagnostically, but low-dimensional modulation is insufficient; the direct temporal model remains necessary.
```

Both outcomes are informative.

---

## Stage 3: Dynamic Equivalent-Arm Moment Head

### Objective

Test whether a better force prior can be converted into better moment predictions using a state-dependent equivalent arm and optional free-moment residual.

Main question:

```text
Given F_hat, can M be modeled better as r_hat(x) x F_hat + tau_free_hat(x) than by direct moment prediction or fixed-arm assumptions?
```

### Models to compare

Use the best force source from Stage 1/2, plus at least one baseline:

```text
C0: fixed arm fitted on train, M = r x F_hat
C1: dynamic arm only, M = r_hat(x) x F_hat
C2: dynamic arm + free moment, M = r_hat(x) x F_hat + tau_free_hat(x)
C3: direct moment head baseline with same input features/backbone where practical
```

The dynamic arm model can be lightweight:

```text
input = phase encoding, flap frequency, AoA, airspeed/q_dyn, controls, body rates, optionally temporal features
output = r_hat and optionally tau_free_hat
```

The implementer may choose MLP, temporal Transformer features, or reuse an existing trained backbone. Do not overconstrain the architecture. The key is that the output structure must expose:

```text
r_hat
tau_free_hat
M_hat
```

### Useful constraints, not mandatory

Potentially useful if training is unstable:

```text
bound r_hat with tanh to a physically plausible range
penalize very large |r_hat|
penalize tau_free when it dominates r x F
project tau_free parallel to F for interpretability
```

These are options, not hard requirements. Let the result guide whether constraints are necessary.

### Deliverables

Create scripts as needed, for example:

```text
scripts/train_dynamic_arm_moment_head.py
scripts/evaluate_dynamic_arm_moment_head.py
tests/test_dynamic_arm_moment_head.py
```

Suggested output root:

```text
artifacts/20260525_dynamic_arm_moment_head_v1
```

Required outputs:

```text
model_config.json
metrics_by_split.csv
moment_metrics_summary.csv
r_hat_distribution.csv
tau_free_energy.csv
per_log_moment_metrics.csv
prediction_curves/
README.md
```

### Evaluation

Report:

```text
mx_b RMSE/R2
my_b RMSE/R2
mz_b RMSE/R2
moment mean RMSE/R2
longitudinal moment my_b
roll/yaw moment mx_b/mz_b
r_hat norm median/p90/p99
tau_free energy fraction
per-log failure cases
```

Also compare:

```text
using ground-truth F_eff
using DeLaurier F_D
using Stage 1/2 corrected F_hat
```

This separates force quality from arm-head quality.

### Interpretation to preserve

Positive result:

```text
Improving force and using a dynamic equivalent arm yields a physically structured moment model.
```

If direct moment head still wins:

```text
The force-arm structure captures a meaningful part of the moment but is not yet expressive enough for all channels, especially roll/yaw or low-SNR moments.
```

---

## Final Synthesis

After all stages, create a short report:

```text
docs/results/2026-05-25-residual-guided-delaurier-greybox-force-arm.md
```

The report should answer:

1. Did force-only DeLaurier recalibration improve `fx_b/fy_b/fz_b`?
2. Did phase/frequency/AoA grey-box modulation improve beyond global recalibration?
3. Did force correction reduce phase-locked, frequency-locked, or condition-dependent residuals?
4. Does a dynamic equivalent arm convert improved force into useful moment prediction?
5. Is `tau_free` small enough to support the force-arm story, or does moment still require direct modeling?
6. What should be claimed in the paper, and what should remain future work?

Use bounded language:

```text
demonstrates
suggests
is consistent with
motivates
remains inconclusive
```

Avoid claiming that learned parameters are true aerodynamic constants unless that is directly validated.

---

## Verification

At minimum:

```bash
python -m pytest tests/test_delaurier_force_recalibration.py tests/test_delaurier_greybox_force_correction.py tests/test_dynamic_arm_moment_head.py -q
```

Also run the main scripts on the full split or a documented subset first, then rerun the best candidates on full data.

Record all commands in the relevant artifact README files.

