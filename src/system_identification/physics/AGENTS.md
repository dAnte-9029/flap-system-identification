# Physics

Make force/moment frames, units, signs, reference points, geometry, and phase/airflow assumptions explicit. Keep physically measured quantities distinct from fitted parameters and modeling assumptions.

Physics models may be substantially redesigned when requested; do not require a separate phase or an approved historical design document first. Do not silently mutate an existing frozen baseline. Where a new force or aerodynamic formulation changes torque, check the corresponding moment consistency.

Use focused analytic, numerical, or physical checks for properties central to the proposed model. Old/new numerical parity is required only for behavior-preserving work, not for a purposely different physical model. No default blanket regression suite, symmetry checklist, or parameter sweep.
