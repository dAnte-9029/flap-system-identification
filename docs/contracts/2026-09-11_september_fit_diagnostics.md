# Frozen September fitting and generalization diagnostics

Use all three frozen checkpoints from september_multihorizon_20260911. No training,
normalization fitting, validation tuning, or September 8 test access. Verify the
registered September dataset and original run source hashes; save checkpoint,
normalization and evaluation-input provenance in a separate immutable run.

Evaluate all 4,198 original training origins at 1/2 seconds. Additionally evaluate
1/2/3/5 seconds on the subset of these origins with registered five-second coverage.
Record that subset explicitly: its short-horizon scores must be shown alongside
its long-horizon scores, rather than splicing different cohorts into one curve.
Use the original 6,835 validation origins and saved frozen-model endpoint errors.
No intermediate real-state resets. Future measured states remain metric targets.

Stratify by initial NED speed magnitude and absolute body z angular rate, using
tertile boundaries calculated only from the original training origins. Body z rate
is a turn-intensity proxy, not flight-path heading rate or curvature. Save exact
boundaries; retain out-of-range validation values in the outer bins. Report per-log
RMSE, equal-log macro RMSE, contributing logs and window counts. Retain nonfinite
failures and window identity. Sparse-bin comparisons are exploratory; overlapping
windows are not independent samples.

Training 1/2-second errors diagnose fitting. Training 3/5-second errors diagnose
extrapolation beyond the training objective. Train/validation differences also
contain cross-day operating-condition differences and cannot by themselves prove
overfitting. This stage does not choose architecture or establish scientific
promotion. Output directory: artifacts/september_fit_diagnostics_20260911.
