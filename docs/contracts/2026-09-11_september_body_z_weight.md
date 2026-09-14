# Matched body-z angular-rate supervision experiment

Authorized after oracle diagnostics: inspect body-axis errors, then execute one
minimal training change to test whether rotation improvement transfers to full
free-run trajectory accuracy. No oracle inputs in this experiment.

## Diagnostic motivation and limits

Frozen history_26 free-run predictions show similar instantaneous body x/y rate
RMSE from 1 to 5s but larger growth of integrated body-z error. Validation 5s
axis-integral RMSE is about x/y/z = .138/.197/.573 rad; train 2s is .089/.061/.140.
These integrals are persistence diagnostics in moving body coordinates, not
SO(3) attitude error or Euler yaw angle. Small mean signed errors across logs can
hide per-window accumulated error. Instantaneous x/y errors are larger than z;
uniformly increasing all rate weights would emphasize a different problem.

A single predeclared candidate sets body_rate_axis_weights=(1,1,4) instead of
(1,1,1). This quadruples the body-z contribution to the existing rate loss after
its unchanged /2 scale. Body z is not Euler yaw rate. Four is an exploratory fixed
coefficient, not an optimized value; no grid search or guarantee of improvement.
Keep x/y rate, attitude, position, velocity, phase and frequency loss terms intact.
No cumulative-integral auxiliary loss, architecture, label, timing or data changes.

## Matching and gates

Same history_26 experiment origins: 3,966 train and 6,669 validation, same saved
normalization, 26 history samples, controlled GRU 64, 2s objective, 80 epochs,
seed 17, batch 256, lr 3e-4, weight decay 1e-5, clip 5, GPU 1. Same initialization,
shuffle and 1,280 optimizer updates. Refit baseline and candidate from scratch.

Default axis weights preserve original loss arithmetic. Require baseline checkpoint
bitwise equality with original history_26.pt before candidate training; otherwise
fail and do not claim a matched experiment. Verify all baseline endpoint error
arrays against original at evaluation. Record deliberate training-source change
and verify other prior source hashes. Save source artifact hashes and dataset
provenance; fail on missing/mismatched sources, no silent fallback.

Training uses future truth only as loss targets. No validation fitting/early
stopping, future oracle signals or September 8 test access. Fixed final epoch 80.

## Evaluation and decision evidence

Train 1/2s and validation continuous 1/2/3/5s without resets. Save all state
predictions, error traces, per-window endpoints, per-log and equal-log curves,
axis signed bias/RMSE and integrated-axis error diagnostics. Preserve original
logged-position error and auxiliary true-velocity-integral-position error with
separate names. Retain nonfinite and large-error fractions; all rotation states
are predicted, so the original position>10m OR attitude>60deg threshold applies.

Inspect whether z error and accumulated error improve, whether this lowers actual
attitude/velocity/position errors, and whether short-term or other-axis accuracy
regresses. A lower weighted training objective is not itself a performance gain.
If z supervision improves but rotation/translation does not, do not blindly raise
weights further. If improvement is broad, replicate across seeds before promotion.

Entry: scripts/run_september_body_z_weight.py. Output:
artifacts/september_body_z_weight_20260911, sibling .run.log/.launch.json. Background
execution without monitoring, no overwrites or automatic retries. Independent
axis analysis: scripts/analyze_september_body_rates.py and
artifacts/september_body_rate_analysis_20260911.
