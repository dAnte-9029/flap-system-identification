# September state-consistency diagnostic contract

Authorized: consolidate the temporary error-source scripts, normalize metric
aggregation and investigate logged position/velocity consistency. No training,
label changes, lag calibration, oracle model promotion or sealed test access.

Use registered September v2 train/validation and exactly the selected windows
of september_history_lengths_20260911: 3,966 train 2s and 6,669 validation 5s.
Default execution verifies cached history_26 predictions against original error
arrays, window IDs, dt and dataset truth. --replay regenerates these predictions
from the frozen checkpoint/normalization on GPU 1 and verifies the same errors.
The consolidated runner replaces /tmp/run_diag.py, analyze_diag.py and
finalize_step2.py; archive their available original text in the independent run.

All endpoint results are Euclidean vector RMSE within each log followed by an
unweighted mean across logs. Separately label pooled-window vector RMSE. Axis
RMSE is diagnostic and must not be presented as vector RMSE. Explicitly retain
nonfinite failures; fail on invalid dt. Distinguish per-step position-increment
residual from its cumulative integral. Verify the signed decomposition:
position_error(t)-position_error(0) = integrated_velocity_error(t)
  + predicted_cumulative_residual(t) - true_cumulative_residual(t).
These vectors are correlated; RMS values cannot be added or interpreted as
independent causal percentages. A true-state consistency residual is not an
irreducible error floor for all forecasting models.

Read only the 15 registered train/validation ULogs, checking their hashes.
Verify every position/velocity sample and both timestamps exactly against
vehicle_local_position instance 0. Compare cumulative residuals using publication
and sample clocks on identical indices. Audit position, velocity and heading
reset counters, and estimator-selector changes within windows. Use z_deriv only
as a diagnostic alternative to vz, not as an updated training label. Count raw
transitions once for correlation/step summaries despite overlapping windows.
Optional active-estimator output-tracking-error correlation uses a causal asof
join with maximum age 250ms, selected by estimator_selector_status; missing
coverage is reported, never silently substituted with another instance.

Predeclared lag sensitivity: velocity(t+lag), lag in {-100,-50,0,50,100}ms,
linear interpolation inside each window; remove 110ms from both ends for all
lags to preserve a common interior span. Report actual span and separate train
from validation. This uses future values for offline diagnostics and must not
become an unreviewed inference alignment, time correction or validation-fitted
hyperparameter. Do not compare cropped-span metrics to full horizons.

Record source data manifest, source artifact hashes, current diagnostic source,
and unchanged source checkpoints. Existing run/data files are not overwritten.
Output: artifacts/september_state_consistency_20260911, with manifest, summary,
per-log and aggregate CSVs, frozen prediction arrays and raw-window audit table.
