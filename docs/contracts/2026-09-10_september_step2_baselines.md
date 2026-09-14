# September Step 2: local prediction versus continuous rollout

Frozen before training. Dataset: registered `trajectory_v2_september_phase_observed`;
cohort: core. Read September 6 train and September 7 validation only. September 8
remains sealed. All data integrity hashes are verified before loading.

Models: constant NED velocity/body rate; ridge without controls (alpha=1); original
64-unit Main V1 GRU with controls; identical GRU without controls. Both GRUs use
26 samples of real history (including origin), 40 epochs, seed 17, batch 256,
AdamW lr=3e-4, weight decay=1e-5, gradient clip=5, and a 50-step/1 s rollout loss.
No new architecture, Hall feature, training-horizon sweep or validation tuning.
GRU phase remains sin/cos(relative phase - origin phase). Ridge retains its old
log-relative phase feature convention; it is a reference, not an absolute-phase
model. Diagnostic logged-phase bins never enter model inputs.

Training uses the registered 1 s windows with >=0.5 s available history and >=25
preceding samples. Validation uses registered 5 s windows with the same history
rule. All models and reset modes use identical parent windows. Shorter-horizon
results are prefixes of these 5 s windows, not a different short-window cohort.
Coverage counts per log and the retained source window tables are saved.

Two modes:

- Continuous: initialize once, then predict 250 steps without real-state updates.
- Reset every second: five independent 50-step predictions from true observed
  states and freshly encoded true histories at offsets 0/50/100/150/200 samples.
  This is a local diagnostic, not a continuous simulation. Each interval has its
  own local horizon; global time is also recorded. At a reset boundary plots use
  the previous interval endpoint error, not the zero error immediately after reset.

Only the known four-channel command tape may enter future rollout. Reset-mode
observations are permitted at each explicitly declared new origin only; continuous
mode never receives them. The reset comparison also renews hidden/flap state, so
its benefit cannot uniquely identify a single cause of error accumulation.

Outputs: every-step (50 Hz) per-log RMSE and nearest-rank p95 curves; equal-log
macro curves; endpoint metrics at 1/2/3/5 s; whole-interval trajectory RMS;
per-origin maneuver and logged-phase-bin diagnostics at integer endpoints;
saved per-window error traces and representative trajectories; checkpoints,
training histories, coverage, plots, and manifest. Large-error diagnostic is
position >10 m OR attitude >60 deg; it is not a deployment tolerance. Nonfinite
predictions count as failures, not dropped rows. Overlapping windows and phase
groups are not independent replicates. Maneuver labels are threshold-based audit
proxies at each mode's origin, not verified mission labels.

Single-seed baseline diagnostic; no significance or causal identification claim.
Data/architecture changes belong to subsequent steps. Fixed configuration and
no-overwrite output root are recorded before the run. GPU 1 selected after checking
available memory. Background run writes `status.json` and `run.log`; exceptions
write failed status and traceback. No automatic retry or agent monitoring.

## Launch

Preflight retained 4,363 train windows and 6,835 validation parent windows, covering
all six train and nine validation logs. GPU allocation preflight succeeded on
`cuda:1`. The 15 targeted diagnostic/model/baseline tests passed before launch.

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -u scripts/run_september_step2.py
```

Default output: `artifacts/september_step2_20260910/`. Detached launch redirects
stdout/stderr to `artifacts/september_step2_20260910.run.log` and records the PID in
`artifacts/september_step2_20260910.launch.json`. The entrypoint refuses to overwrite
an existing run directory. `status.json` states loading/training/evaluating/completed
or failed, and failed runs also save `traceback.txt`. Training CSV/checkpoint files
are saved after each model finishes; stdout is not a per-epoch progress stream.
