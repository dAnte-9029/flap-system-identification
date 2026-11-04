# Slow/fast feasibility study

Run from the research worktree:

```bash
bash experiments/slow_fast_separation_v1/run.sh docs/analysis/results/slow_fast_separation_v1/reproduction
```

The destination must not exist. Uses the registered local dataset and the specified `flap-train-gpu` Python environment, with CPU BLAS threads limited to one. Only train and validation parquet paths are constructed. No model training or formal labels.

`methods.py` implements midpoint quadrature, within-cycle half-cell resampling, cycle-balanced harmonic regression, masked cubic synchronous fitting, and past-cycle linear extrapolation. `run.py` evaluates flight by flight and writes coverage, paired prediction, preservation and runtime diagnostics. `summarize.py` produces a report and two figures. Four core tests are in `test_methods.py`.

The definitive run is `final_run`. Earlier local interrupted runs are development artifacts and are not publication evidence. Findings and remaining validation limitations are explicitly stated in the report. Do not interpret a partial feasibility result as authorization for GRU training.
