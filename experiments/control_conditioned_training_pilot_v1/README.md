# Control-change W1 training pilot

Frozen historical `velocity_rate_multistep_v1/main_50hz` experiment: three fixed folds × M0/M1 × W0/W1 × seed17, six epochs each. Expected 12 runs, 72 epochs, 6,384 updates. No W2, new architecture, parameter search or test partition.

All new outputs reside here. Model definition comes unchanged from `../structured_control_effectiveness_v1/model.py`; W1 loss is imported unchanged from `../control_conditioned_training_weights_v1/weighted_loss.py`. Training clones the previous fixed-budget orchestration without editing it. The initial parameter hash is saved so W0/W1 initialization equality can be verified within each fold/architecture.

## Reproduce

Use the existing environment, no dependency installation:

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/control_conditioned_training_pilot_v1
$PY "$EXP/prepare.py" > "$EXP/results/prepare.log" 2>&1
$PY "$EXP/preflight.py"
$PY "$EXP/train.py" --device cuda:1 > "$EXP/results/training.log" 2>&1
$PY "$EXP/evaluate.py" --device cuda:1 > "$EXP/results/evaluation.log" 2>&1
$PY "$EXP/analyze.py" > "$EXP/results/analysis.log" 2>&1
$PY "$EXP/describe.py" > "$EXP/results/description.log" 2>&1
$PY "$EXP/reuse_support.py" > "$EXP/results/support.log" 2>&1
$PY "$EXP/verify.py" > "$EXP/results/verification.log" 2>&1
$PY "$EXP/verify_pairing.py" > "$EXP/results/verification_pairing.log" 2>&1
$PY -m pytest -q "$EXP/test_contract.py" "$EXP/test_results.py" > "$EXP/results/tests_final.log" 2>&1
$PY "$EXP/plot.py" > "$EXP/results/plot.log" 2>&1
$PY "$EXP/write_report.py"
$PY "$EXP/archive.py"
```

The checked-in result directory is an immutable completed record. Reproduction should use a fresh experiment output copy/worktree; do not overwrite completed artifacts. `train.py` skips completed runs only after checking completion/checkpoint hash. Incomplete runs resume from `last.pt` only with matching config, trainer, model, weights and normalization, restoring optimizer/scheduler/Python/NumPy/CPU/CUDA RNG and the deterministic epoch permutation. Resume is not a new best-of-many trial.

## Contracts

- Full registered H1000 fitting windows: 87,239 / 95,627 / 87,750; 500 ms histories, unchanged native future dt and exact ZOH canonicalization.
- Per-fold normalization and loss scales reuse the archived independently verified fit-only results, with source cache, fit-index and fit-function hashes checked. No full-training normalization is used.
- Frozen W1 arrays are matched by `registered_index`, `window_id` and `flight_id`. Every batch uses the same local index tensor to gather input, target, mask and weights; `training.batch` passes only the original eight permitted input fields to the model.
- Stored W1 float64 weights are retained exactly. Model/targets/original component losses remain float32; the W1 scalar weighted reduction is float64. W0 and all-one W1 dispatch to the original uniform loss with exact value/gradient equivalence. Weight dtype is not a learned hyperparameter.
- AdamW 0.001, weight decay 0.0001, clip5, batch1024, cosine `T_max=6`, no early stop. Same epoch permutation across all four combinations, independent of model RNG. Initial parameters match within architecture/W0-W1.
- Best checkpoint: **uniform W0** original four-component validation loss, equal weights for the original 27 validation flights; minimum improvement 1e−4. No response metric chooses an epoch. Fold-normalized losses are not compared as physical-unit cross-fold accuracy.
- Training uses original TF32 settings; evaluation disables TF32 as in the source evaluation. No teacher forcing or true future phase input.
- GPU0 was occupied by another task; this experiment uses GPU1 without changing batch size or stopping other processes. GPU snapshots and per-epoch wall times/peak allocations are saved.
- B/D at300ms, direction threshold0.1 (sensitivities0.05/0.2); D uses the frozen270ms centered average. Original A retained. Missing endpoint/filter support stays missing; no extrapolation.
- All48 main pairs are scored once by their assigned held-flight checkpoint per combination;11 dependence components and all fixed subsets/background rules remain unchanged. Background pairs crossing folds remain unavailable. `validation:44` is retained only as a separately labelled selection-exposed supplemental case.
- `CA/CB/C/H` are internal model decompositions; no real swapped trajectories or counterfactual accuracy labels exist.

## Outputs

- `report_zh.md`: conclusions and limits; `config.json`: immutable training/evaluation settings. `results/resolved_contract.json` identifies unused inherited descriptive fields (no 2000ms experiment or M2 run).
- `folds/fold*/normalization*.json`, `weight_alignment.json`: fit-only provenance and exact source weight identity.
- `folds/fold*/training/{M0W0,M0W1,M1W0,M1W1}/seed17/`: best/last checkpoints, epoch JSONL, provenance, summary and effective configuration.
- `results/training_history.csv`, `training_summary.csv`, `training_completion.json`: all72epochs/12runs and budget checks; each JSONL epoch also contains every batch's weight mean/sum/ESS.
- `results/predictions_fold*.npz`: four-combination AA/AB/BA/BB native predictions with explicit case indices.
- `factual_curves/points.csv.gz`, `internal_curves/points.csv.gz`, `window_points.csv.gz`: recorded-control responses, internal swaps and individual window increments.
- `factual_summary.csv`, `paired_changes.csv.gz`, `paired_summary.csv`: same-architecture W1−W0 comparisons, folds, dependence components, pair/flight/component weighting; positive/negative amplitude bias and threshold sensitivity.
- `per_group_300.csv`, `component_assignments.csv`, `peak_metrics.csv.gz`: all components/flight/config/date and supported peak diagnostics.
- `heldout_trajectory.csv.gz`, `trajectory_per_flight.csv`, `trajectory_summary.csv`: all held H1000 windows in physical units, including p/q/r errors; not merely selected matched windows.
- `inference_checks.json`, `independent_verification.json`, `first_epoch_regression.json`, test logs: input causality, ZOH redundancy, interpolation/filter integration, metric/graph checks and historical epoch1 regression.
- `source_receipt.json`, `archive_manifest.json`, `environment.json`: data, code, normalization and checkpoint SHA256 plus runtime environment.
- `figures/*.png`: four combinations, paired gains/losses, all11 components, q/r, trajectories, internal C/H and all five prespecified cases.

These repeatedly used development folds are not a new blind test. Pilot findings do not validate physical causal control effectiveness or MPC. Stop after this pilot; a 24-epoch multi-seed experiment needs a separate research decision.
