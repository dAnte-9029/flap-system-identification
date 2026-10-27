# Control-conditioned training weights v1

Offline CPU-only audit on explicitly requested historical `velocity_rate_multistep_v1/main_50hz` train. **Training runs: 0.** No model/checkpoint changes, no test/validation data reads, no dependencies installed.

Run from this worktree with the existing interpreter (the complete weight audit takes a few minutes):

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/control_conditioned_training_weights_v1
$PY "$EXP/analyze.py" > "$EXP/results/run.log" 2>&1
$PY "$EXP/describe.py" > "$EXP/results/description.log" 2>&1
$PY -m pytest -q "$EXP/test_weights.py" > "$EXP/results/tests.log" 2>&1
$PY "$EXP/verify.py" > "$EXP/results/verification.log" 2>&1
$PY "$EXP/record.py"
```

The registered local data/cache paths are in `config.json`. `analyze.py` checks their registered SHA256 before use. Missing/mismatched artifacts fail; there is no download, dataset rebuilding or fallback. It loads only `dt/future_mask` members of train.npz, plus named train columns/files. Historical scripts are imported for constants/native ZOH utilities, not executed as training programs.

- `report_zh.md`: evidence, caveats, Q1–Q5 and selected W1 plan.
- `weighting.py`: pure command/time weighting, no file/model/label access. Fixed thresholds match config; changing them defines a different experiment.
- `weighted_loss.py`: reusable scalar `weighted_loss` and per-window `window_loss`, importing the original error definitions.
- `results/{full_train,fold0,fold1,fold2}/weights.npz`: complete W0/W1/W2, `registered_index`, `window_id`, `flight_id`, `future_mask`, relative integer `output_us`, support flags, exposure, delta and four-coordinate statistics. W0 scalar per window implies uniform valid steps; padding has raw weight1 but coefficient0.
- `window_audit.csv.gz`: exact origin/time identity, frozen past eligibility/features, reason strings, sign/duration and ESS. Use `registered_index` to gather weights with an original dataset index; do not assume fold-local rows match an arbitrary data-loader order.
- Each fold `isolation.json`: original train/held/validation members and independent pointwise generation. No fold thresholds are estimated from global data.
- `coverage.csv`, `group_contributions.csv`, `weight_distribution.csv`, `time_contribution.csv`, `step_contribution.csv`, `control_coupling.csv`: full and per-fold numerical audits. W2 `added` is raw step-weight mass, **not** window training share.
- `sign_and_clock_coverage.csv.gz`: positive/negative W2 rolling exposures, unique horizon exposure durations and history/decision command comparison; rolling durations overlap and are not independent events.
- `state_distribution.csv`, `summary.json`, `supplementary_summary.json`: descriptive results; output labels and model errors never enter weight eligibility.
- `figures/*.png`: ESS, flight/sign coverage, time contribution, state/configuration and command coupling.
- `source_receipt.json`, `artifact_manifest.json`, `environment.json`: data/cache/source and generated-file hashes/environment; manifest excludes itself.
- `tests.log`, `independent_verification.json`, `verification.log`: tests and independent quadrature/ESS checks (native quadrature on 4,458 windows/all94 flights; full identity/ESS checks).
- `next_stage_protocol.json`: frozen **plan only**; 12 future runs, none executed.

Minimal future trainer call (not executed here):

```python
# Import weighted_loss.py via this directory / importlib.
# Preserve original model, autonomous rollout, four component weights and fold-fit scales.
loss = weighted_loss(pred, target, future_mask, loss_scales, component_weights,
                     scheme="W1", window_weight=batch_w1)
# W0: scheme="W0". W2: scheme="W2", step_weight=batch_w2.
```

Pass tensors on the same device/dtype as the loss. W1 normalizes in each batch; W2 normalizes valid time weights inside each window. `window_loss(..., scheme="W1")` returns unweighted original window losses; use scalar `weighted_loss` for W1 aggregation. Weights are detached. Validation checkpoint selection must use **uniform original** flight-equal loss for every arm, even when training with W1.

W2 raw weights are prefix-causal, but its whole-window normalization is an offline loss definition. W1 starts at decision publication; W2 evaluates at unchanged native state-output measurement timestamps. No shift of model inputs, future truth, controls, or phase. Weights do not create independent excitation or physical causal labels.
