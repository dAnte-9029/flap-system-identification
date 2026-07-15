# Phase 0E Implementation Report

## Migration

- Branch: `refactor/phase0b-foundation-migration`
- HEAD: `ad807e239b0f3b0cfc3136065e5176f1059a635d`
- Starting point: uncommitted Phase 0B through Phase 0D worktree
- Cumulative tracked-diff backup: `/tmp/flap_phase0_before_phase0e.patch`

Thirty-two existing top-level functions were migrated without source-body changes:

- `models/bundles.py`: 11 bundle schema, model-type, model-factory, and frozen-state reconstruction helpers.
- `evaluation/prediction.py`: 12 device resolution, frozen normalization application, loader, and CPU/GPU batch-inference helpers.
- `evaluation/metrics.py`: 3 pure metric, bin-validation, and disjoint-target aggregation helpers.
- `evaluation/reports.py`: 4 history, flattened-metric, metric-row, and target-group table helpers.
- `plotting/figures.py`: 2 existing ablation and baseline-comparison figure writers.

`training.py` imports and re-exports the same function objects. All 32 canonical definitions match their baseline AST source exactly. The new modules do not import `training.py`, so no circular dependency was introduced.

## Compatibility and remaining responsibilities

`training.py` is 3,539 lines. It still owns frame/window/sequence/rollout preparation, normalization fitting, target transformation, losses, fit loops, early stopping, candidate selection, high-level bundle prediction, per-log and regime evaluation orchestration, diagnostic execution, prediction/residual/training plots, checkpoint file I/O, and run artifact writing.

Checkpoint keys, bundle field names, defaults, tensor conversion, state-dict loading, and file format are unchanged. The existing `_to_serializable_bundle` and model reconstruction functions moved exactly; checkpoint read/write control flow remains in `training.py`.

Deferred because moving them would require a reverse dependency on `training.py` or moving training-coupled frame preparation:

- `predict_model_bundle`, prediction metadata construction, and bundle-specific frame/target array builders;
- `evaluate_model_bundle`, per-log/regime evaluation, and diagnostic orchestration;
- prediction/residual and training-history plots;
- checkpoint file I/O and full run orchestration.

No existing non-training test directly isolates those internal helpers; existing high-level coverage performs fitting or a training job and was not run in this phase.

## Verification

- Collect-only: 295 tests collected successfully.
- Focused Phase 0B–0E compatibility suite: 20 passed in 1.49 seconds.
- Representative metric dictionaries are identical.
- Representative CPU batch predictions are identical.
- Bundle serialization keys, tensor dtypes, values, and untouched metadata are identical.
- Repeated legacy/canonical plotting calls produce byte-identical PNG output.
- `git diff --check`: passed.

No training, selection, test protocol, metric formula, plot content/style, serialization, physics, labels, split, CLI, or output behavior was changed. No training, data generation, or experiment command was run, and the pre-existing `outputs/` tree was not modified.
