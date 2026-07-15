# Phase 0 Migration Summary

## Outcome

Phase 0B–0I converted the previously concentrated package into explicit conventions, data, labels, physics, models, training, evaluation, plotting, and artifact boundaries while preserving legacy imports. The tracked baseline `training.py` contained 5,470 lines; the final compatibility facade contains 213 lines.

No scientific method, model definition, split, normalization, training/test protocol, metric, plotting behavior, checkpoint schema, or artifact filename was intentionally changed.

## Migration map

| Phase | Canonical boundary | Main migration |
|---|---|---|
| 0B | `conventions/`, `data/` | Phase conventions, resampling, preprocessing, and split utilities; legacy wrappers retained. |
| 0C | `labels/`, `physics/` | Effective-wrench reconstruction and frozen DeLaurier/wing-only physics implementations. |
| 0D | `models/features.py`, `models/neural.py` | Deterministic feature definitions and existing neural/model structures. |
| 0E | `models/bundles.py`, `evaluation/`, `plotting/` | Bundle inference/build helpers, metrics, prediction, reports, and reusable figures. |
| 0F | `training/windows.py`, `normalization.py`, `losses.py`, `loaders.py` | Training foundation utilities. |
| 0G | `training/loop.py`, `early_stopping.py`, `selection.py` | Epoch loops, validation loss evaluation, patience, and best-epoch selection. |
| 0H | `training/model_factory.py`, `optimizer_factory.py`, `history.py`, `bundle_assembly.py` | Training-facing model construction, optimizer/scheduler construction, validation history, and bundle schemas. |
| 0I | `training/recipes.py`, `orchestration.py`, `data_preparation.py`, `evaluation/diagnostics.py`, `artifacts/io.py` | High-level recipes, run dispatch, diagnostics, selected artifact writers, and thin CLI imports. |

## Directory responsibilities

- `conventions/`: phase and future coordinate/unit/time identity conventions.
- `data/`: generic resampling, preprocessing, and split structures.
- `labels/`: whole-aircraft effective-wrench reconstruction.
- `physics/`: frozen physical priors and pure aerodynamic helpers.
- `models/`: feature schemas, model structures, and inference bundle contracts.
- `training/`: data preparation, optimization foundations, fit recipes, validation selection, and run orchestration.
- `evaluation/`: frozen prediction, metrics, reporting, and diagnostics.
- `plotting/`: reusable figure construction.
- `artifacts/`: existing standalone artifact writers and future schema-preserving I/O extraction.

## Compatibility strategy

Legacy module files remain explicit wrappers where modules were renamed. `system_identification.training` remains the public legacy entry point and exposes the sibling `training/` directory through `__path__`; no conflicting `training/__init__.py` exists. Old and canonical imports resolve to the same implementation objects, as checked by Phase 0B–0I compatibility tests.

## Thin core scripts

The following scripts retain their argparse interface and main function but import canonical package orchestration directly:

- `train_baseline_torch.py`
- `run_baseline_comparison.py`
- `run_feature_ablation.py`
- `run_training_diagnostics.py`

The other historical and experiment-specific scripts were intentionally not reorganized.

## Verification

- Collect-only: 315 tests.
- Focused cumulative compatibility and orchestration run: 50 passed.
- Full repository suite: 315 passed in 70.29 seconds, with no skips or xfails.
- Existing 106 untracked `outputs/` files remained untouched.
- No repository cache, model, data, figure, or experiment output was intentionally generated.

## Deferred and known constraints

- `training.py` uses an intentional module-as-package compatibility mechanism; a future package-name/API transition should replace it only with an explicit deprecation plan.
- Feature-derived dimension resolution remains in model-specific recipes to preserve array-dependent call order.
- JSON/CSV/checkpoint/config writes remain coupled to `run_training_job`; only the existing standalone figure writers moved to `artifacts/io.py`.
- `pipeline.py` retains mixed ULog assembly and higher-level data orchestration outside the effective-wrench core.
- Keyed prior/label alignment services and strict duplicate/missing-key contracts remain deferred.
- Strict test isolation and changes to automatic test evaluation were explicitly outside Phase 0.
- Historical experiment scripts remain available and may still contain experiment-specific orchestration or fixed defaults.
- The existing broad `.gitignore` rule `artifacts/` hides the new `src/system_identification/artifacts/` package. The three files are present and tested but require an explicit review decision before commit; Phase 0 did not alter `.gitignore` or stage ignored files.

## Recommended next priorities

1. Perform one independent, read-only cumulative Phase 0 diff audit before commit.
2. Freeze the full-suite result and compatibility mapping in the review record.
3. Commit Phase 0 as one identifiable migration boundary only after review.
4. In later separately authorized phases, address remaining artifact schema extraction, keyed alignment contracts, and test-isolation policy without combining them with file movement.
