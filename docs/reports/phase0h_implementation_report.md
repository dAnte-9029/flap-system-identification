# Phase 0H Implementation Report

## Baseline

- Branch: `refactor/phase0b-foundation-migration`
- Baseline HEAD: `ad807e239b0f3b0cfc3136065e5176f1059a635d`
- Starting point: cumulative, uncommitted Phase 0B–0G worktree
- Pre-Phase 0H backup: `/tmp/flap_phase0_before_phase0h.patch`

## Implemented migration

Eleven symbols are exposed through four new modules and re-exported by legacy `system_identification.training`:

- `training/model_factory.py` (3): `_build_regressor`, `_build_sequence_regressor`, and `_build_rollout_regressor`. Their implementation was already canonical in `models/bundles.py` from Phase 0E, so this module provides training-facing aliases to the same objects rather than duplicating model construction.
- `training/optimizer_factory.py` (2): `build_adamw_optimizer` and `build_training_scheduler`, preserving the existing AdamW and optional warmup-cosine construction.
- `training/history.py` (3): standard, sequence, and rollout validation-history row builders with the existing field order and float conversion.
- `training/bundle_assembly.py` (3): standard, sequence, and rollout trained-bundle schema assembly without file-system I/O.

The three fit functions retain dimension derivation, model-builder invocation order, training and validation orchestration, and final state loading. They delegate only optimizer/scheduler construction, history row construction, and final bundle dictionary assembly.

The existing module-as-package compatibility mechanism remains unchanged. No `training/__init__.py` was created, and all legacy imports reference the canonical objects.

## Remaining `training.py` responsibilities

`training.py` is 3,061 lines. It retains data preparation and dimension resolution, model invocation, training/validation orchestration, best-state loading, frozen inference/evaluation entry points, diagnostics, split loading, artifact writing, ablations, and baseline recipe execution.

## Compatibility evidence

- Model builders are identical objects through `training`, `training.model_factory`, and `models.bundles`; fixed-seed model type, parameter names, shapes, and initialized tensors match exactly.
- AdamW parameter groups and warmup-cosine scheduler state match direct pre-migration construction.
- History row names, insertion order, values, and per-target expansion match the former inline schemas.
- The 59-field sequence, 47-field rollout, and 31-field standard bundle dictionaries are AST-equivalent to the pre-migration dictionary expressions.
- Bundle normalization arrays retain their dtype and identity; model state dictionaries and feature/target schemas are unchanged.
- Existing ordinary, sequence prior-anchor, rollout, and sequence training-tricks tests passed. The last test directly covers scheduler, gradient clipping, EMA, history output, and serialized bundle metadata.

## Verification

- Full collection: 310 tests collected with pytest cache disabled.
- Focused run: 39 passed, covering all Phase 0B–0H compatibility tests and four directly relevant existing training tests.
- Phase 0H compatibility test: 5 passed.
- `git diff --check`: passed before report creation and is repeated at completion.
- No unresolved circular dependency was introduced.

Model classes, forward behavior, optimizer/scheduler parameters, random order, training and selection, test timing, checkpoint serialization, evaluation, physics, labels, and splits were not changed. `outputs/` was not modified.

## Deferred

- Feature-derived input dimension resolution remains beside the prepared arrays in each fit function.
- Model-specific validation prediction and metric computation remain in each fit loop; only the pure history row construction moved.
- Split loading, test evaluation, run directories, artifacts, diagnostics, recipes, and cross-model scheduling remain in `training.py`.
