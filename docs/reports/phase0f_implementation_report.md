# Phase 0F Implementation Report

## Baseline

- Branch: `refactor/phase0b-foundation-migration`
- Baseline HEAD: `ad807e239b0f3b0cfc3136065e5176f1059a635d`
- Starting point: cumulative, uncommitted Phase 0B–0E worktree
- Pre-Phase 0F backup: `/tmp/flap_phase0_before_phase0f.patch`

## Implemented migration

Twenty-three top-level symbols were moved without changing their AST bodies:

- `training/windows.py` (7): `prepare_feature_target_frames`, `_normalized_window_mode`, `_window_offsets`, `_window_feature_name`, `prepare_windowed_feature_target_frames`, `prepare_causal_sequence_feature_target_frames`, and `prepare_causal_rollout_feature_target_frames`.
- `training/normalization.py` (9): `_fit_feature_stats`, `_fit_target_stats`, `_transform_targets`, `_fit_sequence_feature_stats`, `_fit_rollout_feature_stats`, `_transform_features`, `_inverse_transform_targets`, `_transform_sequence_features`, and `_transform_rollout_features`.
- `training/losses.py` (4): `resolve_target_loss_weights`, `_target_loss_weights_as_dict`, `_normalized_loss_type`, and `regression_loss`.
- `training/loaders.py` (3): `_make_loader`, `_make_sequence_loader`, and `_make_rollout_loader`.

`system_identification.training` remains the legacy public import and re-exports the canonical objects. Because Python gives a same-named package directory precedence over `training.py`, no `training/__init__.py` was created. Instead, the legacy module exposes the new directory as its submodule search path. This preserves both `from system_identification.training import X` and `from system_identification.training.<module> import X`.

Phase 0E's `evaluation.prediction` paths for moved normalization and loader helpers remain compatible through lazy re-exports. Its prediction functions import the same canonical loaders at call time, avoiding an import cycle without changing inference behavior.

## Remaining `training.py` responsibilities

`training.py` is 3,226 lines. It retains random seeding, scheduler and EMA helpers, training/evaluation loops, inline early stopping, candidate selection, run orchestration, diagnostics, and artifact/checkpoint orchestration.

No independent early-stopping state/helper or low-level checkpoint save/load helper exists in the current code. Extracting the inline state transitions or high-level checkpoint control flow would change control-flow boundaries, so these are deferred. Existing bundle serialization remains canonical in `models/bundles.py` from Phase 0E.

## Verification

- Full collection: 300 tests collected with pytest cache disabled.
- Focused run: 37 passed, covering Phase 0B–0F compatibility plus the existing window, sequence, rollout, target-weight, and weighted-Huber tests.
- Phase 0F compatibility test: 5 passed, including legacy/canonical object identity, representative window construction, normalization fit/transform/inverse, loss and loader parity, and an unchanged bundle checkpoint round trip.
- Static behavior check: all 23 moved definitions are AST-equivalent to the pre-Phase 0F definitions saved before migration.
- `git diff --check`: passed.
- No unresolved import cycle remains.

Checkpoint keys, dtype conversion, state dictionaries, filenames, and serialization formats were not changed. Training protocol, normalization scope, windows, loss weighting, batch ordering, early stopping, selection, test evaluation, physics, labels, and split behavior were not changed. `outputs/` was not modified.

## Deferred

- Inline early-stopping bookkeeping in the three training loops.
- Scheduler, warmup, EMA, and scaled-loss evaluation helpers that do not fit the requested pure-module boundaries cleanly.
- High-level checkpoint and run-directory orchestration.
