# Phase 0G Implementation Report

## Baseline

- Branch: `refactor/phase0b-foundation-migration`
- Baseline HEAD: `ad807e239b0f3b0cfc3136065e5176f1059a635d`
- Starting point: cumulative, uncommitted Phase 0B–0F worktree
- Pre-Phase 0G backup: `/tmp/flap_phase0_before_phase0g.patch`

## Implemented migration

Thirteen symbols are canonical in three new modules and re-exported by legacy `system_identification.training`:

- `training/loop.py` (9): `_normalized_lr_scheduler`, `_warmup_cosine_lr_lambda`, `_ema_update_state`, `_train_scaled_epoch`, `_train_sequence_scaled_epoch`, `_train_rollout_scaled_epoch`, `_evaluate_scaled_loss`, `_evaluate_sequence_scaled_loss`, and `_evaluate_rollout_scaled_loss`.
- `training/early_stopping.py` (2): `EarlyStoppingState` and `update_early_stopping`.
- `training/selection.py` (2): `BestEpochSelection` and `update_best_epoch_selection`.

The three model-specific fit functions still prepare train/validation arrays, create the model and optimizer, compute validation metrics, and assemble their returned bundle. Their batch training is delegated to the matching single-epoch helper. Best-epoch selection still uses validation loss only, minimizes with the exact `val_loss < best_val_loss - 1e-8` rule, and deep-copies the same raw or EMA state. Patience still resets on improvement, increments once per non-improving epoch, and stops when the count reaches the configured patience.

No independent cross-candidate selector exists in `training.py`; baseline recipes are executed and summarized, but not selected there. That high-level orchestration was therefore not reclassified or moved.

The existing module-as-package compatibility mechanism remains unchanged. No `training/__init__.py` was created.

## Remaining `training.py` responsibilities

`training.py` is 3,065 lines. It retains data preparation around fitting, model and optimizer construction, validation metric/history construction, fit-result bundle assembly, frozen inference/evaluation entry points, diagnostics, split loading, run/artifact orchestration, ablations, and baseline recipe execution.

## Verification

- Full collection: 305 tests collected with pytest cache disabled.
- Focused run: 33 passed, covering all Phase 0B–0G compatibility tests plus the existing ordinary-regressor, sequence prior-anchor, and rollout training paths.
- Phase 0G compatibility test: 5 passed.
- A tiny CPU epoch was compared against the pre-migration inline algorithm: loss, optimizer update, and model parameters matched exactly.
- Synthetic validation scores matched the former best-epoch tolerance, tie behavior, patience count, and stop epoch.
- Six moved pre-existing helpers are AST-equivalent to their pre-Phase 0G definitions.
- Ordinary, sequence, and rollout fit smoke tests passed without GPU or external data.
- `git diff --check`: passed before report creation and is repeated at completion.

No unresolved circular dependency was introduced. Random seeds, batch order, optimizer and scheduler parameters, gradient clipping, EMA timing, early stopping, selection metric, test timing, normalization, loss, loaders, checkpoint/bundle formats, evaluation, physics, labels, and splits were not changed. `outputs/` was not modified.

## Deferred

- High-level validation metric/history table construction remains in each fit function because it depends on model-specific prediction and inverse transformation.
- Data preparation, model construction, and returned bundle assembly remain in the fit functions.
- Baseline recipe and run-directory orchestration remain in `training.py`; no cross-candidate selection implementation was present to migrate.
