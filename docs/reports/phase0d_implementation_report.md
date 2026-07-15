# Phase 0D Implementation Report

## Migration

- Branch: `refactor/phase0b-foundation-migration`
- HEAD: `ad807e239b0f3b0cfc3136065e5176f1059a635d`
- Starting point: uncommitted Phase 0B and Phase 0C worktree
- Cumulative tracked-diff backup: `/tmp/flap_phase0_before_phase0d.patch`

`models/features.py` now owns 22 existing feature symbols: the default and paper feature schemas, phase-conditioning schemas, window/sequence feature-mode schemas, feature-set resolvers, deterministic derived-column construction, sequence-order transformation, and related pure column resolvers.

`models/neural.py` now owns 17 existing symbols: all current `nn.Module` model structures plus their forward-only mathematical helpers, including MLP, GRU, LSTM, TCN, Transformer/FiLM, TCN-GRU, SUBNET, ASL, PFNN, Catmull-Rom weights, and private layer helpers.

All 39 migrated definitions match their baseline AST source exactly. `training.py` imports and re-exports the same objects, so legacy and canonical imports are identical. No circular dependency was introduced: feature and neural modules do not import `training.py`.

`structured.py` and `bundles.py` were not created because `training.py` contains no low-coupling structured model definition, while its bundle reconstruction, prediction, checkpoint conversion, and evaluation helpers remain coupled to training normalization and evaluation control flow.

## Remaining training.py responsibilities

`training.py` is 4,166 lines after extraction. It still owns target/frame preparation, window and rollout sample construction, normalization fitting and transforms, loaders, losses, fit loops, early stopping, model builders, bundle reconstruction and prediction orchestration, metrics/evaluation, plotting, diagnostics, candidate/recipe selection, and run-level artifact writing.

Deferred work includes those responsibilities, feature/target frame builders, normalization, losses, training and validation selection, bundle I/O/inference orchestration, and structured models currently implemented in scripts.

## Verification

- Collect-only: 290 tests collected successfully.
- Phase compatibility tests: 15 passed.
- Direct pure model/feature tests: 37 passed, 58 unrelated `test_training.py` tests deselected.
- Total focused tests executed: 52 passed.
- Representative fixed-seed MLP outputs are exactly equal through legacy and canonical imports.
- Representative derived-feature frames are exactly equal.
- `git diff --check`: passed.

No training loop, normalization source, early stopping, selection, test use, evaluation, physics, labels, split, checkpoint format, CLI, plotting, or output behavior was changed. No training or experiment command was run. The pre-existing `outputs/` tree was not modified.
