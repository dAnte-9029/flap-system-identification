# Phase 0I Implementation Report

## Baseline

- Branch: `refactor/phase0b-foundation-migration`
- Baseline HEAD: `ad807e239b0f3b0cfc3136065e5176f1059a635d`
- Starting point: cumulative, uncommitted Phase 0B–0H worktree
- Pre-Phase 0I backup: `/tmp/flap_phase0_before_phase0i.patch`

## Implemented migration

Thirty-seven existing top-level constants and functions were moved with legacy re-exports:

- `training/data_preparation.py` (3): target schema, deterministic seeding, and existing split-frame loading/subsampling.
- `training/recipes.py` (3): standard, sequence, and rollout fit recipes, including their unchanged data preparation, validation, selection, and bundle-return order.
- `training/orchestration.py` (12): baseline/ablation configuration, recipe-name resolution, run-level train/validation/test dispatch, artifact calls, ablation execution, and baseline comparison.
- `evaluation/diagnostics.py` (16): frozen prediction/evaluation, metadata alignment, audit flags, per-log/regime diagnostics, and diagnostic report execution.
- `artifacts/io.py` (3): existing training-curve, prediction-vs-true, and residual-histogram writers.

`training.py` is now a 213-line compatibility facade. It retains the module-as-package `__path__` mechanism and imports/re-exports canonical symbols; no `training/__init__.py` was created.

## Thin CLI updates

Four already argparse-oriented core scripts now import their callable from the canonical module while retaining all arguments, defaults, paths, printed outputs, and main control flow:

- `scripts/train_baseline_torch.py`
- `scripts/run_baseline_comparison.py`
- `scripts/run_feature_ablation.py`
- `scripts/run_training_diagnostics.py`

## Verification

- Full collection: 315 tests collected with pytest cache disabled.
- Focused run: 50 passed, covering Phase 0B–0I compatibility, all three fit recipes, run artifacts, diagnostics, ablation, baseline comparison, and CLI argument behavior.
- Full suite: 315 passed in 70.29 seconds; no tests were skipped or xfailed.
- Existing standard, sequence, rollout, scheduler/EMA, diagnostics, artifact, ablation, and baseline-comparison tests passed.
- Canonical/legacy identity was checked for all 37 Phase 0I symbols.
- `git diff --check`: passed before report creation and is repeated at completion.

No unresolved import cycle was introduced. Scientific calculations, model/optimizer behavior, split and test timing, metrics, plots, checkpoint/bundle schemas, filenames, and run ordering were not intentionally changed. The existing 106 files under `outputs/` were not modified.

## Deferred

- Feature-derived dimension resolution remains inside the three recipes because it is coupled to the exact prepared array shapes and model-specific call order.
- JSON, CSV, Torch checkpoint, and training-config writes remain in `run_training_job`; separating them further would require a large schema object and wider control-flow change. The migrated artifact module therefore covers only the three previously standalone writers.
- Historical and experiment-specific scripts outside the four selected core CLIs remain unchanged.
- Strict test isolation and any changes to current automatic test evaluation remain outside Phase 0.
- The existing `.gitignore` line `artifacts/` also matches `src/system_identification/artifacts/`. Its three required Phase 0I files exist and pass tests but appear as ignored (`!!`) rather than untracked. Because this phase forbids `.gitignore` changes and staging, review must explicitly decide between a narrow ignore exception and force-adding those three files before commit.
