# Phase 0C Implementation Report

## Baseline

- Branch: `refactor/phase0b-foundation-migration`
- HEAD: `ad807e239b0f3b0cfc3136065e5176f1059a635d`
- Starting point: the complete, uncommitted Phase 0B worktree plus the pre-existing untracked `outputs/` tree
- Phase 0B tracked-diff backup: `/tmp/flap_phase0b_before_phase0c.patch`
- Environment: `flap-train-gpu`, Python 3.11.14

## Production-module migrations

Five production modules were migrated, below the six-module phase limit.

| Existing path or owner | Canonical implementation |
|---|---|
| `system_identification.pipeline._compute_effective_wrench_labels` | `system_identification.labels.effective_wrench.compute_effective_wrench_labels` |
| `system_identification.physics.delaurier_airflow` | `system_identification.physics.delaurier.airflow` |
| `system_identification.physics.delaurier_dynamic_twist` | `system_identification.physics.delaurier.dynamic_twist` |
| `system_identification.physics.delaurier_strip_wrench` | `system_identification.physics.delaurier.strip_wrench` |
| `system_identification.baselines.isaaclab_wing_only_baseline` | `system_identification.physics.baselines.wing_only` |

The label implementation represents the existing whole-aircraft effective external force and moment reconstruction. It was not reinterpreted as an isolated wing load. The current frame, units, signs, aircraft-CG moment reference, validity mask, and invalid-row behavior are unchanged.

The three legacy DeLaurier modules and the legacy wing-only baseline module are explicit compatibility wrappers. Existing public imports resolve to the canonical objects. The tested private baseline helper `_wing_polar_transforms_frd` also remains available because an existing regression test imports it directly. `pipeline._compute_effective_wrench_labels` delegates by direct object import to the canonical label implementation, preserving repository scripts that use the old private path.

## Numerical behavior evidence

- The three canonical DeLaurier primitive files and canonical wing-only baseline file compare byte-for-byte with their pre-migration implementations.
- The four extracted label function bodies compare exactly with their baseline source:
  - `_as_float`
  - `_frame_columns_or_none`
  - `_rotation_body_to_world_from_quaternions`
  - `_compute_effective_wrench_labels`
- Legacy and canonical symbol identity checks pass.
- Representative label, airflow, phase mapping, wrench translation, and baseline calls match.
- The existing frozen DeLaurier regression suite passes, including the `1e-10` Isaac fixture and frame/sign/reference tests.

No formula, constant, default, physical parameter, airflow mode, phase mapping, dynamic twist behavior, separation behavior, quaternion behavior, label definition, mask, or serialization behavior was changed.

## Verification

- Final `python -m pytest -p no:cacheprovider --collect-only`: 286 tests collected successfully.
- Focused test command: 31 passed in 0.59 seconds.
  - `tests/test_phase0c_labels_physics_import_compatibility.py`
  - `tests/test_pipeline.py`
  - `tests/test_delaurier_wing_wrench.py`
  - `tests/test_phase0b_foundation_import_compatibility.py`
- `git diff --check`: passed.

An initial collect-only run exposed the existing test dependency on the private `_wing_polar_transforms_frd` symbol. The compatibility wrapper was corrected to re-export that same canonical object; no algorithm was changed.

## Deferred scope

- Smoothed, lateral-only, time-aligned, and lagged label variants remain in scripts because extracting them would also move CLI, manifest, file-writing, or experiment-specific control flow.
- Prior/label alignment remains in residual-building scripts. A canonical keyed alignment service should be migrated separately with explicit duplicate and missing-key behavior; equal row count alone must not become an alignment contract.
- The rest of `pipeline.py` remains mixed data assembly, ULog, derivative, and orchestration code. Only the behavior-preserving effective-wrench core moved here.
- Calibration, structured corrections, residual models, plotting, evaluation, and training remain in their current locations.

No existing bug was changed or newly identified beyond these structural deferrals.

## Scope and worktree confirmation

- Phase 0B files remain present and unchanged by Phase 0C.
- `training.py`, split behavior, evaluation behavior, scripts, metadata values, `pyproject.toml`, and `.gitignore` were not modified.
- No training, data generation, complete evaluation, plotting, calibration, or model-development command was run.
- The pre-existing `outputs/` metadata fingerprint remained `51420029589b0b8b549190f05157e04e71fc03c98c656ad345c93946c7cca9a1`.
- No cache directory was created in `labels/`, `physics/delaurier/`, or `physics/baselines/`; pre-existing ignored cache directories were left untouched.
