# Phase 0A Repository Read-Only Audit

Date: 2026-07-15

Scope: repository state at `main` / `c0ee4725597fe149b2ad4697d9a54841b5579e6d`

Status: read-only audit; no migration or model implementation

## 1. Executive summary

### Observed facts

The tracked repository contains 223 files: 79 documentation files, 68 flat Python scripts, 46 test modules plus one JSON fixture, 18 package files, two aircraft-metadata files, and nine root assets/configuration files. The installed editable package is `system-identification==0.1.0`. The package contains stable data, phase, split, physics, training, plotting, and one analysis implementation, but much of the reusable research logic remains in `scripts/`. Static inspection found 38 of 46 test modules (38/46) directly importing `scripts.*`, versus 10 importing `system_identification.*`; 18 script modules are imported by other scripts, with no static import cycle but strong hub coupling.

The implemented flow is real and testable, but it is not one single pipeline. It is a chain of versioned dataset and artifact transformations connected by paths, CLI defaults, and script-to-script imports. Whole-log and cycle-block split implementations coexist. Current neural training fits normalization on training rows and selects epochs on validation loss. Several structured-correction tests explicitly enforce validation-only selection. Test metrics and test figures are nevertheless emitted by the ordinary training job, so procedural discipline is still required to prevent human test-set iteration.

The highest-priority migration need is to move reusable behavior out of scripts while freezing behavior and artifact schemas first. Package renaming is not the current bottleneck. The recommended target therefore retains `system_identification`, introduces domain subpackages, thin categorized CLIs, explicit configs and run manifests, and defers all new research behavior until the migration stages are frozen.

### Recommendations

Adopt the conservative, staged architecture in the companion architecture document. Begin Phase 0B only after independent review. Do not combine file moves, behavior changes, and model development in one phase.

## 2. Scope, method, and limitations

### Observed facts

Read: all tracked filenames; all package modules at structural/API level and critical implementations in detail; every script through AST/import/I/O inspection with detailed reading of hubs and active flows; all test module imports/test names; root configuration; aircraft metadata/geometry; principal contracts, state, insights, plans/results indexes, and the current wing-wrench analysis. The inventory marks uncertain items `REVIEW` rather than asserting full understanding.

Commands were read-only except creation of the five allowed documents. No training, data regeneration, artifact export, checkout, cleanup, dependency operation, or formatting was performed. `pytest --collect-only` was used; the full 275-test suite was not executed because Phase 0A does not require runtime regeneration and many tests write temporary artifacts or train smoke models.

### Limitations and unknowns

- Untracked/ignored `dataset/` (14 GiB) and `artifacts/` (38 GiB) were not exhaustively content-audited.
- Existing untracked `outputs/` (42 MiB, 106 files) was sampled through its current manifest, not exhaustively validated.
- External IsaacLab, PX4, paper, and raw-log repositories were not audited.
- Runtime invocation frequency of historical scripts is unknown; static import absence does not prove obsolescence.
- No external consumers of `system_identification` were found inside this repository; consumers outside it remain unknown.

## 3. Git and environment baseline

### Observed facts

| Item | Value |
|---|---|
| Branch | `main` |
| HEAD | `c0ee4725597fe149b2ad4697d9a54841b5579e6d` |
| Initial status | `?? outputs/` |
| Tracked files | 223 |
| Python | `/home/zn/anaconda3/envs/flap-train-gpu/bin/python` |
| Version | Python 3.11.14 |
| Conda environment | `flap-train-gpu` |
| Pip packages | 271 |
| Editable install | `system-identification==0.1.0` at this checkout |
| Key versions | NumPy 1.26.0; pandas 2.3.3; SciPy 1.15.3; matplotlib 3.10.3; PyArrow 23.0.1; PyTorch 2.7.0+cu128; pytest 9.0.2 |

Other Conda environments observed: `base`, `env_isaaclab`, `flap-train`, and `paper_figures`. The complete `python -m pip list` output was captured during the audit; no package operation was performed. Recent commits, newest first: `c0ee472`, `a87dad0`, `609b183`, `d1d94f1`, `01b3f4b`, `bd99aff`, `1484c94`, `f04b6b1`, `51b1e25`, `60def00`.

The initial workspace already contained `.pytest_cache/` and seven `__pycache__/` trees. They are ignored and predate Phase 0A. `outputs/` was already untracked and is not ignored by `.gitignore`; `dataset/`, `artifacts/`, pytest cache, and Python bytecode are ignored.

## 4. Current tracked structure

```text
.
├── src/system_identification/        # 18 files; core package
│   ├── analysis/ baselines/ physics/ plotting/
│   ├── pipeline.py dataset_split.py training.py
│   └── metadata.py phase.py resample.py signal_preprocessing.py
├── scripts/                          # 68 flat Python scripts
├── tests/                            # 46 test modules + 1 JSON frozen fixture
├── metadata/aircraft/flapper_01/    # YAML + frozen wing geometry
├── docs/                             # contracts/drafts, plans, results, insights, analysis
├── dataset/                          # ignored, existing, 14 GiB
├── artifacts/                        # ignored, existing, 38 GiB
└── outputs/                          # untracked, existing, 42 MiB
```

`src/system_identification.egg-info/` exists locally and is ignored by `src/*.egg-info/`.

## 5. Current system data flow

### Observed facts

| Step | Implementation and entry | Input | Output/config | Dependencies and downstream users | Risks/duplication |
|---|---|---|---|---|---|
| ULog ingestion | `pipeline.extract_topic_frames_from_ulog`, `run_ulog_to_canonical`; `scripts/ulg_to_canonical_parquet.py` | PX4 `.ulg`, aircraft YAML | per-log samples/segments Parquet, source manifest/report; fixed 100 Hz | `pyulog`, metadata, resample; consumed by regeneration/split builders | `pipeline.py` mixes I/O, alignment, labels, and writes; only 100 Hz; absolute source paths |
| Metadata/field mapping | `metadata.py`, `pipeline.topic_dataframe`; YAML | PX4 topics and aircraft constants | canonical topic columns, freshness flags | phase, label, physics code | metadata is `draft_in_use`; actuator signs null; schema validation is minimal |
| Alignment/resampling/filtering | `resample.py`, `signal_preprocessing.py`, pipeline; smoothed-label scripts | topic frames or existing split Parquets | aligned canonical rows; filtered/aligned variants | labels, training, diagnostics | docs call for quaternion SLERP, code linearly resamples quaternion components; filtering rules also live in scripts |
| Phase/cycles | `phase.py`, pipeline | encoder/wing phase/frequency + metadata ratio/sign | mechanical/drive phase, stroke, cycle IDs/validity, cycle mean frequency | split, priors, models | fallback hierarchy is embedded; phase semantics repeated in correction scripts |
| Effective labels | `_compute_effective_wrench_labels`; smoothed-label builders | NED accelerations, body-to-NED wxyz quaternion, body rates/derivatives, mass/inertia | FRD force and CG moment labels plus validity | splits, priors, training, replay | raw and smoothed label generation paths coexist; no explicit label-contract version object |
| Split/folds | `dataset_split.py`; build/materialize/filter/k-fold scripts | canonical datasets/manifests | whole-log or cycle-block train/val/test, outer folds | all fitting/evaluation | multiple split policies; cycle-block mixing within a log is non-causal despite purge; assignment lineage is path-based |
| Physics prior | `physics/*`, `baselines/isaaclab_wing_only_baseline.py`; theta sweep and external exporter/sweep scripts | canonical rows, geometry, metadata, prior params | keyed or legacy prior predictions, component diagnostics | residual builders/calibration | external Isaac/Python paths hard-coded in sweep scripts; some downstream loaders still assume row order |
| Calibration/correction | force recalibration, structured/deployable/phase correction, nested Exp1–4 scripts | split + aligned priors | parameters, predictions, selection metrics/manifests | evaluation/replay, prior-vs-TCN | core algorithms are in scripts; duplicate metrics/features/loaders; physical and learned parameters can share experiment files |
| Neural/residual models | `training.py`; training/screen/prior-vs-TCN scripts | train/val/test Parquets | bundles, config, histories, metrics, plots | diagnostics and paper results | 5,470-line mixed module with 15 classes and 90 top-level functions; constants, models, transforms, fitting, evaluation, and plotting combined |
| Selection | fitting functions and experiment scripts | train and validation | best epoch/candidate | final evaluation | train-only statistics and val selection are implemented; ordinary runs also materialize test results/plots |
| Test evaluation/replay | training evaluator; residual, short-horizon, rotational, Level-2 scripts | locked model/prior and test rows | metrics, predictions, replay tables | results docs/paper | no central evaluation contract; window selection/test-use rules vary by workflow |
| Plots/tables/diagnostics | package plotting + many scripts | predictions/metrics/splits | figures, CSV, Markdown, manifests | docs/results | repeated metric and plotting code; output schemas vary; many defaults point to dated artifacts |

The assumed linear flow in the task is therefore an approximation. In code, label smoothing/alignment, prior generation, correction, and diagnostics form versioned branches, and some priors are produced by an external IsaacLab exporter rather than this package.

## 6. Package module responsibilities and dependency hotspots

### Observed facts

- `pipeline.py` owns ULog extraction, canonical grid construction, field mapping, phase selection, resampling, label computation, segmentation, and artifact writing. This is the largest data-boundary hotspot.
- `training.py` is 5,470 lines with 15 classes and 90 top-level functions. It owns feature contracts, 12+ neural architectures, window/sequence/rollout adapters, normalization, losses, fitting, early stopping, persistence, evaluation, diagnostics, plots, ablations, and baseline recipes. It is the dominant mixed-responsibility hotspot.
- `dataset_split.py` implements both cycle-block and whole-log splitting/materialization; its private functions are imported by `materialize_split_from_log_assignment.py`.
- `physics/` is relatively cohesive and pure NumPy; `baselines/isaaclab_wing_only_baseline.py` adds metadata/frame adaptation and component aggregation.
- `analysis/wing_wrench_theta_sweep.py` is a 970-line reusable analysis library already separated from its 83-line CLI, a useful target pattern.
- No static package or script import cycle was found. The risk is architectural: scripts depend on other scripts' private functions and constants, making layer direction unstable.

Top script hubs by direct script dependents: `build_delaurier_residual_split` (7), `run_nested_prior_shaping_ablation_exp1` (7), `train_fx_fz_correction` (6), residual phase/frequency analyzers (5 each), and `train_deployable_wrench_correction_v2` (5).

## 7. Scripts classification

### Observed facts

| Class | Examples | Current character | Recommended disposition |
|---|---|---|---|
| Thin CLI | `ulg_to_canonical_parquet`, `build_log_split`, `build_cycle_block_split`, `analyze_wing_wrench_theta_sweep` | argument parsing plus package call | keep as thin compatibility entry, later categorize |
| Build/materialize | residual/smoothed splits, keyed priors, accepted-log regeneration | substantial transformation and manifest logic | move reusable implementation to `data/` or `labels/`; retain CLI |
| Physics/calibration | force recalibration, prior regression/nonlinear calibration, parameter sweeps | scientific core and orchestration mixed | move solvers/contracts to package; archive dated experiment orchestration |
| Structured models | deployable, dynamic-arm, phase-structured, Fx/Fz corrections | model classes, features, selection, I/O in entry files | split into `models/structured`, training service, and CLI |
| Neural experiments | baseline, backbone screens, prior-vs-TCN | package calls plus grids and result collation | configs + experiment runners; stable training stays in package |
| Evaluation/replay | residual evaluation, short-horizon, rotational, Level 2 | reusable integrators and metrics | package `evaluation/`; CLI only writes a run |
| Diagnose/analyze/plot | residual phase/frequency/condition, label quality, lateral diagnostics | reusable metrics and plots | package evaluation/plotting, keep dated report configs |
| Audit | ratio-8 lineage/frequency | useful gates | package validation helpers + `scripts/audit/` CLI |

No script is recommended for deletion. `ARCHIVE` means immutable historical entry plus command mapping, not removal.

Inventory action enum: `KEEP` retains the current path; `MOVE` has one primary destination; `SPLIT` has one primary `proposed_target` and records every additional destination in `notes`; `WRAP` retains the old command as a compatibility wrapper around an already packaged implementation; `REVIEW` defers the decision pending named evidence; `ARCHIVE` preserves a historical entry plus its command/artifact mapping. `proposed_target` is always one literal path, never brace shorthand or a `+`-joined list.

## 8. Tests classification and coverage

### Observed facts

`pytest -p no:cacheprovider --collect-only` collected 275 tests successfully from 46 test modules; the tracked test tree also contains one JSON frozen fixture. The suite covers canonical assembly, resampling, phase ratio, split isolation, DeLaurier fixture parity/frame transforms, label preprocessing, prior alignment, structured model selection, neural shapes/training smokes, artifact writers, diagnostics, and replay integrators.

Provisional taxonomy:

- Unit: pure math, feature construction, metrics, transforms, solvers.
- Integration: temp Parquet/JSON writers, CLI subprocesses, training job smokes, materializers.
- Regression: frozen Isaac fixture, phase/frequency contract, validation-only selection, protected output roots, replay closure.
- External/contract: PX4 ratio test reads an external checkout and should be explicitly marked/skippable.

Critical gaps: end-to-end artifact schema compatibility; resolved-config completeness; overwrite behavior across all scripts; external data/version hashes; central split/evaluation contract; row-key alignment in every prior consumer; output-to-figure provenance; and a regression snapshot for the main canonical pipeline.

## 9. Documentation and contract status

### Observed facts

- The three March contract/spec files are explicitly drafts. They contain authoritative intent but are not fully current contracts.
- Confirmed matches include FRD/NED, effective wrench equations, 100 Hz, metadata-sourced mass/inertia/phase ratio, ZOH freshness, and canonical-versus-model-adapter separation.
- Confirmed drift: the dataset/spec docs prescribe quaternion SLERP and additional reset/dropout segmentation; the current pipeline linearly resamples quaternion elements and segments only on selected nav/arming/landed changes. Several documented hard/soft filters are not centralized in pipeline code.
- `PROJECT_STATE.md` is current but only two bullets; it is not an index of datasets, frozen baselines, risks, or contracts.
- `docs/plans/` and `docs/results/` are valuable chronological experiment records, but current/frozen/superseded status is mostly implicit.
- The current wing-wrench analysis is unusually strong in frame, reference, provenance, limitation, and command documentation and should inform future run-report templates.

Future placement: reconciled rules in `docs/contracts/`; architectural boundaries in `docs/architecture/`; method designs in `docs/design/`; read-only findings in `docs/audits/`; dated work in `docs/experiments/`; compact stable evidence in `docs/results/`; decisions in `docs/decisions/`; operational commands in `docs/runbooks/`.

## 10. Metadata, configuration, and outputs

### Observed facts

Aircraft metadata combines identity, PX4 provenance/path, frames, logging coverage, mass/CG/inertia, drive/phase, actuator semantics, sensors, label definition, and open items. Geometry is a separate CSV in the same aircraft directory. This boundary is preferable to moving aircraft constants into experiment configs, but schema/version validation is weak and the YAML remains `draft_in_use`.

Configuration currently comes from package constants, script constants, CLI arguments, JSON overrides, aircraft YAML, resource snapshots, and absolute/default artifact paths. Hydra is installed indirectly but not used by the repository and is not required. Resolved training configuration exists, but a uniform resolved configuration, environment snapshot, data/split hashes, Git status, and output manifest are not guaranteed across workflows.

Outputs are split across ignored `dataset/`, ignored `artifacts/`, and unignored `outputs/`. The sampled wing-sweep manifest records many high-value fields and hashes, but its recorded Git commit (`609b183`) differs from the current HEAD and the workspace was dirty, demonstrating why artifact provenance must be treated as artifact-specific, not inferred from current checkout state.

## 11. Risk register

| Severity | Risk | Evidence / affected files | Consequence | Migration implication |
|---|---|---|---|---|
| CRITICAL | None confirmed | No observed fact establishes current test leakage into fitted parameters across the main locked protocols | Do not inflate uncertain concerns | Preserve evidence-based severity |
| HIGH | Scripts are an implicit library layer | 38/46 test modules directly import scripts; 18 script modules are imported by other scripts; hubs listed above | fragile private APIs and flat namespace block safe moves | extract one hub at a time with wrappers and characterization tests |
| HIGH | Prior/label row alignment is inconsistent | keyed alignment exists in `build_delaurier_residual_split.py`, but phase-structured `_load_split` pairs equal-length rows directly; legacy fallback exists | silent residual target corruption if ordering differs | centralize a required sample-key join before calibration migration |
| HIGH | Data/split/evaluation contracts are not executable and current | draft docs; multiple split policies; no central evaluation contract | protocol drift and incomparable results | Phase 0B drafts contracts/indexes; later stages add validators |
| HIGH | `training.py` mixes too many boundaries | 5,470 lines, 15 classes, and 90 top-level functions spanning features, models, fitting, selection, evaluation, plots, artifacts | behavior changes are difficult to isolate/freeze | split only after characterization; preserve public re-exports |
| HIGH | Test-set procedural isolation is incomplete | ordinary `run_training_job` loads/evaluates test and writes test figures unless `skip_test_eval`; standard comparisons report test | human iteration can use test feedback even if code selection uses val | require selection runs to omit test; separate one-shot final evaluation command |
| HIGH | Output overwrite/provenance policy is uneven | `outputs/` unignored; some scripts guard nonempty roots, many `mkdir(...exist_ok=True)` and fixed filenames | overwrite or ambiguous lineage | central run allocator, immutable run IDs, manifest and Git/data/split hashes |
| MEDIUM | Documentation/code drift in preprocessing | quaternion linear resampling vs documented SLERP; missing documented reset/dropout segmentation | label/alignment assumptions may be overstated | reconcile contract from observed behavior before any behavior change |
| MEDIUM | Multiple split policies can be misreported | cycle-block random assignment within logs and whole-log split coexist | same-log temporal context mixing may be called leakage-resistant incorrectly | make split policy/version mandatory in config and report |
| MEDIUM | Sequence subsampling can distort time adjacency | `_load_split_frame.sample` precedes sequence grouping/sorting; sequence builder does not establish a gap contract | smoke results with max samples may not represent causal contiguous windows | characterize and define sampling/window contract before refactor |
| MEDIUM | Physical conventions are distributed | metadata, pipeline, physics adapter, scripts, docs | sign/reference regressions during moves | create coordinate/physics-prior contracts and regression fixtures |
| MEDIUM | Hard-coded external paths | Isaac/Python/paper defaults in sweep and nested scripts; PX4 path in YAML | nonportable runs and hidden external-version dependency | path config plus explicit external dependency manifest |
| MEDIUM | Metadata boundary is broad and partly unresolved | YAML `draft_in_use`; control surface signs null | ambiguous control semantics | retain aircraft path; validate status/open items; add subgroups only as assets grow |
| LOW | Root image provenance unknown | `inertia.png` has no confirmed tracked consumer | accidental loss or misleading asset | `REVIEW`, never delete without provenance check |
| INFORMATIONAL | No static circular imports found | AST graph of package/scripts | current issue is layering, not a demonstrated cycle | prevent future reverse dependencies through AGENTS and tests |

## 12. Confirmed facts, inferences, and unknowns

### Confirmed facts

Train-derived feature/target statistics are used for validation/test transforms. Early stopping uses validation loss. Whole-log split keeps log IDs disjoint. Cycle-block split uses random blocks plus train purge around validation/test. Existing tests explicitly protect several structured selections from test data. The frozen DeLaurier implementation has detailed frame/reference regression coverage.

### Inferences

The repository evolved through rapid experiment-driven additions, which explains the strong chronological evidence and weak library/CLI boundary. Static no-cycle results reduce immediate circular-import risk, but private cross-script dependencies make future cycles likely unless dependency direction is enforced.

### Unknowns

External consumers, authoritative current dataset/split IDs, which historical entries must remain executable indefinitely, whether `outputs/` should replace or coexist with `artifacts/`, and who approves contract promotion require user/project decisions.

## 13. Ranked structural findings and next step

1. Freeze timestamp, phase, sample identity, alignment keys, split semantics, selection semantics, and artifact schemas.
2. Freeze conventions before migrating data; migrate label builders/alignment under `labels/`, not ordinary `data/`.
3. Extract script hubs into package modules with compatibility wrappers.
4. Split `pipeline.py` and `training.py` by boundary, not by arbitrary file size.
5. Decompose training behavior-preservingly in Phase 0C7A, then isolate fit/validation selection/final test evaluation as the explicit protocol change in Phase 0C7B.
6. Establish immutable run directories and uniform manifests.
7. Reconcile draft contracts with code facts before changing behavior.
8. Categorize individual tests only after runtime/cost classification; split mixed test modules rather than moving them mechanically.
9. Preserve the existing package name through this migration.
10. Archive historical experiments only with a command/artifact mapping.

The reconciled migration map contains 13 independently gated phases: 0B, 0C1-0C6, 0C7A, 0C7B, 0C8-0C10, and 0D. Recommendation only: proceed to Phase 0B after the independent-audit conditions are corrected and checked. No Phase 0B action was performed.
