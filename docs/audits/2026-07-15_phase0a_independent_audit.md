# Phase 0A Independent Audit

Date: 2026-07-15

Reviewed repository state: `main` / `c0ee4725597fe149b2ad4697d9a54841b5579e6d`

## 1. Verdict

**PASS WITH CONDITIONS**

The Phase 0A work is substantially grounded in the repository: the tracked-file inventory is complete, the dominant structural risks are real, the conservative package decision is supported, and the proposed domain boundaries are broadly suitable. Phase 0A does not need to be repeated.

Entry to Phase 0B is nevertheless conditional. Before Phase 0B creates contracts or AGENTS files, the five deliverables must be reconciled in four areas: the scripts/tests counting language, label-versus-data ownership in the inventory, the effective scope of aspirational AGENTS rules, and migration phases that currently combine behavior preservation with protocol-changing acceptance criteria. The exact corrections and acceptance checks are in Sections 9-11.

## 2. Scope and independence statement

This is an independent review. I did not rely on the Phase 0A report as proof of its own claims. I directly read all five Phase 0A deliverables, parsed every CSV record, inspected the relevant package modules, scripts, tests, metadata, configuration, contracts, and ignore rules, and reproduced static counts from the current checkout.

I did not modify any Phase 0A deliverable, tracked file, production code, test, configuration, metadata, root `AGENTS.md`, dataset, artifact, or pre-existing output. I did not enter Phase 0B, create a target skeleton, create an AGENTS file, migrate code, regenerate data, train a model, evaluate a model, make plots, clean the worktree, or create a Git commit. The only repository write made by this review is this report.

The existing `dataset/`, `artifacts/`, ignored caches, and untracked `outputs/` predate this review and are not attributed to Phase 0A or to this task.

## 3. Environment and Git baseline

| Item | Independently observed value |
|---|---|
| Branch | `main` |
| HEAD | `c0ee4725597fe149b2ad4697d9a54841b5579e6d` |
| Python | `/home/zn/anaconda3/envs/flap-train-gpu/bin/python` |
| Python version | `3.11.14` |
| Conda environment | `flap-train-gpu` |
| Initial short status | `?? docs/architecture/`, `?? docs/audits/`, `?? docs/plans/2026-07-15_repository_migration_plan.md`, `?? outputs/` |
| Tracked files | 223 |
| Phase 0A untracked deliverables | exactly the five named files |
| Pre-existing untracked output files | 106 under `outputs/`, 42 MiB |

Baseline commands included `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, `git ls-files`, and `find . -maxdepth 3 -type f | sort`. The latter also exposed pre-existing ignored `.pytest_cache`, `__pycache__`, package egg-info, `dataset/`, and `artifacts/`; their presence was recorded, not changed.

Before review, SHA-256 hashes of the five Phase 0A files were:

| File | SHA-256 |
|---|---|
| `docs/audits/2026-07-15_repository_readonly_audit.md` | `37abf60b03587110d698db8274567134da7b3c39bfead970c83255a81371e03b` |
| `docs/audits/2026-07-15_repository_file_inventory.csv` | `ddb4b72864a105f081e78e644a351928786f9dfa6d4221cfafa30ec926b8be0d` |
| `docs/architecture/2026-07-15_proposed_repository_architecture.md` | `123d91ff2d091c5c8881033f0b7f48bcfae37324c58fe0b170f9c5e65405a074` |
| `docs/architecture/2026-07-15_agents_hierarchy_design.md` | `f736009e73e2ed8e8fbecea2f2b74b43cd78cb3c5e5220babe7dd6f4a93baaa2` |
| `docs/plans/2026-07-15_repository_migration_plan.md` | `11b77b10071fc7a1ec2aaef75430396f01f9673df005aa6b28673e63df7afd10` |

With the required environment variables set, `python -m pytest -p no:cacheprovider --collect-only` completed successfully and collected 275 tests under pytest 9.0.2. No test body or full suite was run.

## 4. Evidence-based verification

### 4.1 Scripts as an implicit library layer — PARTIALLY CONFIRMED

The structural conclusion is correct; two counting statements need tighter wording.

- `git ls-files 'scripts/*'` returns 68 entries. All 68 are top-level Python files. There is no `scripts/__init__.py`, nested tracked script directory, or tracked non-Python file. Therefore “68 flat scripts” is accurate under the tracked top-level `.py` definition.
- There are 47 tracked entries under `tests/`, but only 46 test modules (`tests/test_*.py`). The remaining entry is `tests/fixtures/isaaclab_delaurier_wing_wrench_3b5d4ec.json`. The audit's “47 test modules” wording at `docs/audits/2026-07-15_repository_readonly_audit.md:13` is incorrect.
- AST inspection finds 38 of the 46 test modules with a direct static `import scripts...` or `from scripts...` statement. There were no dynamic-import candidates in the test modules. Thus the reproducible statistic is **38/46 test modules**, not 38/47 modules. The 38 matching files are all listed by the AST command; representative examples are `tests/test_training.py`, `tests/test_pipeline.py`, `tests/test_build_delaurier_residual_split.py`, and `tests/test_phase_structured_wrench_correction.py`.
- Ten test modules directly import `system_identification.*`; overlap is possible and exists, so 38 plus 10 is not intended to partition the test suite.
- Direct script-to-script AST imports comprise 59 unique edges into 18 imported script modules, not 19. There are 27 importing script modules. The top dependent hubs reproduce the report's ordering: `build_delaurier_residual_split` and `run_nested_prior_shaping_ablation_exp1` each have seven direct dependents, followed by `train_fx_fz_correction` with six. No static cycle was found.

Reproducible method: enumerate tracked paths with `git ls-files`, parse each Python file with `ast`, and count only `Import`/`ImportFrom` nodes whose resolved module begins with `scripts`. This explicitly excludes indirect imports and dynamic imports.

Conclusion: the implicit-library diagnosis and HIGH priority are supported, while the test-module denominator and 19-module subclaim require correction.

### 4.2 `training.py` responsibilities and scale — CONFIRMED

`src/system_identification/training.py` has exactly 5,470 physical lines (`wc -l` and final numbered line both agree), not 5,471. “Approximately 5,471” is harmless, but an exact inventory should say 5,470.

AST inspection finds 15 top-level classes, 90 top-level functions, and 128 function/method definitions in total (37 class methods plus one nested function beyond the simple top-level-plus-method count). The file contains all responsibilities claimed by Phase 0A:

- feature contracts and derived features: constants and `resolve_feature_set_columns`, `_with_derived_columns`, `prepare_feature_target_frames` (`training.py:465`, `1534`, `1631`);
- window, causal sequence, and rollout construction: `prepare_windowed_feature_target_frames`, `prepare_causal_sequence_feature_target_frames`, and `prepare_causal_rollout_feature_target_frames` (`1789`, `1849`, `1914`);
- 15 model classes covering MLP, GRU/LSTM, ASL, TCN, Transformer/FiLM, TCN-GRU, SUBNET, and PFNN (`514-1516`);
- train-only imputation/normalization and target scaling (`2003-2039`);
- loss, fitting, early stopping, selected best state, and schedulers (`2200-3968`);
- bundle serialization/reconstruction and prediction (`2048`, `3970-4273`);
- metrics, per-log/regime diagnostics, and evaluation (`2075`, `4275-4584`);
- plotting (`2409-2534`, `4981-5004`);
- job orchestration, ablation, and baseline comparison (`4595-5470`).

The architecture's proposed boundaries can carry these concerns, provided feature/window code remains under `training/` as the target tree says, model definitions move to `models/neural/`, metrics/plots move to `evaluation/`, and bundle/manifest ownership is explicitly divided between `training/bundles.py` and `artifacts/`. The inventory currently contradicts this by targeting `src/system_identification/{features,models,training,evaluation}/` (`inventory.csv:177`), putting `features` at package root rather than `training/features.py` as in the architecture.

A safe split order is: characterize public names and deterministic tiny bundles; extract pure feature/window transforms; extract model definitions; extract normalization/losses; extract fit/selection; extract bundle serialization; finally extract evaluation/plotting. Moving orchestration or changing test-loading behavior before frozen bundle and selection fixtures would be the wrong order.

### 4.3 Prior/label alignment — CONFIRMED

Alignment policy is materially inconsistent and the silent-misalignment risk is real.

Stable or explicit key alignment exists in:

- `scripts/build_delaurier_residual_split.py:24-103`: one-to-one key merge using `log_id`, optional `segment_id`, and rounded 100 Hz `time_s`, with uniqueness and missing-row checks;
- `scripts/train_fx_fz_correction.py:107`, `scripts/run_delaurier_force_recalibration.py:239-243`, `scripts/diagnose_delaurier_prior_conventions.py:48`, `scripts/diagnose_tail_model_parameters.py:38`, and `scripts/sweep_tail_geometry_alignment.py:152`, which call that helper with row-order fallback disabled;
- `scripts/run_nested_prior_shaping_ablation_exp1.py:55,120-135`, which performs a similar one-to-one rounded-time merge;
- `scripts/run_prior_vs_tcn_comparison.py:447-487`, which builds and validates `log_id`/`segment_id`/rounded-time keys;
- `scripts/build_level2_replay_ready_table.py:14,98-126,153-184`, which uses exact `outer_fold`/`log_id`/`segment_id`/`time_s` joins;
- `scripts/materialize_keyed_prior_for_split.py:44-80`, which transfers source sample identity to prior rows and then aligns to a target split.

Equal-length or positional pairing remains in:

- `scripts/train_phase_structured_wrench_correction.py:544-560`: checks only equal length, then assigns `prior[target].to_numpy()` by position;
- `scripts/train_deployable_wrench_correction_v2.py:851-878` and `scripts/train_delaurier_greybox_force_correction.py:315-316`: row-count gates precede positional use;
- `scripts/analyze_component_residual_attribution.py:266-274,1289-1330`: validates counts, resets indexes, and pairs multiple frames positionally;
- `scripts/diagnose_wing_tail_my_alignment.py:371` and other prediction consumers with equal-row assertions but no identity join.

The residual builder also retains an explicit legacy row-order fallback (`build_delaurier_residual_split.py:50-70`); the CLI default is fail-closed, but the lower-level `build_residual_frame` default remains permissive (`:106-119`). This is safer than a silent default at the main CLI boundary, but still easy for library callers to misuse.

Timestamp alignment is not uniform: some consumers round `time_s` to a 100 Hz integer key, Level 2 uses exact floating `time_s`, and some paths rely on already materialized order. Cycle alignment is used to create cycle blocks (`dataset_split.py:79-129,177-206`) and phase/frequency products, but it is not a universal prior/label join key. Resampling creates a new uniform index; subsequent concatenations commonly reset indexes, so index equality alone is not sample identity.

A shuffled equal-length prior can therefore silently corrupt residual targets in the positional consumers. HIGH severity is reasonable. The migration plan correctly puts a keyed alignment service before model migration in principle, but the inventory incorrectly assigns the residual and smoothed-label builders to 0C2 `data/` instead of 0C4 `labels/` (`inventory.csv:105,110-111`). This must be reconciled before Phase 0B freezes contracts and target ownership.

### 4.4 Ordinary training entry reads test data — PARTIALLY CONFIRMED

The Phase 0A distinction between numerical leakage and procedural leakage is accurate, but “ordinary entry” needs entry-specific detail.

`run_training_job` loads train and validation frames, and by default also loads test before calling a fit function (`training.py:4655-4662`). Fitting functions receive only train and validation frames; train-derived normalization is applied to validation, and early stopping selects by validation loss. After fitting, the same job automatically evaluates train/validation/test and writes test metrics plus two test plots (`training.py:4765-4788`). No inspected control-flow path passes test arrays into weight fitting, normalization, epoch selection, or candidate selection.

`scripts/train_baseline_torch.py:30-133` exposes no `--skip-test-eval` option and calls `run_training_job` without setting the flag (`:136-175`), so this normal CLI always requires and reports test data. `scripts/run_baseline_comparison.py:104-155` has an opt-out `--skip-test-eval`, but its default still evaluates test. Several later temporal-screen stages opt out by default (`scripts/run_temporal_backbone_screen.py:755-765`), showing improved but non-uniform practice.

Structured correction entries such as `train_phase_structured_wrench_correction.py:784-803` load all three partitions before fitting and write test predictions/summary, while their actual selectors query validation rows (`:249`, `:360`, `:798-799`). Tests explicitly perturb or inspect test metrics to confirm validation-only selection. That supports “no confirmed code leakage into selected numerical parameters” for the checked paths.

Human iteration remains exposed: ordinary jobs write test scores and plots, baseline comparison plots can include test performance, and downstream diagnostic scripts read test outputs. Repository history cannot prove whether a person used those results to choose features or hyperparameters. This is a procedural/model-development leakage risk, not evidence that the fit code directly uses test values.

The target should separate `fit`, `select`, and `final-evaluate`. However, changing the default command so selection does not load test is a protocol behavior change. It must not be hidden inside a phase described as mechanical, behavior-preserving extraction.

### 4.5 Coexisting split protocols — PARTIALLY CONFIRMED

Whole-log and cycle-block splits are both implemented in `src/system_identification/dataset_split.py`:

- whole-log creation uses `assign_log_splits` and `materialize_log_split` (`:209-238`, `:514-617`), records `split_policy: whole_log`, and keeps a log in one partition;
- cycle-block creation extracts contiguous cycle blocks, randomly assigns blocks, and purges train cycles around validation/test blocks (`:79-129`, `:177-206`, `:241-293`, `:366-500`). It can place different blocks from the same log in different partitions. The current cycle-block manifest does **not** record a `split_policy` field, although the audit recommends mandatory policy reporting;
- `scripts/materialize_split_from_log_assignment.py:39-129` reuses an external whole-log assignment and records `whole_log_reused_assignment`;
- `scripts/build_delaurier_residual_kfold_splits.py` creates log-level outer folds used by nested calibration/prior-vs-TCN workflows. These are additional evaluation folds layered over split artifacts, not a third row-sampling primitive in `dataset_split.py`.

The Phase 0A statement that cycle-block splitting is not whole-log causal isolation is accurate. Purging protects nearby training cycles but does not remove shared-log context or all correlated operating conditions.

No inspected split builder fits normalization across partitions. Neural normalization and target scaling are fitted on train only (`training.py:2003-2039` and calls in each fit function). The time-aligned label builder selects lag on train only (`scripts/build_time_aligned_smoothed_label_split.py:76-109`) and applies the selected lag to all splits; configured low-pass filters are applied group-locally rather than fitted across splits. Admission/validity masks are deterministic per log, though their semantic contract is not centralized.

Phase 0A correctly identifies the need for a split contract, but it underreports the missing cycle-block `split_policy` manifest field and the reused-assignment/outer-fold variants. The split contract must be fixed before training migration, and its semantic draft should precede any 0C data move.

### 4.6 Quaternion SLERP documentation/implementation drift — PARTIALLY CONFIRMED

The preprocessing spec requires quaternion SLERP followed by normalization (`docs/2026-03-23-ulg-to-canonical-parquet-preprocessing-spec.md:259-297`, especially `:267,290-297`; the dataset contract repeats it at `docs/2026-03-23-flapping-dataset-contract-draft.md:431-452`).

The implementation places all four `vehicle_attitude.q[i]` fields in `linear_topics` and independently calls scalar `linear_resample` (`pipeline.py:690-707`; `resample.py:20-31`). Therefore the interpolation itself is componentwise linear, not SLERP.

The audit omits one important nuance: the interpolated quaternion is normalized later when effective-wrench rotation matrices are constructed (`pipeline.py:191-218`, called at `:422`). Thus it is not generally consumed unnormalized by label reconstruction. Normalizing after linear interpolation still does not make it SLERP. More seriously, because `q` and `-q` represent the same attitude, componentwise interpolation across an uncorrected sign change approaches or reaches the zero quaternion. The exact midpoint becomes invalid; nearby normalized samples can take a long or unstable path. This can invalidate labels or produce physically wrong intermediate attitudes.

The issue is both a model/data-behavior risk and a refactor regression risk. The present behavior must be characterized so a structural move does not change it accidentally, then SLERP/sign-continuity correction should be an explicitly approved behavior-fix subphase with before/after label deltas. The migration plan correctly excludes interpolation changes from 0C2/0C3 (`migration_plan.md:109-134`), so Phase 0A did not improperly authorize the fix during structural migration. Its MEDIUM classification is defensible for migration planning, but the data-quality impact should be separately assessed before calling it merely documentation drift.

### 4.7 Output directory dispersion — CONFIRMED

Current local roles are mixed but distinguishable:

- `dataset/` (14 GiB, 1,035 files) contains canonical datasets, materialized splits, and derived label/split tables. It is ignored and is not purely immutable raw input.
- `artifacts/` (38 GiB, 18,357 files) contains model bundles, parameters, predictions, metrics, figures, diagnostics, and dated experiment outputs. It is ignored.
- `outputs/` (42 MiB, 106 files) currently contains the wing-wrench theta-sweep run with predictions, tables, manifests, figures, logs, and report-like material. It is untracked and not ignored.
- Compact reviewed plans, results, analyses, a manifest CSV, metadata, and one root image are tracked under `docs/`, `metadata/`, and the repository root.

`.gitignore:1-10` ignores caches, `dataset/`, `artifacts/`, egg-info, and Codex state; it does not ignore `outputs/`. `git ls-files dataset artifacts outputs` returns no tracked files.

The architecture's immutable `outputs/runs/<run_id>` plus compact reviewed `docs/results` concept is supported, but wholesale merging of `dataset/`, `artifacts/`, and `outputs/` is not. They carry different lifecycle meanings and historical paths are embedded in configs, manifests, and reports. The safer decision is to define roles and manifests first: source/canonical/derived datasets, immutable run/model assets, and reviewed Git evidence. Preserve legacy roots and content hashes; introduce a new run allocator without relocating old history. The architecture itself leaves outputs-versus-artifacts convergence unresolved (`architecture.md:190-196,211-213`), which is appropriately cautious.

### 4.8 DeLaurier regression protection — PARTIALLY CONFIRMED

The cohesive physics implementation is in `src/system_identification/physics/{delaurier_airflow,delaurier_dynamic_twist,delaurier_strip_wrench}.py`; the real-aircraft/Isaac adapter is `src/system_identification/baselines/isaaclab_wing_only_baseline.py`.

`tests/test_delaurier_wing_wrench.py` contains ten tests. It includes:

- a float64 numerical comparison against a frozen IsaacLab fixture at tolerance `1e-10` (`:37-156`);
- zero-twist parity and analytic twist derivative checks (`:159-203`);
- FRD/FLU and vertical-sign checks (`:205-223`);
- real-attitude ground-minus-wind reconstruction, including quaternion sign invariance (`:226-251`);
- an attitude-airflow end-to-end wrench difference and diagnostic check (`:254-297`);
- polar/axial right-wing reflection symmetry (`:300-322`);
- hand-calculated moment-reference translation (`:325-333`);
- canonical-to-DeLaurier phase mapping and metadata CG/stroke checks (`:336-346`).

The fixture records source repository, branch, commit `3b5d4ec1d28f1384cf042402992ad7ea59995f49`, generation method, and float64 dtype. It freezes strip components, twist, Wang-frame wrench, and body/CG left-right totals. This is unusually good provenance.

It is not a complete behavior freeze: the frozen fixture has only one sample/phase point; the end-to-end fixture does not sweep phase, wind, attitude, reverse/near-zero airspeed, density, or geometry; units are documented and numerically implicit but do not have an independent dimensional/scale test; and the attitude-airflow test asserts only selected diagnostics plus a changed `fz`, not a frozen full wrench. Before 0C5 moves the implementation, add a compact multi-case regression table spanning phase quadrants, attitude/wind, and at least two airspeeds, while retaining analytic frame/reference tests. Existing tests are sufficient to start characterization, but not sufficient to declare the entire migrated physics baseline frozen without that addition.

## 5. Inventory validation

The CSV was parsed in full; no sampling or `head`-only inspection was used.

### 5.1 Coverage and counts

| Measure | Result |
|---|---:|
| Git tracked files | 223 |
| CSV data rows | 223 |
| Unique `current_path` values | 223 |
| Missing tracked paths | 0 |
| Extra paths not in tracked baseline | 0 |
| Duplicate `current_path` values | 0 |
| Nonexistent `current_path` values | 0 |
| Phase 0A deliverables incorrectly included as original inventory | 0 |
| Duplicate complete target strings | 0 |
| Nonexistent referenced test paths | 0 |

Root hidden tracked files are covered, including five resource snapshots and `.gitignore`. The Phase 0A files are correctly excluded from the original tracked inventory.

### 5.2 `proposed_action` and phase statistics

| Action | Count |
|---|---:|
| `KEEP` | 44 |
| `MOVE` | 105 |
| `SPLIT` | 51 |
| `WRAP` | 8 |
| `REVIEW` | 9 |
| `ARCHIVE` | 6 |

| Migration phase | Count |
|---|---:|
| `0B` | 10 |
| `0C1` | 0 |
| `0C2` | 17 |
| `0C3` | 1 |
| `0C4` | 0 |
| `0C5` | 7 |
| `0C6` | 9 |
| `0C7` | 1 |
| `0C8` | 33 |
| `0C9` | 19 |
| `0C10` | 47 |
| `0D` | 79 |

All phase labels occur in the migration plan, but the zero inventory entries for 0C1 and 0C4 expose a mapping gap: those phases create new infrastructure or absorb responsibilities assigned elsewhere, and the inventory does not say which source responsibilities feed them.

The inventory uses six action values, including `WRAP`, but none of the companion documents defines the authoritative action enum or semantics for each value. `WRAP` is a reasonable action, not inherently invalid, but it is currently undocumented. Add an enum definition and distinguish `MOVE` from “new implementation plus retained wrapper.”

### 5.3 Target and evidence issues

There are no literal whole-cell target collisions, but 51 `proposed_target` values are not machine-actionable paths: 49 contain two paths joined by ` + `, and two use brace shorthand (`pipeline.py` and `training.py`). A migration mapping needs one row per resulting target or a structured multi-target field with a declared primary implementation, compatibility wrapper, owner phase, and acceptance test.

Material inconsistencies include:

1. Label builders and alignment are assigned to 0C2 `data/` (`inventory.csv:104-111`), while the architecture assigns effective-wrench variants/alignment to `labels/` and the migration plan assigns them to 0C4 (`architecture.md:82-95,157-163`; `migration_plan.md:136-148`).
2. `pipeline.py` targets `data/{ulog,canonical,labels}.py` in 0C2 (`inventory.csv:172`), incorrectly placing labels inside data and prematurely extracting label behavior that the migration plan says is not changed/moved until 0C4.
3. Architecture filenames are `data/resampling.py` and `data/preprocessing.py`; inventory targets `data/resample.py` and `data/signal_preprocessing.py` (`inventory.csv:175-176`). Either choice can work, but one canonical mapping is required.
4. Architecture's DeLaurier targets are `physics/delaurier/{airflow,dynamic_twist,strip_wrench}.py` and `physics/baselines/wing_only.py`; inventory retains the old verbose filenames inside the new directories (`inventory.csv:164,169-171`).
5. Audit CLIs target `scripts/analysis/` in inventory (`inventory.csv:101-102`), while the architecture explicitly has `scripts/audit/`.
6. `training.py` targets package-root `{features,models,training,evaluation}` rather than the architecture's `training/features.py`, `training/windows.py`, `models/neural/*`, and `evaluation/*` (`inventory.csv:177`).
7. `tests/test_training.py` is marked one `MOVE` to integration although it contains many pure unit tests, model shape tests, training smokes, CLI/config tests, and protocol/regression checks. `tests/test_delaurier_wing_wrench.py` similarly mixes pure unit and frozen regression tests. These should be `SPLIT` or retain a mixed module until 0C10 classifies individual tests. The current notes honestly say “provisional,” so this is a mapping correction, not evidence of concealment.

`test_coverage` is nonempty for 86 rows and blank for 137. All explicit `tests/...` paths exist. The field is useful evidence for directly imported modules but is not a coverage measurement: `indirect/partial` is undefined, blank differs ambiguously from `none identified`, and an importing test does not prove every responsibility is asserted. Define evidence categories before using this column as a migration gate.

`stability` uses only `stable` (155) and `mixed/experiment-specific` (68); `risk` uses HIGH (66), MEDIUM (76), and LOW (81). Vocabulary is consistent, but criteria are not defined. Stability currently describes file character, not API maturity, while risk mixes migration coupling, scientific semantics, and provenance. Add short definitions and allow an evidence note for exceptions.

The inventory is appropriately conservative about `inertia.png` and draft documents with `REVIEW`. It does not prematurely mark scripts `ARCHIVE`; only six historical resource/document items use that action. The 0D placement of many dated docs/scripts still needs a command/status map, as the migration plan states.

## 6. Architecture review

### Conclusion: SUPPORTED WITH CONDITIONS

The architecture is based on this repository rather than a generic or IsaacLab-style template. It accounts for the existing src package, flat scripts, draft contracts, aircraft metadata, multiple model families, dated experiment history, and current test surface. It explicitly rejects copying IsaacLab application/extension structure (`architecture.md:202-209`). The scale is reasonable if directories are created only with their first real artifact, as required at `:116`.

The data/conventions/labels/physics/models/training/evaluation/artifacts/scripts/configs/metadata/docs/tests boundaries are useful and generally clear. In particular, labels reconstruct measured effective whole-aircraft wrench, physics produces priors, and residual targets are explicit aligned join products (`architecture.md:157-163`), which prevents a labels↔physics cycle. Training consumes adapters/models; evaluation consumes frozen predictions and must not fit/select (`:142-153`).

Gain-bias, bounded calibration, phase discrepancy, dynamic-arm models, neural baselines, and residual TCN all have plausible homes. Historical commands remain wrappers; no one-shot package rename or new framework is required. The lightweight typed YAML/JSON strategy is adequate for current composition depth (`:169-184`).

Conditions:

- Make sample-key ownership singular. The tree places `sample_keys.py` in conventions, while 0C1 describes sample-key utilities under artifact infrastructure and 0C3 centralizes them again. Artifact hashing may consume a sample-key contract, but it should not define sample identity.
- Reconcile exact target filenames with inventory before creating directories.
- Clarify the boundary between `training/bundles.py` and `artifacts/schemas.py`, and between evaluation report data and tracked `docs/results`.
- Keep `outputs`, `artifacts`, and datasets as distinct roles until a storage/retention ADR exists; do not migrate historical roots merely to match the target tree.
- Prevent `training/` from becoming a new monolith by enforcing dependency direction among features/windows, models, fit/selection, bundles, and evaluation, plus per-subgate size/API review.
- Do not encode residual TCN or new phase discrepancy research as Phase 0B/0C implementation work. The tree can reserve a destination, but research remains out of scope.

### Package-name decision: SUPPORTED

Keeping `system_identification` is the lowest-risk choice. `pyproject.toml:5-18` declares distribution `system-identification` and the src package already works in editable mode. Static search finds 35 test/script files and 50 source occurrences referring to `system_identification`; the package is also referenced by current wrappers and artifacts/configuration. No in-repository collision or required public namespace was found. External consumers remain unknown, which argues for preserving rather than renaming. A later rename should require a separate ADR and forwarding facade exactly as the architecture proposes.

## 7. AGENTS hierarchy review

The proposed 12-file hierarchy is not excessive for the final target, but it is only safe if child files are created alongside real scoped directories and if aspirational target invariants are clearly marked by activation phase. As written, universal rules such as “selection code cannot load test,” “every run has a manifest,” and “no script-to-script scientific imports” conflict with current behavior until 0C7-0C9. Making them immediately mandatory in Phase 0B would block normal maintenance or invite routine exceptions.

| Proposed/recommended path | Decision | Independent finding |
|---|---|---|
| `AGENTS.md` | **REVISE** | Keep skill policy; add universal safety/stage rules, but distinguish current enforced rules from target invariants with activation phase and legacy exception process. |
| `src/system_identification/AGENTS.md` | **REVISE** | Dependency/API/behavior-preservation rules belong here. Move detailed model-specific rules to a model child rather than growing the package parent indefinitely. |
| `src/system_identification/data/AGENTS.md` | **KEEP** | Data identity, field-specific interpolation, group-local operations, and immutable split assignments are distinct local boundaries. |
| `src/system_identification/physics/AGENTS.md` | **KEEP** | Frame, parity, unit, reference point, frozen baseline, and regression requirements are sufficiently specialized. |
| `src/system_identification/labels/AGENTS.md` | **KEEP** | Effective-label semantics and fail-closed prior alignment require a boundary independent of physics and data. |
| `src/system_identification/training/AGENTS.md` | **KEEP** | Train/validation/test roles, causal windows, selection, seeds, and bundle requirements are specialized and testable. |
| `src/system_identification/evaluation/AGENTS.md` | **KEEP** | Locked evaluation, replay modes, metric aggregation, and plot provenance differ materially from training. Plotting does not need another child initially. |
| `scripts/AGENTS.md` | **REVISE** | Thin-CLI is the correct target; add a staged legacy clause so existing shared script logic is permitted only until its mapped migration phase and may not expand. |
| `configs/AGENTS.md` | **KEEP** | Configs contain experiment choices; this is distinct from aircraft facts in metadata. |
| `tests/AGENTS.md` | **REVISE** | Keep taxonomy/pollution rules, but classify individual tests rather than assuming one module has one class; specify how mixed modules and node IDs are handled. |
| `docs/AGENTS.md` | **REVISE** | Fact/inference/decision/result distinctions are valuable. Replace an absolute “every high-risk claim has file/line” rule with reproducible evidence requirements appropriate to code, artifact, or external source. |
| `metadata/AGENTS.md` | **KEEP** | Aircraft facts, provenance, units, uncertainty, and change-impact rules are distinct from configs. |
| `src/system_identification/models/AGENTS.md` | **ADD** | Model files do not inherit sibling `training/AGENTS.md`. Add model-local inference/deployability, structured-vs-neural, physical-vs-learned parameter, serialization, and no-path-I/O rules. |

No separate `plotting/AGENTS.md` is needed because plotting is under evaluation. No `outputs/AGENTS.md` is needed because outputs should be generated state, not a code-edit scope. A conventions child can remain unnecessary if package rules plus executable contracts are sufficient; revisit only if convention code gains independent complexity. An artifacts child is also optional initially if root/package contracts precisely cover immutability and provenance.

The hierarchy already includes stage freeze and independent audit requirements (`agents_hierarchy_design.md:46-61,137-146`); these were not omitted. The missing element is staged enforceability against the current repository.

## 8. Migration-plan review

The 12 gated phases have useful uniform state, evidence, rollback, compatibility, and independent-audit requirements (`migration_plan.md:38-79,244-262`). Phase 0B explicitly forbids production moves and behavior changes (`:83-94`). No phase authorizes new residual TCN, phase-residual research, dependency replatforming, or large regeneration. Those are strong foundations.

### Phase-by-phase assessment

| Phase | Assessment | Required revision or condition |
|---|---|---|
| 0B skeleton/contracts/AGENTS | Conditional | Enumerate the exact allowed new paths/files. Contracts must be drafts/status records, not executable behavior changes. Reconcile inventory/architecture/AGENTS first. |
| 0C1 characterization/paths/manifests | Revise ownership | Keep characterization, hashing, run paths, and manifest schema here. Define sample identity in conventions/contracts, not independently in artifacts. Include before-state fixtures for pipeline, training, split, and physics. |
| 0C2 data/metadata/preprocessing/splits | Reorder/revise | Prefer conventions/sample identity before final data extraction, or explicitly retain old convention facades. Do not move label reconstruction, residual alignment, or smoothed-label builders here. Add `split_policy` to cycle-block schema only in a separately approved manifest-compatible subgate. |
| 0C3 conventions | Move earlier | Frames, units, phase, quaternion ordering, and sample-key contract are dependencies of data, labels, physics, and models. Make this 0C2 and move data extraction to 0C3, after characterization. Keep SLERP correction out. |
| 0C4 labels/alignment | Keep, fix inventory | Own effective-wrench extraction, variants, smoothing/lag provenance, and fail-closed joins. A change from permissive positional fallback to fail-closed behavior needs an explicit compatibility/protocol subgate, not a silent “move.” |
| 0C5 frozen physics | Keep with added fixture | The scope and exclusions are good. Add a multi-case phase/attitude/wind/airspeed regression before declaring frozen. No parameter tuning. |
| 0C6 structured models | Revise acceptance | One family per subgate is good. Model core can move while evaluation/report wrappers remain temporarily. “Scripts no longer shared library” cannot be true for all affected evaluation helpers until 0C8; narrow the completion claim or move minimal shared metric/prediction protocols earlier. |
| 0C7 neural models/training/selection | Split into two gates | First perform behavior-neutral decomposition with old default wrappers. Then, in an explicitly approved protocol-hardening subphase, make selection unable to load test and create separate final evaluation. Current plan simultaneously requires identical defaults (`:177-187`) and a changed command boundary (`:188`), which is contradictory. |
| 0C8 evaluation/replay/plots | Keep | Preserve source tables/metrics and isolate plotting nondeterminism. Define locked-manifest input before test evaluation. Some minimal evaluation interfaces may need to exist earlier for 0C6/0C7. |
| 0C9 thin CLIs/configs | Keep with legacy staging | Correct end position. Existing commands stay wrappers. Do not require all historical commands to adopt new outputs without a compatibility map. |
| 0C10 test taxonomy | Keep, classify per test | Do not mechanically move mixed modules as single unit. Preserve node mapping, count, marks, and fixture provenance. |
| 0D legacy/archive | Keep | Archive only after usage/consumer decision and command/artifact mapping. No deletion by naming convention. |

### Revised dependency sequence

A more executable sequence is:

1. 0B: approved draft contracts, decisions/status index, and staged AGENTS only.
2. 0C1: characterization fixtures plus hashing/run-path/manifest foundation.
3. 0C2: conventions and canonical sample identity (behavior-preserving; no SLERP fix).
4. 0C3: data I/O, metadata loading, resampling/preprocessing, and split services.
5. 0C4: effective labels, variants, lag/filter provenance, and keyed alignment.
6. 0C5: frozen raw physics with expanded regression cases.
7. 0C6: existing structured model cores, one family per gate.
8. 0C7a: neural definitions and behavior-neutral training decomposition; 0C7b: separately authorized fit/select/final-evaluate protocol hardening.
9. 0C8: complete evaluation/replay/diagnostics/plot/report extraction.
10. 0C9: thin categorized CLIs and resolved configs.
11. 0C10: per-test taxonomy and repository regression freeze.
12. 0D: legacy decisions and archive mapping.

This moves conventions before the consumers in the target dependency graph, ensures split/sample identity is fixed before training, and prevents structural migration from silently changing the ordinary training protocol.

Rollback is generally phase-commit based and compatibility wrappers are retained. Add artifact/schema rollback notes for any phase that writes a newer manifest version; reverting code alone cannot reinterpret already-written artifacts safely.

## 9. Cross-document consistency

Legend: `Y` = explicitly present and consistent at high level; `P` = partial/implicit; `C` = material conflict; `—` = not applicable to that document.

| Item | Audit | Inventory | Architecture | AGENTS | Migration | Result |
|---|---:|---:|---:|---:|---:|---|
| Package-name decision | Y | Y | Y | P | Y | Consistent: retain `system_identification`. |
| Target directories | Y | C | Y | Y | C | Inventory exact names/owners differ; migration data/label targets conflict. |
| Migration phase names | P | Y | — | P | Y | Labels all valid, but inventory has no 0C1/0C4 sources and misassigns label builders. |
| AGENTS paths | P | — | P | Y | Y | Target tree contains paths; missing model child remains a design issue. |
| Scripts strategy | Y | Y | Y | Y | Y | Thin wrappers/compatibility consistent; `scripts/audit` vs inventory `scripts/analysis` conflicts. |
| Outputs/artifacts strategy | Y | — | P | Y | P | Roles/provenance consistent; convergence intentionally unresolved. |
| Metadata strategy | Y | Y | Y | Y | Y | Keep aircraft facts outside configs; geometry move needs alias/hash. |
| Test classification | Y | C | Y | Y | Y | Inventory maps mixed modules wholesale despite provisional status. |
| Legacy strategy | Y | Y | Y | Y | Y | No deletion; archive after mapping/decision. |
| Split contract | Y | P | Y | Y | Y | High-level consistent; cycle-block manifest omission and sample-key owner need correction. |
| Physics freeze | Y | Y | Y | Y | Y | Scope consistent; regression breadth needs expansion before freeze. |

Specific contradictions and gaps:

1. Audit says 47 “test modules”; inventory and Git show 46 modules plus one fixture.
2. Audit says 19 imported script modules; independent AST count is 18.
3. Inventory puts residual/smoothed-label work in data/0C2, but architecture and migration put it in labels/0C4.
4. Inventory's `pipeline.py` target includes `data/labels.py`, violating the architecture's explicit data/labels boundary.
5. Inventory target spellings (`resample.py`, `signal_preprocessing.py`, verbose DeLaurier names, `isaaclab_wing_only_baseline.py`) differ from the architecture tree.
6. Inventory routes audit scripts to `scripts/analysis`; architecture defines `scripts/audit`.
7. Inventory's training split puts features at package root; architecture puts them under training.
8. AGENTS design makes target invariants immediate without a migration-stage activation mechanism, although audit and plan acknowledge current violations.
9. Migration 0C7 requires both identical defaults and a selection command that no longer loads test.
10. Migration 0C1 and 0C3 both claim sample-key responsibility.
11. Migration 0C6 claims shared script logic is gone before evaluation helpers move in 0C8.
12. Cycle-block materialization does not save a `split_policy`, despite the audit/architecture requirement that policy/version be mandatory.
13. Architecture leaves outputs-versus-artifacts convergence open; no migration phase records the required ADR owner/timing. This is acceptable for Phase 0B but must be a decision before output-root migration.
14. The audit's HIGH prior-alignment risk is present in 0C4 conceptually, but inventory prevents that plan from being executed as written.

No large-scale contradiction was found in the package decision, overall domain model, compatibility strategy, or prohibition on new research. The conflicts are correctable without rerunning Phase 0A.

## 10. Required corrections

### BLOCKING

None requires discarding or rerunning the entire Phase 0A audit.

### REQUIRED BEFORE PHASE 0B

1. **Correct reproducible counts.**
   - Documents: `repository_readonly_audit.md`.
   - Change: say 46 test modules plus one fixture; report 38/46 direct script imports; correct script-to-script imported-module count from 19 to 18 or publish the exact alternative counting rule; use 5,470 lines for `training.py` when exact.
   - Owner: Phase 0A documentation owner.
   - Acceptance: tracked-path plus AST counting command reproduces every number and emits the matching file list.

2. **Reconcile inventory ownership and target paths.**
   - Documents: `repository_file_inventory.csv`, `proposed_repository_architecture.md`, `repository_migration_plan.md`.
   - Change: move residual/label-variant/alignment builders to `labels/`/0C4; remove `labels` from the 0C2 pipeline target; choose canonical filenames; route audit CLIs to `scripts/audit`; align the training target; define the action enum; replace compound/brace targets with structured mappings.
   - Owner: architecture and migration owner.
   - Acceptance: automated check shows every target belongs to the target tree, every action is defined, every source responsibility has one owning phase, and expanded targets have no collision.

3. **Make AGENTS rules stage-aware and settle the model boundary.**
   - Document: `agents_hierarchy_design.md`.
   - Change: distinguish immediately enforced safety rules from target invariants activated by later frozen phases; document bounded legacy exceptions; add `src/system_identification/models/AGENTS.md` or explicitly justify and test an equivalent parent-only rule set.
   - Owner: Phase 0B AGENTS design owner.
   - Acceptance: walk through one current legacy script change, one model change, one training change, and one evaluation change without contradiction or reliance on a sibling AGENTS file.

4. **Remove phase-contract contradictions.**
   - Document: `repository_migration_plan.md`.
   - Change: establish a single sample-key owner; put conventions before final data extraction or document the temporary facade; split 0C7 behavior-neutral decomposition from test-isolation protocol change; narrow 0C6's “scripts no longer shared library” completion; add manifest rollback/version handling.
   - Owner: migration-plan owner.
   - Acceptance: each phase has one behavior status, independently satisfiable entry/exit criteria, and no acceptance item forbidden by its own “not done” list.

5. **Freeze the exact Phase 0B allowlist.**
   - Documents: migration plan and architecture/AGENTS design.
   - Change: list exact directories/files Phase 0B may add, state that config schemas cannot change runtime behavior, and prohibit placeholder skeleton files without an approved purpose.
   - Owner: Phase 0B owner.
   - Acceptance: a proposed Phase 0B diff can be mechanically compared with the allowlist and contains no production code/import/config behavior change.

### REQUIRED DURING PHASE 0B

1. Draft executable-status contracts for sample identity, split policy/version, train/validation/test roles, label/prior semantics, coordinates/units, and raw physics provenance. Each must state owner, status, unresolved behavior, and activation phase; none may claim SLERP or fail-closed alignment is already implemented.
2. Record an ADR/status item that `dataset/`, `artifacts/`, and `outputs/` remain distinct legacy roots until retention/storage/convergence is decided. Do not move or ignore them in Phase 0B.
3. Record the cycle-block missing `split_policy` field and quaternion SLERP drift as deferred behavior work, not as Phase 0B fixes.
4. Validate AGENTS inheritance using representative tasks and ensure current legacy workflows are bounded rather than globally outlawed before their migration phase.

Owner: Phase 0B documentation/contract owner. Acceptance: documentation/link/inheritance audit plus byte-identical tracked production files and the allowed collect-only command.

### NON-BLOCKING

1. Before 0C5 freeze, add multi-case DeLaurier regression coverage for phase, attitude/wind, airspeed, and full wrench outputs.
2. Before 0C10, classify individual tests in mixed modules instead of moving whole modules mechanically.
3. Before output-root migration, decide retention/storage and URI policy through an ADR.
4. Before a quaternion behavior fix, create a dedicated plan with sign-continuity/SLERP tests and quantified label deltas.
5. Define `stability`, `risk`, blank coverage, and `indirect/partial` semantics in the inventory data dictionary.

## 11. Phase 0B entry decision

**PASS WITH CONDITIONS: Phase 0B is not authorized to begin yet.**

Phase 0B may begin after all five “REQUIRED BEFORE PHASE 0B” corrections are applied to the Phase 0A deliverables and independently checked. Those corrections are bounded documentation/inventory/plan work; they do not require another repository-wide audit.

Once satisfied, Phase 0B may create only the exact approved directory skeleton, contract/status documents, and stage-aware AGENTS hierarchy. It may not move production code, change imports, alter runtime configuration, create executable behavioral contracts that disagree with current code, fix SLERP/alignment/test loading, or start model research.

The Phase 0B audit gate must verify:

- tracked production code/tests/config/metadata are byte-identical;
- the diff matches the exact allowlist;
- contracts distinguish observed behavior, target behavior, unknowns, and activation phase;
- AGENTS inheritance is non-contradictory for legacy and target paths;
- collect-only still reports the same 275 tests;
- no cache, output, data, model, plot, or log was created in the repository.

## 12. Final repository status

Final verification commands were `git status --short`, `git diff --check`, direct SHA-256 comparison of all five untracked Phase 0A deliverables, and explicit inspection of untracked paths. `git diff` is not used as evidence for the content of untracked files.

Expected final short status after this report is added:

```text
?? docs/architecture/
?? docs/audits/
?? docs/plans/2026-07-15_repository_migration_plan.md
?? outputs/
```

Expanded status must show the same five Phase 0A deliverables, this one independent audit report, and the 106 pre-existing `outputs/` files. The five Phase 0A SHA-256 hashes must match Section 3. No tracked file or existing untracked file was modified. No Phase 0B directory, AGENTS file, contract, cache, model, data, image, or log was created by this review.

Phase 0A independent audit completed.
Phase 0B may begin only after the listed conditions are satisfied.
No Phase 0B work was performed.
