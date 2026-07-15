# AGENTS.md Hierarchy Design

Date: 2026-07-15

Status: design only; no AGENTS.md was created or modified

## 1. Current root AGENTS.md audit

### Observed facts

The root file contains only a skill policy: allowed core/parallel/paper/analysis skill families, disabled unrelated scientific domains, and guidance to use the smallest relevant skill set. It applies repository-wide and should remain as the skill-selection section.

It does not currently define repository invariants: read/write boundaries, environment, train/validation/test isolation, coordinate/unit conventions, label/prior semantics, script thinness, output immutability, manifests, physics regression requirements, stage gates, or stop-after-stage behavior.

### Recommendation

Extend the root in Phase 0B with universal rules and an index of child files; retain the skill policy without broadening it. Detailed formulas, file lists, experiment results, or temporary task prompts do not belong in AGENTS files; link to contracts/runbooks instead.

## 2. Inheritance and conflict principles

The nearest AGENTS file adds local constraints but cannot weaken parent safety, split isolation, provenance, or stage-gate rules. Parent rules win on conflict unless the parent explicitly delegates a choice. Child files may narrow allowed writes, tests, dependencies, and completion checks. A task reads the root plus every AGENTS file on the path to each file it will change. Cross-directory changes require reading all affected branches before editing.

Every rule should be testable or point to a contract. Unknown semantics cause a stop and decision request, never an inferred convention. A completed phase stops after its audit package; it does not automatically enter the next phase.

Target-directory rules activate for a production file only when that file is created in or migrated into the governed directory. Before migration, existing files remain governed by the root and nearest existing AGENTS files plus the active phase contract. Target rules may forbid adding new legacy coupling, but they must not retroactively make unmigrated code noncompliant merely because a later migration has not occurred. Each child file states its activation boundary and the phase that introduces it.

## 3. Minimum sufficient hierarchy

| Proposed path | Necessary? | Scope and parent relation |
|---|---:|---|
| `AGENTS.md` | yes | universal repository rules; retains skill policy |
| `src/system_identification/AGENTS.md` | yes | all library code; adds dependency/API/behavior-preservation rules |
| `src/system_identification/data/AGENTS.md` | yes | ULog, metadata loading, resampling, canonical data, splits |
| `src/system_identification/physics/AGENTS.md` | yes | physical models, priors, frames/reference points and regression |
| `src/system_identification/labels/AGENTS.md` | yes | inverse-dynamics labels, alignment, validity and variants |
| `src/system_identification/models/AGENTS.md` | yes | structured/neural inference, deployability, parameter separation and serialization |
| `src/system_identification/training/AGENTS.md` | yes | adapters, fitting, normalization and selection; inference rules live in models child |
| `src/system_identification/evaluation/AGENTS.md` | yes | locked evaluation, replay, metrics, plots, reports |
| `scripts/AGENTS.md` | yes | all CLIs/experiments/legacy wrappers; thin-entry rule |
| `configs/AGENTS.md` | yes | ownership, schema, resolved configs and overrides |
| `tests/AGENTS.md` | yes | taxonomy, fixtures, runtime marks, regression behavior |
| `docs/AGENTS.md` | yes | fact/inference/recommendation separation and document status |
| `metadata/AGENTS.md` | yes | aircraft facts, units, provenance, approval and schema |

Create `models/AGENTS.md`: model files do not inherit the sibling `training/AGENTS.md`, and deployability, structured-versus-neural boundaries, physical-versus-learned parameter separation, inference purity, and serialization are distinct from fitting/selection. Fitting, normalization, and selection remain in `training/AGENTS.md`. Do not create separate initial files for `models/structured` and `models/neural`; add them only if their rules materially diverge. Do not create files for `plotting`, `artifacts`, `fixtures`, individual experiment folders, `outputs`, or every config subtype; their parent rules are sufficient and extra files would fragment invariants.

## 4. Detailed outlines

### Root `AGENTS.md`

Must contain:

- repository purpose and supported environment (`flap-train-gpu`), no dependency changes without explicit phase scope;
- universal contract precedence: coordinate, data, split, evaluation, physics-prior contracts;
- train/validation/test target roles: train fits, validation selects, and test is one-shot final evidence that never feeds selection, normalization, phase/filter/calibration fitting, or plot choice; this target protocol activates in 0C7B, while documented legacy automatic test reporting remains a bounded compatibility exception through 0C7A and must not be expanded;
- no silent invalid-row deletion; emit reason/count/mask and preserve group/sample identity;
- no overwrite of existing datasets, models, results, or runs;
- every experiment stores resolved config and manifest with Git/data/split/metadata provenance;
- physical parameters and learned parameters are separate blocks with different approval semantics;
- stage states and “do not automatically enter the next stage” rule;
- required independent read-only audit before `FROZEN`;
- child AGENTS index and conflict rule.

Must not contain formulas copied from contracts, current best metrics, dated artifact paths, or directory-specific implementation details. Typical task: coordinate a multi-directory migration phase. Completion: scope diff, tests/smoke/regression, manifest, docs, independent audit, identifiable commit.

### `src/system_identification/AGENTS.md`

Must contain: library code owns reusable behavior; no import from scripts or dated configs; public API compatibility/re-export policy; pure functions where practical; explicit input/output types and sample-key preservation; model inference cannot read split paths; model features must be deployable and may not contain target/derivative leakage unless an explicitly diagnostic model says so; physics and learned parameters remain distinct; behavior changes require a separate phase from moves.

Models section: raw prior, calibrated prior, shaped prior, corrected prediction, and residual prediction are distinct named types; no ambiguous `pred`; neural models consume adapters, not canonical storage directly; causal models must prove window grouping/order/gap behavior; no future samples in deployed features. Do not include per-architecture hyperparameters. Typical task: extract a reusable script helper. Completion: old API parity, focused tests, import-direction check, serialization compatibility.

### `src/system_identification/models/AGENTS.md`

Must contain: models expose inference and serialization contracts but do not read train/validation/test paths; structured and neural implementations remain separate; raw physical parameters, calibrated physical parameters, and learned correction parameters use distinct typed/serialized blocks; deployable inference uses only declared available features; causal models preserve grouping/order/gap semantics; model modules do not select candidates, fit normalization, create run directories, or write reports.

Rules become active for a model implementation when it is introduced or migrated under `models/`. Until then, existing script or `training.py` implementations remain under their current parent rules and phase contract. Typical task: move one frozen model definition. Completion: prediction and serialization parity, deployable-feature check, old import facade, and no split/path I/O.

### `src/system_identification/data/AGENTS.md`

Must contain: canonical sample identity and sort key are never discarded; topic units/frames and freshness are explicit; quaternion ordering is PX4 `wxyz` body-FRD to NED unless a contract says otherwise; interpolation method is field-specific; no cross-log/segment filtering; invalid rows receive flags/reasons/counts; split assignments are immutable artifacts; train/val/test are disjoint according to the selected policy; test cannot set normalization, phase alignment, filters, admission thresholds, or split choices.

Real-airflow rule: compute NED ground velocity minus NED wind, then rotate using the measured body-to-NED attitude into body FRD; do not substitute pitch-only/scalar airspeed silently. Record vertical-wind availability. Do not include model training guidance. Typical task: migrate resampling. Completion: schema/key/count parity, boundary/gap tests, split lineage and no new silent drops.

### `src/system_identification/physics/AGENTS.md`

Must contain: declare every input/output frame, vector parity, unit, sign, and moment reference; preserve FRD/NED definitions; translate moments with an explicit source/target reference; phase/frequency/twist/strip geometry definitions reference the physics contract; air-relative velocity follows the real-attitude rule; aircraft geometry and physical parameters come from versioned metadata/config with provenance; learned corrections cannot mutate the frozen raw baseline.

Any physics-model change requires analytic/unit checks, frozen numerical regression, symmetry/parity checks, reference-point tests, and documented expected deltas before merging. Do not put fit/search choices or paper claims here. Typical task: move DeLaurier airflow code. Completion: frozen fixture parity, metadata hashes, frame inventory, no behavior delta unless separately approved.

### `src/system_identification/labels/AGENTS.md`

Must contain: labels are effective whole-aircraft external wrench, not automatically pure wing aerodynamics; force/moment equations, gravity subtraction, quaternion order, FRD/NED, CG moment reference and units reference the label contract; derivative/filter variants are versioned, group-local, and recorded; raw labels remain available; residual targets require keyed one-to-one joins and fail closed; equal row count is not alignment evidence; invalid labels carry masks/reasons and are never silently removed.

No prior fitting or model selection belongs here. Typical task: extract smoothed label reconstruction. Completion: raw/current parity, key uniqueness, valid/invalid counts, boundary tests, manifest label version.

### `src/system_identification/training/AGENTS.md`

Must contain phase-specific activation: in 0C7A, extraction preserves the existing command defaults and automatic test-reporting boundary through explicit compatibility wrappers while pure feature/model/fit components do not acquire new test dependencies. In 0C7B, train fits normalization/imputation/calibration/model weights; validation selects epoch, form, features and hyperparameters; test becomes unavailable to fit/selection code; selection and final evaluation become separate commands/artifacts. In both phases, causal windows stay within log/segment and enforce time-gap/order contracts; subsampling preserves sequence semantics; random seeds and resolved features are saved; checkpoints/bundles include schema/version and physical-versus-learned parameter blocks.

Do not include evaluation-only plot rules or aircraft constants. Typical task: split `training.py`. Completion in 0C7A: zero/near-zero prediction and metric parity, same best epoch under deterministic fixture, train-only statistics assertion, serialization round trip, and unchanged legacy command behavior. Additional completion in 0C7B: test-not-loaded fit/selection tests and isolated final-evaluation artifacts.

### `src/system_identification/evaluation/AGENTS.md`

Must contain: evaluator never fits or selects; test evaluation requires a frozen model/config/split manifest; plots are downstream of saved prediction/metric tables and cannot become implicit selectors; replay modes/horizons and teacher-forcing/oracle inputs are explicit; per-log and aggregate metrics coexist; sample keys and partition labels persist; test-window selection must be predetermined or declared diagnostic and cannot identify model parameters.

Do not embed training loops. Typical task: migrate short-horizon replay. Completion: synthetic closure, locked-input manifest, metric parity, plot provenance, no writes outside new run.

### `scripts/AGENTS.md`

Must contain: scripts parse/validate config, call package API, allocate a unique run, and report outputs; reusable scientific logic and script-to-script private imports are forbidden after migration; no `sys.path` manipulation in new scripts; all defaults are explicit/configurable; no nonempty output overwrite; old commands remain wrappers with mapping/deprecation status; experiment scripts may orchestrate but not become shared libraries.

Typical task: thin one builder CLI. Completion: old/new `--help`, tiny fixture output parity, command mapping, overwrite refusal, resolved config/manifest.

### `configs/AGENTS.md`

Must contain ownership boundaries; schema/version field; units on physical quantities; no aircraft facts duplicated from metadata; no absolute workstation paths in reusable configs; overrides recorded; resolved config always persisted; config changes that alter behavior require a separate reviewed experiment. Do not place secrets, results, or generated resolved configs here. Completion: loader validation, round-trip/hash, old-default equivalence.

### `tests/AGENTS.md`

Must contain taxonomy definitions; unit tests cannot require external data/GPU/network; integration tests use `tmp_path`; regression fixtures are immutable with generator/provenance/tolerance; external contracts are marked/skippable with explicit reason; tests never write repository outputs; physics changes require regression; migration tests compare old/new behavior; expensive tests are marked and excluded from default smoke.

Typical task: classify tests after a module move. Completion: collection unchanged, node mapping recorded, focused suite plus smoke, no cache/output pollution.

### `docs/AGENTS.md`

Must contain Observed facts/Inferences/Recommendations/Unknowns separation; every high-risk claim cites file/line/artifact evidence; status labels (`draft`, `active contract`, `historical`, `superseded`); no claim that test was held out without manifest evidence; commands and artifact provenance must be reproducible; results copied into Git are compact and reviewed. Do not rewrite historical results to appear current. Completion: links, status, contract consistency, no unsupported claims.

### `metadata/AGENTS.md`

Must contain aircraft-wide versus experiment-specific boundary; required schema version/status/unit/source/uncertainty; FRD/NED/origin definitions; actuator signs/open items; geometry/mass/sensor assets hashed and referenced; measured/assumed/fitted values clearly distinguished; learned parameters forbidden; changes require validation and an impact report listing affected labels/priors/artifacts.

Typical task: add a geometry asset. Completion: schema validation, units, provenance/hash, consumer path tests, no silent default fallback.

## 5. AGENTS reading matrix by migration phase

| Work | Required AGENTS files |
|---|---|
| Phase 0B hierarchy/contracts | root, docs; plus every new child being created |
| 0C1 characterization/paths/manifests | root, package, tests, docs |
| 0C2 conventions/timestamp/phase/sample identity/alignment keys | root, package, metadata, tests, docs |
| 0C3 data/metadata/preprocessing/splits | root, package, data, metadata, configs, tests, docs |
| 0C4 labels/alignment | root, package, labels, data, metadata, tests, docs |
| 0C5 physics baseline | root, package, physics, metadata, configs, tests, docs |
| 0C6 structured models | root, package, models, training, configs, tests, docs |
| 0C7A neural/training decomposition | root, package, models, training, configs, tests, docs |
| 0C7B fit/select/final-evaluation isolation | root, package, models, training, evaluation, configs, tests, docs |
| 0C8 evaluation/replay/plots | root, package, evaluation, configs, tests, docs |
| 0C9 CLI/config | root, scripts, package target, configs, tests, docs |
| 0C10 test taxonomy/regression freeze | root, tests, package targets, docs |
| 0D legacy decisions | root, scripts, docs, tests |

## 6. Phase 0B creation order

1. Extend root universal rules while preserving the skill policy.
2. Add package parent rules.
3. Add data, physics, labels, models, training, and evaluation children only alongside the approved target skeleton.
4. Add scripts/configs/tests/docs/metadata rules.
5. Validate inheritance with representative hypothetical tasks.
6. Perform an independent read-only audit; revise; freeze Phase 0B.

No AGENTS file should be created from this design until Phase 0B is independently authorized.
