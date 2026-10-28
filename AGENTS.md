# Flapping-Wing System Identification: Agent Guidelines

## Goal and working style

This repository develops bird-scale flapping-wing dynamics models from flight data, with emphasis on control-relevant prediction. Prioritize the requested scientific result over procedural completeness. An approved research idea may require substantial changes to state representation, model architecture, learning targets, training, and evaluation; do not reduce it to a small patch just to preserve the old design.

Inspect relevant code and contracts, then implement the requested scope. Do not add exploratory ablations, multi-seed sweeps, audits, manifests, or documentation tasks unless requested or essential to the scientific claim. Keep experiment design separate from implementation: Codex should execute the chosen design, not quietly replace it with incremental variants.

## Context and environment

- Read this file and the nearest applicable nested `AGENTS.md` for paths being changed. Do not recursively read every `AGENTS.md` or every historical plan.
- Consult the relevant active configuration, contract, or prior run only when it matters to the task. Historical documents are evidence, not permanent workflow gates.
- Use `/home/zn/anaconda3/envs/flap-train-gpu/bin/python` for repository Python, tests, and training. Do not silently switch environments or modify dependencies; report an unavailable environment rather than guessing.
- Preserve unrelated files and uncommitted user work. Do not expand the task into an unrelated cleanup.

## Data and physical contracts

- Select the data registry for the **requested dataset family**. Canonical flight-sample experiments use `configs/data/canonical_dataset_registry.yaml`; native velocity/rate experiments use `configs/data/velocity_rate_dataset_registry.yaml`. Follow the selected versioned manifest and explicit user overrides. Do not force an unrelated experiment back to the canonical sample registry.
- For canonical ratio-8 data, use the registered `mechanical_phase_rad` and `flap_frequency_hz` definitions. For other datasets, follow their declared phase, frequency, timestamp, and sample contracts. Never silently mix conventions.
- For DeLaurier-based studies, resolve the applicable prior using `configs/physics/delaurier_prior_registry.yaml` and the experiment contract. New non-DeLaurier models do not need to load a DeLaurier prior. Never silently replace a missing selected artifact with an older or superseded artifact.
- Keep train/validation/test roles explicit. Fit statistics and learn parameters from train data; use validation for development and selection; keep sealed test data out of development unless a final-evaluation protocol explicitly authorizes access. Do not open test data or compute its hashes as an incidental check.
- Preserve log/sample identity, time ordering, causal input availability, and segment/gap boundaries. During autonomous rollout, do not inject future measured states or labels unless the experiment is explicitly identified as an oracle/teacher-forced diagnostic.
- State frames, units, signs, moment reference points, and mass/geometry sources when they affect an experiment. Distinguish whole-aircraft effective wrench from isolated wing loads.

## Experiments and verification

- Run the experiment that answers the user's research question, with meaningful baseline comparison and the requested metrics. Do not automatically add further ablations, thresholds, parameter sweeps, or repeated runs.
- Validate proportional to risk: a focused executable check for a local code change; targeted numerical or data checks for changed physics, labels, datasets, training, or evaluation contracts. Recheck after failures or relevant subsequent changes, not by default after every step.
- Preserve existing artifact provenance and integrity checks where necessary, but do not repeatedly calculate SHA256, produce hash inventories, or generate audit receipts for unrelated new runs. Use hashes when artifact identity or corruption is genuinely at issue.
- Do not overwrite prior datasets, checkpoints, or results without instruction. Record only the experiment configuration, dataset/split identity, code revision, seed when relevant, and metrics needed to understand and reproduce the result. Reuse existing machinery rather than creating another verification framework.

## Completion

Report concisely what changed, what was actually run, the key measurements, and material limitations. Do not claim unexecuted tests. Stop when the requested deliverable is complete; do not launch follow-on stages without a request.
