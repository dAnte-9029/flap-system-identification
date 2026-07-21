# Repository Guidelines

## Skill Policy

For work in this repository, treat the following skills as the default allowed set. These should stay enabled by default because they match the main workflows here: IsaacSim simulation development, PX4/control code work, and robotics paper writing.

- Core process skills: `brainstorming`, `writing-plans`, `test-driven-development`, `systematic-debugging`, `verification-before-completion`, `requesting-code-review`, `receiving-code-review`, `using-git-worktrees`
- Parallel execution skills: `dispatching-parallel-agents`, `subagent-driven-development`, `executing-plans`
- Paper and documentation skills: `literature-review`, `citation-management`, `venue-templates`, `scientific-writing`, `scientific-visualization`, `markdown-mermaid-writing`, `scientific-slides`
- Analysis and plotting skills: `get-available-resources`, `matplotlib`, `seaborn`, `plotly`, `statistical-analysis`, `statsmodels`

For this repository, treat the following skill groups as disabled by default unless the user explicitly asks for them or the task clearly requires them:

- Bioinformatics, genomics, proteomics, single-cell, and pathway analysis skills
- Drug discovery, medicinal chemistry, molecular docking, cheminformatics, and compound database skills
- Clinical medicine, treatment planning, clinical reporting, medical regulatory, and trial database skills
- Finance, macroeconomics, SEC, Treasury, hedge fund, and market data skills
- Astronomy, quantum computing, and materials science skills
- Wet-lab automation, LIMS, ELN, microscopy, pathology, flow cytometry, and laboratory platform integration skills
- General scientific database skills unrelated to robotics, controls, simulation, or academic writing in this project

When working in this repository:

- Prefer the smallest relevant skill set instead of loading broad scientific skill chains.
- Do not invoke unrelated scientific database or life-science skills for IsaacSim, PX4, RL, controls, or robotics writing tasks.
- Keep context tight: read only the minimum skill content needed for the current task.
- If a task is purely IsaacSim/PX4 coding, default to the core process skills only.
- If a task is paper-writing related, add only the specific writing or citation skills needed for that document step.

## DeLaurier prior policy

For new longitudinal-force analysis, resolve the DeLaurier prior through
`configs/physics/delaurier_prior_registry.yaml`. Use the entry selected by
`default_prior_id` unless the user explicitly requests a newer manifest-backed
active prior.

- Do not use entries marked `legacy` or `superseded` except for explicitly
  requested historical reproduction.
- Do not infer authority from a dated directory name or from an older
  correction artifact that happens to reference a prior.
- If the authoritative prior artifact is missing for the requested train or
  validation partition, fail and report the missing artifact. Never fall back
  to a legacy prior.
- Before analysis, record the resolved prior ID, absolute path, lifecycle
  status, physics source commit, frame/airflow/phase contracts, and partition
  coverage in the run manifest.
- Test-window diagnostic outputs are not training-ready prior artifacts and
  must not be used to choose a correction structure.

## Canonical dataset policy

For all new experiments, resolve the canonical dataset through
`configs/data/canonical_dataset_registry.yaml`. Use the entry selected by
`default_dataset_id` unless the user explicitly requests a versioned
historical reproduction.

- The current authoritative dataset is
  `canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`.
- Do not select a dataset by directory timestamp, modification time, or a
  legacy script default.
- Do not use the corresponding `v1` rebuild: its two no-`hall_event` logs used
  an inaccurate fallback near the Hall crossing.
- Do not use the source v0.2 ratio-8 dataset directly for new experiments. It
  remains provenance input only; its logged phase/frequency contract is
  superseded by the v0.3 dataset.
- Use `mechanical_phase_rad` as the only phase coordinate and
  `flap_frequency_hz` as the ratio-8 corrected frequency. Do not reconstruct a
  parallel phase from legacy columns.
- This immutable dataset contains the preserved whole-log train, validation,
  and test assignments. EDA0/C1 and all pre-C7 correction work must request
  train/validation explicitly and must not load the test Parquet; test remains
  locked until the applicable gate.
- Record the resolved dataset ID, repository-relative path, manifest hash,
  sample artifact hashes, partitions, phase contract, and frequency contract
  in every downstream run manifest.
- Fail if the registered manifest or Git LFS sample artifacts are missing or
  their hashes do not match. Never fall back silently to an older dataset.
