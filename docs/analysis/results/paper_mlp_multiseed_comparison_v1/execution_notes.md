# Execution and reuse

Environment: `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`.

Entry point: `scripts/run_paper_mlp_multiseed_comparison.py --phase <phase>`.

Executed phases in order:

1. `audit`: verify only required frozen sources, prepared train/validation cache, origin CSVs, normalization, original MLP seed17 provenance, and existing comparator metrics. Freeze protocol before new training.
2. Scoped pytest suite, with results recorded in `tests.json`.
3. `launch`: run disposable small-batch two-stage GPU smoke tests and fixed128-origin seed17 prediction parity; start only MLP seed23/42 workers on GPU1. Each worker starts a fresh seeded model, trains40+25 epochs and resets AdamW between stages. The orchestration process waits for both workers and records their exit codes.
4. After both complete, `evaluate`: infer Actual commands only for the two new checkpoints, reuse frozen seed17/GRU/B0 results, compute the common endpoint metrics and reports, and verify all frozen input hashes again.

The runner refuses an existing protocol, duplicate launch, existing worker directory, failed training completion, or an existing completed evaluation. It does not select the best validation epoch, automatically retry training, or overwrite previous experiments. The training launcher is detached for process durability, but this task continues through evaluation and final reporting.

Each new seed folder under `artifacts/paper_mlp_multiseed_comparison_v1/` retains base and continuation histories/checkpoints, final model, runtime/config/status, complete predictions and per-origin errors. Historical paths are referenced in the protocol/training table without copying old checkpoints into new run folders.

The report script was implemented during training; its final hash is recorded in completion.json. Metric, aggregation and paired-gain definitions were frozen in protocol.json before training. The inherited Step1 single-pilot metadata wording is explicitly scoped in metric_metadata_clarification.json; the current protocol's top-level aggregation rule specifies three-seed sample SD.

No sealed/reserved input, prediction or evaluation artifact is opened. Only exclusion names from already authorized manifest metadata are inspected. No training data generation, normalization fitting, historical rerun or independent-test preparation is performed.
