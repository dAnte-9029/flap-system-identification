# Evaluation

Evaluate an explicit model/checkpoint against an explicit split; do not use sealed test data to tune weights, normalizers, hyperparameters, selection criteria, or reporting examples.

Preserve per-flight and aggregate results, sample identity, and the definition of each metric. Distinguish autonomous closed-loop rollout from teacher forcing, oracle inputs, and factual versus changed-control/counterfactual comparisons. Do not present trajectory accuracy alone as evidence of control-response accuracy.

Generate diagnostics that answer the requested question. Do not automatically add test matrices, plotting variants, or verification reports unrelated to a claim. New evaluation designs may change existing methods when that is the research task.
