# Training

Fit learned parameters, normalization, and train-derived statistics on the authorized training split. Validation may guide development; sealed test data must not inform fit or selection. Keep causal training windows within each log/segment and valid time gaps, and preserve time/sample identity.

Support requested new losses, target definitions, multi-step objectives, or training architectures without forcing historical algorithm/default parity. Distinguish genuine autonomous rollout from injected ground-truth states and report that distinction in metrics.

For each research run, preserve the chosen config, seed(s), split identities, and necessary checkpoint/output references. Do not default to multi-seed repeats, broad sweeps, all-tests runs, repeated hashes, or elaborate verification pipelines; run the focused checks needed to trust the changed computation.
