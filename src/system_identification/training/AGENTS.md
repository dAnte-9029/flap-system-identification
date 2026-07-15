# Training Foundation Guidelines

- Fit normalization and other statistics from training data only; validation is for early stopping and model selection.
- This migration does not change the existing test invocation protocol.
- Causal windows must not cross log, segment, or invalid time-gap boundaries.
- Loaders must preserve sample identity and the existing time-order semantics.
- Keep model structures, physics, plotting, and paper reporting outside this directory.
- Behavior-preserving moves must not change algorithms, defaults, checkpoint formats, or silently fix bugs.
