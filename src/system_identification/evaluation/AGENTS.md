# Evaluation Guidelines

Evaluation accepts a frozen model and an explicit partition. It must not fit models, select hyperparameters, or update normalization, model state, or configuration from test data. Preserve both per-log and aggregate metrics, and retain sample and log identity in prediction and metric tables.

Plotting must not become an implicit model selector. This migration preserves current behavior only; strict test isolation requires a separately authorized stage.
