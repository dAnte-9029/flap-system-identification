# Artifact I/O

Own paths, manifests, checkpoints, predictions, and report serialization. Avoid embedding training, model selection, or metric computation here.

Do not silently overwrite existing results. Preserve enough data/model/split/version information to identify scientific outputs. Reuse existing provenance when available; do not introduce checksum scans, redundant receipts, or new audit formats without a concrete need. Artifact format changes are allowed when required by the requested research design, with affected readers updated in the same task.
