# Artifact I/O Guidelines

- Own only paths and manifest, checkpoint, prediction, and report I/O; do not train, select models, or compute evaluation metrics.
- Never overwrite a non-empty existing run unless the established caller explicitly allows it.
- Preserve schemas and provenance for versions, Git state, splits, metadata, and parameter sources.
- Artifact behavior changes require separate authorization; this phase only migrates existing behavior.
