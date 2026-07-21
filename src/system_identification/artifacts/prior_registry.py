"""Resolve authoritative and legacy DeLaurier prior artifacts.

This module owns only artifact identity, lifecycle, and provenance validation.
It does not evaluate physics or select a model from data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml


DEFAULT_REGISTRY_PATH = Path("configs/physics/delaurier_prior_registry.yaml")
LEGACY_STATUSES = frozenset({"legacy", "superseded", "historical"})


@dataclass(frozen=True)
class PriorResolution:
    """Resolved prior identity with lifecycle and physics contract metadata."""

    prior_id: str
    artifact_root: Path
    lifecycle_status: str
    physics_source_commit: str
    frame_contract: str
    airflow_contract: str
    phase_contract: str
    required_partitions: tuple[str, ...]
    registry_path: Path | None
    registry_entry: Mapping[str, object]
    manifest: Mapping[str, object]

    @property
    def is_legacy(self) -> bool:
        return self.lifecycle_status in LEGACY_STATUSES


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required mapping file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _normal_partition(value: str) -> str:
    return "validation" if value == "val" else str(value)


def _manifest_commit(manifest: Mapping[str, object]) -> str:
    source = manifest.get("physics_source")
    if isinstance(source, Mapping) and source.get("commit"):
        return str(source["commit"])
    isaaclab = manifest.get("isaaclab_git")
    if isinstance(isaaclab, Mapping) and isaaclab.get("commit"):
        return str(isaaclab["commit"])
    return str(manifest.get("physics_source_commit", "unknown"))


def _contract_value(
    manifest: Mapping[str, object],
    entry: Mapping[str, object],
    key: str,
    default: str = "unknown",
) -> str:
    contracts = manifest.get("contracts")
    if isinstance(contracts, Mapping) and key in contracts:
        return str(contracts[key])
    if key in manifest:
        return str(manifest[key])
    return str(entry.get(key, default))


def load_prior_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> tuple[Path, dict[str, object]]:
    """Load the schema-checked DeLaurier prior registry."""

    source = Path(path)
    if not source.is_absolute():
        source = _project_root() / source
    source = source.resolve()
    registry = _read_mapping(source)
    if registry.get("schema_version") != "delaurier_prior_registry_v1":
        raise ValueError(f"Unsupported prior registry schema in {source}: {registry.get('schema_version')!r}")
    if not isinstance(registry.get("priors"), dict):
        raise ValueError(f"prior registry has no priors mapping: {source}")
    return source, registry


def resolve_delaurier_prior(
    *,
    prior_root: str | Path | None = None,
    prior_id: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    allow_legacy: bool = False,
    requested_partitions: Sequence[str] = ("train", "validation"),
) -> PriorResolution:
    """Resolve one prior and fail closed for legacy or incomplete artifacts.

    An explicitly supplied prior outside the registry is accepted only when its
    own manifest declares ``lifecycle_status: active``. Manifests without a
    lifecycle field are treated as legacy/unclassified.
    """

    if prior_root is not None and prior_id is not None:
        raise ValueError("Provide at most one of prior_root and prior_id")
    resolved_registry_path, registry = load_prior_registry(registry_path)
    entries = registry["priors"]
    assert isinstance(entries, dict)

    selected_id = prior_id
    entry: Mapping[str, object] = {}
    if prior_root is None:
        selected_id = selected_id or str(registry.get("default_prior_id", ""))
        if not selected_id or selected_id not in entries:
            raise ValueError(f"Unknown or missing default prior ID {selected_id!r} in {resolved_registry_path}")
        raw_entry = entries[selected_id]
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Registry entry {selected_id!r} must be a mapping")
        entry = raw_entry
        prior_root = str(entry.get("artifact_root", ""))
        if not prior_root:
            raise ValueError(f"Registry entry {selected_id!r} has no artifact_root")

    artifact_root = Path(prior_root)
    if not artifact_root.is_absolute():
        artifact_root = _project_root() / artifact_root
    artifact_root = artifact_root.resolve()

    if not entry:
        for candidate_id, raw_entry in entries.items():
            if not isinstance(raw_entry, dict) or not raw_entry.get("artifact_root"):
                continue
            candidate = Path(str(raw_entry["artifact_root"]))
            if not candidate.is_absolute():
                candidate = _project_root() / candidate
            if candidate.resolve() == artifact_root:
                selected_id = str(candidate_id)
                entry = raw_entry
                break

    manifest_path = artifact_root / "manifest.json"
    manifest = _read_mapping(manifest_path)
    selected_id = str(selected_id or manifest.get("artifact_id") or artifact_root.name)
    # Registry lifecycle is authoritative after promotion/supersession. An
    # immutable historical manifest legitimately retains the status it had at
    # creation, so preferring that field would silently reactivate a prior the
    # registry has superseded.
    lifecycle_status = str(
        entry.get("lifecycle_status")
        or manifest.get("lifecycle_status")
        or "legacy_unclassified"
    )
    is_legacy = lifecycle_status in LEGACY_STATUSES or lifecycle_status == "legacy_unclassified"
    if is_legacy and not allow_legacy:
        raise ValueError(
            f"Refusing legacy DeLaurier prior {selected_id!r} at {artifact_root}; "
            "use --allow-legacy-prior only for explicit historical reproduction"
        )
    if not is_legacy and lifecycle_status != "active":
        raise ValueError(f"Prior {selected_id!r} has unsupported lifecycle_status={lifecycle_status!r}")

    manifest_partitions = {
        _normal_partition(str(value)) for value in manifest.get("partitions", manifest.get("row_counts", {}).keys())
    }
    requested = {_normal_partition(value) for value in requested_partitions}
    missing = sorted(requested - manifest_partitions)
    if missing:
        raise ValueError(f"Prior {selected_id!r} is missing requested partitions: {missing}")
    if bool(manifest.get("test_partition_loaded", False)) and "test" not in requested:
        raise ValueError(f"Prior {selected_id!r} manifest reports test_partition_loaded=true")

    expected_commit = str(entry.get("physics_source_commit", ""))
    actual_commit = _manifest_commit(manifest)
    if expected_commit and actual_commit != expected_commit:
        raise ValueError(
            f"Prior {selected_id!r} physics commit mismatch: registry={expected_commit}, manifest={actual_commit}"
        )

    return PriorResolution(
        prior_id=selected_id,
        artifact_root=artifact_root,
        lifecycle_status=lifecycle_status,
        physics_source_commit=actual_commit,
        frame_contract=_contract_value(manifest, entry, "frame_contract"),
        airflow_contract=_contract_value(manifest, entry, "airflow_contract"),
        phase_contract=_contract_value(manifest, entry, "phase_contract"),
        required_partitions=tuple(sorted(manifest_partitions)),
        registry_path=resolved_registry_path,
        registry_entry=entry,
        manifest=manifest,
    )
