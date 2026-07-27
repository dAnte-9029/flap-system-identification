"""Immutable directory I/O for static correction model bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from system_identification.artifacts.io import write_json
from system_identification.models.correction.bundles import (
    StaticCorrectionBundle,
    _solution_payload,
    compute_bundle_hash,
)
from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.models.correction.static_models import RidgeSolution


REQUIRED_FILES = frozenset(
    {
        "bundle_manifest.json",
        "model_spec.json",
        "mean_coefficients.json",
        "waveform_coefficients.json",
        "feature_schema.json",
        "normalization.json",
        "training_provenance.json",
        "fit_diagnostics.json",
    }
)


def _manifest(bundle: StaticCorrectionBundle) -> dict[str, object]:
    spec = bundle.spec
    provenance = bundle.training_provenance
    feature_names = {
        "mean": list(bundle.mean_solution.feature_names) if bundle.mean_solution else [],
        "waveform": list(bundle.waveform_solution.feature_names) if bundle.waveform_solution else [],
    }
    return {
        "bundle_schema_version": bundle.bundle_schema_version,
        "model_id": bundle.model_id,
        "created_at": bundle.created_at,
        "bundle_hash": bundle.bundle_hash,
        "status": bundle.status,
        "git_commit": provenance.get("git_commit", "not_recorded"),
        "git_dirty": provenance.get("git_dirty", "not_recorded"),
        "correction_ready_artifact_id": provenance["correction_ready_artifact_id"],
        "correction_ready_manifest_hash": provenance["correction_ready_manifest_hash"],
        "dataset_id": provenance["dataset_id"],
        "dataset_hash": provenance["dataset_hash"],
        "prior_id": provenance["prior_id"],
        "prior_hash": provenance["prior_hash"],
        "ratio_contract": provenance["ratio_contract"],
        "phase_contract": provenance["phase_contract"],
        "included_partitions": provenance["included_partitions"],
        "model_type": spec.model_type,
        "force_component": spec.force_component,
        "harmonic_order": spec.harmonic_order,
        "condition_set": spec.condition_set,
        "mean_condition_set": spec.mean_condition_set,
        "waveform_condition_set": spec.waveform_condition_set,
        "mean_prior_retention": spec.mean_prior_retention,
        "waveform_prior_retention": spec.waveform_prior_retention,
        "ridge_lambda_mean": spec.ridge_lambda_mean,
        "ridge_lambda_waveform": spec.ridge_lambda_waveform,
        "mean_weighting": spec.mean_weighting,
        "waveform_weighting": spec.waveform_weighting,
        "physical_component": spec.physical_component,
        "coefficient_constraints": (
            dict(spec.coefficient_constraints) if spec.coefficient_constraints is not None else None
        ),
        "component_scale": bundle.component_scale,
        "feature_names": feature_names,
        "mean_fit_diagnostics": (
            bundle.mean_solution.diagnostics.to_dict() if bundle.mean_solution else None
        ),
        "waveform_fit_diagnostics": (
            bundle.waveform_solution.diagnostics.to_dict() if bundle.waveform_solution else None
        ),
        **dict(bundle.fit_summary),
    }


def save_static_bundle(bundle: StaticCorrectionBundle, path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite static model bundle: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "bundle_manifest.json", _manifest(bundle))
    write_json(destination / "model_spec.json", bundle.spec.to_dict())
    write_json(destination / "mean_coefficients.json", _solution_payload(bundle.mean_solution))
    waveform_payload = _solution_payload(bundle.waveform_solution)
    waveform_payload["component_scale"] = bundle.component_scale
    waveform_payload["coefficient_constraints"] = (
        dict(bundle.spec.coefficient_constraints) if bundle.spec.coefficient_constraints is not None else None
    )
    write_json(destination / "waveform_coefficients.json", waveform_payload)
    write_json(
        destination / "feature_schema.json",
        {
            "mean_feature_names": list(bundle.mean_solution.feature_names) if bundle.mean_solution else [],
            "waveform_feature_names": list(bundle.waveform_solution.feature_names) if bundle.waveform_solution else [],
            "feature_order_contract": "exact_order_required_extra_input_columns_ignored",
        },
    )
    write_json(destination / "normalization.json", dict(bundle.normalization))
    write_json(destination / "training_provenance.json", dict(bundle.training_provenance))
    write_json(
        destination / "fit_diagnostics.json",
        {
            "mean": bundle.mean_solution.diagnostics.to_dict() if bundle.mean_solution else None,
            "waveform": bundle.waveform_solution.diagnostics.to_dict() if bundle.waveform_solution else None,
            **dict(bundle.fit_summary),
        },
    )
    return destination


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping in {path}")
    return value


def _solution(value: Mapping[str, object]) -> RidgeSolution | None:
    if not value.get("feature_names"):
        return None
    return RidgeSolution.from_dict(value)


def load_static_bundle(path: str | Path) -> StaticCorrectionBundle:
    root = Path(path)
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"Static model bundle is incomplete; missing={missing}")
    manifest = _read_json(root / "bundle_manifest.json")
    raw_spec = _read_json(root / "model_spec.json")
    spec = StaticCorrectionSpec.from_dict(raw_spec)
    mean = _solution(_read_json(root / "mean_coefficients.json"))
    waveform_payload = _read_json(root / "waveform_coefficients.json")
    waveform = _solution(waveform_payload)
    provenance = _read_json(root / "training_provenance.json")
    normalization = _read_json(root / "normalization.json")
    diagnostics = _read_json(root / "fit_diagnostics.json")
    fit_summary = {
        key: diagnostics[key]
        for key in (
            "train_cycle_count",
            "train_waveform_row_count",
            "coefficient_count",
            "finite_checks",
            "selection_performed",
        )
    }
    bundle = StaticCorrectionBundle(
        bundle_schema_version=str(manifest["bundle_schema_version"]),
        model_id=str(manifest["model_id"]),
        created_at=str(manifest["created_at"]),
        status=str(manifest["status"]),
        spec=spec,
        mean_solution=mean,
        waveform_solution=waveform,
        component_scale=(
            None if waveform_payload.get("component_scale") is None else float(waveform_payload["component_scale"])
        ),
        normalization=normalization,
        training_provenance=provenance,
        fit_summary=fit_summary,
        bundle_hash=str(manifest["bundle_hash"]),
    )
    actual_hash = compute_bundle_hash(bundle.hash_payload())
    if actual_hash != bundle.bundle_hash:
        legacy_payload = bundle.hash_payload()
        if "mean_condition_set" not in raw_spec and "waveform_condition_set" not in raw_spec:
            legacy_spec = dict(legacy_payload["spec"])
            legacy_spec.pop("mean_condition_set", None)
            legacy_spec.pop("waveform_condition_set", None)
            legacy_payload["spec"] = legacy_spec
        legacy_hash = compute_bundle_hash(legacy_payload)
        if legacy_hash != bundle.bundle_hash:
            raise ValueError(f"Static model bundle hash mismatch: expected={bundle.bundle_hash}, actual={actual_hash}")
    return bundle
