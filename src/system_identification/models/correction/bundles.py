"""Immutable in-memory schema for fitted static correction candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.models.correction.static_models import RidgeSolution


ALLOWED_BUNDLE_STATUSES = frozenset({"candidate", "smoke_test", "selected_static_train_only"})


def _solution_payload(solution: RidgeSolution | None) -> dict[str, object]:
    return solution.to_dict() if solution is not None else {
        "coefficients": [],
        "feature_names": [],
        "ridge_lambda": 0.0,
        "penalize_mask": [],
        "intercept_penalty": "not_applicable",
        "diagnostics": None,
    }


@dataclass(frozen=True)
class StaticCorrectionBundle:
    bundle_schema_version: str
    model_id: str
    created_at: str
    status: str
    spec: StaticCorrectionSpec
    mean_solution: RidgeSolution | None
    waveform_solution: RidgeSolution | None
    component_scale: float | None
    normalization: Mapping[str, object]
    training_provenance: Mapping[str, object]
    fit_summary: Mapping[str, object]
    bundle_hash: str

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_BUNDLE_STATUSES:
            raise ValueError(f"Bundle status must be one of {sorted(ALLOWED_BUNDLE_STATUSES)}")
        if self.training_provenance.get("included_partitions") != ["train"]:
            raise ValueError("Static correction bundles must contain train-only provenance")

    def hash_payload(self) -> dict[str, object]:
        return {
            "bundle_schema_version": self.bundle_schema_version,
            "model_id": self.model_id,
            "status": self.status,
            "spec": self.spec.to_dict(),
            "mean_solution": _solution_payload(self.mean_solution),
            "waveform_solution": _solution_payload(self.waveform_solution),
            "component_scale": self.component_scale,
            "normalization": dict(self.normalization),
            "training_provenance": dict(self.training_provenance),
            "fit_summary": dict(self.fit_summary),
        }


def compute_bundle_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
