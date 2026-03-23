from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_aircraft_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def nested_value(metadata: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = metadata

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    if isinstance(current, dict) and "value" in current:
        return current.get("value", default)

    return current


def _matrix_has_all_values(matrix: Any) -> bool:
    if not isinstance(matrix, list) or not matrix:
        return False

    for row in matrix:
        if not isinstance(row, list) or not row:
            return False
        for item in row:
            if item is None:
                return False
    return True


def metadata_has_complete_labels(metadata: dict[str, Any]) -> bool:
    mass = nested_value(metadata, "mass_properties", "mass_kg")
    inertia = nested_value(metadata, "mass_properties", "inertia_b_kg_m2")

    return mass is not None and _matrix_has_all_values(inertia)


def metadata_open_warnings(metadata: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    if nested_value(metadata, "mass_properties", "mass_kg") is None:
        warnings.append("mass_kg is placeholder; force labels will be NaN")

    if not _matrix_has_all_values(nested_value(metadata, "mass_properties", "inertia_b_kg_m2")):
        warnings.append("inertia_b_kg_m2 is placeholder; moment labels will be NaN")

    if nested_value(metadata, "flapping_drive", "encoder_to_drive_sign") is None:
        warnings.append("encoder_to_drive_sign missing; defaulting to +1.0")

    if nested_value(metadata, "flapping_drive", "drive_phase_zero_offset_rad") is None:
        warnings.append("drive_phase_zero_offset_rad missing; defaulting to 0.0")

    return warnings
