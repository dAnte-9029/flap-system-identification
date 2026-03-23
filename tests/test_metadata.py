from pathlib import Path

from system_identification.metadata import load_aircraft_metadata, metadata_has_complete_labels


def _metadata_path() -> Path:
    return Path(__file__).resolve().parents[1] / "metadata" / "aircraft" / "flapper_01" / "aircraft_metadata.yaml"


def test_load_aircraft_metadata_keeps_confirmed_and_placeholder_values():
    metadata_path = _metadata_path()

    metadata = load_aircraft_metadata(metadata_path)

    assert metadata["aircraft_id"] == "flapper_01"
    assert metadata["flapping_drive"]["encoder_to_drive_ratio"]["value"] == 7.5
    assert metadata["mass_properties"]["mass_kg"]["value"] is None
    assert metadata["flapping_drive"]["positive_wing_stroke_direction"] == "upstroke"


def test_metadata_has_complete_labels_false_for_placeholder_mass_and_inertia():
    metadata_path = _metadata_path()

    metadata = load_aircraft_metadata(metadata_path)

    assert metadata_has_complete_labels(metadata) is False
