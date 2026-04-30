from pathlib import Path

from system_identification.metadata import load_aircraft_metadata, metadata_has_complete_labels


def _metadata_path() -> Path:
    return Path(__file__).resolve().parents[1] / "metadata" / "aircraft" / "flapper_01" / "aircraft_metadata.yaml"


def test_load_aircraft_metadata_keeps_confirmed_and_measured_values():
    metadata_path = _metadata_path()

    metadata = load_aircraft_metadata(metadata_path)

    assert metadata["aircraft_id"] == "flapper_01"
    assert metadata["flapping_drive"]["encoder_to_drive_ratio"]["value"] == 7.5
    assert metadata["mass_properties"]["mass_kg"]["value"] == 0.95
    assert metadata["mass_properties"]["cg_b_m"]["value"] == [0.027, 0.0, -0.025]
    assert metadata["flapping_drive"]["positive_wing_stroke_direction"] == "upstroke"


def test_metadata_has_complete_labels_true_for_seed_mass_and_inertia():
    metadata_path = _metadata_path()

    metadata = load_aircraft_metadata(metadata_path)

    assert metadata_has_complete_labels(metadata) is True
