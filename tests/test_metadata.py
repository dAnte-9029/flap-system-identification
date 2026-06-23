from pathlib import Path

from system_identification.metadata import load_aircraft_metadata, metadata_has_complete_labels


def _metadata_path() -> Path:
    return Path(__file__).resolve().parents[1] / "metadata" / "aircraft" / "flapper_01" / "aircraft_metadata.yaml"


def test_load_aircraft_metadata_keeps_confirmed_and_measured_values():
    metadata_path = _metadata_path()

    metadata = load_aircraft_metadata(metadata_path)

    assert metadata["aircraft_id"] == "flapper_01"
    assert metadata["flapping_drive"]["encoder_to_drive_ratio"]["value"] == 8.0
    assert metadata["flapping_drive"]["encoder_to_drive_sign"]["value"] == 1.0
    assert metadata["flapping_drive"]["drive_phase_zero_offset_rad"]["value"] == 0.0
    assert metadata["mass_properties"]["mass_kg"]["value"] == 0.90415
    assert metadata["mass_properties"]["cg_b_m"]["value"] == [-0.12154, 0.00541, -0.04298]
    assert metadata["mass_properties"]["inertia_b_kg_m2"]["value"] == [
        [0.02329, 0.0, 0.0],
        [0.0, 0.02573, 0.0],
        [0.0, 0.0, 0.04270],
    ]
    assert metadata["flapping_drive"]["positive_wing_stroke_direction"] == "upstroke"
    open_item_fields = {item["field"] for item in metadata.get("open_items", [])}
    assert "flapping_drive.encoder_to_drive_sign" not in open_item_fields
    assert "flapping_drive.drive_phase_zero_offset_rad" not in open_item_fields


def test_metadata_has_complete_labels_true_for_seed_mass_and_inertia():
    metadata_path = _metadata_path()

    metadata = load_aircraft_metadata(metadata_path)

    assert metadata_has_complete_labels(metadata) is True
