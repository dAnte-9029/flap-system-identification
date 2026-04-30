import numpy as np
import pandas as pd

from system_identification.pipeline import assemble_canonical_samples


def _base_topic_frames(grid_us: np.ndarray) -> dict[str, pd.DataFrame]:
    counts = 512 * np.arange(len(grid_us), dtype=float)
    servo_span = 0.1 * np.arange(len(grid_us), dtype=float)
    return {
        "encoder_count": pd.DataFrame(
            {
                "event_time_us": grid_us,
                "total_count": counts,
                "position_raw": counts,
            }
        ),
        "rpm": pd.DataFrame(
            {
                "event_time_us": grid_us,
                "rpm_raw": np.full(len(grid_us), 100.0),
                "rpm_estimate": np.full(len(grid_us), 95.0),
            }
        ),
        "flap_frequency": pd.DataFrame(
            {
                "event_time_us": grid_us,
                "frequency_hz": np.full(len(grid_us), 2.0),
            }
        ),
        "actuator_motors": pd.DataFrame(
            {
                "event_time_us": grid_us + 1_000,
                "control[0]": np.linspace(0.1, 0.1 * len(grid_us), len(grid_us)),
            }
        ),
        "actuator_servos": pd.DataFrame(
            {
                "event_time_us": grid_us + 1_000,
                "control[0]": servo_span,
                "control[1]": -servo_span,
                "control[2]": np.full(len(grid_us), 0.2),
            }
        ),
    }


def test_assemble_canonical_samples_emits_phase_and_nan_labels_when_metadata_incomplete():
    grid_us = np.array([0, 10_000, 20_000], dtype=np.int64)
    metadata = {
        "mass_properties": {
            "mass_kg": {"value": None},
            "cg_b_m": {"value": [None, None, None]},
            "inertia_b_kg_m2": {"value": [[None, None, None], [None, None, None], [None, None, None]]},
        },
        "flapping_drive": {
            "encoder_counts_per_rev": 4096,
            "encoder_to_drive_ratio": {"value": 7.5},
            "encoder_to_drive_sign": 1.0,
            "drive_phase_zero_offset_rad": 0.0,
            "wing_stroke_amplitude_rad": {"value": float(np.deg2rad(30.0))},
            "wing_stroke_phase_offset_rad": {"value": 0.0},
        },
    }
    topic_frames = _base_topic_frames(grid_us)

    samples = assemble_canonical_samples(grid_us=grid_us, topic_frames=topic_frames, metadata=metadata)

    assert "drive_phase_rad" in samples.columns
    assert "phase_source" in samples.columns
    assert "phase_corrected_rad" in samples.columns
    assert "phase_corrected_unwrapped_rad" in samples.columns
    assert "cycle_id" in samples.columns
    assert "cycle_valid" in samples.columns
    assert "wing_stroke_angle_rad" in samples.columns
    assert "motor_cmd_0" in samples.columns
    assert samples["phase_source"].eq("encoder_count_fallback").all()
    assert samples["fx_b"].isna().all()
    assert samples["mx_b"].isna().all()


def test_assemble_canonical_samples_computes_effective_wrench_labels_from_complete_metadata():
    grid_us = np.array([0, 10_000], dtype=np.int64)
    metadata = {
        "mass_properties": {
            "mass_kg": {"value": 2.0},
            "cg_b_m": {"value": [0.0, 0.0, 0.0]},
            "inertia_b_kg_m2": {"value": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]},
        },
        "flapping_drive": {
            "encoder_counts_per_rev": 4096,
            "encoder_to_drive_ratio": {"value": 7.5},
            "encoder_to_drive_sign": 1.0,
            "drive_phase_zero_offset_rad": 0.0,
            "wing_stroke_amplitude_rad": {"value": float(np.deg2rad(30.0))},
            "wing_stroke_phase_offset_rad": {"value": 0.0},
        },
    }
    topic_frames = _base_topic_frames(grid_us)
    topic_frames["vehicle_local_position"] = pd.DataFrame(
        {
            "event_time_us": grid_us,
            "x": np.zeros(len(grid_us)),
            "y": np.zeros(len(grid_us)),
            "z": np.zeros(len(grid_us)),
            "vx": np.zeros(len(grid_us)),
            "vy": np.zeros(len(grid_us)),
            "vz": np.zeros(len(grid_us)),
            "ax": np.array([2.0, 0.0]),
            "ay": np.array([0.0, 2.0]),
            "az": np.array([9.81, 9.81]),
            "heading": np.zeros(len(grid_us)),
        }
    )
    topic_frames["vehicle_attitude"] = pd.DataFrame(
        {
            "event_time_us": grid_us,
            "q[0]": np.array([1.0, np.sqrt(0.5)]),
            "q[1]": np.array([0.0, 0.0]),
            "q[2]": np.array([0.0, 0.0]),
            "q[3]": np.array([0.0, np.sqrt(0.5)]),
        }
    )
    topic_frames["vehicle_angular_velocity"] = pd.DataFrame(
        {
            "event_time_us": grid_us,
            "xyz[0]": np.array([1.0, 1.0]),
            "xyz[1]": np.array([0.0, 0.0]),
            "xyz[2]": np.array([1.0, 1.0]),
            "xyz_derivative[0]": np.array([0.1, 0.1]),
            "xyz_derivative[1]": np.array([0.2, 0.2]),
            "xyz_derivative[2]": np.array([0.3, 0.3]),
        }
    )

    samples = assemble_canonical_samples(grid_us=grid_us, topic_frames=topic_frames, metadata=metadata)

    np.testing.assert_allclose(samples["fx_b"].to_numpy(), np.array([4.0, 4.0]), atol=1e-6)
    np.testing.assert_allclose(samples["fy_b"].to_numpy(), np.array([0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(samples["fz_b"].to_numpy(), np.array([0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(samples["mx_b"].to_numpy(), np.array([0.1, 0.1]), atol=1e-6)
    np.testing.assert_allclose(samples["my_b"].to_numpy(), np.array([-1.6, -1.6]), atol=1e-6)
    np.testing.assert_allclose(samples["mz_b"].to_numpy(), np.array([0.9, 0.9]), atol=1e-6)
    assert samples["label_valid"].all()


def test_assemble_canonical_samples_prefers_wing_phase_and_emits_corrected_cycle_annotations():
    grid_us = np.arange(11, dtype=np.int64) * 20_000
    metadata = {
        "mass_properties": {
            "mass_kg": {"value": None},
            "cg_b_m": {"value": [None, None, None]},
            "inertia_b_kg_m2": {"value": [[None, None, None], [None, None, None], [None, None, None]]},
        },
        "flapping_drive": {
            "encoder_counts_per_rev": 4096,
            "encoder_to_drive_ratio": {"value": 7.5},
            "encoder_to_drive_sign": 1.0,
            "drive_phase_zero_offset_rad": 0.0,
            "wing_stroke_amplitude_rad": {"value": float(np.deg2rad(30.0))},
            "wing_stroke_phase_offset_rad": {"value": 0.0},
        },
    }
    topic_frames = _base_topic_frames(grid_us)
    topic_frames["flap_frequency"] = pd.DataFrame(
        {
            "event_time_us": grid_us,
            "frequency_hz": np.array([0.0] + [4.8] * 10),
        }
    )
    topic_frames["wing_phase"] = pd.DataFrame(
        {
            "event_time_us": grid_us,
            "phase_rad": np.array([0.0, 0.2, 3.0, 6.6, 0.1, 2.1, 4.2, 6.4, 0.15, 3.2, 6.5]),
            "phase_unwrapped_rad": np.array([0.0, 0.2, 3.0, 6.6, 0.1, 2.1, 4.2, 6.4, 0.15, 3.2, 6.5]),
            "phase_sin": np.sin(np.array([0.0, 0.2, 3.0, 6.6, 0.1, 2.1, 4.2, 6.4, 0.15, 3.2, 6.5])),
            "phase_cos": np.cos(np.array([0.0, 0.2, 3.0, 6.6, 0.1, 2.1, 4.2, 6.4, 0.15, 3.2, 6.5])),
            "flap_frequency_hz": np.array([0.0] + [4.8] * 10),
            "encoder_position_raw": np.arange(11),
            "encoder_total_count": np.arange(11),
            "phase_valid": np.array([1] * 11),
        }
    )

    samples = assemble_canonical_samples(grid_us=grid_us, topic_frames=topic_frames, metadata=metadata)

    assert samples["phase_source"].eq("wing_phase").all()
    assert samples["cycle_id"].tolist() == [-1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    assert samples["cycle_valid"].tolist() == [False, False, False, False, True, True, True, True, False, False, False]
    expected_corrected = 2.0 * np.pi * np.array([0.0, 2.0, 4.1, 6.3]) / 6.3
    np.testing.assert_allclose(samples.loc[4:7, "phase_corrected_rad"].to_numpy(), expected_corrected, atol=1e-6)
    np.testing.assert_allclose(
        samples.loc[4:7, "phase_corrected_unwrapped_rad"].to_numpy(),
        2.0 * np.pi + expected_corrected,
        atol=1e-6,
    )
