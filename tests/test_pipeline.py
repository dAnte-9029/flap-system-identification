import numpy as np
import pandas as pd

from system_identification.pipeline import assemble_canonical_samples


def _base_topic_frames(grid_us: np.ndarray) -> dict[str, pd.DataFrame]:
    return {
        "encoder_count": pd.DataFrame(
            {
                "event_time_us": grid_us,
                "total_count": np.array([0, 512, 1024])[: len(grid_us)],
                "position_raw": np.array([0, 512, 1024])[: len(grid_us)],
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
                "control[0]": np.linspace(0.0, 0.1 * (len(grid_us) - 1), len(grid_us)),
                "control[1]": -np.linspace(0.0, 0.1 * (len(grid_us) - 1), len(grid_us)),
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
    assert "wing_stroke_angle_rad" in samples.columns
    assert "motor_cmd_0" in samples.columns
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
