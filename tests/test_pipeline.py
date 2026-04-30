from pathlib import Path

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


def test_assemble_canonical_samples_resamples_direct_airspeed_fields():
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
    topic_frames["airspeed_validated"] = pd.DataFrame(
        {
            "event_time_us": grid_us,
            "indicated_airspeed_m_s": np.array([4.0, 5.0, 6.0]),
            "calibrated_airspeed_m_s": np.array([4.1, 5.1, 6.1]),
            "true_airspeed_m_s": np.array([4.4, 5.4, 6.4]),
            "calibrated_ground_minus_wind_m_s": np.array([4.2, 5.2, 6.2]),
            "true_ground_minus_wind_m_s": np.array([4.5, 5.5, 6.5]),
            "airspeed_derivative_filtered": np.array([0.1, 0.2, 0.3]),
            "throttle_filtered": np.array([0.45, 0.5, 0.55]),
            "pitch_filtered": np.array([0.01, 0.02, 0.03]),
        }
    )

    samples = assemble_canonical_samples(grid_us=grid_us, topic_frames=topic_frames, metadata=metadata)

    expected_columns = [
        "airspeed_validated.calibrated_ground_minus_wind_m_s",
        "airspeed_validated.true_ground_minus_wind_m_s",
        "airspeed_validated.airspeed_derivative_filtered",
        "airspeed_validated.throttle_filtered",
        "airspeed_validated.pitch_filtered",
    ]
    assert set(expected_columns).issubset(samples.columns)
    np.testing.assert_allclose(
        samples["airspeed_validated.true_ground_minus_wind_m_s"].to_numpy(),
        np.array([4.5, 5.5, 6.5]),
    )
    np.testing.assert_allclose(
        samples["airspeed_validated.pitch_filtered"].to_numpy(),
        np.array([0.01, 0.02, 0.03]),
    )
    assert samples["airspeed_validated.pitch_filtered_valid"].all()


def test_assemble_canonical_samples_emits_nan_for_missing_direct_airspeed_fields():
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
    topic_frames["airspeed_validated"] = pd.DataFrame(
        {
            "event_time_us": grid_us,
            "indicated_airspeed_m_s": np.array([4.0, 5.0, 6.0]),
            "calibrated_airspeed_m_s": np.array([4.1, 5.1, 6.1]),
            "true_airspeed_m_s": np.array([4.4, 5.4, 6.4]),
            "calibrated_ground_minus_wind_m_s": np.array([4.2, 5.2, 6.2]),
            "pitch_filtered": np.array([0.01, 0.02, 0.03]),
        }
    )

    samples = assemble_canonical_samples(grid_us=grid_us, topic_frames=topic_frames, metadata=metadata)

    assert "airspeed_validated.true_ground_minus_wind_m_s" in samples.columns
    assert samples["airspeed_validated.true_ground_minus_wind_m_s"].isna().all()
    assert not samples["airspeed_validated.true_ground_minus_wind_m_s_valid"].any()


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


def test_compute_smoothed_kinematic_derivatives_by_log():
    from system_identification.pipeline import compute_smoothed_kinematic_derivatives

    time_s = np.arange(21, dtype=float) * 0.1
    samples = pd.DataFrame(
        {
            "log_id": ["a"] * len(time_s),
            "time_s": time_s,
            "vehicle_local_position.vx": 2.0 * time_s + 0.15 * np.sin(50.0 * time_s),
            "vehicle_local_position.vy": -3.0 * time_s,
            "vehicle_local_position.vz": 0.5 * time_s,
            "vehicle_angular_velocity.xyz[0]": 4.0 * time_s + 0.2 * np.sin(45.0 * time_s),
            "vehicle_angular_velocity.xyz[1]": -2.0 * time_s,
            "vehicle_angular_velocity.xyz[2]": 0.25 * time_s,
        }
    )

    derivatives = compute_smoothed_kinematic_derivatives(samples, window_s=0.5, polyorder=1)

    assert list(derivatives.columns) == [
        "vehicle_local_position.ax_smooth",
        "vehicle_local_position.ay_smooth",
        "vehicle_local_position.az_smooth",
        "vehicle_angular_velocity.xyz_derivative_smooth[0]",
        "vehicle_angular_velocity.xyz_derivative_smooth[1]",
        "vehicle_angular_velocity.xyz_derivative_smooth[2]",
    ]
    np.testing.assert_allclose(
        derivatives["vehicle_local_position.ay_smooth"].iloc[3:-3].to_numpy(),
        -3.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        derivatives["vehicle_angular_velocity.xyz_derivative_smooth[1]"].iloc[3:-3].to_numpy(),
        -2.0,
        atol=1e-6,
    )
    assert np.nanstd(derivatives["vehicle_angular_velocity.xyz_derivative_smooth[0]"]) < 2.5


def test_compute_effective_wrench_labels_accepts_replacement_derivatives():
    from system_identification.pipeline import _compute_effective_wrench_labels

    metadata = {
        "mass_properties": {
            "mass_kg": {"value": 1.0},
            "cg_b_m": {"value": [0.0, 0.0, 0.0]},
            "inertia_b_kg_m2": {"value": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]},
        },
    }
    samples = pd.DataFrame(
        {
            "vehicle_local_position.ax": [0.0, 0.0],
            "vehicle_local_position.ay": [0.0, 0.0],
            "vehicle_local_position.az": [9.81, 9.81],
            "vehicle_attitude.q[0]": [1.0, 1.0],
            "vehicle_attitude.q[1]": [0.0, 0.0],
            "vehicle_attitude.q[2]": [0.0, 0.0],
            "vehicle_attitude.q[3]": [0.0, 0.0],
            "vehicle_angular_velocity.xyz[0]": [0.0, 0.0],
            "vehicle_angular_velocity.xyz[1]": [0.0, 0.0],
            "vehicle_angular_velocity.xyz[2]": [0.0, 0.0],
            "vehicle_angular_velocity.xyz_derivative[0]": [10.0, 10.0],
            "vehicle_angular_velocity.xyz_derivative[1]": [20.0, 20.0],
            "vehicle_angular_velocity.xyz_derivative[2]": [30.0, 30.0],
            "vehicle_angular_velocity.xyz_derivative_smooth[0]": [1.0, 1.0],
            "vehicle_angular_velocity.xyz_derivative_smooth[1]": [2.0, 2.0],
            "vehicle_angular_velocity.xyz_derivative_smooth[2]": [3.0, 3.0],
        }
    )

    _, moment_b, label_valid = _compute_effective_wrench_labels(
        samples,
        metadata,
        angular_acceleration_columns=[
            "vehicle_angular_velocity.xyz_derivative_smooth[0]",
            "vehicle_angular_velocity.xyz_derivative_smooth[1]",
            "vehicle_angular_velocity.xyz_derivative_smooth[2]",
        ],
    )

    np.testing.assert_allclose(moment_b, np.array([[1.0, 4.0, 9.0], [1.0, 4.0, 9.0]]))
    assert label_valid.all()


def test_build_smoothed_label_split_rewrites_labels_and_preserves_split(tmp_path: Path):
    from scripts.build_smoothed_label_split import build_smoothed_label_split

    split_root = tmp_path / "split"
    output_root = tmp_path / "smooth_split"
    split_root.mkdir()
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(
        """
mass_properties:
  mass_kg:
    value: 1.0
  inertia_b_kg_m2:
    value:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
      - [0.0, 0.0, 1.0]
label_definition:
  gravity_m_s2: 9.81
""",
        encoding="utf-8",
    )

    time_s = np.arange(9, dtype=float) * 0.1
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": ["log_a"] * len(time_s),
            "split": ["train"] * len(time_s),
            "vehicle_local_position.ax": np.zeros(len(time_s)),
            "vehicle_local_position.ay": np.zeros(len(time_s)),
            "vehicle_local_position.az": np.full(len(time_s), 9.81),
            "vehicle_local_position.vx": 0.0 * time_s,
            "vehicle_local_position.vy": 0.0 * time_s,
            "vehicle_local_position.vz": 0.0 * time_s,
            "vehicle_attitude.q[0]": np.ones(len(time_s)),
            "vehicle_attitude.q[1]": np.zeros(len(time_s)),
            "vehicle_attitude.q[2]": np.zeros(len(time_s)),
            "vehicle_attitude.q[3]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[0]": 2.0 * time_s,
            "vehicle_angular_velocity.xyz[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[2]": -3.0 * time_s,
            "vehicle_angular_velocity.xyz_derivative[0]": np.full(len(time_s), 99.0),
            "vehicle_angular_velocity.xyz_derivative[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[2]": np.full(len(time_s), 99.0),
            "fx_b": np.zeros(len(time_s)),
            "fy_b": np.zeros(len(time_s)),
            "fz_b": np.zeros(len(time_s)),
            "mx_b": np.full(len(time_s), 99.0),
            "my_b": np.zeros(len(time_s)),
            "mz_b": np.full(len(time_s), 99.0),
            "label_valid": np.ones(len(time_s), dtype=bool),
        }
    )

    for split in ["train", "val", "test"]:
        split_frame = frame.copy()
        split_frame["split"] = split
        split_frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)
        pd.DataFrame({"log_id": ["log_a"], "split": [split], "valid_sample_count": [len(frame)]}).to_csv(
            split_root / f"{split}_logs.csv",
            index=False,
        )
    pd.DataFrame({"log_id": ["log_a"], "split": ["train"], "valid_sample_count": [len(frame)]}).to_csv(
        split_root / "all_logs.csv",
        index=False,
    )

    outputs = build_smoothed_label_split(
        split_root=split_root,
        metadata_path=metadata_path,
        output_root=output_root,
        window_s=0.5,
        polyorder=1,
    )

    rewritten = pd.read_parquet(output_root / "train_samples.parquet")
    assert Path(outputs["manifest_path"]).exists()
    assert rewritten["split"].eq("train").all()
    assert not np.allclose(rewritten["mx_b"].to_numpy(), 99.0)
    assert not np.allclose(rewritten["mz_b"].to_numpy(), 99.0)
    assert rewritten["label_valid"].all()


def test_build_smoothed_label_split_can_keep_raw_force_labels(tmp_path: Path):
    from scripts.build_smoothed_label_split import build_smoothed_label_split

    split_root = tmp_path / "split"
    output_root = tmp_path / "raw_force_smooth_moment"
    split_root.mkdir()
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(
        """
mass_properties:
  mass_kg:
    value: 1.0
  inertia_b_kg_m2:
    value:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
      - [0.0, 0.0, 1.0]
label_definition:
  gravity_m_s2: 9.81
""",
        encoding="utf-8",
    )

    time_s = np.arange(9, dtype=float) * 0.1
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": ["log_a"] * len(time_s),
            "split": ["train"] * len(time_s),
            "vehicle_local_position.ax": np.full(len(time_s), 5.0),
            "vehicle_local_position.ay": np.zeros(len(time_s)),
            "vehicle_local_position.az": np.full(len(time_s), 9.81),
            "vehicle_local_position.vx": np.zeros(len(time_s)),
            "vehicle_local_position.vy": np.zeros(len(time_s)),
            "vehicle_local_position.vz": np.zeros(len(time_s)),
            "vehicle_attitude.q[0]": np.ones(len(time_s)),
            "vehicle_attitude.q[1]": np.zeros(len(time_s)),
            "vehicle_attitude.q[2]": np.zeros(len(time_s)),
            "vehicle_attitude.q[3]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[0]": 2.0 * time_s,
            "vehicle_angular_velocity.xyz[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[2]": -3.0 * time_s,
            "vehicle_angular_velocity.xyz_derivative[0]": np.full(len(time_s), 99.0),
            "vehicle_angular_velocity.xyz_derivative[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[2]": np.full(len(time_s), 99.0),
            "fx_b": np.zeros(len(time_s)),
            "fy_b": np.zeros(len(time_s)),
            "fz_b": np.zeros(len(time_s)),
            "mx_b": np.full(len(time_s), 99.0),
            "my_b": np.zeros(len(time_s)),
            "mz_b": np.full(len(time_s), 99.0),
            "label_valid": np.ones(len(time_s), dtype=bool),
        }
    )

    for split in ["train", "val", "test"]:
        split_frame = frame.copy()
        split_frame["split"] = split
        split_frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    build_smoothed_label_split(
        split_root=split_root,
        metadata_path=metadata_path,
        output_root=output_root,
        window_s=0.5,
        polyorder=1,
        force_label_source="raw",
        moment_label_source="smooth",
    )

    rewritten = pd.read_parquet(output_root / "train_samples.parquet")
    np.testing.assert_allclose(rewritten["fx_b"].to_numpy(), 5.0)
    assert not np.allclose(rewritten["mx_b"].to_numpy(), 99.0)


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
