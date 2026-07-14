from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_level2_replay_ready_table import _pivot_predictions, roll_from_quaternion
from scripts.evaluate_level2_translational_replay import (
    evaluate_level2_replay,
    integrate_force_window,
    local_along_track_axis,
    select_candidate_windows,
)


def _quat_identity_rows(n: int) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0] for _ in range(n)]


def _synthetic_replay_frame(n: int = 11, force_x: float = 2.0, force_z: float = -9.81) -> pd.DataFrame:
    dt = 0.1
    time = np.arange(n, dtype=float) * dt
    mass = 2.0
    ax = force_x / mass
    az = force_z / mass + 9.81
    data = {
        "outer_fold": np.zeros(n, dtype=int),
        "log_id": ["log_a"] * n,
        "segment_id": np.ones(n, dtype=int),
        "time_s": time,
        "vehicle_local_position.x": 5.0 * time + 0.5 * ax * time**2,
        "vehicle_local_position.y": np.zeros(n),
        "vehicle_local_position.z": 0.5 * az * time**2,
        "vehicle_local_position.vx": 5.0 + ax * time,
        "vehicle_local_position.vy": np.zeros(n),
        "vehicle_local_position.vz": az * time,
        "vehicle_angular_velocity.xyz[0]": np.zeros(n),
        "vehicle_angular_velocity.xyz[1]": np.zeros(n),
        "vehicle_angular_velocity.xyz[2]": np.zeros(n),
        "roll_rad": np.zeros(n),
        "label_fx_b": np.full(n, force_x),
        "label_fz_b": np.full(n, force_z),
        "raw_prior_fx_b": np.full(n, 0.0),
        "raw_prior_fz_b": np.full(n, force_z),
        "gain_bias_fx_b": np.full(n, force_x),
        "gain_bias_fz_b": np.full(n, force_z),
        "pure_tcn_fx_b": np.full(n, force_x * 0.5),
        "pure_tcn_fz_b": np.full(n, force_z),
    }
    quat = np.asarray(_quat_identity_rows(n))
    for idx, column in enumerate(
        ["vehicle_attitude.q[0]", "vehicle_attitude.q[1]", "vehicle_attitude.q[2]", "vehicle_attitude.q[3]"]
    ):
        data[column] = quat[:, idx]
    return pd.DataFrame(data)


def test_roll_from_quaternion_identity_is_zero():
    q = np.asarray(_quat_identity_rows(3), dtype=float)
    assert np.allclose(roll_from_quaternion(q), 0.0)


def test_pivot_predictions_requires_all_models():
    rows = []
    for model in ["Raw prior", "Conditioned gain-bias", "Pure TCN"]:
        rows.append(
            {
                "outer_fold": 0,
                "log_id": "log_a",
                "segment_id": 1,
                "time_s": 0.0,
                "model": model,
                "label_fx_b": 1.0,
                "label_fz_b": -2.0,
                "pred_fx_b": 3.0,
                "pred_fz_b": -4.0,
            }
        )
    table = _pivot_predictions(
        pd.DataFrame(rows),
        {"Raw prior": "raw_prior", "Conditioned gain-bias": "gain_bias", "Pure TCN": "pure_tcn"},
    )
    assert set(["raw_prior_fx_b", "gain_bias_fx_b", "pure_tcn_fx_b"]).issubset(table.columns)
    with pytest.raises(ValueError, match="missing required models"):
        _pivot_predictions(pd.DataFrame(rows[:2]), {"Raw prior": "raw_prior", "Pure TCN": "pure_tcn"})


def test_local_along_track_axis_uses_horizontal_velocity():
    axis = local_along_track_axis(np.array([3.0, 4.0, 2.0]), min_speed_m_s=1.0)
    assert np.allclose(axis, [0.6, 0.8, 0.0])
    assert local_along_track_axis(np.array([0.1, 0.0, 2.0]), min_speed_m_s=1.0) is None


def test_integrate_force_window_matches_constant_acceleration():
    frame = _synthetic_replay_frame(n=11)
    _, velocity = integrate_force_window(frame, "oracle", mass_kg=2.0, gravity_m_s2=9.81)
    assert velocity[0] == pytest.approx(frame.iloc[-1]["vehicle_local_position.vx"])
    assert velocity[2] == pytest.approx(frame.iloc[-1]["vehicle_local_position.vz"])


def test_select_candidate_windows_rejects_high_roll():
    frame = _synthetic_replay_frame(n=11)
    assert select_candidate_windows(frame, [1.0], stride_s=0.5, roll_threshold_deg=10.0, min_ground_speed_m_s=1.0)
    high_roll = frame.copy()
    high_roll["roll_rad"] = np.deg2rad(20.0)
    assert not select_candidate_windows(high_roll, [1.0], stride_s=0.5, roll_threshold_deg=10.0, min_ground_speed_m_s=1.0)


def test_evaluate_level2_replay_applies_oracle_gate_and_compares_sources():
    frame = _synthetic_replay_frame(n=11)
    metrics, gates, summary = evaluate_level2_replay(
        frame,
        mass_kg=2.0,
        gravity_m_s2=9.81,
        horizons_s=[1.0],
        stride_s=0.5,
        roll_threshold_deg=10.0,
        oracle_max_velocity_error_m_s=1e-9,
        min_ground_speed_m_s=1.0,
    )
    assert not gates.empty
    assert gates["passed_oracle_gate"].all()
    by_source = metrics.set_index("force_source")
    assert by_source.loc["gain_bias", "velocity_increment_error_m_s"] == pytest.approx(0.0)
    assert by_source.loc["raw_prior", "velocity_increment_error_m_s"] > 0.0
    assert set(summary["force_source"]) == {"raw_prior", "gain_bias", "pure_tcn"}


def test_evaluate_level2_replay_blocks_inconsistent_oracle():
    frame = _synthetic_replay_frame(n=11)
    frame["label_fx_b"] = 0.0
    metrics, gates, summary = evaluate_level2_replay(
        frame,
        mass_kg=2.0,
        gravity_m_s2=9.81,
        horizons_s=[1.0],
        stride_s=0.5,
        roll_threshold_deg=10.0,
        oracle_max_velocity_error_m_s=0.1,
        min_ground_speed_m_s=1.0,
    )
    assert not gates["passed_oracle_gate"].any()
    assert metrics.empty
    assert "candidate_windows" in summary.columns
