from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import system_identification.training as training_module
from system_identification.training import (
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_TARGET_COLUMNS,
    HybridPFNNRegressor,
    NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS,
    PAPER_NO_ACCEL_V2_FEATURE_COLUMNS,
    PAPER_PFNN_10_FEATURE_COLUMNS,
    LEAKAGE_RESISTANT_BASELINE_PROTOCOL,
    cyclic_catmull_rom_weights,
    evaluate_model_bundle,
    evaluate_model_bundle_by_log,
    evaluate_model_bundle_by_regime_bins,
    fit_torch_regressor,
    prepare_feature_target_frames,
    prepare_windowed_feature_target_frames,
    regression_loss,
    resolve_target_loss_weights,
    resolve_feature_set_columns,
    resolve_ablation_variants,
    run_baseline_comparison,
    run_diagnostic_evaluation,
    run_ablation_study,
    run_training_job,
)


def _synthetic_frame(n_rows: int = 256, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_rows)
    frame = pd.DataFrame(
        {
            "phase_corrected_rad": phase,
            "wing_stroke_angle_rad": 0.45 * np.sin(phase),
            "flap_frequency_hz": rng.normal(4.5, 0.2, size=n_rows),
            "cycle_flap_frequency_hz": rng.normal(4.5, 0.2, size=n_rows),
            "motor_cmd_0": rng.uniform(0.45, 0.85, size=n_rows),
            "servo_left_elevon": rng.uniform(-0.3, 0.3, size=n_rows),
            "servo_right_elevon": rng.uniform(-0.3, 0.3, size=n_rows),
            "servo_rudder": rng.uniform(-0.1, 0.1, size=n_rows),
            "vehicle_local_position.vx": rng.normal(0.4, 0.7, size=n_rows),
            "vehicle_local_position.vy": rng.normal(0.0, 0.7, size=n_rows),
            "vehicle_local_position.vz": rng.normal(-0.1, 0.4, size=n_rows),
            "vehicle_local_position.heading": rng.normal(0.0, 0.2, size=n_rows),
            "vehicle_local_position.ax": rng.normal(0.0, 1.0, size=n_rows),
            "vehicle_local_position.ay": rng.normal(0.0, 1.0, size=n_rows),
            "vehicle_local_position.az": rng.normal(0.0, 1.2, size=n_rows),
            "vehicle_angular_velocity.xyz[0]": rng.normal(0.0, 0.5, size=n_rows),
            "vehicle_angular_velocity.xyz[1]": rng.normal(0.0, 0.5, size=n_rows),
            "vehicle_angular_velocity.xyz[2]": rng.normal(0.0, 0.2, size=n_rows),
            "vehicle_angular_velocity.xyz_derivative[0]": rng.normal(0.0, 1.0, size=n_rows),
            "vehicle_angular_velocity.xyz_derivative[1]": rng.normal(0.0, 1.0, size=n_rows),
            "vehicle_angular_velocity.xyz_derivative[2]": rng.normal(0.0, 0.7, size=n_rows),
            "vehicle_attitude.q[0]": rng.normal(0.9, 0.03, size=n_rows),
            "vehicle_attitude.q[1]": rng.normal(0.0, 0.05, size=n_rows),
            "vehicle_attitude.q[2]": rng.normal(0.0, 0.05, size=n_rows),
            "vehicle_attitude.q[3]": rng.normal(0.0, 0.05, size=n_rows),
            "airspeed_validated.indicated_airspeed_m_s": rng.normal(7.6, 0.6, size=n_rows),
            "airspeed_validated.calibrated_airspeed_m_s": rng.normal(7.8, 0.6, size=n_rows),
            "airspeed_validated.true_airspeed_m_s": rng.normal(8.0, 0.6, size=n_rows),
            "airspeed_validated.calibrated_ground_minus_wind_m_s": rng.normal(7.7, 0.6, size=n_rows),
            "airspeed_validated.true_ground_minus_wind_m_s": rng.normal(8.1, 0.6, size=n_rows),
            "airspeed_validated.pitch_filtered": rng.normal(0.05, 0.1, size=n_rows),
            "vehicle_air_data.rho": rng.normal(1.18, 0.02, size=n_rows),
            "wind.windspeed_north": rng.normal(-0.2, 0.4, size=n_rows),
            "wind.windspeed_east": rng.normal(0.1, 0.4, size=n_rows),
        }
    )

    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)
    frame["fx_b"] = 4.0 * phase_sin + 1.5 * frame["motor_cmd_0"] + 0.3 * frame["vehicle_local_position.ax"]
    frame["fy_b"] = 0.5 * frame["servo_rudder"] + 0.2 * frame["vehicle_local_position.ay"]
    frame["fz_b"] = -9.8 + 1.2 * phase_cos + 0.2 * frame["vehicle_local_position.az"]
    frame["mx_b"] = 0.01 * frame["servo_left_elevon"] + 0.002 * frame["vehicle_angular_velocity.xyz_derivative[0]"]
    frame["my_b"] = 0.01 * frame["servo_right_elevon"] + 0.002 * frame["vehicle_angular_velocity.xyz_derivative[1]"]
    frame["mz_b"] = 0.01 * frame["servo_rudder"] + 0.001 * frame["vehicle_angular_velocity.xyz_derivative[2]"]
    return frame


def test_prepare_feature_target_frames_adds_phase_encoding():
    frame = _synthetic_frame(n_rows=8, seed=1)

    features, targets = prepare_feature_target_frames(frame)

    assert list(features.columns) == DEFAULT_FEATURE_COLUMNS
    assert list(targets.columns) == DEFAULT_TARGET_COLUMNS
    np.testing.assert_allclose(features["phase_corrected_sin"].to_numpy(), np.sin(frame["phase_corrected_rad"].to_numpy()))
    np.testing.assert_allclose(features["phase_corrected_cos"].to_numpy(), np.cos(frame["phase_corrected_rad"].to_numpy()))


def test_prepare_feature_target_frames_adds_phase_harmonics():
    frame = _synthetic_frame(n_rows=8, seed=123)

    features, _ = prepare_feature_target_frames(
        frame,
        feature_columns=[
            "phase_corrected_h2_sin",
            "phase_corrected_h2_cos",
            "phase_corrected_h3_sin",
            "phase_corrected_h3_cos",
        ],
    )

    phase = frame["phase_corrected_rad"].to_numpy()
    np.testing.assert_allclose(features["phase_corrected_h2_sin"].to_numpy(), np.sin(2.0 * phase))
    np.testing.assert_allclose(features["phase_corrected_h2_cos"].to_numpy(), np.cos(2.0 * phase))
    np.testing.assert_allclose(features["phase_corrected_h3_sin"].to_numpy(), np.sin(3.0 * phase))
    np.testing.assert_allclose(features["phase_corrected_h3_cos"].to_numpy(), np.cos(3.0 * phase))


def test_prepare_feature_target_frames_derives_sign_invariant_gravity_vector():
    frame = _synthetic_frame(n_rows=4, seed=13)
    flipped = frame.copy()
    attitude_columns = [
        "vehicle_attitude.q[0]",
        "vehicle_attitude.q[1]",
        "vehicle_attitude.q[2]",
        "vehicle_attitude.q[3]",
    ]
    flipped.loc[:, attitude_columns] = -flipped.loc[:, attitude_columns]

    features, _ = prepare_feature_target_frames(frame)
    flipped_features, _ = prepare_feature_target_frames(flipped)

    np.testing.assert_allclose(features[["gravity_b.x", "gravity_b.y", "gravity_b.z"]].to_numpy(), flipped_features[["gravity_b.x", "gravity_b.y", "gravity_b.z"]].to_numpy())
    assert "gravity_b.x" in features.columns
    assert "gravity_b.y" in features.columns
    assert "gravity_b.z" in features.columns


def test_prepare_windowed_feature_target_frames_keeps_windows_inside_log_and_segment():
    frame = _synthetic_frame(n_rows=8, seed=17)
    frame["log_id"] = ["a", "a", "a", "a", "b", "b", "b", "b"]
    frame["segment_id"] = [0, 0, 1, 1, 0, 0, 0, 0]
    feature_columns = ["motor_cmd_0", "servo_rudder"]
    target_columns = ["fx_b"]

    features, targets = prepare_windowed_feature_target_frames(
        frame,
        feature_columns,
        target_columns,
        window_mode="centered",
        window_radius=1,
    )

    assert list(features.columns) == [
        "motor_cmd_0@t-1",
        "servo_rudder@t-1",
        "motor_cmd_0@t+0",
        "servo_rudder@t+0",
        "motor_cmd_0@t+1",
        "servo_rudder@t+1",
    ]
    assert len(features) == 2
    np.testing.assert_allclose(features["motor_cmd_0@t+0"].to_numpy(), frame["motor_cmd_0"].iloc[[5, 6]].to_numpy())
    np.testing.assert_allclose(targets["fx_b"].to_numpy(), frame["fx_b"].iloc[[5, 6]].to_numpy())


def test_prepare_windowed_feature_target_frames_supports_causal_history():
    frame = _synthetic_frame(n_rows=5, seed=18)
    frame["log_id"] = ["a"] * len(frame)

    features, targets = prepare_windowed_feature_target_frames(
        frame,
        ["motor_cmd_0"],
        ["fx_b"],
        window_mode="causal",
        window_radius=2,
    )

    assert list(features.columns) == ["motor_cmd_0@t-2", "motor_cmd_0@t-1", "motor_cmd_0@t+0"]
    assert len(features) == 3
    np.testing.assert_allclose(features["motor_cmd_0@t+0"].to_numpy(), frame["motor_cmd_0"].iloc[2:].to_numpy())
    np.testing.assert_allclose(targets["fx_b"].to_numpy(), frame["fx_b"].iloc[2:].to_numpy())


def test_prepare_windowed_feature_target_frames_can_window_only_selected_features():
    frame = _synthetic_frame(n_rows=4, seed=19)
    frame["log_id"] = ["a"] * len(frame)

    features, targets = prepare_windowed_feature_target_frames(
        frame,
        ["motor_cmd_0", "servo_rudder"],
        ["fx_b"],
        window_mode="causal",
        window_radius=1,
        window_feature_columns=["motor_cmd_0"],
    )

    assert list(features.columns) == [
        "motor_cmd_0@t-1",
        "motor_cmd_0@t+0",
        "servo_rudder@t+0",
    ]
    assert len(features) == 3
    np.testing.assert_allclose(features["motor_cmd_0@t-1"].to_numpy(), frame["motor_cmd_0"].iloc[:-1].to_numpy())
    np.testing.assert_allclose(features["motor_cmd_0@t+0"].to_numpy(), frame["motor_cmd_0"].iloc[1:].to_numpy())
    np.testing.assert_allclose(features["servo_rudder@t+0"].to_numpy(), frame["servo_rudder"].iloc[1:].to_numpy())
    np.testing.assert_allclose(targets["fx_b"].to_numpy(), frame["fx_b"].iloc[1:].to_numpy())


def test_prepare_causal_sequence_feature_target_frames_keeps_windows_inside_log_and_segment():
    frame = _synthetic_frame(n_rows=12, seed=1)
    frame["log_id"] = ["a"] * 6 + ["b"] * 6
    frame["segment_id"] = [0] * 3 + [1] * 3 + [0] * 6
    frame["time_s"] = list(reversed(range(6))) + list(reversed(range(6)))

    seq, current, targets, meta = training_module.prepare_causal_sequence_feature_target_frames(
        frame,
        sequence_feature_columns=["phase_corrected_sin", "phase_corrected_cos", "motor_cmd_0"],
        current_feature_columns=["velocity_b.x"],
        target_columns=["fx_b", "fz_b"],
        history_size=3,
    )

    assert seq.shape == (6, 3, 3)
    assert current.shape == (6, 1)
    assert list(targets.columns) == ["fx_b", "fz_b"]
    assert set(meta.columns) >= {"log_id", "segment_id", "time_s"}
    assert not meta.duplicated(["log_id", "segment_id", "time_s"]).any()


def test_prepare_causal_sequence_feature_target_frames_aligns_target_to_last_timestep():
    frame = _synthetic_frame(n_rows=5, seed=2)
    frame["log_id"] = "a"
    frame["segment_id"] = 0
    frame["time_s"] = np.arange(5, dtype=float)
    frame["fx_b"] = frame["time_s"] * 10.0

    _, _, targets, meta = training_module.prepare_causal_sequence_feature_target_frames(
        frame,
        sequence_feature_columns=["phase_corrected_sin"],
        current_feature_columns=[],
        target_columns=["fx_b"],
        history_size=3,
    )

    np.testing.assert_allclose(meta["time_s"].to_numpy(), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(targets["fx_b"].to_numpy(), [20.0, 30.0, 40.0])


def test_prepare_causal_rollout_feature_target_frames_aligns_context_rollout_and_targets():
    frame = _synthetic_frame(n_rows=8, seed=1)
    frame["log_id"] = "a"
    frame["segment_id"] = 0
    frame["time_s"] = np.arange(8, dtype=float)
    frame["fx_b"] = frame["time_s"] * 10.0

    context, rollout, current, targets, meta = training_module.prepare_causal_rollout_feature_target_frames(
        frame,
        context_feature_columns=["phase_corrected_sin"],
        rollout_feature_columns=["phase_corrected_sin", "motor_cmd_0"],
        current_feature_columns=["velocity_b.x"],
        target_columns=["fx_b"],
        history_size=3,
        rollout_size=2,
        rollout_stride=2,
    )

    assert context.shape == (2, 3, 1)
    assert rollout.shape == (2, 2, 2)
    assert current.shape == (2, 2, 1)
    assert targets.shape == (2, 2, 1)
    np.testing.assert_allclose(meta["time_s"].to_numpy(), [3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(targets.reshape(-1), [30.0, 40.0, 50.0, 60.0])


def test_prepare_causal_rollout_feature_target_frames_never_crosses_log_or_segment():
    frame = _synthetic_frame(n_rows=12, seed=2)
    frame["log_id"] = ["a"] * 6 + ["b"] * 6
    frame["segment_id"] = [0] * 3 + [1] * 3 + [0] * 6
    frame["time_s"] = list(reversed(range(6))) + list(reversed(range(6)))

    context, rollout, current, targets, meta = training_module.prepare_causal_rollout_feature_target_frames(
        frame,
        context_feature_columns=["phase_corrected_sin"],
        rollout_feature_columns=["phase_corrected_sin"],
        current_feature_columns=[],
        target_columns=["fx_b", "fz_b"],
        history_size=2,
        rollout_size=2,
        rollout_stride=1,
    )

    assert context.shape[0] == 3
    assert current.shape == (3, 2, 0)
    assert targets.shape == (3, 2, 2)
    assert set(meta["log_id"]) == {"b"}
    assert set(meta["segment_id"]) == {0}


def test_resolve_sequence_feature_columns_defaults_to_leakage_resistant_history():
    columns = resolve_feature_set_columns("paper_no_accel_v2")

    sequence_columns = training_module.resolve_sequence_feature_columns(columns, "phase_actuator_airdata")

    assert "phase_corrected_sin" in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns
    assert "velocity_b.x" not in sequence_columns
    assert "vehicle_angular_velocity.xyz[0]" not in sequence_columns
    assert "alpha_rad" not in sequence_columns


def test_resolve_sequence_feature_columns_supports_phase_harmonics():
    columns = resolve_feature_set_columns("paper_no_accel_v2_phase_harmonic")

    sequence_columns = training_module.resolve_sequence_feature_columns(columns, "phase_harmonic_actuator_airdata")

    assert "phase_corrected_sin" in sequence_columns
    assert "phase_corrected_h2_sin" in sequence_columns
    assert "phase_corrected_h3_cos" in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns
    assert "velocity_b.x" not in sequence_columns


def test_resolve_sequence_feature_columns_supports_raw_phase_airdata():
    columns = resolve_feature_set_columns("paper_no_accel_v2_raw_phase")

    sequence_columns = training_module.resolve_sequence_feature_columns(columns, "raw_phase_actuator_airdata")

    assert "phase_corrected_rad" in sequence_columns
    assert "phase_corrected_sin" not in sequence_columns
    assert "phase_corrected_cos" not in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns


def test_resolve_sequence_feature_columns_supports_no_phase_actuator_airdata():
    columns = resolve_feature_set_columns("paper_no_accel_v2")

    sequence_columns = training_module.resolve_sequence_feature_columns(columns, "no_phase_actuator_airdata")

    assert "phase_corrected_rad" not in sequence_columns
    assert "phase_corrected_sin" not in sequence_columns
    assert "phase_corrected_cos" not in sequence_columns
    assert "wing_stroke_angle_rad" not in sequence_columns
    assert "motor_cmd_0" in sequence_columns
    assert "airspeed_validated.true_airspeed_m_s" in sequence_columns


def test_resolve_sequence_feature_columns_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown sequence_feature_mode"):
        training_module.resolve_sequence_feature_columns(resolve_feature_set_columns("paper_no_accel_v2"), "bad")


def test_causal_gru_regressor_forward_shape_with_current_features():
    model = training_module.CausalGRURegressor(
        sequence_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    seq = torch.randn(5, 8, 4)
    current = torch.randn(5, 3)

    out = model(seq, current)

    assert out.shape == (5, 6)


def test_causal_gru_regressor_forward_shape_without_current_features():
    model = training_module.CausalGRURegressor(
        sequence_input_dim=4,
        current_input_dim=0,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    out = model(torch.randn(5, 8, 4), None)

    assert out.shape == (5, 6)


def test_causal_lstm_regressor_forward_shape_with_current_features():
    model = training_module.CausalLSTMRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)


def test_causal_tcn_regressor_forward_shape_with_current_features():
    model = training_module.CausalTCNRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        channels=16,
        num_blocks=3,
        kernel_size=3,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)


def test_causal_transformer_regressor_forward_shape_with_current_features():
    model = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)


def test_causal_transformer_head_film_forward_shape_with_current_features():
    model = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="head",
        phase_conditioning_indices=(0, 1),
        film_hidden_size=16,
        film_scale=0.1,
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)


def test_causal_transformer_input_film_forward_shape_with_current_features():
    model = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="input",
        phase_conditioning_indices=(0, 1),
        film_hidden_size=16,
        film_scale=0.1,
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)


def test_phase_film_zero_initialized_as_identity():
    torch.manual_seed(12)
    base = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="none",
    )
    film = training_module.CausalTransformerRegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        head_hidden_sizes=(8,),
        film_mode="head",
        phase_conditioning_indices=(0, 1),
        film_hidden_size=16,
        film_scale=0.1,
    )

    base_state = base.state_dict()
    compatible = {key: value for key, value in base_state.items() if key in film.state_dict()}
    film.load_state_dict({**film.state_dict(), **compatible})

    sequence = torch.randn(2, 16, 5)
    current = torch.randn(2, 3)
    torch.testing.assert_close(film(sequence, current), base(sequence, current), atol=1e-6, rtol=1e-6)


def test_causal_tcn_gru_regressor_forward_shape_with_current_features():
    model = training_module.CausalTCNGRURegressor(
        sequence_input_dim=5,
        current_input_dim=3,
        output_dim=6,
        tcn_channels=16,
        tcn_num_blocks=2,
        tcn_kernel_size=3,
        gru_hidden_size=16,
        gru_num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(8,),
    )

    sequence = torch.randn(4, 64, 5)
    current = torch.randn(4, 3)
    output = model(sequence, current)

    assert output.shape == (4, 6)


def test_subsection_gru_regressor_forward_shape():
    model = training_module.SubsectionGRUWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    out = model(
        torch.randn(5, 8, 4),
        torch.randn(5, 6, 4),
        torch.randn(5, 6, 3),
    )

    assert out.shape == (5, 6, 6)


def test_subsection_gru_regressor_forward_shape_without_current_features():
    model = training_module.SubsectionGRUWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=0,
        output_dim=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        head_hidden_sizes=(12,),
    )

    out = model(torch.randn(5, 8, 4), torch.randn(5, 6, 4), None)

    assert out.shape == (5, 6, 6)


def test_discrete_subnet_wrench_regressor_forward_shape():
    model = training_module.DiscreteSUBNETWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        latent_size=10,
        hidden_sizes=(16,),
        dropout=0.0,
    )

    out = model(
        torch.randn(5, 8, 4),
        torch.randn(5, 6, 4),
        torch.randn(5, 6, 3),
    )

    assert out.shape == (5, 6, 6)


def test_ct_subnet_euler_wrench_regressor_forward_shape_and_tau_config():
    model = training_module.ContinuousTimeSUBNETWrenchRegressor(
        context_input_dim=4,
        rollout_input_dim=4,
        current_input_dim=3,
        output_dim=6,
        latent_size=10,
        hidden_sizes=(16,),
        dropout=0.0,
        dt_over_tau=0.03,
        integrator="euler",
    )

    out = model(
        torch.randn(5, 8, 4),
        torch.randn(5, 6, 4),
        torch.randn(5, 6, 3),
    )

    assert out.shape == (5, 6, 6)
    assert model.dt_over_tau == pytest.approx(0.03)


def test_ct_subnet_euler_rejects_bad_dt_over_tau():
    with pytest.raises(ValueError, match="dt_over_tau"):
        training_module.ContinuousTimeSUBNETWrenchRegressor(
            context_input_dim=4,
            rollout_input_dim=4,
            current_input_dim=0,
            output_dim=6,
            latent_size=10,
            hidden_sizes=(16,),
            dt_over_tau=0.0,
        )


def test_fit_and_evaluate_torch_regressor_smoke():
    train_frame = _synthetic_frame(n_rows=192, seed=2)
    val_frame = _synthetic_frame(n_rows=96, seed=3)

    bundle = fit_torch_regressor(
        train_frame=train_frame,
        val_frame=val_frame,
        hidden_sizes=(32, 32),
        batch_size=64,
        max_epochs=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=7,
        num_workers=0,
        use_amp=False,
    )
    metrics = evaluate_model_bundle(bundle, val_frame, split_name="val", batch_size=64)

    assert bundle["feature_columns"] == DEFAULT_FEATURE_COLUMNS
    assert bundle["target_columns"] == DEFAULT_TARGET_COLUMNS
    assert bundle["device_type"] == "cpu"
    assert metrics["split"] == "val"
    assert metrics["sample_count"] == len(val_frame)
    assert set(metrics["per_target"].keys()) == set(DEFAULT_TARGET_COLUMNS)
    assert np.isfinite(metrics["overall_rmse"])
    assert np.isfinite(metrics["overall_mae"])
    assert np.isfinite(metrics["overall_r2"])


def test_run_training_job_writes_artifacts(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    _synthetic_frame(n_rows=160, seed=4).to_parquet(split_root / "train_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=5).to_parquet(split_root / "val_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=6).to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "artifacts"
    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        hidden_sizes=(32, 32),
        batch_size=64,
        max_epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=11,
        num_workers=0,
        use_amp=False,
    )

    assert Path(outputs["model_bundle_path"]).exists()
    assert Path(outputs["metrics_path"]).exists()
    assert Path(outputs["training_config_path"]).exists()

    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {"train", "val", "test"}
    assert metrics["test"]["sample_count"] == 80
    payload = torch.load(outputs["model_bundle_path"], map_location="cpu")
    assert payload["target_columns"] == DEFAULT_TARGET_COLUMNS


def test_run_training_job_supports_causal_gru_model(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 10, "train_log"), ("val", 11, "val_log"), ("test", 12, "test_log")]:
        frame = _synthetic_frame(n_rows=48, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "run",
        feature_set_name="paper_no_accel_v2",
        model_type="causal_gru",
        sequence_history_size=8,
        sequence_feature_mode="phase_actuator_airdata",
        hidden_sizes=(16,),
        batch_size=16,
        max_epochs=1,
        device="cpu",
        use_amp=False,
    )

    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))

    assert cfg["model_type"] == "causal_gru"
    assert cfg["sequence_history_size"] == 8
    assert cfg["sequence_feature_mode"] == "phase_actuator_airdata"
    assert "velocity_b.x" not in cfg["sequence_feature_columns"]
    assert metrics["test"]["sample_count"] == 41


@pytest.mark.parametrize(
    "model_type",
    ["causal_lstm", "causal_tcn", "causal_transformer", "causal_tcn_gru"],
)
def test_run_training_job_supports_temporal_sequence_model_types(tmp_path: Path, model_type: str):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 20, "train_log"), ("val", 21, "val_log"), ("test", 22, "test_log")]:
        frame = _synthetic_frame(n_rows=48, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / model_type,
        feature_set_name="paper_no_accel_v2",
        model_type=model_type,
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        loss_type="huber",
        huber_delta=1.5,
        sequence_history_size=4,
        sequence_feature_mode="phase_actuator_airdata",
        current_feature_mode="remaining_current",
        device="cpu",
        use_amp=False,
    )

    assert Path(outputs["model_bundle_path"]).exists()
    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    assert cfg["model_type"] == model_type
    assert cfg["has_acceleration_inputs"] is False
    assert cfg["has_centered_window"] is False


@pytest.mark.parametrize("model_type", ["causal_transformer_head_film", "causal_transformer_input_film"])
def test_run_training_job_supports_phase_film_transformers(tmp_path: Path, model_type: str):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 520, "train_log"), ("val", 521, "val_log"), ("test", 522, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / model_type,
        feature_set_name="paper_no_accel_v2",
        model_type=model_type,
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_dim_feedforward=32,
        device="cpu",
        num_workers=0,
        use_amp=False,
    )

    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    bundle = torch.load(outputs["model_bundle_path"], map_location="cpu", weights_only=False)
    assert cfg["model_type"] == model_type
    assert bundle["feature_set_name"] == "paper_no_accel_v2"
    assert cfg["film_mode"] in {"head", "input"}
    assert cfg["phase_conditioning_columns"] == ["phase_corrected_sin", "phase_corrected_cos"]
    assert metrics["test"]["sample_count"] > 0


def test_run_training_job_supports_rollout_model_config(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 1, "train_log"), ("val", 2, "val_log"), ("test", 3, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "run",
        feature_set_name="paper_no_accel_v2",
        model_type="subsection_gru",
        sequence_history_size=8,
        rollout_size=4,
        rollout_stride=4,
        sequence_feature_mode="phase_actuator_airdata",
        hidden_sizes=(16,),
        batch_size=8,
        max_epochs=1,
        device="cpu",
        use_amp=False,
    )

    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))

    assert cfg["model_type"] == "subsection_gru"
    assert cfg["rollout_size"] == 4
    assert cfg["rollout_stride"] == 4
    assert cfg["has_acceleration_inputs"] is False
    assert cfg["has_velocity_history"] is False
    assert metrics["test"]["sample_count"] > 0


def test_adaptive_spectrum_layer_preserves_sequence_shape():
    layer = training_module.AdaptiveSpectrumLayer(input_dim=4, hidden_size=8, dropout=0.0, max_frequency_bins=5)
    x = torch.randn(3, 16, 4)

    y = layer(x)

    assert y.shape == x.shape


def test_causal_gru_asl_regressor_forward_shape():
    model = training_module.CausalGRUASLRegressor(
        sequence_input_dim=4,
        current_input_dim=2,
        output_dim=6,
        gru_hidden_size=16,
        gru_num_layers=1,
        asl_hidden_size=8,
        asl_dropout=0.0,
        asl_max_frequency_bins=5,
        head_hidden_sizes=(12,),
    )

    out = model(torch.randn(5, 16, 4), torch.randn(5, 2))

    assert out.shape == (5, 6)


def test_run_training_job_writes_history_and_diagnostics(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    _synthetic_frame(n_rows=128, seed=7).to_parquet(split_root / "train_samples.parquet", index=False)
    _synthetic_frame(n_rows=64, seed=8).to_parquet(split_root / "val_samples.parquet", index=False)
    _synthetic_frame(n_rows=64, seed=9).to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "artifacts"
    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        hidden_sizes=(32, 32),
        batch_size=64,
        max_epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=12,
        num_workers=0,
        use_amp=False,
    )

    assert Path(outputs["history_path"]).exists()
    assert Path(outputs["training_curves_path"]).exists()
    assert Path(outputs["pred_vs_true_test_path"]).exists()
    assert Path(outputs["residual_hist_test_path"]).exists()

    history = pd.read_csv(outputs["history_path"])
    assert {"epoch", "train_loss", "val_loss", "val_overall_rmse", "val_overall_r2"}.issubset(history.columns)


def test_default_ablation_variants_resolve_expected_feature_sets():
    assert set(DEFAULT_FEATURE_GROUPS.keys()) == {
        "phase",
        "actuators",
        "linear_kinematics",
        "angular_kinematics",
        "attitude",
        "aero",
    }
    assert set(DEFAULT_ABLATION_VARIANTS.keys()) >= {"full", "no_phase", "no_actuators", "no_attitude", "no_aero"}

    variants = resolve_ablation_variants(["full", "no_phase", "no_attitude"])

    assert variants["full"] == DEFAULT_FEATURE_COLUMNS
    assert "phase_corrected_sin" not in variants["no_phase"]
    assert "phase_corrected_cos" not in variants["no_phase"]
    assert "wing_stroke_angle_rad" not in variants["no_phase"]
    assert "phase_corrected_sin" in variants["full"]
    assert "gravity_b.x" in variants["full"]
    assert "gravity_b.x" not in variants["no_attitude"]


def test_no_accel_no_alpha_feature_set_excludes_label_derivative_inputs():
    excluded = {
        "vehicle_local_position.ax",
        "vehicle_local_position.ay",
        "vehicle_local_position.az",
        "vehicle_angular_velocity.xyz_derivative[0]",
        "vehicle_angular_velocity.xyz_derivative[1]",
        "vehicle_angular_velocity.xyz_derivative[2]",
    }

    feature_columns = resolve_feature_set_columns("no_accel_no_alpha")

    assert feature_columns == NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS
    assert excluded.isdisjoint(feature_columns)
    assert {"phase_corrected_sin", "phase_corrected_cos", "wing_stroke_angle_rad"}.issubset(feature_columns)
    assert {"motor_cmd_0", "servo_left_elevon", "servo_right_elevon", "servo_rudder"}.issubset(feature_columns)
    assert {"vehicle_local_position.vx", "vehicle_local_position.vy", "vehicle_local_position.vz"}.issubset(feature_columns)
    assert {
        "vehicle_angular_velocity.xyz[0]",
        "vehicle_angular_velocity.xyz[1]",
        "vehicle_angular_velocity.xyz[2]",
    }.issubset(feature_columns)
    assert {"gravity_b.x", "gravity_b.y", "gravity_b.z"}.issubset(feature_columns)
    assert {"airspeed_validated.true_airspeed_m_s", "vehicle_air_data.rho"}.issubset(feature_columns)


def test_paper_no_accel_v2_feature_set_excludes_label_derivative_inputs():
    excluded = {
        "vehicle_local_position.ax",
        "vehicle_local_position.ay",
        "vehicle_local_position.az",
        "vehicle_angular_velocity.xyz_derivative[0]",
        "vehicle_angular_velocity.xyz_derivative[1]",
        "vehicle_angular_velocity.xyz_derivative[2]",
    }

    feature_columns = resolve_feature_set_columns("paper_no_accel_v2")

    assert feature_columns == PAPER_NO_ACCEL_V2_FEATURE_COLUMNS
    assert excluded.isdisjoint(feature_columns)
    assert {
        "airspeed_validated.indicated_airspeed_m_s",
        "airspeed_validated.calibrated_airspeed_m_s",
        "airspeed_validated.true_airspeed_m_s",
        "airspeed_validated.calibrated_ground_minus_wind_m_s",
        "airspeed_validated.true_ground_minus_wind_m_s",
        "airspeed_validated.pitch_filtered",
        "vehicle_local_position.heading",
    }.issubset(feature_columns)
    assert {
        "roll_rad",
        "pitch_rad",
        "velocity_b.x",
        "velocity_b.y",
        "velocity_b.z",
        "relative_air_velocity_b.x",
        "relative_air_velocity_b.y",
        "relative_air_velocity_b.z",
        "alpha_rad",
        "beta_rad",
        "dynamic_pressure_pa",
        "elevator_like",
        "aileron_like",
    }.issubset(feature_columns)


def test_paper_no_accel_v2_phase_harmonic_feature_set_adds_harmonics():
    feature_columns = resolve_feature_set_columns("paper_no_accel_v2_phase_harmonic")

    assert set(PAPER_NO_ACCEL_V2_FEATURE_COLUMNS).issubset(feature_columns)
    assert {
        "phase_corrected_h2_sin",
        "phase_corrected_h2_cos",
        "phase_corrected_h3_sin",
        "phase_corrected_h3_cos",
    }.issubset(feature_columns)


def test_paper_pfnn_10_feature_set_matches_paper_inputs():
    excluded = {
        "vehicle_local_position.ax",
        "vehicle_local_position.ay",
        "vehicle_local_position.az",
        "vehicle_angular_velocity.xyz_derivative[0]",
        "vehicle_angular_velocity.xyz_derivative[1]",
        "vehicle_angular_velocity.xyz_derivative[2]",
        "phase_corrected_sin",
        "phase_corrected_cos",
    }

    feature_columns = resolve_feature_set_columns("paper_pfnn_10")

    assert feature_columns == PAPER_PFNN_10_FEATURE_COLUMNS
    assert feature_columns == [
        "phase_corrected_rad",
        "velocity_b.x",
        "velocity_b.y",
        "velocity_b.z",
        "pitch_rad",
        "roll_rad",
        "alpha_rad",
        "beta_rad",
        "cycle_flap_frequency_hz",
        "elevator_like",
        "servo_rudder",
    ]
    assert excluded.isdisjoint(feature_columns)


def test_cyclic_catmull_rom_weights_are_periodic_and_normalized():
    phase = torch.tensor([0.0, np.pi / 3.0, 2.0 * np.pi - 0.1], dtype=torch.float32)

    indices, weights = cyclic_catmull_rom_weights(phase, num_control_points=6)
    shifted_indices, shifted_weights = cyclic_catmull_rom_weights(phase + 2.0 * np.pi, num_control_points=6)

    assert indices.shape == (3, 4)
    assert weights.shape == (3, 4)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(indices, shifted_indices)
    torch.testing.assert_close(weights, shifted_weights, atol=1e-6, rtol=1e-6)


def test_hybrid_pfnn_regressor_forward_shape_and_phase_periodicity():
    model = HybridPFNNRegressor(
        input_dim=len(PAPER_PFNN_10_FEATURE_COLUMNS),
        output_dim=len(DEFAULT_TARGET_COLUMNS),
        hidden_sizes=(8, 8),
        phase_feature_index=0,
        expanded_input_dim=12,
        phase_node_count=3,
        phase_control_points=6,
        dropout=0.0,
    )
    inputs = torch.randn(5, len(PAPER_PFNN_10_FEATURE_COLUMNS), dtype=torch.float32)
    inputs[:, 0] = torch.linspace(0.0, 2.0 * np.pi, 5)

    outputs = model(inputs)
    shifted = inputs.clone()
    shifted[:, 0] = shifted[:, 0] + 2.0 * np.pi
    shifted_outputs = model(shifted)

    assert outputs.shape == (5, len(DEFAULT_TARGET_COLUMNS))
    torch.testing.assert_close(outputs, shifted_outputs, atol=1e-5, rtol=1e-5)


def test_prepare_feature_target_frames_derives_paper_no_accel_v2_features():
    frame = _synthetic_frame(n_rows=1, seed=17)
    frame.loc[:, "vehicle_attitude.q[0]"] = 1.0
    frame.loc[:, "vehicle_attitude.q[1]"] = 0.0
    frame.loc[:, "vehicle_attitude.q[2]"] = 0.0
    frame.loc[:, "vehicle_attitude.q[3]"] = 0.0
    frame.loc[:, "vehicle_local_position.vx"] = 10.0
    frame.loc[:, "vehicle_local_position.vy"] = 2.0
    frame.loc[:, "vehicle_local_position.vz"] = -1.0
    frame.loc[:, "wind.windspeed_north"] = 1.0
    frame.loc[:, "wind.windspeed_east"] = -1.0
    frame.loc[:, "airspeed_validated.true_airspeed_m_s"] = 10.0
    frame.loc[:, "vehicle_air_data.rho"] = 1.2
    frame.loc[:, "servo_left_elevon"] = 0.4
    frame.loc[:, "servo_right_elevon"] = 0.2

    features, _ = prepare_feature_target_frames(frame, PAPER_NO_ACCEL_V2_FEATURE_COLUMNS)

    np.testing.assert_allclose(features["roll_rad"].to_numpy(), [0.0], atol=1e-7)
    np.testing.assert_allclose(features["pitch_rad"].to_numpy(), [0.0], atol=1e-7)
    np.testing.assert_allclose(features["velocity_b.x"].to_numpy(), [10.0], atol=1e-7)
    np.testing.assert_allclose(features["velocity_b.y"].to_numpy(), [2.0], atol=1e-7)
    np.testing.assert_allclose(features["velocity_b.z"].to_numpy(), [-1.0], atol=1e-7)
    np.testing.assert_allclose(features["relative_air_velocity_b.x"].to_numpy(), [9.0], atol=1e-7)
    np.testing.assert_allclose(features["relative_air_velocity_b.y"].to_numpy(), [3.0], atol=1e-7)
    np.testing.assert_allclose(features["relative_air_velocity_b.z"].to_numpy(), [-1.0], atol=1e-7)
    np.testing.assert_allclose(features["alpha_rad"].to_numpy(), [np.arctan2(-1.0, 9.0)], atol=1e-7)
    np.testing.assert_allclose(features["beta_rad"].to_numpy(), [np.arcsin(3.0 / np.sqrt(91.0))], atol=1e-7)
    np.testing.assert_allclose(features["dynamic_pressure_pa"].to_numpy(), [60.0], atol=1e-7)
    np.testing.assert_allclose(features["elevator_like"].to_numpy(), [0.3], atol=1e-7)
    np.testing.assert_allclose(features["aileron_like"].to_numpy(), [0.1], atol=1e-7)


def test_run_ablation_study_writes_summary_outputs(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    _synthetic_frame(n_rows=160, seed=10).to_parquet(split_root / "train_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=11).to_parquet(split_root / "val_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=12).to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "ablation"
    outputs = run_ablation_study(
        split_root=split_root,
        output_dir=output_dir,
        variant_names=["full", "no_phase"],
        hidden_sizes=(32, 32),
        batch_size=64,
        max_epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=21,
        num_workers=0,
        use_amp=False,
    )

    assert Path(outputs["summary_csv_path"]).exists()
    assert Path(outputs["summary_plot_path"]).exists()
    summary = pd.read_csv(outputs["summary_csv_path"])
    assert {"variant_name", "val_overall_r2", "test_overall_r2"}.issubset(summary.columns)
    assert set(summary["variant_name"]) == {"full", "no_phase"}


def test_leakage_resistant_baseline_protocol_excludes_label_derivative_inputs():
    forbidden = set(LEAKAGE_RESISTANT_BASELINE_PROTOCOL["forbidden_feature_columns"])

    assert LEAKAGE_RESISTANT_BASELINE_PROTOCOL["split_policy"] == "whole_log"
    assert LEAKAGE_RESISTANT_BASELINE_PROTOCOL["model_type"] == "mlp"
    assert LEAKAGE_RESISTANT_BASELINE_PROTOCOL["feature_set_name"] == "paper_no_accel_v2"
    assert LEAKAGE_RESISTANT_BASELINE_PROTOCOL["loss_type"] == "huber"
    assert forbidden.isdisjoint(PAPER_NO_ACCEL_V2_FEATURE_COLUMNS)


def test_diagnostic_evaluation_writes_per_log_and_regime_metrics(tmp_path: Path):
    train_frame = _synthetic_frame(n_rows=160, seed=60)
    val_frame = _synthetic_frame(n_rows=96, seed=61)
    test_frame = _synthetic_frame(n_rows=120, seed=62)
    test_frame["log_id"] = ["test_log_a"] * 60 + ["test_log_b"] * 60
    test_frame["airspeed_validated.true_airspeed_m_s"] = np.r_[np.full(60, 7.0), np.full(60, 9.0)]
    test_frame["phase_corrected_rad"] = np.linspace(0.0, 2.0 * np.pi, len(test_frame), endpoint=False)

    bundle = fit_torch_regressor(
        train_frame=train_frame,
        val_frame=val_frame,
        feature_columns=PAPER_NO_ACCEL_V2_FEATURE_COLUMNS,
        hidden_sizes=(16, 16),
        batch_size=64,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=60,
        num_workers=0,
        use_amp=False,
        loss_type="huber",
    )

    per_log = evaluate_model_bundle_by_log(bundle, test_frame, split_name="test", batch_size=64, device="cpu")
    assert set(per_log["log_id"]) == {"test_log_a", "test_log_b"}
    assert {"test_fx_b_r2", "test_overall_rmse"}.issubset(per_log.columns)

    per_regime = evaluate_model_bundle_by_regime_bins(
        bundle,
        test_frame,
        split_name="test",
        bin_specs={
            "airspeed_validated.true_airspeed_m_s": [0.0, 8.0, 12.0],
            "phase_corrected_rad": [0.0, np.pi, 2.0 * np.pi],
        },
        batch_size=64,
        device="cpu",
    )
    assert set(per_regime["regime_column"]) == {"airspeed_validated.true_airspeed_m_s", "phase_corrected_rad"}
    assert {"test_fz_b_rmse", "test_overall_r2"}.issubset(per_regime.columns)

    split_root = tmp_path / "split"
    split_root.mkdir()
    train_frame.to_parquet(split_root / "train_samples.parquet", index=False)
    val_frame.to_parquet(split_root / "val_samples.parquet", index=False)
    test_frame.to_parquet(split_root / "test_samples.parquet", index=False)
    bundle_path = tmp_path / "model_bundle.pt"
    torch.save(bundle, bundle_path)

    outputs = run_diagnostic_evaluation(
        model_bundle_path=bundle_path,
        split_root=split_root,
        output_dir=tmp_path / "diagnostics",
        split_names=("test",),
        bin_specs={"airspeed_validated.true_airspeed_m_s": [0.0, 8.0, 12.0]},
        batch_size=64,
        device="cpu",
    )
    assert Path(outputs["per_log_metrics_path"]).exists()
    assert Path(outputs["per_regime_metrics_path"]).exists()


def test_run_baseline_comparison_writes_protocol_summary(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    train = _synthetic_frame(n_rows=128, seed=70)
    val = _synthetic_frame(n_rows=72, seed=71)
    test = _synthetic_frame(n_rows=72, seed=72)
    for frame, log_id in [(train, "train_log"), (val, "val_log"), (test, "test_log")]:
        frame["log_id"] = log_id
    train.to_parquet(split_root / "train_samples.parquet", index=False)
    val.to_parquet(split_root / "val_samples.parquet", index=False)
    test.to_parquet(split_root / "test_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=["mlp_paper_no_accel_v2"],
        hidden_sizes=(16, 16),
        batch_size=64,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=70,
        num_workers=0,
        use_amp=False,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert summary.loc[0, "recipe_name"] == "mlp_paper_no_accel_v2"
    assert summary.loc[0, "feature_set_name"] == "paper_no_accel_v2"
    assert summary.loc[0, "model_type"] == "mlp"
    assert summary.loc[0, "loss_type"] == "huber"
    assert {"test_overall_r2", "test_fx_b_rmse"}.issubset(summary.columns)


def test_run_baseline_comparison_supports_split_axis_independent_models(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    train = _synthetic_frame(n_rows=144, seed=80)
    val = _synthetic_frame(n_rows=72, seed=81)
    test = _synthetic_frame(n_rows=72, seed=82)
    for frame, log_id in [(train, "train_log"), (val, "val_log"), (test, "test_log")]:
        frame["log_id"] = log_id
    train.to_parquet(split_root / "train_samples.parquet", index=False)
    val.to_parquet(split_root / "val_samples.parquet", index=False)
    test.to_parquet(split_root / "test_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=["split_axis_mlp_paper_no_accel_v2"],
        hidden_sizes=(16, 16),
        batch_size=64,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=80,
        num_workers=0,
        use_amp=False,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert summary.loc[0, "recipe_name"] == "split_axis_mlp_paper_no_accel_v2"
    assert summary.loc[0, "model_type"] == "split_axis_mlp"
    assert summary.loc[0, "target_groups"] == "longitudinal:fx_b|fz_b|my_b;lateral:fy_b|mx_b|mz_b"
    assert {"test_fx_b_r2", "test_fy_b_r2", "test_mz_b_rmse", "test_overall_r2"}.issubset(summary.columns)

    recipe_dir = Path(summary.loc[0, "output_dir"])
    longitudinal_config = json.loads((recipe_dir / "longitudinal" / "training_config.json").read_text(encoding="utf-8"))
    lateral_config = json.loads((recipe_dir / "lateral" / "training_config.json").read_text(encoding="utf-8"))
    assert longitudinal_config["target_columns"] == ["fx_b", "fz_b", "my_b"]
    assert lateral_config["target_columns"] == ["fy_b", "mx_b", "mz_b"]


def test_run_baseline_comparison_supports_causal_sequence_recipes(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 90, "train_log"), ("val", 91, "val_log"), ("test", 92, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=[
            "causal_gru_paper_no_accel_v2_phase_actuator_airdata",
            "causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata",
        ],
        hidden_sizes=(16,),
        batch_size=16,
        max_epochs=1,
        device="cpu",
        random_seed=90,
        num_workers=0,
        use_amp=False,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])

    assert set(summary["recipe_name"]) == {
        "causal_gru_paper_no_accel_v2_phase_actuator_airdata",
        "causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata",
    }
    assert set(summary["model_type"]) == {"causal_gru", "causal_gru_asl"}
    assert {"sequence_history_size", "sequence_feature_mode", "test_overall_r2", "test_fx_b_rmse"}.issubset(
        summary.columns
    )
    assert set(summary["sequence_feature_mode"]) == {"phase_actuator_airdata"}


def test_run_baseline_comparison_supports_temporal_backbone_recipes(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 120, "train_log"), ("val", 121, "val_log"), ("test", 122, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=[
            "causal_lstm_paper_no_accel_v2_phase_actuator_airdata",
            "causal_tcn_paper_no_accel_v2_phase_actuator_airdata",
            "causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
            "causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata",
        ],
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        device="cpu",
        random_seed=120,
        num_workers=0,
        use_amp=False,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert set(summary["model_type"]) == {
        "causal_lstm",
        "causal_tcn",
        "causal_transformer",
        "causal_tcn_gru",
    }


def test_run_baseline_comparison_supports_phase_harmonic_transformer_recipe(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 430, "train_log"), ("val", 431, "val_log"), ("test", 432, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "runs",
        recipe_names=["causal_transformer_paper_no_accel_v2_phase_harmonic_airdata"],
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_dim_feedforward=32,
        device="cpu",
        num_workers=0,
        use_amp=False,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert summary.loc[0, "feature_set_name"] == "paper_no_accel_v2_phase_harmonic"
    assert summary.loc[0, "sequence_feature_mode"] == "phase_harmonic_actuator_airdata"


def test_run_baseline_comparison_supports_phase_film_transformer_recipes(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 540, "train_log"), ("val", 541, "val_log"), ("test", 542, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "runs",
        recipe_names=[
            "causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata",
            "causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata",
        ],
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_dim_feedforward=32,
        device="cpu",
        num_workers=0,
        use_amp=False,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])
    assert set(summary["model_type"]) == {"causal_transformer_head_film", "causal_transformer_input_film"}
    assert set(summary["sequence_feature_mode"]) == {"phase_actuator_airdata"}


def test_temporal_screen_quick_grid_contains_reference_and_candidates():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="quick")
    names = {config.recipe_name for config in configs}

    assert "mlp_paper_no_accel_v2" in names
    assert "causal_gru_paper_no_accel_v2_phase_actuator_airdata" in names
    assert "causal_lstm_paper_no_accel_v2_phase_actuator_airdata" in names
    assert "causal_tcn_paper_no_accel_v2_phase_actuator_airdata" in names
    assert "causal_transformer_paper_no_accel_v2_phase_actuator_airdata" in names


def test_classify_temporal_candidate_promotes_clear_rmse_win():
    from scripts.run_temporal_backbone_screen import classify_candidate

    decision = classify_candidate(
        candidate_rmse=0.96,
        reference_rmse=1.00,
        candidate_r2=0.72,
        reference_r2=0.70,
        hard_target_improvements=0,
        worst_regime_rmse_improvement=0.0,
    )

    assert decision == "promote"


def test_temporal_screen_tcn_gru_focused_grid_has_12_configs():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="tcn_gru_focused")

    assert len(configs) == 12
    assert {config.recipe_name for config in configs} == {
        "causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata"
    }
    assert len({config.config_id for config in configs}) == 12
    assert all(config.stage == "tcn_gru_focused" for config in configs)
    assert {config.sequence_history_size for config in configs} == {96, 128, 160}


def test_temporal_screen_tcn_gru_focused_final_grid_uses_full_budget():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="tcn_gru_focused_final")

    assert len(configs) == 12
    assert all(config.max_epochs == 50 for config in configs)
    assert all(config.early_stopping_patience == 8 for config in configs)
    assert all(config.dropout == 0.0 for config in configs)


def test_temporal_screen_transformer_focused_grid_has_12_configs():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="transformer_focused")

    assert len(configs) == 12
    assert {config.recipe_name for config in configs} == {
        "causal_transformer_paper_no_accel_v2_phase_actuator_airdata"
    }
    assert len({config.config_id for config in configs}) == 12
    assert all(config.stage == "transformer_focused" for config in configs)
    assert {config.sequence_history_size for config in configs} == {96, 128, 160, 192}
    assert "transformer_focused_hist128_d64_l2_h4_do0" in {config.config_id for config in configs}


def test_temporal_screen_transformer_focused_final_grid_uses_full_budget():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="transformer_focused_final")

    assert len(configs) == 12
    assert all(config.max_epochs == 50 for config in configs)
    assert all(config.early_stopping_patience == 8 for config in configs)
    assert "transformer_focused_final_hist128_d64_l2_h4_do0" in {config.config_id for config in configs}


def test_temporal_screen_phase_harmonic_grid_has_four_ablation_configs():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="phase_harmonic")

    assert len(configs) == 4
    assert {config.stage for config in configs} == {"phase_harmonic"}
    assert {
        "phase_harmonic_no_phase",
        "phase_harmonic_raw_phase",
        "phase_harmonic_sin_cos",
        "phase_harmonic_harmonic3",
    } == {config.config_id for config in configs}
    assert all(config.sequence_history_size == 128 for config in configs)
    assert all(config.dropout == 0.05 for config in configs)


def test_temporal_screen_phase_film_grid_contains_baseline_head_and_input():
    from scripts.run_temporal_backbone_screen import build_screen_configs

    configs = build_screen_configs(stage="phase_film")

    assert {
        "phase_film_baseline",
        "phase_film_head",
        "phase_film_input",
    } == {config.config_id for config in configs}
    assert {config.stage for config in configs} == {"phase_film"}
    assert all(config.sequence_history_size == 128 for config in configs)
    assert all(config.dropout == 0.05 for config in configs)


def test_train_baseline_torch_cli_supports_phase_harmonic_sequence_mode(monkeypatch):
    from scripts import train_baseline_torch

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_baseline_torch.py",
            "--split-root",
            "split",
            "--output-dir",
            "runs",
            "--sequence-feature-mode",
            "phase_harmonic_actuator_airdata",
        ],
    )

    args = train_baseline_torch.parse_args()

    assert args.sequence_feature_mode == "phase_harmonic_actuator_airdata"


def test_train_baseline_torch_cli_supports_phase_film_model_type(monkeypatch):
    from scripts import train_baseline_torch

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_baseline_torch.py",
            "--split-root",
            "split",
            "--output-dir",
            "runs",
            "--model-type",
            "causal_transformer_head_film",
        ],
    )

    args = train_baseline_torch.parse_args()

    assert args.model_type == "causal_transformer_head_film"


def test_run_baseline_comparison_can_skip_test_eval(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 210, "train_log"), ("val", 211, "val_log"), ("test", 212, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=["causal_transformer_paper_no_accel_v2_phase_actuator_airdata"],
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        device="cpu",
        random_seed=210,
        num_workers=0,
        use_amp=False,
        skip_test_eval=True,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])

    assert "val_overall_rmse" in summary.columns
    assert "test_overall_rmse" not in summary.columns


def test_run_baseline_comparison_supports_subnet_rollout_recipes(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 130, "train_log"), ("val", 131, "val_log"), ("test", 132, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=[
            "subsection_gru_paper_no_accel_v2_phase_actuator_airdata",
            "subnet_discrete_paper_no_accel_v2_phase_actuator_airdata",
            "ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata",
        ],
        hidden_sizes=(16,),
        batch_size=8,
        max_epochs=1,
        device="cpu",
        random_seed=130,
        num_workers=0,
        use_amp=False,
        sequence_history_size=8,
        rollout_size=4,
        rollout_stride=4,
        latent_size=8,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])

    assert set(summary["model_type"]) == {"subsection_gru", "subnet_discrete", "ct_subnet_euler"}
    assert {"rollout_size", "rollout_stride", "latent_size", "test_overall_r2", "test_fx_b_rmse"}.issubset(
        summary.columns
    )
    assert set(summary["sequence_feature_mode"]) == {"phase_actuator_airdata"}


def test_run_baseline_comparison_sequence_history_size_overrides_recipe_default(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 94, "train_log"), ("val", 95, "val_log"), ("test", 96, "test_log")]:
        frame = _synthetic_frame(n_rows=24, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_baseline_comparison(
        split_root=split_root,
        output_dir=tmp_path / "comparison",
        recipe_names=["causal_gru_paper_no_accel_v2_phase_actuator_airdata"],
        hidden_sizes=(16,),
        batch_size=8,
        max_epochs=1,
        device="cpu",
        random_seed=94,
        num_workers=0,
        use_amp=False,
        sequence_history_size=8,
    )

    summary = pd.read_csv(outputs["summary_csv_path"])

    assert int(summary.loc[0, "sequence_history_size"]) == 8
    assert int(summary.loc[0, "test_sample_count"]) == 17


def test_causal_gru_config_records_no_dangerous_history(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 100, "train_log"), ("val", 101, "val_log"), ("test", 102, "test_log")]:
        frame = _synthetic_frame(n_rows=48, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "run",
        feature_set_name="paper_no_accel_v2",
        model_type="causal_gru",
        sequence_history_size=8,
        sequence_feature_mode="phase_actuator_airdata",
        hidden_sizes=(16,),
        batch_size=16,
        max_epochs=1,
        device="cpu",
        use_amp=False,
    )

    cfg = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))

    assert cfg["has_velocity_history"] is False
    assert cfg["has_angular_velocity_history"] is False
    assert cfg["has_alpha_beta_history"] is False
    assert cfg["has_acceleration_inputs"] is False


def test_sequence_regime_diagnostics_skip_bins_without_complete_history(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 110, "train_log"), ("val", 111, "val_log"), ("test", 112, "test_log")]:
        frame = _synthetic_frame(n_rows=30, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "run",
        feature_set_name="paper_no_accel_v2",
        model_type="causal_gru",
        sequence_history_size=8,
        sequence_feature_mode="phase_actuator_airdata",
        hidden_sizes=(16,),
        batch_size=16,
        max_epochs=1,
        device="cpu",
        use_amp=False,
    )
    bundle = torch.load(outputs["model_bundle_path"], map_location="cpu", weights_only=False)
    test_frame = pd.read_parquet(split_root / "test_samples.parquet")

    diagnostics = evaluate_model_bundle_by_regime_bins(
        bundle,
        test_frame,
        split_name="test",
        bin_specs={"time_s": [0.0, 0.05, 1.0]},
        batch_size=16,
    )

    assert len(diagnostics) == 1
    assert diagnostics.loc[0, "test_sample_count"] == 23


def test_run_training_job_accepts_no_accel_no_alpha_feature_set(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    _synthetic_frame(n_rows=160, seed=14).to_parquet(split_root / "train_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=15).to_parquet(split_root / "val_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=16).to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "artifacts"
    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        feature_set_name="no_accel_no_alpha",
        hidden_sizes=(32, 32),
        batch_size=64,
        max_epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=22,
        num_workers=0,
        use_amp=False,
    )

    training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))

    assert training_config["feature_set_name"] == "no_accel_no_alpha"
    assert training_config["feature_columns"] == NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS
    assert "vehicle_local_position.ax" not in training_config["feature_columns"]
    assert "vehicle_angular_velocity.xyz_derivative[0]" not in training_config["feature_columns"]


def test_resolve_target_loss_weights_from_mapping():
    weights = resolve_target_loss_weights(
        DEFAULT_TARGET_COLUMNS,
        "fx_b=1,fy_b=0.5,fz_b=1,mx_b=0.25,my_b=1,mz_b=0.25",
    )

    np.testing.assert_allclose(weights, np.array([1.0, 0.5, 1.0, 0.25, 1.0, 0.25], dtype=np.float32))


def test_regression_loss_supports_weighted_huber():
    predictions = torch.tensor([[0.0, 3.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

    loss = regression_loss(
        predictions,
        targets,
        target_loss_weights=weights,
        loss_type="huber",
        huber_delta=1.0,
    )

    expected = (1.0 * 0.0 + 2.0 * 2.5) / (1.0 + 2.0)
    np.testing.assert_allclose(float(loss.item()), expected, rtol=1e-6)


def test_run_training_job_records_target_loss_weights(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    _synthetic_frame(n_rows=96, seed=34).to_parquet(split_root / "train_samples.parquet", index=False)
    _synthetic_frame(n_rows=64, seed=35).to_parquet(split_root / "val_samples.parquet", index=False)
    _synthetic_frame(n_rows=64, seed=36).to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "artifacts"
    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        feature_set_name="paper_no_accel_v2",
        hidden_sizes=(16, 16),
        batch_size=32,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=33,
        num_workers=0,
        use_amp=False,
        target_loss_weights="fx_b=1,fy_b=0.5,fz_b=1,mx_b=0.5,my_b=1,mz_b=0.5",
    )

    training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    payload = torch.load(outputs["model_bundle_path"], map_location="cpu")

    assert training_config["target_loss_weights"] == {
        "fx_b": 1.0,
        "fy_b": 0.5,
        "fz_b": 1.0,
        "mx_b": 0.5,
        "my_b": 1.0,
        "mz_b": 0.5,
    }
    np.testing.assert_allclose(payload["target_loss_weights"].numpy(), np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5]))


def test_run_training_job_records_huber_loss_config(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    _synthetic_frame(n_rows=96, seed=44).to_parquet(split_root / "train_samples.parquet", index=False)
    _synthetic_frame(n_rows=64, seed=45).to_parquet(split_root / "val_samples.parquet", index=False)
    _synthetic_frame(n_rows=64, seed=46).to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "artifacts"
    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        feature_set_name="paper_no_accel_v2",
        hidden_sizes=(16, 16),
        batch_size=32,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=43,
        num_workers=0,
        use_amp=False,
        target_loss_weights="fx_b=1,fy_b=0.5,fz_b=1,mx_b=0.5,my_b=1,mz_b=0.5",
        loss_type="huber",
        huber_delta=0.75,
    )

    training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    payload = torch.load(outputs["model_bundle_path"], map_location="cpu")

    assert training_config["loss_type"] == "huber"
    assert training_config["huber_delta"] == 0.75
    assert payload["loss_type"] == "huber"
    assert payload["huber_delta"] == 0.75


def test_run_training_job_records_sequence_training_tricks(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 310, "train_log"), ("val", 311, "val_log"), ("test", 312, "test_log")]:
        frame = _synthetic_frame(n_rows=96, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "artifacts",
        feature_set_name="paper_no_accel_v2",
        model_type="causal_transformer",
        hidden_sizes=(16, 8),
        batch_size=16,
        max_epochs=2,
        early_stopping_patience=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=310,
        num_workers=0,
        use_amp=False,
        sequence_history_size=4,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_dim_feedforward=32,
        lr_scheduler="warmup_cosine",
        lr_warmup_ratio=0.25,
        gradient_clip_norm=1.0,
        ema_decay=0.9,
    )

    training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    history = pd.read_csv(outputs["history_path"])
    payload = torch.load(outputs["model_bundle_path"], map_location="cpu")

    assert training_config["lr_scheduler"] == "warmup_cosine"
    assert training_config["lr_warmup_ratio"] == 0.25
    assert training_config["gradient_clip_norm"] == 1.0
    assert training_config["ema_decay"] == 0.9
    assert payload["lr_scheduler"] == "warmup_cosine"
    assert payload["gradient_clip_norm"] == 1.0
    assert payload["ema_decay"] == 0.9
    assert "learning_rate" in history.columns


def test_prediction_metadata_frame_for_sequence_bundle_aligns_rows(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    for split, seed, log_id in [("train", 410, "train_log"), ("val", 411, "val_log"), ("test", 412, "test_log")]:
        frame = _synthetic_frame(n_rows=80, seed=seed)
        frame["log_id"] = log_id
        frame["segment_id"] = 0
        frame["time_s"] = np.arange(len(frame), dtype=float) * 0.01
        frame.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    outputs = run_training_job(
        split_root=split_root,
        output_dir=tmp_path / "model",
        feature_set_name="paper_no_accel_v2",
        model_type="causal_transformer",
        hidden_sizes=(16, 8),
        batch_size=8,
        max_epochs=1,
        early_stopping_patience=1,
        sequence_history_size=4,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_dim_feedforward=32,
        device="cpu",
        random_seed=410,
        num_workers=0,
        use_amp=False,
    )
    bundle = torch.load(outputs["model_bundle_path"], map_location="cpu", weights_only=False)
    test_frame = pd.read_parquet(split_root / "test_samples.parquet")

    aligned = training_module.prediction_metadata_frame_for_bundle(
        bundle,
        test_frame,
        split_name="test",
        batch_size=16,
    )

    assert len(aligned) == 77
    assert {"log_id", "segment_id", "time_s"}.issubset(aligned.columns)
    assert {"servo_rudder", "airspeed_validated.true_airspeed_m_s", "cycle_flap_frequency_hz"}.issubset(aligned.columns)
    assert {"true_fy_b", "pred_fy_b", "resid_fy_b"}.issubset(aligned.columns)
    assert set(aligned["split"]) == {"test"}


def test_lateral_diagnostics_compute_target_scale_table():
    from scripts.run_lateral_diagnostics import compute_target_scale_table

    frame = pd.DataFrame(
        {
            "true_fy_b": [0.0, 1.0, -1.0],
            "pred_fy_b": [0.0, 0.5, -0.5],
            "true_fx_b": [0.0, 2.0, -2.0],
            "pred_fx_b": [0.0, 1.0, -1.0],
        }
    )

    table = compute_target_scale_table(frame, target_columns=["fy_b", "fx_b"])

    assert set(table["target"]) == {"fy_b", "fx_b"}
    assert {"true_std", "rmse", "rmse_over_std", "mean_abs_true", "p95_abs_true"}.issubset(table.columns)


def test_lateral_diagnostics_compute_per_log_table():
    from scripts.run_lateral_diagnostics import compute_per_log_lateral_table

    frame = pd.DataFrame(
        {
            "log_id": ["a", "a", "b", "b"],
            "true_fy_b": [0.0, 1.0, 0.0, 2.0],
            "pred_fy_b": [0.0, 1.0, 0.0, 0.0],
            "true_mx_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mx_b": [0.0, 1.0, 0.0, 0.0],
            "true_mz_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mz_b": [0.0, 1.0, 0.0, 0.0],
        }
    )

    table = compute_per_log_lateral_table(frame, lateral_targets=["fy_b", "mx_b", "mz_b"])

    assert list(table["log_id"]) == ["b", "a"]
    assert {"fy_b_rmse", "mx_b_r2", "lateral_rmse_mean", "lateral_r2_mean"}.issubset(table.columns)
    assert table.loc[0, "sample_count"] == 2


def test_lateral_diagnostics_compute_regime_table():
    from scripts.run_lateral_diagnostics import compute_regime_lateral_table

    frame = pd.DataFrame(
        {
            "airspeed_validated.true_airspeed_m_s": [5.0, 7.0, 9.0, 11.0],
            "cycle_flap_frequency_hz": [2.0, 4.0, 6.0, 8.0],
            "phase_corrected_rad": [0.1, 1.0, 3.0, 5.0],
            "servo_rudder": [-0.2, 0.0, 0.2, 0.3],
            "servo_left_elevon": [0.2, 0.1, -0.2, -0.3],
            "servo_right_elevon": [0.0, 0.1, 0.0, 0.2],
            "true_fy_b": [0.0, 1.0, 0.0, 2.0],
            "pred_fy_b": [0.0, 1.0, 0.0, 0.0],
            "true_mx_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mx_b": [0.0, 1.0, 0.0, 0.0],
            "true_mz_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mz_b": [0.0, 1.0, 0.0, 0.0],
        }
    )

    table = compute_regime_lateral_table(frame, lateral_targets=["fy_b", "mx_b", "mz_b"], min_samples=1)

    assert {"regime", "bin", "sample_count", "lateral_rmse_mean", "lateral_r2_mean"}.issubset(table.columns)
    assert set(table["regime"]) >= {"airspeed_validated.true_airspeed_m_s", "elevon_diff"}


def test_lateral_diagnostics_compute_with_without_suspect_log_table():
    from scripts.run_lateral_diagnostics import compute_with_without_suspect_log_table

    frame = pd.DataFrame(
        {
            "log_id": ["bad", "bad", "good", "good"],
            "true_fy_b": [0.0, 2.0, 0.0, 2.0],
            "pred_fy_b": [0.0, 0.0, 0.0, 2.0],
            "true_mx_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mx_b": [0.0, 0.0, 0.0, 1.0],
            "true_mz_b": [0.0, 1.0, 0.0, 1.0],
            "pred_mz_b": [0.0, 0.0, 0.0, 1.0],
        }
    )

    table = compute_with_without_suspect_log_table(frame, suspect_logs=["bad"], target_columns=["fy_b", "mx_b", "mz_b"])

    assert set(table["case"]) == {"all", "without_suspect", "suspect_only"}
    assert table.loc[table["case"] == "without_suspect", "fy_b_r2"].iloc[0] == pytest.approx(1.0)
    assert table.loc[table["case"] == "suspect_only", "fy_b_r2"].iloc[0] < 0.0


def test_lateral_diagnostics_estimates_integer_lag():
    from scripts.run_lateral_diagnostics import estimate_best_lag

    true = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
    pred = np.roll(true, 3)

    result = estimate_best_lag(true, pred, max_lag=8)

    assert result["best_lag"] == 3
    assert result["best_corr"] > result["zero_lag_corr"]
    assert result["best_rmse"] < result["zero_lag_rmse"]


def test_lateral_diagnostics_phase_lag_table_preserves_log_ids():
    from scripts.run_lateral_diagnostics import compute_phase_lag_lateral_table

    frame = pd.DataFrame(
        {
            "log_id": ["log_a"] * 8 + ["log_b"] * 8,
            "segment_id": [0] * 16,
            "time_s": list(range(8)) + list(range(8)),
            "true_fy_b": [0, 0, 1, 0, 0, 0, 0, 0] * 2,
            "pred_fy_b": [0, 0, 0, 1, 0, 0, 0, 0] * 2,
        }
    )

    table = compute_phase_lag_lateral_table(frame, lateral_targets=["fy_b"], max_lag=2)

    assert set(table["log_id"]) == {"log_a", "log_b"}
    assert set(table["best_lag"]) == {1}


def test_lateral_diagnostics_residual_correlation_table():
    from scripts.run_lateral_diagnostics import compute_residual_correlation_table

    feature = np.linspace(-1.0, 1.0, 20)
    frame = pd.DataFrame(
        {
            "servo_rudder": feature,
            "phase_corrected_rad": np.linspace(0.0, 2.0 * np.pi, 20),
            "true_fy_b": feature,
            "pred_fy_b": np.zeros_like(feature),
            "true_mx_b": np.ones_like(feature),
            "pred_mx_b": np.ones_like(feature),
        }
    )

    table = compute_residual_correlation_table(frame, targets=["fy_b", "mx_b"], feature_columns=["servo_rudder"])

    top = table.sort_values("abs_corr", ascending=False).iloc[0]
    assert top["target"] == "fy_b"
    assert top["feature"] == "servo_rudder"
    assert top["corr"] == pytest.approx(1.0)


def test_run_training_job_records_window_config(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    train = _synthetic_frame(n_rows=96, seed=54)
    val = _synthetic_frame(n_rows=64, seed=55)
    test = _synthetic_frame(n_rows=64, seed=56)
    for frame, log_id in [(train, "train_log"), (val, "val_log"), (test, "test_log")]:
        frame["log_id"] = log_id

    train.to_parquet(split_root / "train_samples.parquet", index=False)
    val.to_parquet(split_root / "val_samples.parquet", index=False)
    test.to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "artifacts"
    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        feature_set_name="paper_no_accel_v2",
        hidden_sizes=(16, 16),
        batch_size=32,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=53,
        num_workers=0,
        use_amp=False,
        window_mode="causal",
        window_radius=2,
        window_feature_mode="phase_actuator",
    )

    training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    payload = torch.load(outputs["model_bundle_path"], map_location="cpu")

    assert training_config["window_mode"] == "causal"
    assert training_config["window_radius"] == 2
    assert training_config["window_feature_mode"] == "phase_actuator"
    assert payload["window_mode"] == "causal"
    assert payload["window_radius"] == 2
    assert payload["window_feature_mode"] == "phase_actuator"
    assert "motor_cmd_0@t-2" in training_config["feature_columns"]
    assert "vehicle_local_position.vx@t-2" not in training_config["feature_columns"]
    assert "vehicle_local_position.vx@t+0" in training_config["feature_columns"]


def test_run_training_job_accepts_pfnn_model_type(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir(parents=True)

    _synthetic_frame(n_rows=160, seed=24).to_parquet(split_root / "train_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=25).to_parquet(split_root / "val_samples.parquet", index=False)
    _synthetic_frame(n_rows=80, seed=26).to_parquet(split_root / "test_samples.parquet", index=False)

    output_dir = tmp_path / "artifacts"
    outputs = run_training_job(
        split_root=split_root,
        output_dir=output_dir,
        feature_set_name="paper_pfnn_10",
        model_type="pfnn",
        hidden_sizes=(16, 16),
        batch_size=64,
        max_epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        random_seed=23,
        num_workers=0,
        use_amp=False,
        pfnn_expanded_input_dim=18,
        pfnn_phase_node_count=3,
        pfnn_control_points=6,
    )

    training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
    payload = torch.load(outputs["model_bundle_path"], map_location="cpu")

    assert training_config["feature_set_name"] == "paper_pfnn_10"
    assert training_config["model_type"] == "pfnn"
    assert training_config["feature_columns"] == PAPER_PFNN_10_FEATURE_COLUMNS
    assert training_config["phase_feature_column"] == "phase_corrected_rad"
    assert payload["model_type"] == "pfnn"
    assert payload["phase_feature_index"] == 0
