from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from system_identification.training import (
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_TARGET_COLUMNS,
    HybridPFNNRegressor,
    NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS,
    PAPER_NO_ACCEL_V2_FEATURE_COLUMNS,
    PAPER_PFNN_10_FEATURE_COLUMNS,
    cyclic_catmull_rom_weights,
    evaluate_model_bundle,
    fit_torch_regressor,
    prepare_feature_target_frames,
    resolve_target_loss_weights,
    resolve_feature_set_columns,
    resolve_ablation_variants,
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
