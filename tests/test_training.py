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
    evaluate_model_bundle,
    fit_torch_regressor,
    prepare_feature_target_frames,
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
            "airspeed_validated.true_airspeed_m_s": rng.normal(8.0, 0.6, size=n_rows),
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
