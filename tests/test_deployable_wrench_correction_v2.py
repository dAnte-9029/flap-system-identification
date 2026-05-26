from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_deployable_wrench_correction_v2 import (
    FORCE_TARGETS,
    MOMENT_TARGETS,
    build_v2_feature_frame,
    cross_arm_force,
    fit_force_correction_models,
    fit_moment_correction_models,
    force_feature_group_specs,
    force_metrics,
    moment_metrics,
    predict_force_correction,
    predict_moment_model,
    v2_feature_groups,
)


def test_build_v2_feature_frame_adds_rates_controls_lateral_and_interactions() -> None:
    frame = pd.DataFrame(
        {
            "phase_corrected_rad": [0.0, np.pi / 2.0],
            "cycle_flap_frequency_hz": [4.0, 5.0],
            "airspeed_validated.true_airspeed_m_s": [10.0, 12.0],
            "vehicle_air_data.rho": [1.2, 1.1],
            "alpha_rad": [0.1, 0.2],
            "vehicle_angular_velocity.xyz[0]": [0.01, 0.02],
            "vehicle_angular_velocity.xyz[1]": [0.03, 0.04],
            "vehicle_angular_velocity.xyz[2]": [0.05, 0.06],
            "servo_left_elevon": [0.2, 0.3],
            "servo_right_elevon": [0.1, 0.2],
            "servo_rudder": [-0.1, 0.2],
            "air_relative_velocity_b_x": [10.0, 12.0],
            "air_relative_velocity_b_y": [1.0, -1.0],
            "air_relative_velocity_b_z": [-0.5, -0.6],
        }
    )

    features, spec = build_v2_feature_frame(frame)
    groups = v2_feature_groups(features.columns)

    assert "body_rate_p" in features
    assert "body_rate_q" in features
    assert "body_rate_r" in features
    assert "q_dyn_x_body_rate_r" in features
    assert "servo_rudder" in features
    assert "elevon_sum_proxy" in features
    assert "elevon_diff_proxy" in features
    assert "beta_proxy_rad" in features
    assert "body_rate_r_x_phase_sin_1" in features
    assert "servo_rudder_x_phase_cos_1" in features
    assert "elevon_diff_x_phase_sin_1" in features
    assert "body_rate_r" in groups["rates"]
    assert "servo_rudder" in groups["controls"]
    assert "beta_proxy_rad" in groups["lateral"]
    assert "body_rate_r_x_phase_sin_1" in groups["interactions"]
    assert spec["uses_true_force"] is False


def _force_frame(rate: np.ndarray, rudder: np.ndarray, base_signal: np.ndarray | None = None) -> pd.DataFrame:
    n = len(rate)
    base_signal = np.zeros(n) if base_signal is None else base_signal
    frame = pd.DataFrame(
        {
            "phase_sin_1": base_signal,
            "phase_cos_1": np.ones(n),
            "phase_sin_2": np.zeros(n),
            "phase_cos_2": np.ones(n),
            "alpha_rad": np.zeros(n),
            "flap_frequency_hz": np.full(n, 5.0),
            "true_airspeed_m_s": np.full(n, 10.0),
            "dynamic_pressure_pa": np.full(n, 60.0),
            "alpha_rad_x_phase_sin_1": np.zeros(n),
            "alpha_rad_x_phase_cos_1": np.zeros(n),
            "flap_frequency_hz_x_phase_sin_1": np.zeros(n),
            "flap_frequency_hz_x_phase_cos_1": np.full(n, 5.0),
            "true_airspeed_m_s_x_phase_sin_1": np.zeros(n),
            "true_airspeed_m_s_x_phase_cos_1": np.full(n, 10.0),
            "alpha_rad_x_flap_frequency_hz": np.zeros(n),
            "body_rate_p": np.zeros(n),
            "body_rate_q": np.zeros(n),
            "body_rate_r": rate,
            "q_dyn_x_body_rate_p": np.zeros(n),
            "q_dyn_x_body_rate_q": np.zeros(n),
            "q_dyn_x_body_rate_r": 60.0 * rate,
            "servo_rudder": rudder,
            "servo_left_elevon": np.zeros(n),
            "servo_right_elevon": np.zeros(n),
            "elevon_sum_proxy": np.zeros(n),
            "elevon_diff_proxy": np.zeros(n),
            "q_dyn_x_servo_rudder": 60.0 * rudder,
            "q_dyn_x_servo_left_elevon": np.zeros(n),
            "q_dyn_x_servo_right_elevon": np.zeros(n),
            "beta_proxy_rad": np.zeros(n),
            "v_air_b_y": np.zeros(n),
            "q_dyn_x_beta_proxy": np.zeros(n),
        }
    )
    for target in FORCE_TARGETS:
        frame[f"prior_{target}"] = 0.0
    frame["label_fx_b"] = 0.0
    frame["label_fy_b"] = 2.0 * rate - 0.5 * rudder
    frame["label_fz_b"] = 0.0
    frame["true_force_fx_b"] = 1000.0
    frame["true_force_fy_b"] = -1000.0
    frame["true_force_fz_b"] = 500.0
    return frame


def test_force_selection_uses_validation_not_test_and_ignores_true_force_columns() -> None:
    train = _force_frame(np.linspace(-1.0, 1.0, 40), np.sin(np.linspace(-2.0, 2.0, 40)))
    val = _force_frame(np.linspace(-0.9, 0.9, 24), np.sin(np.linspace(-1.8, 1.8, 24)))
    test = _force_frame(
        np.linspace(-1.0, 1.0, 24),
        np.linspace(0.3, -0.3, 24),
        base_signal=np.linspace(-1.0, 1.0, 24),
    )
    test["label_fy_b"] = 0.0
    groups = force_feature_group_specs(v2_feature_groups(train.columns))

    metrics, selected = fit_force_correction_models(
        {"train": train, "val": val, "test": test},
        groups,
        alphas=(0.0, 1.0),
        variants=("additive",),
    )
    predicted = predict_force_correction(selected["model"], test)
    changed_true_force = test.copy()
    changed_true_force.loc[:, ["true_force_fx_b", "true_force_fy_b", "true_force_fz_b"]] = 1.0e9

    assert selected["feature_group"] == "base+rates+controls"
    assert selected["selection_split"] == "val"
    assert selected["uses_true_force_for_inference"] is False
    assert np.allclose(predicted, predict_force_correction(selected["model"], changed_true_force))
    val_base = metrics.query("split == 'val' and feature_group == 'base' and target == 'force_mean'")["rmse"].min()
    val_selected = metrics.query("split == 'val' and is_selected and target == 'force_mean'")["rmse"].iloc[0]
    assert val_selected < val_base
    test_best_group = (
        metrics.query("split == 'test' and target == 'force_mean' and feature_group != 'prior'")
        .sort_values("rmse")
        .iloc[0]["feature_group"]
    )
    assert test_best_group != selected["feature_group"]


def test_force_metrics_reports_mean_row() -> None:
    true_force = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]])
    pred_force = np.array([[0.0, 2.0, 2.0], [0.0, 1.0, 5.0]])

    metrics = force_metrics(split="unit", model_name="demo", true_force=true_force, predicted_force=pred_force)

    assert set(metrics["target"]) == {*FORCE_TARGETS, "force_mean"}
    assert metrics.loc[metrics["target"].eq("force_mean"), "rmse"].iloc[0] > 0.0


def test_metric_mean_rows_do_not_warn_when_all_r2_values_are_nan() -> None:
    constant_force = np.ones((3, 3))
    constant_moment = np.ones((3, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        force = force_metrics(split="unit", model_name="constant", true_force=constant_force, predicted_force=constant_force)
        moment = moment_metrics(split="unit", model_name="constant", y_true=constant_moment, y_pred=constant_moment)

    assert np.isnan(force.loc[force["target"].eq("force_mean"), "r2"].iloc[0])
    assert np.isnan(moment.loc[moment["target"].eq("moment_mean"), "r2"].iloc[0])


def _moment_frame(
    rate: np.ndarray,
    force_y: np.ndarray,
    base_signal: np.ndarray | None = None,
    rudder: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(rate)
    base_signal = np.zeros(n) if base_signal is None else base_signal
    rudder = np.zeros(n) if rudder is None else rudder
    frame = _force_frame(rate, rudder, base_signal=base_signal)
    frame["force_v2_fx_b"] = 1.0
    frame["force_v2_fy_b"] = force_y
    frame["force_v2_fz_b"] = 2.0
    for target in MOMENT_TARGETS:
        frame[f"prior_{target}"] = 0.0
    arm_signal = rate + 0.25 * rudder
    frame["label_mx_b"] = 0.0
    frame["label_my_b"] = -2.0 * arm_signal
    frame["label_mz_b"] = arm_signal * force_y
    frame["true_force_fx_b"] = -123.0
    frame["true_force_fy_b"] = -456.0
    frame["true_force_fz_b"] = -789.0
    return frame


def test_moment_forms_and_selection_are_deployable_and_validation_only() -> None:
    train_rate = np.linspace(-1.0, 1.0, 50)
    val_rate = np.linspace(-0.8, 0.8, 30)
    train_rudder = np.sin(np.linspace(-2.0, 2.0, 50))
    val_rudder = np.sin(np.linspace(-1.8, 1.8, 30))
    train = _moment_frame(train_rate, 3.0 + 0.5 * train_rate**2, rudder=train_rudder)
    val = _moment_frame(val_rate, 3.0 + 0.5 * val_rate**2, rudder=val_rudder)
    test = _moment_frame(
        np.linspace(-1.0, 1.0, 30),
        np.full(30, 3.0),
        base_signal=np.linspace(-1.0, 1.0, 30),
    )
    test["label_my_b"] = 0.0
    test["label_mz_b"] = 0.0
    groups = force_feature_group_specs(v2_feature_groups(train.columns))

    metrics, selected = fit_moment_correction_models(
        {"train": train, "val": val, "test": test},
        groups,
        alphas=(0.0, 1.0),
        model_forms=("direct_residual", "force_arm", "hybrid"),
    )
    prediction = predict_moment_model(selected["model"], test)
    changed_true_force = test.copy()
    changed_true_force.loc[:, ["true_force_fx_b", "true_force_fy_b", "true_force_fz_b"]] = 1.0e9

    assert np.allclose(cross_arm_force(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 3.0, 2.0]])), [[0.0, -2.0, 3.0]])
    assert {"direct_residual", "force_arm", "hybrid"}.issubset(set(metrics["model_form"]))
    assert selected["model_form"] in {"force_arm", "hybrid"}
    assert selected["feature_group"] == "base+rates+controls"
    assert selected["selection_split"] == "val"
    assert selected["uses_true_force_for_inference"] is False
    assert np.allclose(prediction["moment"], predict_moment_model(selected["model"], changed_true_force)["moment"])
    test_best_group = (
        metrics.query("split == 'test' and target == 'moment_mean' and feature_group != 'prior'")
        .sort_values("rmse")
        .iloc[0]["feature_group"]
    )
    assert test_best_group != selected["feature_group"]


def test_force_arm_model_includes_free_torque_term() -> None:
    rate = np.linspace(-1.0, 1.0, 50)
    frame = _moment_frame(rate, np.full_like(rate, 3.0))
    frame["label_mx_b"] = rate
    frame["label_my_b"] = 3.0 * rate
    frame["label_mz_b"] = 2.0 * rate
    groups = force_feature_group_specs(v2_feature_groups(frame.columns))

    _, selected = fit_moment_correction_models(
        {"train": frame, "val": frame, "test": frame},
        {"base+rates+controls": groups["base+rates+controls"]},
        alphas=(0.0,),
        model_forms=("force_arm",),
    )
    prediction = predict_moment_model(selected["model"], frame)

    assert selected["model_form"] == "force_arm"
    assert np.linalg.norm(prediction["tau_free"]) > 1.0e-6
    assert np.allclose(
        prediction["moment"],
        np.column_stack([frame["label_mx_b"], frame["label_my_b"], frame["label_mz_b"]]),
        atol=1.0e-8,
    )


def _write_cli_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    split_root = root / "split"
    prior_root = root / "prior"
    force_v1_root = root / "force_v1"
    moment_v1_root = root / "moment_v1"
    for path in (split_root, prior_root, force_v1_root / "prediction_parquets", moment_v1_root):
        path.mkdir(parents=True, exist_ok=True)
    for split, offset in (("train", 0.0), ("val", 0.1), ("test", 0.2)):
        n = 12
        rate = np.linspace(-1.0, 1.0, n) + offset
        samples = pd.DataFrame(
            {
                "timestamp_us": np.arange(n),
                "time_s": np.arange(n, dtype=float) * 0.01,
                "log_id": [f"{split}_log"] * n,
                "segment_id": 0,
                "cycle_id": np.arange(n),
                "phase_corrected_rad": np.linspace(0.0, 2.0 * np.pi, n),
                "cycle_flap_frequency_hz": 5.0,
                "airspeed_validated.true_airspeed_m_s": 10.0,
                "vehicle_air_data.rho": 1.2,
                "alpha_rad": 0.0,
                "vehicle_angular_velocity.xyz[0]": rate,
                "vehicle_angular_velocity.xyz[1]": 0.0,
                "vehicle_angular_velocity.xyz[2]": rate,
                "servo_left_elevon": 0.0,
                "servo_right_elevon": 0.0,
                "servo_rudder": 0.0,
                "air_relative_velocity_b_x": 10.0,
                "air_relative_velocity_b_y": 0.0,
                "air_relative_velocity_b_z": 0.0,
                "fx_b": 0.0,
                "fy_b": 2.0 * rate,
                "fz_b": 0.0,
                "mx_b": 0.0,
                "my_b": -2.0 * rate,
                "mz_b": 3.0 * rate,
            }
        )
        samples.to_parquet(split_root / f"{split}_samples.parquet", index=False)
        prior = pd.DataFrame({target: np.zeros(n) for target in (*FORCE_TARGETS, *MOMENT_TARGETS)})
        prior.to_parquet(prior_root / f"{split}_predictions.parquet", index=False)
        force_v1 = pd.DataFrame(
            {
                **{f"label_{target}": samples[target] for target in FORCE_TARGETS},
                **{f"prior_{target}": 0.0 for target in FORCE_TARGETS},
                **{f"corrected_{target}": 0.0 for target in FORCE_TARGETS},
            }
        )
        force_v1.to_parquet(force_v1_root / "prediction_parquets" / f"{split}_predictions.parquet", index=False)
    pd.DataFrame(
        [
            {"split": "test", "model": "dynamic_arm_linear", "force_source": "corrected_force", "target": "moment_mean", "rmse": 1.0},
            {"split": "test", "model": "dynamic_arm_linear", "force_source": "true_force", "target": "moment_mean", "rmse": 0.1},
        ]
    ).to_csv(moment_v1_root / "metrics_by_split.csv", index=False)
    (moment_v1_root / "model_config.json").write_text(json.dumps({"best_model": {"force_source": "true_force"}}), encoding="utf-8")
    return split_root, prior_root, force_v1_root, moment_v1_root


def test_cli_writes_outputs_and_protects_existing_baselines(tmp_path: Path) -> None:
    split_root, prior_root, force_v1_root, moment_v1_root = _write_cli_fixture(tmp_path)
    output_root = tmp_path / "out"
    command = [
        sys.executable,
        "scripts/train_deployable_wrench_correction_v2.py",
        "--split-root",
        str(split_root),
        "--prior-root",
        str(prior_root),
        "--force-v1-root",
        str(force_v1_root),
        "--moment-v1-root",
        str(moment_v1_root),
        "--output-root",
        str(output_root),
        "--alphas",
        "0,1,10",
    ]

    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    assert result.returncode == 0, result.stderr
    for relative in (
        "force_metrics_by_split.csv",
        "moment_metrics_by_split.csv",
        "model_selection.csv",
        "feature_group_ablation.csv",
        "per_log_metrics.csv",
        "prediction_parquets/train_predictions.parquet",
        "prediction_parquets/val_predictions.parquet",
        "prediction_parquets/test_predictions.parquet",
        "coefficients_or_model.npz",
        "inference_model_state.json",
        "model_config.json",
        "README.md",
    ):
        assert (output_root / relative).exists(), relative
    config = json.loads((output_root / "model_config.json").read_text(encoding="utf-8"))
    assert config["uses_true_force_for_inference"] is False
    assert config["selection_split"] == "val"
    assert config["selected_force_source"] == "corrected_force_v2"
    assert config["excluded_diagnostic_only_rows"]
    state = json.loads((output_root / "inference_model_state.json").read_text(encoding="utf-8"))
    assert state["uses_true_force_for_inference"] is False
    assert state["force_model"]["feature_mean"]
    assert state["force_model"]["feature_scale"]
    assert state["moment_model"]["feature_transform"]["mean"]
    assert state["moment_model"]["free_coefficients"]

    protected = subprocess.run(
        command[:-6]
        + [
            "--output-root",
            str(force_v1_root),
            "--alphas",
            "0,1,10",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert protected.returncode != 0
    assert "refusing to write" in protected.stderr
