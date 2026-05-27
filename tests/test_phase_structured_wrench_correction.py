from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.train_phase_structured_wrench_correction as phase_correction
from scripts.train_phase_structured_wrench_correction import (
    FORCE_TARGETS,
    MOMENT_TARGETS,
    _force_design_for_family,
    build_phase_structured_feature_families,
    cross_arm_force,
    fit_phase_structured_force_models,
    fit_phase_structured_moment_models,
    phase_structured_force_family_specs,
    predict_force_correction,
    run_phase_structured_experiment,
    select_force_candidate,
    select_moment_candidate,
)


def _raw_feature_frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase_corrected_rad": np.linspace(0.0, np.pi, n),
            "cycle_flap_frequency_hz": np.linspace(4.0, 5.0, n),
            "airspeed_validated.true_airspeed_m_s": np.linspace(10.0, 12.0, n),
            "vehicle_air_data.rho": np.full(n, 1.2),
            "alpha_rad": np.linspace(0.05, 0.15, n),
            "vehicle_angular_velocity.xyz[0]": np.linspace(-0.2, 0.2, n),
            "vehicle_angular_velocity.xyz[1]": np.linspace(0.1, -0.1, n),
            "vehicle_angular_velocity.xyz[2]": np.linspace(-0.4, 0.4, n),
            "servo_left_elevon": np.linspace(0.1, 0.2, n),
            "servo_right_elevon": np.linspace(-0.1, -0.2, n),
            "servo_rudder": np.linspace(-0.3, 0.3, n),
            "air_relative_velocity_b_x": np.full(n, 10.0),
            "air_relative_velocity_b_y": np.linspace(-1.0, 1.0, n),
            "air_relative_velocity_b_z": np.linspace(0.2, -0.2, n),
            "fx_b": np.arange(n),
            "fy_b": np.arange(n),
            "fz_b": np.arange(n),
            "mx_b": np.arange(n),
            "my_b": np.arange(n),
            "mz_b": np.arange(n),
            "label_fx_b": np.arange(n),
        }
    )


def test_phase_structured_feature_families_are_disjoint_and_deployable() -> None:
    features, families, metadata = build_phase_structured_feature_families(_raw_feature_frame())

    assert metadata["uses_true_force"] is False
    assert set(families) == {
        "prior",
        "slow_only",
        "phase_only",
        "phase_structured",
        "phase_structured_plus_rates_controls",
    }
    assert families["prior"] == []
    assert not any(column.startswith(("phase_sin_", "phase_cos_")) for column in families["slow_only"])
    assert all(column.startswith(("phase_sin_", "phase_cos_")) for column in families["phase_only"])
    assert {"phase_sin_1", "phase_cos_1", "phase_sin_2", "phase_cos_2"}.issubset(families["phase_structured"])
    assert any("_x_phase_sin_1" in column for column in families["phase_structured"])
    assert {"body_rate_p", "body_rate_q", "body_rate_r", "servo_rudder", "beta_proxy_rad", "v_air_b_y"}.issubset(
        families["phase_structured_plus_rates_controls"]
    )
    assert any(column.endswith("_x_phase_cos_1") for column in families["phase_structured_plus_rates_controls"])

    forbidden = {"fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"}
    for columns in families.values():
        assert not forbidden.intersection(columns)
        assert not any(column.startswith("label_") for column in columns)
        assert set(columns).issubset(features.columns)

    rebuilt = phase_structured_force_family_specs(features.columns, {})
    assert rebuilt == families


def _force_frame(rate: np.ndarray, rudder: np.ndarray, phase_signal: np.ndarray | None = None) -> pd.DataFrame:
    n = len(rate)
    phase_signal = np.zeros(n) if phase_signal is None else phase_signal
    frame = pd.DataFrame(
        {
            "phase_sin_1": phase_signal,
            "phase_cos_1": np.ones(n),
            "phase_sin_2": np.zeros(n),
            "phase_cos_2": np.ones(n),
            "alpha_rad": np.zeros(n),
            "flap_frequency_hz": np.full(n, 5.0),
            "true_airspeed_m_s": np.full(n, 10.0),
            "dynamic_pressure_pa": np.full(n, 60.0),
            "alpha_rad_x_phase_sin_1": np.zeros(n),
            "alpha_rad_x_phase_cos_1": np.zeros(n),
            "flap_frequency_hz_x_phase_sin_1": 5.0 * phase_signal,
            "flap_frequency_hz_x_phase_cos_1": np.full(n, 5.0),
            "true_airspeed_m_s_x_phase_sin_1": 10.0 * phase_signal,
            "true_airspeed_m_s_x_phase_cos_1": np.full(n, 10.0),
            "alpha_rad_x_flap_frequency_hz": np.zeros(n),
            "body_rate_p": rate,
            "body_rate_q": np.zeros(n),
            "body_rate_r": rate,
            "q_dyn_x_body_rate_p": 60.0 * rate,
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
            "body_rate_r_x_phase_sin_1": rate * phase_signal,
            "body_rate_r_x_phase_cos_1": rate,
            "servo_rudder_x_phase_sin_1": rudder * phase_signal,
            "servo_rudder_x_phase_cos_1": rudder,
            "beta_proxy_x_phase_sin_1": np.zeros(n),
            "beta_proxy_x_phase_cos_1": np.zeros(n),
        }
    )
    for target in FORCE_TARGETS:
        frame[f"prior_{target}"] = 0.0
    frame["label_fx_b"] = 0.0
    frame["label_fy_b"] = 2.0 * rate - 0.5 * rudder
    frame["label_fz_b"] = 0.0
    frame[["true_force_fx_b", "true_force_fy_b", "true_force_fz_b"]] = 123.0
    return frame


def test_force_design_and_selection_are_validation_only() -> None:
    features, families, _ = build_phase_structured_feature_families(_force_frame(np.linspace(-1.0, 1.0, 8), np.zeros(8)))
    prior_force = np.column_stack([np.ones(len(features)), np.full(len(features), 2.0), np.full(len(features), 3.0)])

    additive = _force_design_for_family(features, prior_force, families["phase_only"], "additive")
    affine = _force_design_for_family(features, prior_force, families["phase_only"], "affine")

    assert additive.columns.tolist() == families["phase_only"]
    assert {"prior_fx_b_x_phase_sin_1", "prior_fy_b_x_phase_sin_1", "prior_fz_b_x_phase_sin_1"}.issubset(affine.columns)

    selection_metrics = pd.DataFrame(
        [
            {"split": "val", "target": "force_mean", "family": "phase_structured", "variant": "affine", "alpha": 0.0, "rmse": 1.0},
            {"split": "val", "target": "force_mean", "family": "phase_only", "variant": "affine", "alpha": 0.0, "rmse": 1.0},
            {"split": "val", "target": "force_mean", "family": "slow_only", "variant": "additive", "alpha": 1.0, "rmse": 1.0},
            {"split": "test", "target": "force_mean", "family": "phase_structured", "variant": "additive", "alpha": 0.0, "rmse": 0.0},
        ]
    )
    selected_row = select_force_candidate(selection_metrics)
    assert selected_row["family"] == "slow_only"
    assert selected_row["variant"] == "additive"


def test_force_training_ignores_test_metric_and_true_force_columns() -> None:
    train = _force_frame(np.linspace(-1.0, 1.0, 48), np.sin(np.linspace(-2.0, 2.0, 48)))
    val = _force_frame(np.linspace(-0.8, 0.8, 32), np.sin(np.linspace(-1.8, 1.8, 32)))
    test = _force_frame(np.linspace(-1.0, 1.0, 32), np.linspace(0.4, -0.4, 32), np.linspace(-1.0, 1.0, 32))
    test["label_fy_b"] = 0.0
    families = phase_structured_force_family_specs(train.columns, {})

    metrics, selected = fit_phase_structured_force_models(
        {"train": train, "val": val, "test": test},
        families,
        alphas=(0.0, 1.0),
        variants=("additive",),
    )
    prediction = predict_force_correction(selected["model"], test)
    changed_true_force = test.copy()
    changed_true_force.loc[:, ["true_force_fx_b", "true_force_fy_b", "true_force_fz_b"]] = 1.0e9

    assert selected["family"] == "phase_structured_plus_rates_controls"
    assert selected["selection_split"] == "val"
    assert selected["uses_true_force_for_inference"] is False
    assert np.allclose(prediction, predict_force_correction(selected["model"], changed_true_force))
    test_best = metrics.query("split == 'test' and target == 'force_mean'").sort_values("rmse").iloc[0]
    assert test_best["family"] != selected["family"] or test_best["variant"] != selected["variant"]


def _moment_frame(rate: np.ndarray, force_y: np.ndarray, rudder: np.ndarray | None = None) -> pd.DataFrame:
    rudder = np.zeros(len(rate)) if rudder is None else rudder
    frame = _force_frame(rate, rudder)
    frame["force_corr_fx_b"] = 1.0
    frame["force_corr_fy_b"] = force_y
    frame["force_corr_fz_b"] = 2.0
    for target in MOMENT_TARGETS:
        frame[f"prior_{target}"] = 0.0
    arm_signal = rate + 0.25 * rudder
    frame["label_mx_b"] = 0.0
    frame["label_my_b"] = -2.0 * arm_signal
    frame["label_mz_b"] = arm_signal * force_y
    return frame


def test_moment_forms_and_selection_are_validation_only() -> None:
    force = np.array([[0.0, 3.0, 2.0]])
    arm = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(cross_arm_force(arm, force), [[0.0, -2.0, 3.0]])

    train_rate = np.linspace(-1.0, 1.0, 48)
    val_rate = np.linspace(-0.8, 0.8, 32)
    train_rudder = np.sin(np.linspace(-2.0, 2.0, 48))
    val_rudder = np.sin(np.linspace(-1.8, 1.8, 32))
    train = _moment_frame(train_rate, 3.0 + 0.5 * train_rate**2, train_rudder)
    val = _moment_frame(val_rate, 3.0 + 0.5 * val_rate**2, val_rudder)
    test = _moment_frame(np.linspace(-1.0, 1.0, 32), np.full(32, 3.0))
    for frame in (train, val, test):
        for target in FORCE_TARGETS:
            frame[f"force_corr_{target}"] = 0.0
    test["label_my_b"] = 0.0
    test["label_mz_b"] = 0.0
    families = phase_structured_force_family_specs(train.columns, {})

    metrics, selected = fit_phase_structured_moment_models(
        {"train": train, "val": val, "test": test},
        families,
        alphas=(0.0, 1.0),
    )

    assert {"direct_residual", "force_arm_only", "force_arm_plus_free", "hybrid_prior_arm_free"}.issubset(set(metrics["form"]))
    assert selected["form"] == "force_arm_plus_free"
    assert selected["selection_split"] == "val"
    assert selected["uses_true_force_for_inference"] is False
    selected_row = select_moment_candidate(
        pd.DataFrame(
            [
                {"split": "val", "target": "moment_mean", "form": "direct_residual", "feature_family": "phase_structured", "alpha": 1.0, "rmse": 1.0},
                {"split": "val", "target": "moment_mean", "form": "force_arm_plus_free", "feature_family": "phase_structured", "alpha": 1.0, "rmse": 1.0},
            ]
        )
    )
    assert selected_row["form"] == "force_arm_plus_free"


def _free_torque_design_matrix(phi: np.ndarray) -> np.ndarray:
    n, p = phi.shape
    design = np.zeros((n, 3, p * 3), dtype=float)
    for idx in range(p):
        for axis in range(3):
            design[:, axis, idx * 3 + axis] = phi[:, idx]
    return design.reshape(n * 3, p * 3)


def test_force_arm_plus_free_uses_joint_ridge_solution() -> None:
    rate = np.linspace(-1.0, 1.0, 48)
    rudder = np.sin(np.linspace(-2.0, 2.0, 48))
    frame = _moment_frame(rate, 3.0 + 0.5 * rate**2, rudder)
    columns = phase_structured_force_family_specs(frame.columns, {})["phase_structured_plus_rates_controls"]
    alpha = 0.1

    model = phase_correction._fit_moment_model(frame, columns, "force_arm_plus_free", alpha)
    features = phase_correction.build_v2_feature_frame(frame)[0].loc[:, columns]
    transform = phase_correction._fit_transform(features, columns)
    phi = transform.transform(features)
    force = phase_correction._array_from_prefixed_columns(frame, "force_corr", FORCE_TARGETS)
    y = phase_correction._array_from_prefixed_columns(frame, "label", MOMENT_TARGETS)
    design = np.hstack(
        [
            phase_correction._build_arm_design_matrix(phi, force),
            _free_torque_design_matrix(phi),
        ]
    )
    expected = phase_correction._ridge_solve(design, y, alpha)
    p = phi.shape[1]

    assert np.allclose(model.arm_coefficients, expected[: p * 3].reshape(p, 3))
    assert np.allclose(model.free_coefficients, expected[p * 3 :].reshape(p, 3))


def _write_cli_fixture(root: Path) -> tuple[Path, Path, Path]:
    split_root = root / "split"
    prior_root = root / "prior"
    v2_root = root / "v2"
    (v2_root / "prediction_parquets").mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)
    prior_root.mkdir(parents=True, exist_ok=True)
    for split, offset in (("train", 0.0), ("val", 0.1), ("test", 0.2)):
        n = 14
        rate = np.linspace(-1.0, 1.0, n) + offset
        samples = _raw_feature_frame(n)
        samples["timestamp_us"] = np.arange(n)
        samples["time_s"] = np.arange(n) * 0.01
        samples["log_id"] = f"{split}_log"
        samples["segment_id"] = 0
        samples["cycle_id"] = np.arange(n)
        samples["vehicle_angular_velocity.xyz[0]"] = rate
        samples["vehicle_angular_velocity.xyz[2]"] = rate
        samples["servo_rudder"] = 0.0
        samples["fx_b"] = 0.0
        samples["fy_b"] = 2.0 * rate
        samples["fz_b"] = 0.0
        samples["mx_b"] = 0.0
        samples["my_b"] = -2.0 * rate
        samples["mz_b"] = 3.0 * rate
        samples.to_parquet(split_root / f"{split}_samples.parquet", index=False)
        pd.DataFrame({target: np.zeros(n) for target in (*FORCE_TARGETS, *MOMENT_TARGETS)}).to_parquet(
            prior_root / f"{split}_predictions.parquet", index=False
        )
        pd.DataFrame({"placeholder": np.zeros(n)}).to_parquet(v2_root / "prediction_parquets" / f"{split}_predictions.parquet", index=False)
    return split_root, prior_root, v2_root


def test_run_phase_structured_experiment_writes_outputs(tmp_path: Path) -> None:
    split_root, prior_root, v2_root = _write_cli_fixture(tmp_path)
    output_root = tmp_path / "out"

    run_phase_structured_experiment(
        split_root=split_root,
        prior_root=prior_root,
        v2_reference_root=v2_root,
        output_root=output_root,
        alphas=(0.0, 1.0),
        overwrite=False,
        command="unit",
    )

    for relative in (
        "force_metrics_by_split.csv",
        "moment_metrics_by_split.csv",
        "force_model_selection.csv",
        "moment_model_selection.csv",
        "model_config.json",
        "inference_model_state.json",
        "prediction_parquets/test_predictions.parquet",
        "README.md",
    ):
        assert (output_root / relative).exists(), relative
    state = json.loads((output_root / "inference_model_state.json").read_text(encoding="utf-8"))
    assert state["uses_true_force_for_inference"] is False
    assert state["selection_protocol"] == "validation metrics select models; test metrics are final reporting only"
