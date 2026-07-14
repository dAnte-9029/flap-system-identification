from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_prior_vs_tcn_comparison import (
    TARGETS,
    apply_ood_split_filter,
    condition_bin_table,
    expand_experiment_configs,
    fit_global_gain_bias,
    fit_matrix_gain_bias,
    overall_metric_row,
    predict_global_gain_bias,
    predict_matrix_gain_bias,
    select_training_fraction,
)


def test_overall_metric_row_reports_component_and_vector_errors() -> None:
    truth = np.array([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]])
    pred = np.array([[0.0, -1.0], [4.0, -4.0], [5.0, -9.0]])

    row = overall_metric_row(
        model="demo",
        fold=2,
        frame=pd.DataFrame({"log_id": ["a", "a", "b"]}),
        truth=truth,
        pred=pred,
    )

    assert row["model"] == "demo"
    assert row["outer_fold"] == 2
    assert row["n_samples"] == 3
    assert row["n_logs"] == 2
    assert np.isclose(row["rmse_fx_b"], np.sqrt((1.0 + 1.0 + 0.0) / 3.0))
    assert np.isclose(row["mae_fz_b"], (1.0 + 0.0 + 3.0) / 3.0)
    assert np.isclose(row["rmse_force_norm"], np.sqrt((2.0 + 1.0 + 9.0) / 3.0))
    assert row["r2_fx_b"] < 1.0
    assert row["r2_fz_b"] < 1.0


def test_condition_bin_table_covers_airspeed_phase_aoa_and_stroke_bins() -> None:
    frame = pd.DataFrame(
        {
            "airspeed_validated.true_airspeed_m_s": [2.0, 5.0, 7.0, 9.0],
            "alpha_rad": np.deg2rad([-15.0, -2.0, 8.0, 20.0]),
            "phase_corrected_rad": [0.1, 1.7, 3.4, 5.2],
            "wing_stroke_direction": [1, -1, 1, -1],
        }
    )
    truth = np.zeros((4, 2))
    pred = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

    table = condition_bin_table(frame=frame, truth=truth, pred=pred, model="demo", fold=0, phase_bins=4)

    assert {"airspeed", "alpha_eff", "phase", "stroke"}.issubset(set(table["bin_family"]))
    assert set(table.loc[table["bin_family"].eq("airspeed"), "bin_label"]) == {"[0,4)", "[4,6)", "[6,8)", "[8,inf)"}
    assert table.loc[table["bin_family"].eq("phase"), "bin_index"].nunique() == 4
    assert set(table.loc[table["bin_family"].eq("stroke"), "bin_label"]) == {"downstroke", "upstroke"}


def test_condition_bin_table_accepts_string_stroke_direction() -> None:
    frame = pd.DataFrame(
        {
            "airspeed_validated.true_airspeed_m_s": [5.0, 5.0],
            "alpha_rad": [0.0, 0.0],
            "phase_corrected_rad": [0.0, 3.14],
            "wing_stroke_direction": ["downstroke", "upstroke"],
        }
    )
    truth = np.zeros((2, 2))
    pred = np.ones((2, 2))

    table = condition_bin_table(frame=frame, truth=truth, pred=pred, model="demo", fold=0, phase_bins=2)

    assert set(table.loc[table["bin_family"].eq("stroke"), "bin_label"]) == {"downstroke", "upstroke"}


def test_select_training_fraction_keeps_whole_segments_and_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "log_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "segment_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "time_s": np.arange(8, dtype=float),
            "value": np.arange(8, dtype=float),
        }
    )

    first = select_training_fraction(frame, fraction=0.5, seed=7)
    second = select_training_fraction(frame, fraction=0.5, seed=7)

    assert first.equals(second)
    assert set(first["log_id"]).issubset({"a", "b", "c", "d"})
    assert first.groupby(["log_id", "segment_id"]).size().nunique() == 1
    assert 0 < first["log_id"].nunique() < frame["log_id"].nunique()


def test_select_training_fraction_one_returns_full_frame_in_order() -> None:
    frame = pd.DataFrame(
        {
            "log_id": ["b", "a", "b"],
            "segment_id": [1, 0, 1],
            "time_s": [0.2, 0.1, 0.3],
        }
    )

    selected = select_training_fraction(frame, fraction=1.0, seed=3)

    pd.testing.assert_frame_equal(selected, frame.reset_index(drop=True))


def test_expand_sample_efficiency_configs_uses_requested_fractions() -> None:
    configs = expand_experiment_configs(
        experiment="sample_efficiency",
        train_fractions=(0.1, 0.2),
        capacity_presets=("base",),
    )

    assert [config["train_fraction"] for config in configs] == [0.1, 0.2]
    assert {config["capacity_preset"] for config in configs} == {"base"}


def test_expand_capacity_configs_uses_full_train_fraction() -> None:
    configs = expand_experiment_configs(
        experiment="capacity",
        train_fractions=(1.0,),
        capacity_presets=("tiny", "small", "base"),
    )

    assert [config["capacity_preset"] for config in configs] == ["tiny", "small", "base"]
    assert {config["train_fraction"] for config in configs} == {1.0}


def test_expand_ood_configs_uses_requested_presets() -> None:
    configs = expand_experiment_configs(
        experiment="ood",
        train_fractions=(1.0,),
        capacity_presets=("base",),
        ood_presets=("airspeed_ge8", "alpha_abs_ge20"),
    )

    assert [config["ood_preset"] for config in configs] == ["airspeed_ge8", "alpha_abs_ge20"]
    assert {config["train_fraction"] for config in configs} == {1.0}


def test_expand_prior_anchor_configs_keeps_requested_loss_weights() -> None:
    configs = expand_experiment_configs(
        experiment="prior_anchor",
        train_fractions=(0.1,),
        capacity_presets=("base",),
        prior_loss_weights=(0.0, 0.01, 0.1),
    )

    assert [config["prior_loss_weight"] for config in configs] == [0.0, 0.01, 0.1]
    assert all("__lambda_" in str(config["config_id"]) for config in configs)
    assert {config["train_fraction"] for config in configs} == {0.1}


def test_apply_ood_split_filter_uses_id_train_val_and_ood_test_for_airspeed() -> None:
    frames = {
        "train": pd.DataFrame({"airspeed_validated.true_airspeed_m_s": [5.0, 7.9, 8.1], "phase_corrected_rad": 0.0}),
        "val": pd.DataFrame({"airspeed_validated.true_airspeed_m_s": [4.0, 8.0, 9.0], "phase_corrected_rad": 0.0}),
        "test": pd.DataFrame({"airspeed_validated.true_airspeed_m_s": [6.0, 8.0, 10.0], "phase_corrected_rad": 0.0}),
    }

    filtered = apply_ood_split_filter(frames, ood_preset="airspeed_ge8")

    assert filtered["train"]["airspeed_validated.true_airspeed_m_s"].tolist() == [5.0, 7.9]
    assert filtered["val"]["airspeed_validated.true_airspeed_m_s"].tolist() == [4.0]
    assert filtered["test"]["airspeed_validated.true_airspeed_m_s"].tolist() == [8.0, 10.0]


def test_apply_ood_split_filter_uses_alpha_abs_threshold() -> None:
    frames = {
        "train": pd.DataFrame({"alpha_rad": np.deg2rad([0.0, 19.0, 21.0]), "phase_corrected_rad": 0.0}),
        "val": pd.DataFrame({"alpha_rad": np.deg2rad([-10.0, -25.0]), "phase_corrected_rad": 0.0}),
        "test": pd.DataFrame({"alpha_rad": np.deg2rad([5.0, -20.0, 30.0]), "phase_corrected_rad": 0.0}),
    }

    filtered = apply_ood_split_filter(frames, ood_preset="alpha_abs_ge20")

    assert np.all(np.abs(np.rad2deg(filtered["train"]["alpha_rad"])) < 20.0)
    assert np.all(np.abs(np.rad2deg(filtered["val"]["alpha_rad"])) < 20.0)
    assert np.all(np.abs(np.rad2deg(filtered["test"]["alpha_rad"])) >= 20.0)


def test_global_and_matrix_gain_bias_fit_expected_affine_forms() -> None:
    prior = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, -1.0], [4.0, -2.0]])
    truth = np.column_stack([2.0 * prior[:, 0] + 1.0, -3.0 * prior[:, 1] - 2.0])

    global_model = fit_global_gain_bias(prior, truth, alpha=0.0)
    global_pred = predict_global_gain_bias(global_model, prior)
    np.testing.assert_allclose(global_pred, truth, atol=1e-10)

    matrix_truth = prior @ np.array([[2.0, -1.0], [0.5, 3.0]]) + np.array([1.0, -2.0])
    matrix_model = fit_matrix_gain_bias(prior, matrix_truth, alpha=0.0)
    matrix_pred = predict_matrix_gain_bias(matrix_model, prior)
    np.testing.assert_allclose(matrix_pred, matrix_truth, atol=1e-10)
    assert TARGETS == ("fx_b", "fz_b")
