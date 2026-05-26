from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_component_residual_attribution import (
    build_candidate_variables,
    build_residual_frame,
    default_feature_groups,
    residual_variable_bin_table,
    residual_feature_group_ablation,
    residual_variable_ranking_table,
    summarize_residual_variable_bins,
)


def test_build_candidate_variables_derives_phase_lateral_and_control_proxies() -> None:
    frame = pd.DataFrame(
        {
            "phase_corrected_rad": [0.0, np.pi / 2.0],
            "cycle_flap_frequency_hz": [4.0, 5.0],
            "airspeed_validated.true_airspeed_m_s": [10.0, 20.0],
            "vehicle_air_data.rho": [1.2, 1.0],
            "alpha_rad": [0.1, 0.2],
            "vehicle_angular_velocity.xyz[0]": [0.01, 0.02],
            "vehicle_angular_velocity.xyz[1]": [0.03, 0.04],
            "vehicle_angular_velocity.xyz[2]": [0.05, 0.06],
            "actuator_servos.servo[0]": [0.1, 0.2],
            "actuator_servos.servo[1]": [0.3, 0.4],
            "actuator_servos.servo[2]": [0.5, 0.6],
            "air_relative_velocity_b_x": [10.0, 20.0],
            "air_relative_velocity_b_y": [1.0, -2.0],
            "air_relative_velocity_b_z": [-0.5, -1.0],
        }
    )

    variables, spec = build_candidate_variables(frame)

    assert "phase_sin_1" in variables
    assert "phase_cos_1" in variables
    assert "dynamic_pressure_pa" in variables
    assert "beta_proxy_rad" in variables
    assert "q_dyn_x_beta_proxy" in variables
    assert "body_rate_p" in variables
    assert "body_rate_q" in variables
    assert "body_rate_r" in variables
    assert "servo_0" in variables
    assert "servo_1" in variables
    assert "servo_2" in variables
    assert variables.loc[0, "dynamic_pressure_pa"] == 0.5 * 1.2 * 10.0**2
    assert np.isclose(variables.loc[0, "beta_proxy_rad"], np.arctan2(1.0, 10.0))
    assert spec["phase_column"] == "phase_corrected_rad"


def test_build_residual_frame_keeps_prior_and_corrected_residuals() -> None:
    samples = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "log_id": ["log_a", "log_a"],
            "fx_b": [10.0, 12.0],
            "fy_b": [1.0, 2.0],
            "fz_b": [-5.0, -6.0],
            "mx_b": [0.1, 0.2],
            "my_b": [0.3, 0.4],
            "mz_b": [0.5, 0.6],
            "phase_corrected_rad": [0.0, 1.0],
            "cycle_flap_frequency_hz": [4.0, 4.0],
            "airspeed_validated.true_airspeed_m_s": [10.0, 11.0],
        }
    )
    force_predictions = pd.DataFrame(
        {
            "label_fx_b": [10.0, 12.0],
            "label_fy_b": [1.0, 2.0],
            "label_fz_b": [-5.0, -6.0],
            "prior_fx_b": [8.0, 10.0],
            "prior_fy_b": [0.0, 1.0],
            "prior_fz_b": [-4.0, -5.0],
            "corrected_fx_b": [9.0, 11.5],
            "corrected_fy_b": [0.5, 1.8],
            "corrected_fz_b": [-4.5, -5.8],
        }
    )
    prior_predictions = pd.DataFrame(
        {
            "prior_mx_b": [0.0, 0.1],
            "prior_my_b": [0.1, 0.2],
            "prior_mz_b": [0.4, 0.5],
        }
    )
    current_moment_predictions = pd.DataFrame(
        {
            "pred_mx_b": [0.05, 0.15],
            "pred_my_b": [0.25, 0.35],
            "pred_mz_b": [0.45, 0.55],
        }
    )

    residual = build_residual_frame(
        split="train",
        samples=samples,
        force_predictions=force_predictions,
        prior_predictions=prior_predictions,
        current_moment_predictions=current_moment_predictions,
    )

    assert residual["split"].tolist() == ["train", "train"]
    assert residual["force_prior_residual_fx_b"].tolist() == [2.0, 2.0]
    assert residual["force_corrected_residual_fy_b"].tolist() == [0.5, 0.2]
    assert np.allclose(residual["moment_prior_residual_my_b"], [0.2, 0.2])
    assert np.allclose(residual["moment_current_residual_mz_b"], [0.05, 0.05])


def test_residual_variable_bin_table_ignores_nonfinite_values_and_reports_metrics() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test"] * 5,
            "force_corrected_residual_fy_b": [1.0, 2.0, 3.0, 4.0, 100.0],
            "beta_proxy_rad": [0.1, 0.2, 0.3, 0.4, np.nan],
        }
    )

    table = residual_variable_bin_table(
        frame,
        residual_columns=("force_corrected_residual_fy_b",),
        variable_columns=("beta_proxy_rad",),
        quantile_bins=2,
        min_samples=1,
    )

    assert table["split"].tolist() == ["test", "test"]
    assert table["residual_kind"].tolist() == ["force_corrected", "force_corrected"]
    assert table["target"].tolist() == ["fy_b", "fy_b"]
    assert table["variable"].tolist() == ["beta_proxy_rad", "beta_proxy_rad"]
    assert table["bin"].tolist() == [0, 1]
    assert table["sample_count"].tolist() == [2, 2]
    assert table.loc[0, "variable_min"] == 0.1
    assert table.loc[0, "variable_max"] == 0.2
    assert table.loc[0, "variable_median"] == 0.15
    assert table.loc[0, "residual_bias"] == 1.5
    assert table.loc[0, "residual_mae"] == 1.5
    assert table.loc[0, "residual_rmse"] == np.sqrt((1.0**2 + 2.0**2) / 2.0)

    summary = summarize_residual_variable_bins(table)
    row = summary.iloc[0]
    assert row["bin_count"] == 2
    assert row["sample_count"] == 4
    assert row["residual_rmse_range"] == table["residual_rmse"].max() - table["residual_rmse"].min()


def test_residual_variable_ranking_table_reports_correlations_and_per_log_groups() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test"] * 6,
            "log_id": ["a", "a", "a", "b", "b", "b"],
            "moment_current_residual_mz_b": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "beta_proxy_rad": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "body_rate_r": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )

    aggregate = residual_variable_ranking_table(
        frame,
        residual_columns=("moment_current_residual_mz_b",),
        variable_columns=("beta_proxy_rad", "body_rate_r"),
        min_samples=3,
    )
    per_log = residual_variable_ranking_table(
        frame,
        residual_columns=("moment_current_residual_mz_b",),
        variable_columns=("beta_proxy_rad",),
        group_columns=("log_id",),
        min_samples=3,
    )

    beta = aggregate.loc[aggregate["variable"].eq("beta_proxy_rad")].iloc[0]
    assert beta["residual_kind"] == "moment_current"
    assert beta["target"] == "mz_b"
    assert beta["sample_count"] == 6
    assert np.isfinite(beta["pearson"])
    assert np.isfinite(beta["spearman"])
    assert beta["abs_pearson"] == abs(beta["pearson"])
    assert beta["combined_rank_score"] >= 0.0
    assert set(per_log["log_id"]) == {"a", "b"}
    assert per_log.loc[per_log["log_id"].eq("a"), "spearman"].iloc[0] > 0.9
    assert per_log.loc[per_log["log_id"].eq("b"), "spearman"].iloc[0] < -0.9


def test_default_feature_groups_include_expected_candidate_buckets() -> None:
    groups = default_feature_groups(
        [
            "phase_sin_1",
            "alpha_rad_x_phase_sin_1",
            "alpha_rad",
            "beta_proxy_rad",
            "body_rate_p",
            "servo_0",
            "q_dyn_x_beta_proxy",
        ]
    )

    assert "phase_sin_1" in groups["phase"]
    assert "alpha_rad_x_phase_sin_1" not in groups["phase"]
    assert "alpha_rad_x_phase_sin_1" in groups["phase_interactions"]
    assert "alpha_rad" in groups["longitudinal"]
    assert "alpha_rad_x_phase_sin_1" not in groups["longitudinal"]
    assert "beta_proxy_rad" in groups["lateral_body"]
    assert "body_rate_p" in groups["body_rates"]
    assert "servo_0" in groups["tail_controls"]
    assert "q_dyn_x_beta_proxy" in groups["lateral_tail"]
    assert set(groups["all_candidate"]) == {
        "phase_sin_1",
        "alpha_rad_x_phase_sin_1",
        "alpha_rad",
        "beta_proxy_rad",
        "body_rate_p",
        "servo_0",
        "q_dyn_x_beta_proxy",
    }


def test_residual_feature_group_ablation_uses_validation_alpha_and_skips_missing_features() -> None:
    x_train = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x_val = np.array([-1.5, -0.5, 0.5, 1.5])
    x_test = np.array([-1.0, 0.0, 1.0])
    frame = pd.DataFrame(
        {
            "split": ["train"] * len(x_train) + ["val"] * len(x_val) + ["test"] * len(x_test),
            "log_id": ["train_log"] * len(x_train) + ["val_log"] * len(x_val) + ["test_log"] * len(x_test),
            "true_signal": np.concatenate([x_train, x_val, x_test]),
            "force_corrected_residual_fy_b": 2.0 * np.concatenate([x_train, x_val, x_test]),
        }
    )
    # Make the test set prefer a heavily regularized model if alpha were selected incorrectly on test.
    frame.loc[frame["split"].eq("test"), "force_corrected_residual_fy_b"] = 0.0

    aggregate, per_log = residual_feature_group_ablation(
        frame,
        residual_columns=("force_corrected_residual_fy_b",),
        feature_groups={"signal": ["true_signal", "missing_feature"]},
        alphas=(0.0, 1000.0),
    )

    zero = aggregate.loc[aggregate["feature_group"].eq("zero_residual")].iloc[0]
    signal = aggregate.loc[aggregate["feature_group"].eq("signal")].iloc[0]
    assert zero["n_features"] == 0
    assert signal["n_features"] == 1
    assert signal["feature_columns"] == "true_signal"
    assert signal["selected_alpha"] == 0.0
    assert signal["val_rmse"] < zero["zero_val_rmse"]
    assert "test_rmse_reduction_fraction" in signal
    assert set(per_log["log_id"]) == {"test_log"}


def _write_tiny_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    split_root = root / "split"
    force_root = root / "force"
    prior_root = root / "prior"
    moment_root = root / "moment"
    for path in (split_root, force_root / "prediction_parquets", prior_root, moment_root / "prediction_parquets"):
        path.mkdir(parents=True, exist_ok=True)

    for split_idx, split in enumerate(("train", "val", "test")):
        n = 4
        x = np.linspace(-1.0, 1.0, n) + split_idx
        samples = pd.DataFrame(
            {
                "time_s": np.arange(n, dtype=float),
                "log_id": [f"{split}_log"] * n,
                "fx_b": 10.0 + x,
                "fy_b": 1.0 + x,
                "fz_b": -5.0 + x,
                "mx_b": 0.1 + 0.1 * x,
                "my_b": 0.2 + 0.1 * x,
                "mz_b": 0.3 + 0.1 * x,
                "phase_corrected_rad": np.linspace(0.0, np.pi, n),
                "cycle_flap_frequency_hz": np.full(n, 4.0),
                "airspeed_validated.true_airspeed_m_s": 10.0 + x,
                "vehicle_air_data.rho": np.full(n, 1.2),
                "air_relative_velocity_b_x": 10.0 + x,
                "air_relative_velocity_b_y": x,
                "air_relative_velocity_b_z": -0.5 * x,
                "vehicle_angular_velocity.xyz[0]": 0.01 * x,
                "vehicle_angular_velocity.xyz[1]": 0.02 * x,
                "vehicle_angular_velocity.xyz[2]": 0.03 * x,
                "actuator_servos.servo[0]": 0.1 * x,
                "actuator_servos.servo[1]": -0.1 * x,
            }
        )
        force = pd.DataFrame(
            {
                "label_fx_b": samples["fx_b"],
                "label_fy_b": samples["fy_b"],
                "label_fz_b": samples["fz_b"],
                "prior_fx_b": samples["fx_b"] - 0.5 * x,
                "prior_fy_b": samples["fy_b"] - x,
                "prior_fz_b": samples["fz_b"] + 0.25 * x,
                "corrected_fx_b": samples["fx_b"] - 0.25 * x,
                "corrected_fy_b": samples["fy_b"] - 0.5 * x,
                "corrected_fz_b": samples["fz_b"] + 0.1 * x,
            }
        )
        prior = pd.DataFrame(
            {
                "fx_b": samples["fx_b"] - 0.5 * x,
                "fy_b": samples["fy_b"] - x,
                "fz_b": samples["fz_b"] + 0.25 * x,
                "mx_b": samples["mx_b"] - 0.05 * x,
                "my_b": samples["my_b"] - 0.08 * x,
                "mz_b": samples["mz_b"] - 0.1 * x,
            }
        )
        moment = pd.DataFrame(
            {
                "label_mx_b": samples["mx_b"],
                "label_my_b": samples["my_b"],
                "label_mz_b": samples["mz_b"],
                "pred_mx_b": samples["mx_b"] - 0.02 * x,
                "pred_my_b": samples["my_b"] - 0.03 * x,
                "pred_mz_b": samples["mz_b"] - 0.04 * x,
            }
        )
        samples.to_parquet(split_root / f"{split}_samples.parquet", index=False)
        force.to_parquet(force_root / "prediction_parquets" / f"{split}_predictions.parquet", index=False)
        prior.to_parquet(prior_root / f"{split}_predictions.parquet", index=False)
        moment.to_parquet(moment_root / "prediction_parquets" / f"{split}_predictions.parquet", index=False)
    return split_root, force_root, prior_root, moment_root


def test_cli_runs_on_tiny_fixture_and_writes_expected_outputs(tmp_path: Path) -> None:
    split_root, force_root, prior_root, moment_root = _write_tiny_fixture(tmp_path)
    output_root = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_component_residual_attribution.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--split-root",
            str(split_root),
            "--force-prediction-root",
            str(force_root),
            "--moment-prediction-root",
            str(moment_root),
            "--prior-root",
            str(prior_root),
            "--output-root",
            str(output_root),
            "--quantile-bins",
            "2",
            "--min-samples",
            "1",
            "--alphas",
            "1.0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "residual_frame.parquet").exists()
    assert (output_root / "residual_variable_bins.csv").exists()
    assert (output_root / "residual_variable_rankings.csv").exists()
    assert (output_root / "residual_feature_group_ablation.csv").exists()
    assert (output_root / "residual_attribution_config.json").exists()
    assert "observational residual attribution" in (output_root / "README.md").read_text(encoding="utf-8")
