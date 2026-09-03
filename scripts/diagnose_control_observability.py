#!/usr/bin/env python3
"""Run Step 4 control-observability diagnostics on Step 1 train/validation data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyulog import ULog
from scipy.stats import ks_2samp, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.analysis.control_observability import (
    control_summary_features,
    distribution_shift_summary,
    fit_standardized_ridge,
    paired_log_bootstrap,
    predict_standardized_ridge,
    quaternion_relative_rotation_vector,
)
from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
from system_identification.data.ulg_audit import _dataset, _zoh_values
from system_identification.evaluation.trajectory import (
    BODY_RATE_COLUMNS,
    POSITION_COLUMNS,
    QUATERNION_COLUMNS,
    VELOCITY_COLUMNS,
)
from system_identification.models.trajectory_main_v1 import offset_invariant_dynamics_features
from system_identification.physics.delaurier.airflow import quaternion_wxyz_to_rotation_body_to_ned
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows


CONTROL_NAMES = ("flap_motor", "left_elevon", "right_elevon", "rudder")
LAG_CONTROL_NAMES = (*CONTROL_NAMES, "symmetric_elevon", "differential_elevon")
RESPONSE_NAMES = (
    "acceleration_body_x",
    "acceleration_body_y",
    "acceleration_body_z",
    "angular_acceleration_body_x",
    "angular_acceleration_body_y",
    "angular_acceleration_body_z",
    "flap_frequency_rate",
    "flap_frequency_hz",
)
PRIMARY_LAG_PAIRS = (
    ("flap_motor", "flap_frequency_rate"),
    ("flap_motor", "flap_frequency_hz"),
    ("symmetric_elevon", "angular_acceleration_body_y"),
    ("differential_elevon", "angular_acceleration_body_x"),
    ("rudder", "angular_acceleration_body_z"),
)
HORIZONS_S = (0.1, 0.2, 0.5, 1.0, 2.0)
HISTORY_TAPS = (0, 15, 20, 25)  # -0.50, -0.20, -0.10, 0.00 s in a 26-sample history.


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _verify_contract(dataset_root: Path, manifest: dict) -> None:
    if manifest.get("dataset_version") != "trajectory_dataset_v1":
        raise ValueError("Step 4 requires trajectory_dataset_v1")
    split = manifest.get("split_contract", {})
    if split.get("sealed_test_opened") is not False:
        raise ValueError("refusing a manifest whose sealed test was opened")
    if set(split.get("materialized_partitions", [])) != {"train", "validation"}:
        raise ValueError("Step 4 requires exactly train and validation materialized")
    if (dataset_root / "samples_sealed_test.parquet").exists():
        raise ValueError("sealed-test samples are present; refusing to continue")
    for split_name in ("train", "validation"):
        for stem in ("samples", "windows"):
            if not (dataset_root / f"{stem}_{split_name}.parquet").is_file():
                raise FileNotFoundError(f"missing Step 1 artifact: {stem}_{split_name}.parquet")


def _history_features(batch, keep: np.ndarray) -> np.ndarray:
    return batch.history_state_features[keep][:, HISTORY_TAPS].reshape(np.sum(keep), -1)


def _trajectory_innovations(batch, keep: np.ndarray, step: int) -> np.ndarray:
    truth = batch.trajectory.truth
    observed_dt = np.sum(batch.trajectory.dt_s[keep, :step], axis=1)
    position = truth.position_n[keep, step] - truth.position_n[keep, 0]
    position -= truth.velocity_n[keep, 0] * observed_dt[:, None]
    velocity = truth.velocity_n[keep, step] - truth.velocity_n[keep, 0]
    attitude = quaternion_relative_rotation_vector(
        truth.quaternion_nb[keep, 0], truth.quaternion_nb[keep, step]
    )
    body_rate = truth.angular_velocity_b[keep, step] - truth.angular_velocity_b[keep, 0]
    return np.column_stack((position, velocity, attitude, body_rate))


def _per_log_standardized_rmse(
    truth: np.ndarray, prediction: np.ndarray, log_ids: np.ndarray, target_std: np.ndarray
) -> dict[str, float]:
    squared = np.square((prediction - truth) / target_std)
    return {
        str(log_id): float(np.sqrt(np.mean(squared[log_ids == log_id])))
        for log_id in sorted(set(log_ids))
    }


def _incremental_information(
    train_batch,
    validation_batch,
    *,
    nominal_rate_hz: float,
    alpha: float,
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_keep = train_batch.history_mask[:, 0]
    validation_keep = validation_batch.history_mask[:, 0]
    train_history = _history_features(train_batch, train_keep)
    validation_history = _history_features(validation_batch, validation_keep)
    train_logs = train_batch.trajectory.log_ids[train_keep]
    validation_logs = validation_batch.trajectory.log_ids[validation_keep]
    metrics: list[dict] = []
    per_log_rows: list[dict] = []
    gain_rows: list[dict] = []
    for horizon_s in HORIZONS_S:
        step = int(round(horizon_s * nominal_rate_hz))
        train_target = _trajectory_innovations(train_batch, train_keep, step)
        validation_target = _trajectory_innovations(validation_batch, validation_keep, step)
        train_control, control_feature_names = control_summary_features(
            train_batch.trajectory.controls[train_keep],
            train_batch.trajectory.dt_s[train_keep],
            steps=step,
            channel_names=CONTROL_NAMES,
        )
        validation_control, _ = control_summary_features(
            validation_batch.trajectory.controls[validation_keep],
            validation_batch.trajectory.dt_s[validation_keep],
            steps=step,
            channel_names=CONTROL_NAMES,
        )
        feature_sets = {"state_history_only": np.zeros(train_control.shape[1], dtype=bool)}
        for channel in CONTROL_NAMES:
            feature_sets[f"state_plus_{channel}"] = np.array(
                [name.startswith(f"{channel}_") for name in control_feature_names]
            )
        feature_sets["state_plus_all_controls"] = np.ones(train_control.shape[1], dtype=bool)
        per_model: dict[str, dict[str, dict[str, float]]] = {}
        for model_name, selected in feature_sets.items():
            train_x = np.column_stack((train_history, train_control[:, selected]))
            validation_x = np.column_stack((validation_history, validation_control[:, selected]))
            validation_prediction, fit = fit_standardized_ridge(
                train_x, train_target, validation_x, alpha=alpha
            )
            train_prediction = predict_standardized_ridge(train_x, fit)
            target_std = np.asarray(fit["target_std"])
            split_values = (
                ("train", train_target, train_prediction, train_logs),
                ("validation", validation_target, validation_prediction, validation_logs),
            )
            per_model[model_name] = {}
            for split_name, truth, prediction, logs in split_values:
                by_log = _per_log_standardized_rmse(truth, prediction, logs, target_std)
                per_model[model_name][split_name] = by_log
                metrics.append(
                    {
                        "horizon_s": horizon_s,
                        "model": model_name,
                        "split": split_name,
                        "standardized_trajectory_innovation_rmse_per_log_macro": float(
                            np.mean(list(by_log.values()))
                        ),
                        "window_count": int(len(truth)),
                        "log_count": len(by_log),
                    }
                )
                per_log_rows.extend(
                    {
                        "horizon_s": horizon_s,
                        "model": model_name,
                        "split": split_name,
                        "log_id": log_id,
                        "standardized_trajectory_innovation_rmse": value,
                    }
                    for log_id, value in by_log.items()
                )
        reference = per_model["state_history_only"]
        for model_name in feature_sets:
            if model_name == "state_history_only":
                continue
            for split_name in ("train", "validation"):
                result = paired_log_bootstrap(
                    reference[split_name],
                    per_model[model_name][split_name],
                    seed=seed + step,
                    draws=bootstrap_draws,
                )
                reference_mean = float(np.mean(list(reference[split_name].values())))
                gain_rows.append(
                    {
                        "horizon_s": horizon_s,
                        "candidate": model_name,
                        "split": split_name,
                        **result,
                        "mean_gain_percent_of_state_only": 100.0
                        * float(result["mean_gain"])
                        / reference_mean,
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(per_log_rows), pd.DataFrame(gain_rows)


def _control_predictability(
    train_batch,
    validation_batch,
    *,
    nominal_rate_hz: float,
    alpha: float,
) -> pd.DataFrame:
    train_keep = train_batch.history_mask[:, 0]
    validation_keep = validation_batch.history_mask[:, 0]
    train_x = _history_features(train_batch, train_keep)
    validation_x = _history_features(validation_batch, validation_keep)
    train_logs = train_batch.trajectory.log_ids[train_keep]
    validation_logs = validation_batch.trajectory.log_ids[validation_keep]
    rows: list[dict] = []
    for horizon_s in HORIZONS_S:
        step = int(round(horizon_s * nominal_rate_hz))
        train_y = np.mean(train_batch.trajectory.controls[train_keep, :step], axis=1)
        validation_y = np.mean(validation_batch.trajectory.controls[validation_keep, :step], axis=1)
        validation_prediction, fit = fit_standardized_ridge(train_x, train_y, validation_x, alpha=alpha)
        train_prediction = predict_standardized_ridge(train_x, fit)
        for split_name, truth, prediction, logs in (
            ("train", train_y, train_prediction, train_logs),
            ("validation", validation_y, validation_prediction, validation_logs),
        ):
            for channel_index, channel in enumerate(CONTROL_NAMES):
                per_log_r2 = []
                per_log_corr = []
                for log_id in sorted(set(logs)):
                    selected = logs == log_id
                    actual = truth[selected, channel_index]
                    predicted = prediction[selected, channel_index]
                    denominator = np.sum(np.square(actual - np.mean(actual)))
                    per_log_r2.append(
                        float(1.0 - np.sum(np.square(actual - predicted)) / denominator)
                        if denominator > 1e-12
                        else np.nan
                    )
                    per_log_corr.append(_correlation(actual, predicted))
                rows.append(
                    {
                        "horizon_s": horizon_s,
                        "split": split_name,
                        "control": channel,
                        "r2_per_log_macro": float(np.nanmean(per_log_r2)),
                        "correlation_fisher_equal_log_macro": _fisher_macro(per_log_corr),
                        "log_count": len(per_log_r2),
                    }
                )
    return pd.DataFrame(rows)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 3 or np.std(x[finite]) < 1e-10 or np.std(y[finite]) < 1e-10:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _fisher_macro(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(finite, -0.999999, 0.999999)))))


def _lag_arrays(samples: pd.DataFrame, lag_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history_blocks: list[np.ndarray] = []
    control_blocks: list[np.ndarray] = []
    response_blocks: list[np.ndarray] = []
    log_blocks: list[np.ndarray] = []
    core = samples.loc[samples["valid_core"] & (samples["segment_id"] >= 0)]
    for (log_id, _), group in core.groupby(["log_id", "segment_id"], sort=False):
        ordered = group.sort_values("sample_in_segment", kind="stable")
        count = len(ordered)
        first = max(25, 2 - lag_steps)
        stop = min(count, count - 2 - lag_steps)
        if stop <= first:
            continue
        command_index = np.arange(first, stop)
        response_index = command_index + lag_steps
        taps = command_index[:, None] + np.array([-25, -10, -5, 0])[None, :]
        velocity = ordered[list(VELOCITY_COLUMNS)].to_numpy(dtype=float)
        quaternion = ordered[list(QUATERNION_COLUMNS)].to_numpy(dtype=float)
        body_rate = ordered[list(BODY_RATE_COLUMNS)].to_numpy(dtype=float)
        phase = ordered["relative_flap_phase_rad"].to_numpy(dtype=float)
        frequency = ordered["flap_frequency_hz"].to_numpy(dtype=float)
        anchor = phase[command_index]
        features = offset_invariant_dynamics_features(
            velocity_n=velocity[taps].reshape(-1, 3),
            quaternion_nb=quaternion[taps].reshape(-1, 4),
            angular_velocity_b=body_rate[taps].reshape(-1, 3),
            relative_phase_rad=phase[taps].reshape(-1),
            phase_anchor_rad=np.repeat(anchor, len(HISTORY_TAPS)),
            flap_frequency_hz=frequency[taps].reshape(-1),
        ).reshape(len(command_index), -1)
        controls = ordered[list(CONTROL_COLUMNS)].to_numpy(dtype=float)[command_index]
        controls = np.column_stack(
            (controls, 0.5 * (controls[:, 1] + controls[:, 2]), 0.5 * (controls[:, 1] - controls[:, 2]))
        )
        timestamps = ordered["timestamp_us"].to_numpy(dtype=np.int64)
        response_dt = (timestamps[response_index + 2] - timestamps[response_index - 2]) * 1e-6
        acceleration_n = (velocity[response_index + 2] - velocity[response_index - 2]) / response_dt[:, None]
        rotation, valid = quaternion_wxyz_to_rotation_body_to_ned(quaternion[response_index])
        if not np.all(valid):
            raise ValueError("invalid quaternion inside valid trajectory core")
        acceleration_b = np.einsum("nji,nj->ni", rotation, acceleration_n)
        angular_acceleration = (
            body_rate[response_index + 2] - body_rate[response_index - 2]
        ) / response_dt[:, None]
        frequency_rate = (
            frequency[response_index + 2] - frequency[response_index - 2]
        ) / response_dt
        response = np.column_stack(
            (acceleration_b, angular_acceleration, frequency_rate, frequency[response_index])
        )
        history_blocks.append(features)
        control_blocks.append(controls)
        response_blocks.append(response)
        log_blocks.append(np.full(len(command_index), str(log_id), dtype=object))
    return (
        np.concatenate(history_blocks),
        np.concatenate(control_blocks),
        np.concatenate(response_blocks),
        np.concatenate(log_blocks),
    )


def _correlation_rows(
    controls: np.ndarray,
    responses: np.ndarray,
    residuals: np.ndarray | None,
    logs: np.ndarray,
    *,
    split: str,
    lag_steps: int,
    nominal_rate_hz: float,
) -> list[dict]:
    rows = []
    for control_index, control_name in enumerate(LAG_CONTROL_NAMES):
        for response_index, response_name in enumerate(RESPONSE_NAMES):
            raw_values = []
            conditioned_values = []
            for log_id in sorted(set(logs)):
                selected = logs == log_id
                raw_values.append(_correlation(controls[selected, control_index], responses[selected, response_index]))
                if residuals is not None:
                    conditioned_values.append(
                        _correlation(controls[selected, control_index], residuals[selected, response_index])
                    )
            rows.append(
                {
                    "split": split,
                    "lag_steps": lag_steps,
                    "lag_s": lag_steps / nominal_rate_hz,
                    "control": control_name,
                    "response": response_name,
                    "raw_correlation_fisher_equal_log_macro": _fisher_macro(raw_values),
                    "state_conditioned_correlation_fisher_equal_log_macro": (
                        _fisher_macro(conditioned_values) if residuals is not None else np.nan
                    ),
                    "sample_count": len(logs),
                    "log_count": len(set(logs)),
                }
            )
    return rows


def _lag_diagnostics(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    *,
    nominal_rate_hz: float,
    alpha: float,
    maximum_lag_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    maximum_steps = int(round(maximum_lag_s * nominal_rate_hz))
    for lag_steps in range(-maximum_steps, maximum_steps + 1):
        train = _lag_arrays(train_samples, lag_steps)
        validation = _lag_arrays(validation_samples, lag_steps)
        fit = None
        if lag_steps >= 0:
            _, fit = fit_standardized_ridge(train[0], train[2], validation[0], alpha=alpha)
        train_residual = train[2] - predict_standardized_ridge(train[0], fit) if fit else None
        validation_residual = (
            validation[2] - predict_standardized_ridge(validation[0], fit) if fit else None
        )
        rows.extend(
            _correlation_rows(
                train[1], train[2], train_residual, train[3], split="train",
                lag_steps=lag_steps, nominal_rate_hz=nominal_rate_hz
            )
        )
        rows.extend(
            _correlation_rows(
                validation[1], validation[2], validation_residual, validation[3], split="validation",
                lag_steps=lag_steps, nominal_rate_hz=nominal_rate_hz
            )
        )
    curves = pd.DataFrame(rows)
    summaries = []
    for control, response in PRIMARY_LAG_PAIRS:
        selected = curves.loc[
            (curves["control"] == control) & (curves["response"] == response)
        ]
        for split_name in ("train", "validation"):
            split_rows = selected.loc[selected["split"] == split_name]
            for metric in (
                "raw_correlation_fisher_equal_log_macro",
                "state_conditioned_correlation_fisher_equal_log_macro",
            ):
                candidates = split_rows.loc[split_rows["lag_s"] >= 0.0].dropna(subset=[metric])
                best = candidates.iloc[int(np.argmax(np.abs(candidates[metric].to_numpy())))]
                summaries.append(
                    {
                        "control": control,
                        "response": response,
                        "split": split_name,
                        "correlation_type": metric.removesuffix("_correlation_fisher_equal_log_macro"),
                        "best_nonnegative_lag_s": float(best["lag_s"]),
                        "correlation_at_best_lag": float(best[metric]),
                    }
                )
    return curves, pd.DataFrame(summaries)


def _distribution_shift(train_samples: pd.DataFrame, validation_samples: pd.DataFrame) -> pd.DataFrame:
    train = train_samples.loc[train_samples["valid_core"]]
    validation = validation_samples.loc[validation_samples["valid_core"]]
    rows = []
    for column, control in zip(CONTROL_COLUMNS, CONTROL_NAMES, strict=True):
        summary = distribution_shift_summary(train[column], validation[column])
        summary["ks_statistic_descriptive_only"] = float(
            ks_2samp(train[column], validation[column]).statistic
        )
        rows.append({"control": control, **summary})
    return pd.DataFrame(rows)


def _control_cross_correlations(samples_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, samples in samples_by_split.items():
        core = samples.loc[samples["valid_core"]]
        for left_index, left_name in enumerate(CONTROL_NAMES):
            for right_index in range(left_index + 1, len(CONTROL_NAMES)):
                right_name = CONTROL_NAMES[right_index]
                per_log = [
                    _correlation(group[CONTROL_COLUMNS[left_index]], group[CONTROL_COLUMNS[right_index]])
                    for _, group in core.groupby("log_id", sort=True)
                ]
                rows.append(
                    {
                        "split": split_name,
                        "control_left": left_name,
                        "control_right": right_name,
                        "correlation_fisher_equal_log_macro": _fisher_macro(per_log),
                        "log_count": len(per_log),
                    }
                )
    return pd.DataFrame(rows)


def _step3_shift_impact(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    step3_per_log_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate per-flight control shift to the already-frozen Step 3 degradation."""
    train = train_samples.loc[train_samples["valid_core"]]
    validation = validation_samples.loc[validation_samples["valid_core"]]
    train_mean = train[list(CONTROL_COLUMNS)].mean().to_numpy(dtype=float)
    train_std = train[list(CONTROL_COLUMNS)].std(ddof=0).to_numpy(dtype=float)
    train_std = np.where(train_std > 1e-8, train_std, 1.0)
    shift_rows = []
    for log_id, group in validation.groupby("log_id", sort=True):
        standardized = (
            group[list(CONTROL_COLUMNS)].mean().to_numpy(dtype=float) - train_mean
        ) / train_std
        shift_rows.append(
            {
                "log_id": str(log_id),
                **{
                    f"{name}_standardized_mean_shift": float(value)
                    for name, value in zip(CONTROL_NAMES, standardized, strict=True)
                },
                "control_mean_shift_l2": float(np.linalg.norm(standardized)),
            }
        )
    shifts = pd.DataFrame(shift_rows)
    step3 = pd.read_csv(step3_per_log_path)
    no_control = step3.loc[step3["model"] == "history_no_control_multistep"]
    controlled = step3.loc[step3["model"] == "main_v1_history_controlled_multistep"]
    keys = ["split", "horizon_s", "log_id"]
    merged = no_control.merge(controlled, on=keys, suffixes=("_no_control", "_controlled"))
    if len(merged) != 25 or set(merged["split"]) != {"validation"}:
        raise ValueError("unexpected frozen Step 3 per-log metric contract")
    metric_names = (
        "position_rmse_m",
        "velocity_rmse_m_s",
        "attitude_rmse_deg",
        "body_rate_rmse_rad_s",
    )
    impact = merged[keys].merge(shifts, on="log_id")
    for metric in metric_names:
        impact[f"{metric}_controlled_minus_no_control"] = (
            merged[f"{metric}_controlled"] - merged[f"{metric}_no_control"]
        )
    associations = []
    for horizon_s, group in impact.groupby("horizon_s", sort=True):
        for metric in metric_names:
            result = spearmanr(
                group["control_mean_shift_l2"],
                group[f"{metric}_controlled_minus_no_control"],
            )
            associations.append(
                {
                    "horizon_s": float(horizon_s),
                    "step3_metric": metric,
                    "spearman_shift_vs_controlled_degradation": float(result.statistic),
                    "validation_log_count": int(len(group)),
                    "interpretation": "descriptive_only_no_sample_level_inference",
                }
            )
    return impact, pd.DataFrame(associations)


def _raw_proxy_diagnostics(
    samples_by_split: dict[str, pd.DataFrame],
    assignments: dict[str, list[str]],
    *,
    log_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rpm_rows = []
    output_rows = []
    output_indices = (2, 0, 1, 4)  # PWM functions 101, 201, 202, 203 in all admitted logs.
    for split_name in ("train", "validation"):
        samples = samples_by_split[split_name]
        for log_id in assignments[split_name]:
            selected = samples.loc[(samples["log_id"] == log_id) & samples["valid_core"]]
            reference_us = selected["timestamp_us"].to_numpy(dtype=np.int64)
            ulog = ULog(str(log_root / log_id))
            ratio = float(ulog.initial_parameters.get("FLAP_RATIO", np.nan))
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(f"invalid FLAP_RATIO in {log_id}: {ratio}")
            rpm = _dataset(ulog, "rpm")
            if rpm is None:
                raise ValueError(f"rpm topic missing in {log_id}")
            raw, raw_valid = _zoh_values(reference_us, rpm, "rpm_raw", freshness_s=0.10)
            estimate, estimate_valid = _zoh_values(reference_us, rpm, "rpm_estimate", freshness_s=0.10)
            logged_frequency = selected["flap_frequency_hz"].to_numpy(dtype=float)
            for proxy_name, rpm_values, valid in (
                ("rpm_raw_ratio_corrected", raw, raw_valid),
                ("rpm_estimate_ratio_corrected", estimate, estimate_valid),
            ):
                proxy = np.abs(rpm_values) / (60.0 * ratio)
                valid &= np.isfinite(logged_frequency)
                rpm_rows.append(
                    {
                        "split": split_name,
                        "log_id": log_id,
                        "proxy": proxy_name,
                        "flap_ratio": ratio,
                        "sample_count": int(np.sum(valid)),
                        "correlation_with_dataset_flap_frequency": _correlation(
                            proxy[valid], logged_frequency[valid]
                        ),
                        "rmse_hz_vs_dataset_flap_frequency": float(
                            np.sqrt(np.mean(np.square(proxy[valid] - logged_frequency[valid])))
                        ),
                    }
                )
            outputs = _dataset(ulog, "actuator_outputs")
            if outputs is None:
                raise ValueError(f"actuator_outputs topic missing in {log_id}")
            for channel_index, (control, output_index) in enumerate(zip(CONTROL_NAMES, output_indices, strict=True)):
                pwm, valid = _zoh_values(reference_us, outputs, f"output[{output_index}]", freshness_s=0.20)
                command = selected[CONTROL_COLUMNS[channel_index]].to_numpy(dtype=float)
                valid &= np.isfinite(command)
                output_rows.append(
                    {
                        "split": split_name,
                        "log_id": log_id,
                        "control": control,
                        "pwm_output_index": output_index,
                        "sample_count": int(np.sum(valid)),
                        "command_pwm_correlation": _correlation(command[valid], pwm[valid]),
                    }
                )
    return pd.DataFrame(rpm_rows), pd.DataFrame(output_rows)


def _plot_lags(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for axis, (control, response) in zip(axes.flat, PRIMARY_LAG_PAIRS, strict=False):
        selected = curves.loc[(curves["control"] == control) & (curves["response"] == response)]
        for split_name, color in (("train", "tab:blue"), ("validation", "tab:orange")):
            rows = selected.loc[selected["split"] == split_name].sort_values("lag_s")
            axis.plot(rows["lag_s"], rows["raw_correlation_fisher_equal_log_macro"],
                      color=color, linestyle=":", alpha=0.8, label=f"{split_name} raw")
            axis.plot(rows["lag_s"], rows["state_conditioned_correlation_fisher_equal_log_macro"],
                      color=color, label=f"{split_name} conditioned")
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.set(title=f"{control} -> {response}", xlabel="command lead lag (s)", ylabel="equal-log correlation")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_incremental_gains(gains: pd.DataFrame, path: Path) -> None:
    selected = gains.loc[gains["split"] == "validation"]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    candidates = list(selected["candidate"].drop_duplicates())
    width = 0.8 / len(candidates)
    x = np.arange(len(HORIZONS_S))
    for index, candidate in enumerate(candidates):
        rows = selected.loc[selected["candidate"] == candidate].set_index("horizon_s").loc[list(HORIZONS_S)]
        axis.bar(x + (index - (len(candidates) - 1) / 2) * width,
                 rows["mean_gain_percent_of_state_only"], width=width, label=candidate)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [str(value) for value in HORIZONS_S])
    axis.set(xlabel="prediction horizon (s)", ylabel="validation gain over state history only (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset/trajectory_v1_august_f5_c4")
    parser.add_argument("--log-root", type=Path, default=Path("/home/zn/QgcLogs"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/control_observability_v1")
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument(
        "--step3-per-log-metrics",
        type=Path,
        default=PROJECT_ROOT / "docs/analysis/results/trajectory_main_v1/validation_per_log_metrics.csv",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--maximum-lag-s", type=float, default=0.5)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    summary_root = args.summary_root.resolve() if args.summary_root else None
    for path in (output_root, summary_root):
        if path is not None and path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output root: {path}")
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    _verify_contract(dataset_root, manifest)
    nominal_rate_hz = float(manifest["sampling"]["nominal_rate_hz"])
    samples = {
        split: pd.read_parquet(dataset_root / f"samples_{split}.parquet")
        for split in ("train", "validation")
    }
    windows = {
        split: pd.read_parquet(dataset_root / f"windows_{split}.parquet")
        for split in ("train", "validation")
    }
    batches = {
        split: assemble_history_trajectory_windows(samples[split], windows[split], history_steps=26)
        for split in ("train", "validation")
    }
    incremental, incremental_per_log, gains = _incremental_information(
        batches["train"], batches["validation"], nominal_rate_hz=nominal_rate_hz,
        alpha=args.ridge_alpha, bootstrap_draws=args.bootstrap_draws, seed=args.seed
    )
    predictability = _control_predictability(
        batches["train"], batches["validation"], nominal_rate_hz=nominal_rate_hz,
        alpha=args.ridge_alpha
    )
    lag_curves, lag_summary = _lag_diagnostics(
        samples["train"], samples["validation"], nominal_rate_hz=nominal_rate_hz,
        alpha=args.ridge_alpha, maximum_lag_s=args.maximum_lag_s
    )
    shift = _distribution_shift(samples["train"], samples["validation"])
    control_cross_correlation = _control_cross_correlations(samples)
    shift_impact, shift_associations = _step3_shift_impact(
        samples["train"], samples["validation"], args.step3_per_log_metrics.resolve()
    )
    rpm_proxy, pwm_proxy = _raw_proxy_diagnostics(
        samples, manifest["split_contract"]["assignments"], log_root=args.log_root.resolve()
    )

    output_root.mkdir(parents=True)
    tables = {
        "incremental_information_metrics.csv": incremental,
        "incremental_information_per_log.csv": incremental_per_log,
        "incremental_information_gains.csv": gains,
        "control_predictability.csv": predictability,
        "lag_correlation_curves.csv": lag_curves,
        "lag_summary.csv": lag_summary,
        "control_distribution_shift.csv": shift,
        "control_cross_correlation.csv": control_cross_correlation,
        "validation_log_shift_step3_impact.csv": shift_impact,
        "shift_step3_associations.csv": shift_associations,
        "rpm_proxy_audit.csv": rpm_proxy,
        "pwm_proxy_audit.csv": pwm_proxy,
    }
    for filename, table in tables.items():
        table.to_csv(output_root / filename, index=False)
    _plot_lags(lag_curves, output_root / "control_response_lags.png")
    _plot_incremental_gains(gains, output_root / "incremental_control_gain.png")
    run_manifest = {
        "experiment": "control_observability_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_head": _git_head(),
        "step3_baseline_head": "46f261856d1eaf1cd56443913b2e46c9acdd1fbd",
        "dataset": {
            "root": str(dataset_root),
            "dataset_version": manifest["dataset_version"],
            "builder_git_head": manifest["builder_git_head"],
            "partitions_read": ["train", "validation"],
            "sealed_test_opened": False,
        },
        "method": {
            "history_steps_including_t0": 26,
            "history_taps_s": [-0.5, -0.2, -0.1, 0.0],
            "ridge_alpha_fixed_without_validation_tuning": args.ridge_alpha,
            "horizons_s": list(HORIZONS_S),
            "lag_range_s": [-args.maximum_lag_s, args.maximum_lag_s],
            "lag_derivative_span_s_nominal": 0.08,
            "bootstrap_unit": "flight log",
            "bootstrap_draws": args.bootstrap_draws,
            "sample_level_p_values_used": False,
            "shift_impact_association": "Spearman across five validation logs; descriptive only",
            "future_state_phase_frequency_airdata_or_wind_used_as_input": False,
        },
        "counts": {
            split: {
                "sample_rows": int(len(samples[split])),
                "valid_core_rows": int(samples[split]["valid_core"].sum()),
                "step1_windows": int(len(windows[split])),
                "full_history_windows": int(np.sum(batches[split].history_mask[:, 0])),
                "logs": int(samples[split]["log_id"].nunique()),
            }
            for split in ("train", "validation")
        },
        "limitations": {
            "phase": "log-local relative phase only; history features subtract phase at t0",
            "servo_feedback": "no measured servo position/current topic in admitted logs",
            "pwm": "actuator_outputs is commanded electrical output, not measured surface state",
            "rpm_estimate": "filtered encoder-derived signal and not independent of dataset flap_frequency_hz",
        },
        "artifacts": {key: key for key in tables} | {
            "lag_figure": "control_response_lags.png",
            "incremental_gain_figure": "incremental_control_gain.png",
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    if summary_root is not None:
        summary_root.mkdir(parents=True)
        for filename in (
            "incremental_information_metrics.csv", "incremental_information_gains.csv",
            "control_predictability.csv", "lag_summary.csv", "control_distribution_shift.csv",
            "control_cross_correlation.csv",
            "validation_log_shift_step3_impact.csv", "shift_step3_associations.csv",
            "rpm_proxy_audit.csv", "pwm_proxy_audit.csv", "control_response_lags.png",
            "incremental_control_gain.png", "manifest.json"
        ):
            source = output_root / filename
            (summary_root / filename).write_bytes(source.read_bytes())
    print(json.dumps(run_manifest["counts"], indent=2))
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
