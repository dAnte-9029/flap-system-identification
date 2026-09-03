#!/usr/bin/env python3
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
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.evaluation.trajectory import (
    aggregate_trajectory_metrics,
    assemble_trajectory_windows,
    evaluate_trajectory_predictions,
)
from system_identification.models.trajectory import (
    ConstantTwistPredictor,
    IntegratedDynamicsPredictor,
    PersistencePredictor,
)
from system_identification.training.trajectory_baselines import (
    fit_mlp_dynamics,
    fit_ridge_dynamics,
)


MODEL_ORDER = (
    "persistence",
    "constant_twist",
    "ridge_no_control",
    "ridge_controlled",
    "mlp_no_control",
    "mlp_controlled",
)


def _parse_horizons(raw: str, *, nominal_rate_hz: float, maximum_steps: int) -> dict[float, int]:
    result: dict[float, int] = {}
    for part in raw.split(","):
        horizon = float(part.strip())
        steps = int(round(horizon * nominal_rate_hz))
        if horizon <= 0.0 or steps < 1 or steps > maximum_steps:
            raise ValueError(f"horizon outside available window: {horizon}")
        if not np.isclose(steps / nominal_rate_hz, horizon, atol=1.0e-9):
            raise ValueError(f"horizon must be an integer number of nominal samples: {horizon}")
        result[horizon] = steps
    if not result:
        raise ValueError("at least one horizon is required")
    return result


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _verify_step1_contract(dataset_root: Path, manifest: dict) -> None:
    if manifest.get("dataset_version") != "trajectory_dataset_v1":
        raise ValueError("Step 2 requires trajectory_dataset_v1")
    split = manifest.get("split_contract", {})
    if split.get("sealed_test_opened") is not False:
        raise ValueError("refusing a dataset manifest that opened sealed test")
    if set(split.get("materialized_partitions", [])) != {"train", "validation"}:
        raise ValueError("Step 2 requires train and validation only")
    if (dataset_root / "samples_sealed_test.parquet").exists():
        raise ValueError("sealed-test samples are present; refusing to continue")
    for partition in ("train", "validation"):
        for stem in ("samples", "windows"):
            if not (dataset_root / f"{stem}_{partition}.parquet").is_file():
                raise FileNotFoundError(f"missing Step 1 artifact: {stem}_{partition}.parquet")


def _control_gain_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "position_rmse_m_per_log_macro",
        "velocity_rmse_m_s_per_log_macro",
        "attitude_rmse_deg_per_log_macro",
        "body_rate_rmse_rad_s_per_log_macro",
    )
    rows: list[dict[str, float | str]] = []
    for family in ("ridge", "mlp"):
        without = aggregate.loc[aggregate["model"] == f"{family}_no_control"].set_index("horizon_s")
        with_control = aggregate.loc[aggregate["model"] == f"{family}_controlled"].set_index("horizon_s")
        for horizon in sorted(set(without.index) & set(with_control.index)):
            row: dict[str, float | str] = {"family": family, "horizon_s": float(horizon)}
            for metric in metric_columns:
                baseline = float(without.loc[horizon, metric])
                controlled = float(with_control.loc[horizon, metric])
                row[f"{metric}_gain_percent"] = 100.0 * (baseline - controlled) / baseline
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_horizon_metrics(aggregate: pd.DataFrame, output_paths: list[Path]) -> None:
    panels = (
        ("position_rmse_m_per_log_macro", "Position RMSE (m)"),
        ("velocity_rmse_m_s_per_log_macro", "Velocity RMSE (m/s)"),
        ("attitude_rmse_deg_per_log_macro", "Attitude RMSE (deg)"),
        ("body_rate_rmse_rad_s_per_log_macro", "Body-rate RMSE (rad/s)"),
    )
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, panels, strict=True):
        for index, model_name in enumerate(MODEL_ORDER):
            model_rows = aggregate.loc[aggregate["model"] == model_name].sort_values("horizon_s")
            axis.plot(
                model_rows["horizon_s"],
                model_rows[metric],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[index],
                label=model_name,
            )
        axis.set_xlabel("Rollout horizon (s)")
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_ridge(path: Path, predictor) -> None:
    np.savez(
        path,
        feature_mean=predictor.feature_mean,
        feature_std=predictor.feature_std,
        target_mean=predictor.target_mean,
        target_std=predictor.target_std,
        coefficients=predictor.coefficients,
        intercept=predictor.intercept,
        use_controls=np.array(predictor.use_controls),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Step 2 trajectory baselines")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/trajectory_v1_august_f5_c4",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/trajectory_baselines_v1",
    )
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--horizons", default="0.10,0.20,0.50,1.00,2.00")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--mlp-hidden-sizes", default="64,64")
    parser.add_argument("--mlp-epochs", type=int, default=40)
    parser.add_argument("--mlp-batch-size", type=int, default=1024)
    parser.add_argument("--mlp-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    if args.summary_root is not None and args.summary_root.exists() and any(args.summary_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty summary root: {args.summary_root}")
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    _verify_step1_contract(dataset_root, manifest)
    nominal_rate_hz = float(manifest["sampling"]["nominal_rate_hz"])
    maximum_steps = int(manifest["sampling"]["horizon_steps"])
    horizon_steps = _parse_horizons(
        args.horizons, nominal_rate_hz=nominal_rate_hz, maximum_steps=maximum_steps
    )
    hidden_sizes = tuple(int(value) for value in args.mlp_hidden_sizes.split(",") if value.strip())
    if not hidden_sizes or any(value <= 0 for value in hidden_sizes):
        raise ValueError("mlp hidden sizes must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    train_samples = pd.read_parquet(dataset_root / "samples_train.parquet")
    validation_samples = pd.read_parquet(dataset_root / "samples_validation.parquet")
    validation_windows = pd.read_parquet(dataset_root / "windows_validation.parquet")
    batch = assemble_trajectory_windows(validation_samples, validation_windows)

    ridge_no_control, ridge_transitions = fit_ridge_dynamics(
        train_samples, alpha=args.ridge_alpha, use_controls=False
    )
    ridge_controlled, _ = fit_ridge_dynamics(
        train_samples, alpha=args.ridge_alpha, use_controls=True
    )
    mlp_no_control, mlp_transitions, history_no_control = fit_mlp_dynamics(
        train_samples,
        use_controls=False,
        hidden_sizes=hidden_sizes,
        epochs=args.mlp_epochs,
        batch_size=args.mlp_batch_size,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        seed=args.seed,
    )
    mlp_controlled, _, history_controlled = fit_mlp_dynamics(
        train_samples,
        use_controls=True,
        hidden_sizes=hidden_sizes,
        epochs=args.mlp_epochs,
        batch_size=args.mlp_batch_size,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        seed=args.seed,
    )
    models = {
        "persistence": PersistencePredictor(),
        "constant_twist": ConstantTwistPredictor(),
        "ridge_no_control": IntegratedDynamicsPredictor(ridge_no_control),
        "ridge_controlled": IntegratedDynamicsPredictor(ridge_controlled),
        "mlp_no_control": IntegratedDynamicsPredictor(mlp_no_control),
        "mlp_controlled": IntegratedDynamicsPredictor(mlp_controlled),
    }

    metric_frames = []
    for model_name, model in models.items():
        prediction = model.rollout(batch.initial_state(), batch.controls, batch.dt_s)
        metric_frames.append(
            evaluate_trajectory_predictions(
                prediction,
                batch.truth,
                model_name=model_name,
                split="validation",
                window_ids=batch.window_ids,
                log_ids=batch.log_ids,
                segment_ids=batch.segment_ids,
                horizon_steps=horizon_steps,
                dt_s=batch.dt_s,
            )
        )
    window_metrics = pd.concat(metric_frames, ignore_index=True)
    aggregate, per_log = aggregate_trajectory_metrics(window_metrics)
    control_gain = _control_gain_table(aggregate)
    history_no_control.insert(0, "model", "mlp_no_control")
    history_controlled.insert(0, "model", "mlp_controlled")
    training_history = pd.concat([history_no_control, history_controlled], ignore_index=True)

    output_root.mkdir(parents=True, exist_ok=True)
    model_root = output_root / "models"
    model_root.mkdir()
    window_metrics.to_parquet(output_root / "validation_window_metrics.parquet", index=False)
    aggregate.to_csv(output_root / "validation_horizon_metrics.csv", index=False)
    per_log.to_csv(output_root / "validation_per_log_metrics.csv", index=False)
    control_gain.to_csv(output_root / "control_gain.csv", index=False)
    training_history.to_csv(output_root / "training_history.csv", index=False)
    _save_ridge(model_root / "ridge_no_control.npz", ridge_no_control)
    _save_ridge(model_root / "ridge_controlled.npz", ridge_controlled)
    for name, predictor in (
        ("mlp_no_control", mlp_no_control),
        ("mlp_controlled", mlp_controlled),
    ):
        torch.save(
            {
                "state_dict": predictor.module.state_dict(),
                "feature_mean": predictor.feature_mean,
                "feature_std": predictor.feature_std,
                "target_mean": predictor.target_mean,
                "target_std": predictor.target_std,
                "hidden_sizes": hidden_sizes,
                "use_controls": predictor.use_controls,
            },
            model_root / f"{name}.pt",
        )
    _plot_horizon_metrics(aggregate, [output_root / "rollout_errors.png"])

    run_manifest = {
        "experiment": "trajectory_baselines_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_head": _git_head(),
        "dataset": {
            "root": str(dataset_root),
            "dataset_version": manifest["dataset_version"],
            "builder_git_head": manifest["builder_git_head"],
            "primary_cohort": manifest["source"]["primary_cohort"],
            "partitions_read": ["train", "validation"],
            "sealed_test_opened": False,
        },
        "task": manifest["roles"],
        "horizon_steps": {str(key): value for key, value in horizon_steps.items()},
        "primary_aggregation": "equal-log macro of per-log window RMSE",
        "dynamics_contract": {
            "base_features": [
                "velocity_body_frd_m_s[3]",
                "angular_velocity_body_frd_rad_s[3]",
                "gravity_body_frd_m_s2[3]",
                "relative_phase_sin_cos[2]",
                "flap_frequency_hz",
            ],
            "controlled_features_append": manifest["roles"][
                "known_future_control_t0_to_tT_exclusive"
            ],
            "one_step_targets": [
                "net_kinematic_acceleration_body_frd_m_s2[3]",
                "angular_acceleration_body_frd_rad_s2[3]",
                "flap_frequency_rate_hz_s",
            ],
            "integration": "actual per-window dt; midpoint body-rate quaternion update",
            "frequency_constraint_hz": [0.5, 20.0],
        },
        "training": {
            "train_transition_count": int(len(ridge_transitions.features)),
            "ridge_alpha": args.ridge_alpha,
            "mlp_hidden_sizes": hidden_sizes,
            "mlp_epochs": args.mlp_epochs,
            "mlp_batch_size": args.mlp_batch_size,
            "mlp_learning_rate": args.mlp_learning_rate,
            "mlp_weight_decay": args.mlp_weight_decay,
            "seed": args.seed,
            "device": "cpu",
            "torch_threads": args.torch_threads,
            "normalization": "fit on train transitions only",
            "validation_used_for_fitting_or_tuning": False,
        },
        "models": list(MODEL_ORDER),
        "validation_window_count": int(len(batch.window_ids)),
        "artifacts": {
            "window_metrics": "validation_window_metrics.parquet",
            "horizon_metrics": "validation_horizon_metrics.csv",
            "per_log_metrics": "validation_per_log_metrics.csv",
            "control_gain": "control_gain.csv",
            "figure": "rollout_errors.png",
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.summary_root is not None:
        summary_root = args.summary_root.resolve()
        summary_root.mkdir(parents=True, exist_ok=True)
        aggregate.to_csv(summary_root / "validation_horizon_metrics.csv", index=False)
        per_log.to_csv(summary_root / "validation_per_log_metrics.csv", index=False)
        control_gain.to_csv(summary_root / "control_gain.csv", index=False)
        _plot_horizon_metrics(
            aggregate,
            [summary_root / "rollout_errors.png", summary_root / "rollout_errors.svg"],
        )
        compact_manifest = {
            **run_manifest,
            "artifact_root": str(output_root),
            "training_final_standardized_mse": {
                "mlp_no_control": float(history_no_control.iloc[-1]["train_standardized_mse"]),
                "mlp_controlled": float(history_controlled.iloc[-1]["train_standardized_mse"]),
            },
        }
        (summary_root / "summary.json").write_text(
            json.dumps(compact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(aggregate.to_string(index=False))
    print(f"artifacts: {output_root}")


if __name__ == "__main__":
    main()
