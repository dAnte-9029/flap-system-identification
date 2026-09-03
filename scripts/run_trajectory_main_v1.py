#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
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
    evaluate_trajectory_predictions,
)
from system_identification.models.trajectory import IntegratedDynamicsPredictor
from system_identification.training.trajectory_baselines import fit_ridge_dynamics
from system_identification.training.trajectory_main_v1 import (
    MainV1Config,
    assemble_history_trajectory_windows,
    fit_history_trajectory_model,
    fit_main_v1_stats,
    predict_history_trajectory_model,
)


MODEL_ORDER = (
    "ridge_no_control",
    "history_controlled_local",
    "no_history_controlled_multistep",
    "history_no_control_multistep",
    "main_v1_history_controlled_multistep",
)

COMPARISONS = {
    "versus_ridge_no_control": ("ridge_no_control", "main_v1_history_controlled_multistep"),
    "history_no_control_versus_ridge": (
        "ridge_no_control",
        "history_no_control_multistep",
    ),
    "history_gain": (
        "no_history_controlled_multistep",
        "main_v1_history_controlled_multistep",
    ),
    "control_gain": (
        "history_no_control_multistep",
        "main_v1_history_controlled_multistep",
    ),
    "multistep_objective_gain": (
        "history_controlled_local",
        "main_v1_history_controlled_multistep",
    ),
}

METRICS = (
    "position_rmse_m_per_log_macro",
    "velocity_rmse_m_s_per_log_macro",
    "attitude_rmse_deg_per_log_macro",
    "body_rate_rmse_rad_s_per_log_macro",
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _verify_step1_contract(dataset_root: Path, manifest: dict) -> None:
    if manifest.get("dataset_version") != "trajectory_dataset_v1":
        raise ValueError("Step 3 requires trajectory_dataset_v1")
    split = manifest.get("split_contract", {})
    if split.get("sealed_test_opened") is not False:
        raise ValueError("refusing a dataset manifest that opened sealed test")
    if set(split.get("materialized_partitions", [])) != {"train", "validation"}:
        raise ValueError("Step 3 requires train and validation only")
    if (dataset_root / "samples_sealed_test.parquet").exists():
        raise ValueError("sealed-test samples are present; refusing to continue")
    for partition in ("train", "validation"):
        for stem in ("samples", "windows"):
            if not (dataset_root / f"{stem}_{partition}.parquet").is_file():
                raise FileNotFoundError(f"missing Step 1 artifact: {stem}_{partition}.parquet")


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


def _gain_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for comparison, (reference_name, candidate_name) in COMPARISONS.items():
        reference = aggregate.loc[aggregate["model"] == reference_name].set_index("horizon_s")
        candidate = aggregate.loc[aggregate["model"] == candidate_name].set_index("horizon_s")
        for horizon in sorted(set(reference.index) & set(candidate.index)):
            row: dict[str, float | str] = {
                "comparison": comparison,
                "reference_model": reference_name,
                "candidate_model": candidate_name,
                "horizon_s": float(horizon),
            }
            for metric in METRICS:
                baseline = float(reference.loc[horizon, metric])
                evaluated = float(candidate.loc[horizon, metric])
                row[f"{metric}_gain_percent"] = 100.0 * (baseline - evaluated) / baseline
            rows.append(row)
    return pd.DataFrame(rows)


def _success_gate(gains: pd.DataFrame) -> dict[str, bool]:
    midlong = gains.loc[gains["horizon_s"].isin([0.5, 1.0, 2.0])]

    def all_positive(comparison: str) -> bool:
        selected = midlong.loc[midlong["comparison"] == comparison]
        columns = [f"{metric}_gain_percent" for metric in METRICS]
        return len(selected) == 3 and bool((selected[columns] > 0.0).all().all())

    beats_ridge = all_positive("versus_ridge_no_control")
    history_no_control_beats_ridge = all_positive("history_no_control_versus_ridge")
    control_gain = all_positive("control_gain")
    multistep_gain = all_positive("multistep_objective_gain")
    return {
        "beats_ridge_all_metrics_at_0p5_1p0_2p0_s": beats_ridge,
        "history_no_control_beats_ridge_all_metrics_at_0p5_1p0_2p0_s": (
            history_no_control_beats_ridge
        ),
        "control_gain_all_metrics_at_0p5_1p0_2p0_s": control_gain,
        "multistep_gain_all_metrics_at_0p5_1p0_2p0_s": multistep_gain,
        "recommend_enter_h1_h2": beats_ridge and control_gain and multistep_gain,
    }


def _plot_metrics(aggregate: pd.DataFrame, path: Path) -> None:
    panels = (
        ("position_rmse_m_per_log_macro", "Position RMSE (m)"),
        ("velocity_rmse_m_s_per_log_macro", "Velocity RMSE (m/s)"),
        ("attitude_rmse_deg_per_log_macro", "Attitude RMSE (deg)"),
        ("body_rate_rmse_rad_s_per_log_macro", "Body-rate RMSE (rad/s)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for axis, (metric, label) in zip(axes.flat, panels, strict=True):
        for index, model_name in enumerate(MODEL_ORDER):
            rows = aggregate.loc[aggregate["model"] == model_name].sort_values("horizon_s")
            axis.plot(
                rows["horizon_s"],
                rows[metric],
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
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate trajectory Main V1")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/trajectory_v1_august_f5_c4",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "artifacts/trajectory_main_v1"
    )
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--horizons", default="0.10,0.20,0.50,1.00,2.00")
    parser.add_argument("--history-steps", type=int, default=26)
    parser.add_argument("--train-rollout-steps", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    summary_root = args.summary_root.resolve() if args.summary_root is not None else None
    for path in (output_root, summary_root):
        if path is not None and path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output root: {path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {args.device}")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    dataset_manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    _verify_step1_contract(dataset_root, dataset_manifest)
    nominal_rate_hz = float(dataset_manifest["sampling"]["nominal_rate_hz"])
    maximum_steps = int(dataset_manifest["sampling"]["horizon_steps"])
    horizon_steps = _parse_horizons(
        args.horizons, nominal_rate_hz=nominal_rate_hz, maximum_steps=maximum_steps
    )
    if args.history_steps < 1 or args.train_rollout_steps < 1:
        raise ValueError("history_steps and train_rollout_steps must be positive")
    if args.train_rollout_steps > maximum_steps:
        raise ValueError("train_rollout_steps exceed Step 1 windows")

    train_samples = pd.read_parquet(dataset_root / "samples_train.parquet")
    train_windows = pd.read_parquet(dataset_root / "windows_train.parquet")
    validation_samples = pd.read_parquet(dataset_root / "samples_validation.parquet")
    validation_windows = pd.read_parquet(dataset_root / "windows_validation.parquet")
    train_batch = assemble_history_trajectory_windows(
        train_samples, train_windows, history_steps=args.history_steps
    )
    validation_batch = assemble_history_trajectory_windows(
        validation_samples, validation_windows, history_steps=args.history_steps
    )
    stats = fit_main_v1_stats(train_samples, train_batch)

    common = {
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "seed": args.seed,
    }
    configs = (
        MainV1Config(
            model_name="history_controlled_local",
            use_history=True,
            use_controls=True,
            objective_steps=1,
            **common,
        ),
        MainV1Config(
            model_name="no_history_controlled_multistep",
            use_history=False,
            use_controls=True,
            objective_steps=args.train_rollout_steps,
            **common,
        ),
        MainV1Config(
            model_name="history_no_control_multistep",
            use_history=True,
            use_controls=False,
            objective_steps=args.train_rollout_steps,
            **common,
        ),
        MainV1Config(
            model_name="main_v1_history_controlled_multistep",
            use_history=True,
            use_controls=True,
            objective_steps=args.train_rollout_steps,
            **common,
        ),
    )

    models = {}
    histories = []
    for config in configs:
        print(f"training {config.model_name}", flush=True)
        model, history = fit_history_trajectory_model(
            train_batch, stats, config, device=args.device
        )
        models[config.model_name] = model
        histories.append(history)

    metric_frames = []
    ridge, ridge_transitions = fit_ridge_dynamics(
        train_samples, alpha=args.ridge_alpha, use_controls=False
    )
    ridge_prediction = IntegratedDynamicsPredictor(ridge).rollout(
        validation_batch.trajectory.initial_state(),
        validation_batch.trajectory.controls,
        validation_batch.trajectory.dt_s,
    )
    predictions = {"ridge_no_control": ridge_prediction}
    for config in configs:
        print(f"evaluating {config.model_name}", flush=True)
        predictions[config.model_name] = predict_history_trajectory_model(
            models[config.model_name],
            validation_batch,
            use_history=config.use_history,
            batch_size=args.batch_size,
            device=args.device,
        )
    for model_name in MODEL_ORDER:
        metric_frames.append(
            evaluate_trajectory_predictions(
                predictions[model_name],
                validation_batch.trajectory.truth,
                model_name=model_name,
                split="validation",
                window_ids=validation_batch.trajectory.window_ids,
                log_ids=validation_batch.trajectory.log_ids,
                segment_ids=validation_batch.trajectory.segment_ids,
                horizon_steps=horizon_steps,
                dt_s=validation_batch.trajectory.dt_s,
            )
        )
    window_metrics = pd.concat(metric_frames, ignore_index=True)
    aggregate, per_log = aggregate_trajectory_metrics(window_metrics)
    gains = _gain_table(aggregate)
    gate = _success_gate(gains)
    training_history = pd.concat(histories, ignore_index=True)

    output_root.mkdir(parents=True)
    model_root = output_root / "models"
    model_root.mkdir()
    window_metrics.to_parquet(output_root / "validation_window_metrics.parquet", index=False)
    aggregate.to_csv(output_root / "validation_horizon_metrics.csv", index=False)
    per_log.to_csv(output_root / "validation_per_log_metrics.csv", index=False)
    gains.to_csv(output_root / "matched_ablation_gains.csv", index=False)
    training_history.to_csv(output_root / "training_history.csv", index=False)
    _plot_metrics(aggregate, output_root / "rollout_errors.png")
    for config in configs:
        torch.save(
            {
                "state_dict": models[config.model_name].state_dict(),
                "config": asdict(config),
                "history_steps": args.history_steps,
                "phase_contract": "sin/cos(relative_phase - phase_at_t0)",
            },
            model_root / f"{config.model_name}.pt",
        )

    run_manifest = {
        "experiment": "trajectory_main_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_head": _git_head(),
        "dataset": {
            "root": str(dataset_root),
            "dataset_version": dataset_manifest["dataset_version"],
            "builder_git_head": dataset_manifest["builder_git_head"],
            "primary_cohort": dataset_manifest["source"]["primary_cohort"],
            "partitions_read": ["train", "validation"],
            "sealed_test_opened": False,
        },
        "history_contract": {
            "version": "trajectory_history_context_v1",
            "history_steps_including_t0": args.history_steps,
            "nominal_history_s": (args.history_steps - 1) / nominal_rate_hz,
            "same_step1_window_ids": True,
            "boundary": "same log and segment; left padding ignored by explicit mask",
            "observations": "past rigid-body state, relative flap state, and matched past controls",
            "future_known": dataset_manifest["roles"][
                "known_future_control_t0_to_tT_exclusive"
            ],
            "future_forbidden": dataset_manifest["roles"]["future_forbidden_as_input"],
            "phase": "sin/cos(relative_phase - phase_at_t0); invariant to log-local zero offset",
        },
        "training": {
            "device": args.device,
            "train_window_count": int(len(train_batch.trajectory.window_ids)),
            "validation_window_count": int(len(validation_batch.trajectory.window_ids)),
            "ridge_transition_count": int(len(ridge_transitions.features)),
            "train_rollout_steps": args.train_rollout_steps,
            "train_rollout_s_nominal": args.train_rollout_steps / nominal_rate_hz,
            "normalization": "fit on train histories, controls, and transitions only",
            "validation_used_for_fitting_tuning_or_early_stopping": False,
            "configs": [asdict(config) for config in configs],
        },
        "evaluation": {
            "horizon_steps": {str(key): value for key, value in horizon_steps.items()},
            "primary_aggregation": "equal-log macro of per-log window RMSE",
            "same_validation_window_count_per_model_horizon": int(
                len(validation_batch.trajectory.window_ids)
            ),
        },
        "success_gate": gate,
        "models": list(MODEL_ORDER),
        "artifacts": {
            "window_metrics": "validation_window_metrics.parquet",
            "horizon_metrics": "validation_horizon_metrics.csv",
            "per_log_metrics": "validation_per_log_metrics.csv",
            "matched_ablation_gains": "matched_ablation_gains.csv",
            "training_history": "training_history.csv",
            "figure": "rollout_errors.png",
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if summary_root is not None:
        summary_root.mkdir(parents=True)
        aggregate.to_csv(summary_root / "validation_horizon_metrics.csv", index=False)
        per_log.to_csv(summary_root / "validation_per_log_metrics.csv", index=False)
        gains.to_csv(summary_root / "matched_ablation_gains.csv", index=False)
        _plot_metrics(aggregate, summary_root / "rollout_errors.png")
        compact = {
            **run_manifest,
            "artifact_root": str(output_root),
            "final_train_trajectory_loss": {
                name: float(group.iloc[-1]["train_trajectory_loss"])
                for name, group in training_history.groupby("model", sort=True)
            },
        }
        (summary_root / "summary.json").write_text(
            json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(aggregate.to_string(index=False))
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"artifacts: {output_root}")


if __name__ == "__main__":
    main()
