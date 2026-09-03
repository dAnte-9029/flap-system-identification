#!/usr/bin/env python3
"""Train and evaluate actuator-aware trajectory Main V2 ablations."""

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
from system_identification.training.trajectory_main_v1 import (
    MainV1Config,
    assemble_history_trajectory_windows,
    fit_history_trajectory_model,
    fit_main_v1_stats,
    predict_history_trajectory_model,
)
from system_identification.training.trajectory_main_v2 import (
    MainV2Config,
    fit_actuator_aware_model,
    fit_main_v2_stats,
)


REFERENCE = "history_no_control_multistep"
DRIVE = "v2_drive_state"
UNGATED = "v2_drive_tail_ungated"
MAIN_V2 = "main_v2_drive_tail_gated"
MODEL_ORDER = (REFERENCE, DRIVE, UNGATED, MAIN_V2)
COMPARISONS = {
    "drive_gain": (REFERENCE, DRIVE),
    "ungated_tail_gain": (DRIVE, UNGATED),
    "gated_tail_gain": (DRIVE, MAIN_V2),
    "gating_gain": (UNGATED, MAIN_V2),
    "main_v2_control_gain": (REFERENCE, MAIN_V2),
}
METRICS = (
    "position_rmse_m",
    "velocity_rmse_m_s",
    "attitude_rmse_deg",
    "body_rate_rmse_rad_s",
)
HORIZON_METRIC_COLUMNS = tuple(f"{metric}_per_log_macro" for metric in METRICS)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _verify_step1_contract(dataset_root: Path, manifest: dict) -> None:
    if manifest.get("dataset_version") != "trajectory_dataset_v1":
        raise ValueError("Step 5 requires trajectory_dataset_v1")
    split = manifest.get("split_contract", {})
    if split.get("sealed_test_opened") is not False:
        raise ValueError("refusing a dataset manifest that opened sealed test")
    if set(split.get("materialized_partitions", [])) != {"train", "validation"}:
        raise ValueError("Step 5 requires train and validation only")
    if (dataset_root / "samples_sealed_test.parquet").exists():
        raise ValueError("sealed-test samples are present; refusing to continue")
    for partition in ("train", "validation"):
        for stem in ("samples", "windows"):
            if not (dataset_root / f"{stem}_{partition}.parquet").is_file():
                raise FileNotFoundError(f"missing Step 1 artifact: {stem}_{partition}.parquet")


def _parse_horizons(raw: str, *, nominal_rate_hz: float, maximum_steps: int) -> dict[float, int]:
    result = {}
    for part in raw.split(","):
        horizon = float(part.strip())
        steps = int(round(horizon * nominal_rate_hz))
        if horizon <= 0.0 or steps < 1 or steps > maximum_steps:
            raise ValueError(f"horizon outside available window: {horizon}")
        if not np.isclose(steps / nominal_rate_hz, horizon, atol=1e-9):
            raise ValueError(f"horizon must be an integer number of nominal samples: {horizon}")
        result[horizon] = steps
    if not result:
        raise ValueError("at least one horizon is required")
    return result


def _gain_tables(
    aggregate: pd.DataFrame, per_log: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    per_log_rows = []
    for comparison, (reference_name, candidate_name) in COMPARISONS.items():
        reference = aggregate.loc[aggregate["model"] == reference_name].set_index("horizon_s")
        candidate = aggregate.loc[aggregate["model"] == candidate_name].set_index("horizon_s")
        for horizon in sorted(set(reference.index) & set(candidate.index)):
            row = {
                "comparison": comparison,
                "reference_model": reference_name,
                "candidate_model": candidate_name,
                "horizon_s": float(horizon),
            }
            for column in HORIZON_METRIC_COLUMNS:
                baseline = float(reference.loc[horizon, column])
                evaluated = float(candidate.loc[horizon, column])
                row[f"{column}_gain_percent"] = 100.0 * (baseline - evaluated) / baseline
            aggregate_rows.append(row)
            reference_logs = per_log.loc[
                (per_log["model"] == reference_name) & (per_log["horizon_s"] == horizon)
            ].set_index("log_id")
            candidate_logs = per_log.loc[
                (per_log["model"] == candidate_name) & (per_log["horizon_s"] == horizon)
            ].set_index("log_id")
            for metric in METRICS:
                shared = sorted(set(reference_logs.index) & set(candidate_logs.index))
                gains = reference_logs.loc[shared, metric] - candidate_logs.loc[shared, metric]
                per_log_rows.append(
                    {
                        "comparison": comparison,
                        "horizon_s": float(horizon),
                        "metric": metric,
                        "logs_improved": int(np.sum(gains > 0.0)),
                        "log_count": len(shared),
                        "mean_absolute_gain": float(np.mean(gains)),
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_log_rows)


def _frequency_metrics(predictions: dict, batch, horizon_steps: dict[float, int]) -> pd.DataFrame:
    rows = []
    truth = batch.trajectory.truth.flap_frequency_hz
    for model_name, prediction in predictions.items():
        for horizon_s, step in horizon_steps.items():
            error = prediction.flap_frequency_hz[:, step] - truth[:, step]
            by_log = []
            for log_id in sorted(set(batch.trajectory.log_ids)):
                selected = batch.trajectory.log_ids == log_id
                by_log.append(float(np.sqrt(np.mean(np.square(error[selected])))))
            rows.append(
                {
                    "model": model_name,
                    "split": "validation",
                    "horizon_s": float(horizon_s),
                    "flap_frequency_rmse_hz_per_log_macro": float(np.mean(by_log)),
                    "flap_frequency_rmse_hz_per_log_min": float(np.min(by_log)),
                    "flap_frequency_rmse_hz_per_log_max": float(np.max(by_log)),
                    "log_count": len(by_log),
                    "window_count": len(error),
                }
            )
    return pd.DataFrame(rows)


def _success_gate(gains: pd.DataFrame, counts: pd.DataFrame) -> dict[str, bool]:
    horizons = {0.5, 1.0, 2.0}

    def macro_all_positive(comparison: str) -> bool:
        selected = gains.loc[
            (gains["comparison"] == comparison) & gains["horizon_s"].isin(horizons)
        ]
        columns = [f"{column}_gain_percent" for column in HORIZON_METRIC_COLUMNS]
        return len(selected) == 3 and bool((selected[columns] > 0.0).all().all())

    def majority_logs(comparison: str) -> bool:
        selected = counts.loc[
            (counts["comparison"] == comparison) & counts["horizon_s"].isin(horizons)
        ]
        return len(selected) == 12 and bool((selected["logs_improved"] >= 3).all())

    drive_positive = macro_all_positive("drive_gain")
    tail_positive = macro_all_positive("gated_tail_gain")
    main_positive = macro_all_positive("main_v2_control_gain")
    main_majority = majority_logs("main_v2_control_gain")
    return {
        "drive_gain_all_metrics_at_0p5_1p0_2p0_s": drive_positive,
        "gated_tail_gain_all_metrics_at_0p5_1p0_2p0_s": tail_positive,
        "main_v2_control_gain_all_metrics_at_0p5_1p0_2p0_s": main_positive,
        "main_v2_improves_at_least_3_of_5_logs_every_metric_horizon": main_majority,
        "recommend_enter_h1_h2": main_positive and main_majority,
    }


def _reference_reproduction(aggregate: pd.DataFrame, step3_metrics_path: Path) -> dict[str, float]:
    current = aggregate.loc[aggregate["model"] == REFERENCE].sort_values("horizon_s")
    frozen = pd.read_csv(step3_metrics_path)
    frozen = frozen.loc[frozen["model"] == REFERENCE].sort_values("horizon_s")
    if len(current) != len(frozen) or not np.array_equal(
        current["horizon_s"].to_numpy(), frozen["horizon_s"].to_numpy()
    ):
        raise ValueError("frozen Step 3 reference horizons do not match")
    differences = {
        column: float(np.max(np.abs(current[column].to_numpy() - frozen[column].to_numpy())))
        for column in HORIZON_METRIC_COLUMNS
    }
    return differences


def _plot_rollout(aggregate: pd.DataFrame, path: Path) -> None:
    panels = tuple(
        zip(
            HORIZON_METRIC_COLUMNS,
            ("Position RMSE (m)", "Velocity RMSE (m/s)", "Attitude RMSE (deg)", "Body-rate RMSE (rad/s)"),
            strict=True,
        )
    )
    labels = {
        REFERENCE: "history only",
        DRIVE: "drive state",
        UNGATED: "drive + tail, ungated",
        MAIN_V2: "Main V2, gated",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.3), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for axis, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        for index, model_name in enumerate(MODEL_ORDER):
            rows = aggregate.loc[aggregate["model"] == model_name].sort_values("horizon_s")
            axis.plot(rows["horizon_s"], rows[metric], marker="o", linewidth=1.8,
                      color=colors[index], label=labels[model_name])
        axis.set(xlabel="Rollout horizon (s)", ylabel=ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_gains(gains: pd.DataFrame, path: Path) -> None:
    selected = gains.loc[
        (gains["comparison"] == "main_v2_control_gain")
        & gains["horizon_s"].isin([0.5, 1.0, 2.0])
    ].sort_values("horizon_s")
    fig, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    x = np.arange(len(selected))
    width = 0.18
    labels = ("position", "velocity", "attitude", "body rate")
    for index, (column, label) in enumerate(zip(HORIZON_METRIC_COLUMNS, labels, strict=True)):
        axis.bar(x + (index - 1.5) * width, selected[f"{column}_gain_percent"], width, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [f"{value:.1f}" for value in selected["horizon_s"]])
    axis.set(xlabel="Rollout horizon (s)", ylabel="Main V2 gain over history only (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset/trajectory_v1_august_f5_c4")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/trajectory_main_v2")
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--step3-horizon-metrics", type=Path,
                        default=PROJECT_ROOT / "docs/analysis/results/trajectory_main_v1/validation_horizon_metrics.csv")
    parser.add_argument("--horizons", default="0.10,0.20,0.50,1.00,2.00")
    parser.add_argument("--history-steps", type=int, default=26)
    parser.add_argument("--train-rollout-steps", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--base-epochs", type=int, default=40)
    parser.add_argument("--actuator-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--actuator-seed", type=int, default=29)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    summary_root = args.summary_root.resolve() if args.summary_root else None
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
    horizon_steps = _parse_horizons(args.horizons, nominal_rate_hz=nominal_rate_hz,
                                     maximum_steps=maximum_steps)
    if args.train_rollout_steps < 1 or args.train_rollout_steps > maximum_steps:
        raise ValueError("train_rollout_steps outside Step 1 windows")
    train_samples = pd.read_parquet(dataset_root / "samples_train.parquet")
    validation_samples = pd.read_parquet(dataset_root / "samples_validation.parquet")
    train_windows = pd.read_parquet(dataset_root / "windows_train.parquet")
    validation_windows = pd.read_parquet(dataset_root / "windows_validation.parquet")
    train_batch = assemble_history_trajectory_windows(train_samples, train_windows,
                                                       history_steps=args.history_steps)
    validation_batch = assemble_history_trajectory_windows(validation_samples, validation_windows,
                                                             history_steps=args.history_steps)
    base_stats = fit_main_v1_stats(train_samples, train_batch)
    base_config = MainV1Config(
        model_name=REFERENCE, use_history=True, use_controls=False,
        objective_steps=args.train_rollout_steps, hidden_size=args.hidden_size,
        epochs=args.base_epochs, batch_size=args.batch_size, seed=args.seed,
    )
    print(f"training {REFERENCE}", flush=True)
    base_model, base_history = fit_history_trajectory_model(
        train_batch, base_stats, base_config, device=args.device
    )
    v2_stats = fit_main_v2_stats(train_batch)
    common = {
        "objective_steps": args.train_rollout_steps,
        "epochs": args.actuator_epochs,
        "batch_size": args.batch_size,
        "seed": args.actuator_seed,
    }
    configs = (
        MainV2Config(model_name=DRIVE, use_drive=True, use_tail=False, gated_tail=False, **common),
        MainV2Config(model_name=UNGATED, use_drive=True, use_tail=True, gated_tail=False,
                     tail_gate_l1=0.0, **common),
        MainV2Config(model_name=MAIN_V2, use_drive=True, use_tail=True, gated_tail=True, **common),
    )
    models = {}
    histories = [base_history]
    for config in configs:
        print(f"training {config.model_name}", flush=True)
        model, history = fit_actuator_aware_model(
            train_batch, base_model, v2_stats, config, device=args.device
        )
        models[config.model_name] = model
        histories.append(history)

    predictions = {
        REFERENCE: predict_history_trajectory_model(
            base_model, validation_batch, use_history=True, batch_size=args.batch_size,
            device=args.device
        )
    }
    for config in configs:
        print(f"evaluating {config.model_name}", flush=True)
        predictions[config.model_name] = predict_history_trajectory_model(
            models[config.model_name], validation_batch, use_history=True,
            batch_size=args.batch_size, device=args.device
        )
    metric_frames = []
    for model_name in MODEL_ORDER:
        metric_frames.append(
            evaluate_trajectory_predictions(
                predictions[model_name], validation_batch.trajectory.truth,
                model_name=model_name, split="validation",
                window_ids=validation_batch.trajectory.window_ids,
                log_ids=validation_batch.trajectory.log_ids,
                segment_ids=validation_batch.trajectory.segment_ids,
                horizon_steps=horizon_steps, dt_s=validation_batch.trajectory.dt_s,
            )
        )
    window_metrics = pd.concat(metric_frames, ignore_index=True)
    aggregate, per_log = aggregate_trajectory_metrics(window_metrics)
    gains, log_counts = _gain_tables(aggregate, per_log)
    frequency_metrics = _frequency_metrics(predictions, validation_batch, horizon_steps)
    gate = _success_gate(gains, log_counts)
    reference_difference = _reference_reproduction(
        aggregate, args.step3_horizon_metrics.resolve()
    )
    training_history = pd.concat(histories, ignore_index=True, sort=False)

    output_root.mkdir(parents=True)
    model_root = output_root / "models"
    model_root.mkdir()
    window_metrics.to_parquet(output_root / "validation_window_metrics.parquet", index=False)
    aggregate.to_csv(output_root / "validation_horizon_metrics.csv", index=False)
    per_log.to_csv(output_root / "validation_per_log_metrics.csv", index=False)
    gains.to_csv(output_root / "matched_ablation_gains.csv", index=False)
    log_counts.to_csv(output_root / "per_log_improvement_counts.csv", index=False)
    frequency_metrics.to_csv(output_root / "flap_frequency_metrics.csv", index=False)
    training_history.to_csv(output_root / "training_history.csv", index=False)
    _plot_rollout(aggregate, output_root / "rollout_errors.png")
    _plot_gains(gains, output_root / "main_v2_control_gains.png")
    torch.save({"state_dict": base_model.state_dict(), "config": asdict(base_config)},
               model_root / f"{REFERENCE}.pt")
    for config in configs:
        torch.save(
            {
                "state_dict": models[config.model_name].state_dict(),
                "config": asdict(config),
                "base_config": asdict(base_config),
                "tail_mean": v2_stats.tail_mean,
                "tail_std": v2_stats.tail_std,
                "phase_contract": "sin/cos(relative_phase - phase_at_t0)",
            },
            model_root / f"{config.model_name}.pt",
        )
    final_gates = {
        name: models[name].tail_gate_values().detach().cpu().tolist()
        for name in (UNGATED, MAIN_V2)
    }
    run_manifest = {
        "experiment": "trajectory_main_v2",
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_head": _git_head(),
        "step4_baseline_head": "fcc661e6ec3a8441d679a4ed0820c234d0458d7b",
        "dataset": {
            "root": str(dataset_root),
            "dataset_version": dataset_manifest["dataset_version"],
            "builder_git_head": dataset_manifest["builder_git_head"],
            "primary_cohort": dataset_manifest["source"]["primary_cohort"],
            "partitions_read": ["train", "validation"],
            "sealed_test_opened": False,
        },
        "contract": {
            "history_version": "trajectory_history_context_v1",
            "history_steps_including_t0": args.history_steps,
            "future_known": dataset_manifest["roles"]["known_future_control_t0_to_tT_exclusive"],
            "future_forbidden": dataset_manifest["roles"]["future_forbidden_as_input"],
            "relative_phase": "sin/cos(relative_phase - phase_at_t0); no cross-log zero assumption",
            "control_path": "future commands update causal actuator states; no realized future actuator state input",
        },
        "architecture": {
            "base": "frozen Step 3 history-only GRU and derivative head",
            "drive": "0.10 s causal first-order motor state; residual restricted to flap-frequency rate",
            "tail": "0.04 s causal states for symmetric, differential, rudder controls",
            "tail_output_masks": {
                "symmetric": ["body_acceleration_x", "body_acceleration_z", "pitch_acceleration"],
                "differential": ["roll_acceleration"],
                "rudder": ["body_acceleration_y", "yaw_acceleration"],
            },
            "fallback": "frozen history-only base plus zero-initialized-small control residuals; gated tail starts at 0.05",
        },
        "training": {
            "device": args.device,
            "train_windows": len(train_windows),
            "validation_windows": len(validation_windows),
            "base_config": asdict(base_config),
            "actuator_configs": [asdict(config) for config in configs],
            "normalization": "train only",
            "validation_used_for_fitting_tuning_or_early_stopping": False,
        },
        "evaluation": {
            "horizon_steps": {str(key): value for key, value in horizon_steps.items()},
            "primary_aggregation": "equal-log macro of per-log window RMSE",
            "reference_reproduction_max_abs_difference": reference_difference,
        },
        "final_tail_gates_symmetric_differential_rudder": final_gates,
        "success_gate": gate,
        "artifacts": {
            "horizon_metrics": "validation_horizon_metrics.csv",
            "per_log_metrics": "validation_per_log_metrics.csv",
            "matched_ablation_gains": "matched_ablation_gains.csv",
            "per_log_improvement_counts": "per_log_improvement_counts.csv",
            "flap_frequency_metrics": "flap_frequency_metrics.csv",
            "training_history": "training_history.csv",
            "rollout_figure": "rollout_errors.png",
            "gain_figure": "main_v2_control_gains.png",
            "window_metrics": "validation_window_metrics.parquet",
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if summary_root is not None:
        summary_root.mkdir(parents=True)
        for filename in (
            "validation_horizon_metrics.csv", "validation_per_log_metrics.csv",
            "matched_ablation_gains.csv", "per_log_improvement_counts.csv",
            "flap_frequency_metrics.csv", "rollout_errors.png", "main_v2_control_gains.png",
        ):
            source = output_root / filename
            (summary_root / filename).write_bytes(source.read_bytes())
        compact = {
            **run_manifest,
            "artifact_root": str(output_root),
            "final_train_losses": {
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
