from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]

DEFAULT_FEATURE_COLUMNS = [
    "phase_corrected_sin",
    "phase_corrected_cos",
    "wing_stroke_angle_rad",
    "flap_frequency_hz",
    "cycle_flap_frequency_hz",
    "motor_cmd_0",
    "servo_left_elevon",
    "servo_right_elevon",
    "servo_rudder",
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "vehicle_local_position.ax",
    "vehicle_local_position.ay",
    "vehicle_local_position.az",
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
    "vehicle_angular_velocity.xyz_derivative[0]",
    "vehicle_angular_velocity.xyz_derivative[1]",
    "vehicle_angular_velocity.xyz_derivative[2]",
    "gravity_b.x",
    "gravity_b.y",
    "gravity_b.z",
    "airspeed_validated.true_airspeed_m_s",
    "vehicle_air_data.rho",
    "wind.windspeed_north",
    "wind.windspeed_east",
]

DEFAULT_FEATURE_GROUPS: dict[str, list[str]] = {
    "phase": [
        "phase_corrected_sin",
        "phase_corrected_cos",
        "wing_stroke_angle_rad",
        "flap_frequency_hz",
        "cycle_flap_frequency_hz",
    ],
    "actuators": [
        "motor_cmd_0",
        "servo_left_elevon",
        "servo_right_elevon",
        "servo_rudder",
    ],
    "linear_kinematics": [
        "vehicle_local_position.vx",
        "vehicle_local_position.vy",
        "vehicle_local_position.vz",
        "vehicle_local_position.ax",
        "vehicle_local_position.ay",
        "vehicle_local_position.az",
    ],
    "angular_kinematics": [
        "vehicle_angular_velocity.xyz[0]",
        "vehicle_angular_velocity.xyz[1]",
        "vehicle_angular_velocity.xyz[2]",
        "vehicle_angular_velocity.xyz_derivative[0]",
        "vehicle_angular_velocity.xyz_derivative[1]",
        "vehicle_angular_velocity.xyz_derivative[2]",
    ],
    "attitude": [
        "gravity_b.x",
        "gravity_b.y",
        "gravity_b.z",
    ],
    "aero": [
        "airspeed_validated.true_airspeed_m_s",
        "vehicle_air_data.rho",
        "wind.windspeed_north",
        "wind.windspeed_east",
    ],
}

DEFAULT_ABLATION_VARIANTS: dict[str, dict[str, list[str]]] = {
    "full": {"include_groups": list(DEFAULT_FEATURE_GROUPS.keys())},
    "no_phase": {"drop_groups": ["phase"]},
    "no_actuators": {"drop_groups": ["actuators"]},
    "no_attitude": {"drop_groups": ["attitude"]},
    "no_aero": {"drop_groups": ["aero"]},
    "phase_plus_kinematics": {"include_groups": ["phase", "linear_kinematics", "angular_kinematics"]},
    "kinematics_plus_actuators": {"include_groups": ["linear_kinematics", "angular_kinematics", "actuators"]},
}


def _ordered_unique_columns(columns: list[str], reference: list[str]) -> list[str]:
    allowed = set(columns)
    return [column for column in reference if column in allowed]


def resolve_ablation_variants(
    variant_names: list[str] | None = None,
    *,
    base_feature_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    feature_columns = base_feature_columns or DEFAULT_FEATURE_COLUMNS
    available_columns = set(feature_columns)
    selected_variant_names = variant_names or list(DEFAULT_ABLATION_VARIANTS.keys())

    resolved: dict[str, list[str]] = {}
    for variant_name in selected_variant_names:
        if variant_name not in DEFAULT_ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant: {variant_name}")
        spec = DEFAULT_ABLATION_VARIANTS[variant_name]
        include_groups = spec.get("include_groups")
        drop_groups = spec.get("drop_groups", [])

        if include_groups is not None:
            selected_columns: list[str] = []
            for group_name in include_groups:
                if group_name not in DEFAULT_FEATURE_GROUPS:
                    raise ValueError(f"Unknown feature group: {group_name}")
                selected_columns.extend(DEFAULT_FEATURE_GROUPS[group_name])
            selected = _ordered_unique_columns(selected_columns, feature_columns)
        else:
            dropped_columns: set[str] = set()
            for group_name in drop_groups:
                if group_name not in DEFAULT_FEATURE_GROUPS:
                    raise ValueError(f"Unknown feature group: {group_name}")
                dropped_columns.update(DEFAULT_FEATURE_GROUPS[group_name])
            selected = [column for column in feature_columns if column not in dropped_columns]

        resolved[variant_name] = [column for column in selected if column in available_columns]

    return resolved


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str | None) -> torch.device:
    if requested and requested.lower() != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _with_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()
    if "phase_corrected_rad" not in derived.columns:
        raise ValueError("Missing required column: phase_corrected_rad")
    phase = derived["phase_corrected_rad"].to_numpy(dtype=float)
    derived["phase_corrected_sin"] = np.sin(phase)
    derived["phase_corrected_cos"] = np.cos(phase)

    quaternion_columns = [
        "vehicle_attitude.q[0]",
        "vehicle_attitude.q[1]",
        "vehicle_attitude.q[2]",
        "vehicle_attitude.q[3]",
    ]
    if all(column in derived.columns for column in quaternion_columns):
        quat = derived.loc[:, quaternion_columns].to_numpy(dtype=float, copy=True)
        quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
        quat_norm = np.where(quat_norm > 1e-8, quat_norm, 1.0)
        quat = quat / quat_norm
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        # Use gravity direction in body FRD to avoid q / -q sign ambiguity.
        derived["gravity_b.x"] = 2.0 * (x * z - w * y)
        derived["gravity_b.y"] = 2.0 * (y * z + w * x)
        derived["gravity_b.z"] = 1.0 - 2.0 * (x * x + y * y)
    return derived


def prepare_feature_target_frames(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    target_cols = target_columns or DEFAULT_TARGET_COLUMNS
    derived = _with_derived_columns(frame)

    missing_features = [column for column in feature_cols if column not in derived.columns]
    missing_targets = [column for column in target_cols if column not in derived.columns]
    if missing_features or missing_targets:
        missing = missing_features + missing_targets
        raise ValueError(f"Missing required training columns: {missing}")

    features = derived.loc[:, feature_cols].copy()
    targets = derived.loc[:, target_cols].copy()
    return features, targets


def _fit_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(features), features, medians)
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    return medians.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


def _fit_target_stats(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(targets).all():
        raise ValueError("Targets contain non-finite values")
    means = targets.mean(axis=0)
    stds = targets.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def _transform_features(features: np.ndarray, medians: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    imputed = np.where(np.isfinite(features), features, medians)
    return ((imputed - means) / stds).astype(np.float32)


def _transform_targets(targets: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return ((targets - means) / stds).astype(np.float32)


def _inverse_transform_targets(targets_scaled: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (targets_scaled * stds + means).astype(np.float32)


def _as_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _to_serializable_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(bundle)
    for key in ["feature_medians", "feature_means", "feature_stds", "target_means", "target_stds"]:
        serializable[key] = torch.as_tensor(bundle[key], dtype=torch.float32)
    return serializable


def _metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_columns: list[str],
    split_name: str,
) -> dict[str, Any]:
    residual = y_pred - y_true
    overall_mae = float(np.mean(np.abs(residual)))
    overall_rmse = float(np.sqrt(np.mean(np.square(residual))))

    per_target: dict[str, dict[str, float]] = {}
    r2_values: list[float] = []
    for idx, target_name in enumerate(target_columns):
        target_true = y_true[:, idx]
        target_pred = y_pred[:, idx]
        target_residual = target_pred - target_true
        ss_res = float(np.sum(np.square(target_residual)))
        ss_tot = float(np.sum(np.square(target_true - target_true.mean())))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        r2_values.append(r2)
        per_target[target_name] = {
            "mae": float(np.mean(np.abs(target_residual))),
            "rmse": float(np.sqrt(np.mean(np.square(target_residual)))),
            "r2": float(r2),
        }

    return {
        "split": split_name,
        "sample_count": int(len(y_true)),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_r2": float(np.mean(r2_values)),
        "per_target": per_target,
    }


def _make_loader(
    features: np.ndarray,
    targets: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
    if targets is None:
        dataset = TensorDataset(feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _predict_scaled_batches(
    model: nn.Module,
    features_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    loader = _make_loader(
        features_scaled,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    outputs: list[np.ndarray] = []
    amp_enabled = use_amp and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_features)
            outputs.append(batch_predictions.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _history_frame(history: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(history)


def _save_training_curves(history: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = history["epoch"].to_numpy()

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Scaled MSE")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_overall_rmse"], label="val_overall_rmse")
    axes[1].plot(epochs, history["val_overall_mae"], label="val_overall_mae")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Wrench Error")
    axes[1].set_title("Validation Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for target_name in DEFAULT_TARGET_COLUMNS:
        r2_column = f"val_{target_name}_r2"
        if r2_column in history.columns:
            axes[2].plot(epochs, history[r2_column], label=target_name)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R^2")
    axes[2].set_title("Validation Per-Target R^2")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pred_vs_true_plot(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    batch_size: int,
    device: str | None = None,
) -> None:
    _, targets_df = prepare_feature_target_frames(frame, bundle["feature_columns"], bundle["target_columns"])
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for idx, target_name in enumerate(bundle["target_columns"]):
        ax = axes_flat[idx]
        y_true = targets_df[target_name].to_numpy()
        y_pred = predictions_df[target_name].to_numpy()
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.scatter(y_true, y_pred, s=5, alpha=0.15)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax.set_title(target_name)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_residual_hist_plot(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    batch_size: int,
    device: str | None = None,
) -> None:
    _, targets_df = prepare_feature_target_frames(frame, bundle["feature_columns"], bundle["target_columns"])
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for idx, target_name in enumerate(bundle["target_columns"]):
        ax = axes_flat[idx]
        residual = predictions_df[target_name].to_numpy() - targets_df[target_name].to_numpy()
        ax.hist(residual, bins=50, alpha=0.8, color="steelblue", edgecolor="black")
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(target_name)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _flatten_split_metrics(split_name: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    flat: dict[str, float | int] = {
        f"{split_name}_sample_count": int(metrics["sample_count"]),
        f"{split_name}_overall_mae": float(metrics["overall_mae"]),
        f"{split_name}_overall_rmse": float(metrics["overall_rmse"]),
        f"{split_name}_overall_r2": float(metrics["overall_r2"]),
    }
    for target_name, target_metrics in metrics["per_target"].items():
        for metric_name, value in target_metrics.items():
            flat[f"{split_name}_{target_name}_{metric_name}"] = float(value)
    return flat


def _save_ablation_summary_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    fig_width = max(8.0, 1.5 * len(summary))
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    x = np.arange(len(summary))
    width = 0.36
    ax.bar(x - width / 2, summary["val_overall_r2"], width=width, label="val_overall_r2")
    ax.bar(x + width / 2, summary["test_overall_r2"], width=width, label="test_overall_r2")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["variant_name"], rotation=20, ha="right")
    ax.set_ylabel("R^2")
    ax.set_title("Feature Ablation Summary")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _evaluate_scaled_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_features)
                loss = torch.nn.functional.mse_loss(predictions, batch_targets, reduction="mean")
            batch_size = len(batch_features)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def fit_torch_regressor(
    *,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    hidden_sizes: tuple[int, ...] = (256, 256),
    dropout: float = 0.0,
    batch_size: int = 4096,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 8,
    device: str | None = None,
    random_seed: int = 42,
    num_workers: int = 0,
    use_amp: bool = True,
) -> dict[str, Any]:
    _set_random_seed(random_seed)
    resolved_device = _resolve_device(device)
    pin_memory = resolved_device.type == "cuda"

    train_features_df, train_targets_df = prepare_feature_target_frames(train_frame, feature_columns, target_columns)
    val_features_df, val_targets_df = prepare_feature_target_frames(val_frame, feature_columns, target_columns)

    train_features = train_features_df.to_numpy(dtype=np.float32, copy=True)
    train_targets = train_targets_df.to_numpy(dtype=np.float32, copy=True)
    val_features = val_features_df.to_numpy(dtype=np.float32, copy=True)
    val_targets = val_targets_df.to_numpy(dtype=np.float32, copy=True)

    feature_medians, feature_means, feature_stds = _fit_feature_stats(train_features)
    target_means, target_stds = _fit_target_stats(train_targets)

    train_features_scaled = _transform_features(train_features, feature_medians, feature_means, feature_stds)
    val_features_scaled = _transform_features(val_features, feature_medians, feature_means, feature_stds)
    train_targets_scaled = _transform_targets(train_targets, target_means, target_stds)
    val_targets_scaled = _transform_targets(val_targets, target_means, target_stds)

    train_loader = _make_loader(
        train_features_scaled,
        train_targets_scaled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        val_features_scaled,
        val_targets_scaled,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MLPRegressor(
        input_dim=train_features_scaled.shape[1],
        output_dim=train_targets_scaled.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and resolved_device.type == "cuda")

    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    amp_enabled = use_amp and resolved_device.type == "cuda"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(resolved_device, non_blocking=True)
            batch_targets = batch_targets.to(resolved_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=resolved_device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_features)
                loss = torch.nn.functional.mse_loss(predictions, batch_targets, reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = len(batch_features)
            train_loss_sum += float(loss.item()) * batch_count
            train_sample_count += batch_count

        train_loss = train_loss_sum / max(train_sample_count, 1)
        val_loss = _evaluate_scaled_loss(model, val_loader, resolved_device, use_amp=use_amp)
        val_predictions_scaled = _predict_scaled_batches(
            model,
            val_features_scaled,
            batch_size=batch_size,
            device=resolved_device,
            use_amp=use_amp,
        )
        val_predictions = _inverse_transform_targets(val_predictions_scaled, target_means, target_stds)
        val_metrics = _metrics_from_arrays(
            val_targets.astype(np.float32, copy=False),
            val_predictions.astype(np.float32, copy=False),
            target_columns=list(train_targets_df.columns),
            split_name="val",
        )
        history_row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_overall_mae": float(val_metrics["overall_mae"]),
            "val_overall_rmse": float(val_metrics["overall_rmse"]),
            "val_overall_r2": float(val_metrics["overall_r2"]),
        }
        for target_name, metrics in val_metrics["per_target"].items():
            history_row[f"val_{target_name}_mae"] = float(metrics["mae"])
            history_row[f"val_{target_name}_rmse"] = float(metrics["rmse"])
            history_row[f"val_{target_name}_r2"] = float(metrics["r2"])
        history.append(history_row)

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    model.load_state_dict(best_state_dict)

    return {
        "model_state_dict": best_state_dict,
        "feature_columns": list(train_features_df.columns),
        "target_columns": list(train_targets_df.columns),
        "feature_medians": feature_medians,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "target_means": target_means,
        "target_stds": target_stds,
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(dropout),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "device_type": resolved_device.type,
        "use_amp": bool(amp_enabled),
        "random_seed": int(random_seed),
    }


def _build_model_from_bundle(bundle: dict[str, Any], device: torch.device) -> nn.Module:
    model = MLPRegressor(
        input_dim=len(bundle["feature_columns"]),
        output_dim=len(bundle["target_columns"]),
        hidden_sizes=tuple(int(v) for v in bundle["hidden_sizes"]),
        dropout=float(bundle["dropout"]),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model


def predict_model_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    resolved_device = _resolve_device(device or bundle.get("device_type"))
    features_df, _ = prepare_feature_target_frames(frame, bundle["feature_columns"], bundle["target_columns"])
    features = features_df.to_numpy(dtype=np.float32, copy=True)
    features_scaled = _transform_features(
        features,
        _as_numpy_array(bundle["feature_medians"]),
        _as_numpy_array(bundle["feature_means"]),
        _as_numpy_array(bundle["feature_stds"]),
    )

    loader = _make_loader(
        features_scaled,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved_device.type == "cuda",
    )

    model = _build_model_from_bundle(bundle, resolved_device)
    predictions_scaled: list[np.ndarray] = []
    amp_enabled = bool(bundle.get("use_amp", False)) and resolved_device.type == "cuda"

    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(resolved_device, non_blocking=True)
            with torch.autocast(device_type=resolved_device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_features)
            predictions_scaled.append(batch_predictions.cpu().numpy())

    predictions_scaled_arr = np.concatenate(predictions_scaled, axis=0)
    predictions = _inverse_transform_targets(
        predictions_scaled_arr,
        _as_numpy_array(bundle["target_means"]),
        _as_numpy_array(bundle["target_stds"]),
    )
    return pd.DataFrame(predictions, columns=bundle["target_columns"])


def evaluate_model_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    batch_size: int = 8192,
    device: str | None = None,
) -> dict[str, Any]:
    _, targets_df = prepare_feature_target_frames(frame, bundle["feature_columns"], bundle["target_columns"])
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)

    y_true = targets_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = predictions_df.to_numpy(dtype=np.float64, copy=False)
    return _metrics_from_arrays(y_true, y_pred, target_columns=bundle["target_columns"], split_name=split_name)


def _load_split_frame(split_root: str | Path, split_name: str, max_samples: int | None, sample_seed: int) -> pd.DataFrame:
    path = Path(split_root) / f"{split_name}_samples.parquet"
    frame = pd.read_parquet(path)
    if max_samples is not None and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=sample_seed).reset_index(drop=True)
    return frame


def run_training_job(
    *,
    split_root: str | Path,
    output_dir: str | Path,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    hidden_sizes: tuple[int, ...] = (256, 256),
    dropout: float = 0.0,
    batch_size: int = 4096,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 8,
    device: str | None = None,
    random_seed: int = 42,
    num_workers: int = 0,
    use_amp: bool = True,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_frame = _load_split_frame(split_root, "train", max_train_samples, random_seed)
    val_frame = _load_split_frame(split_root, "val", max_val_samples, random_seed + 1)
    test_frame = _load_split_frame(split_root, "test", max_test_samples, random_seed + 2)

    bundle = fit_torch_regressor(
        train_frame=train_frame,
        val_frame=val_frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
        random_seed=random_seed,
        num_workers=num_workers,
        use_amp=use_amp,
    )

    metrics = {
        "train": evaluate_model_bundle(bundle, train_frame, split_name="train", batch_size=batch_size, device=device),
        "val": evaluate_model_bundle(bundle, val_frame, split_name="val", batch_size=batch_size, device=device),
        "test": evaluate_model_bundle(bundle, test_frame, split_name="test", batch_size=batch_size, device=device),
    }

    model_bundle_path = output_path / "model_bundle.pt"
    metrics_path = output_path / "metrics.json"
    training_config_path = output_path / "training_config.json"
    history_path = output_path / "history.csv"
    training_curves_path = output_path / "training_curves.png"
    pred_vs_true_test_path = output_path / "pred_vs_true_test.png"
    residual_hist_test_path = output_path / "residual_hist_test.png"

    torch.save(_to_serializable_bundle(bundle), model_bundle_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    history = _history_frame(bundle["history"])
    history.to_csv(history_path, index=False)
    _save_training_curves(history, training_curves_path)
    _save_pred_vs_true_plot(bundle, test_frame, pred_vs_true_test_path, batch_size=batch_size, device=device)
    _save_residual_hist_plot(bundle, test_frame, residual_hist_test_path, batch_size=batch_size, device=device)
    training_config_path.write_text(
        json.dumps(
            {
                "split_root": str(split_root),
                "feature_columns": bundle["feature_columns"],
                "target_columns": bundle["target_columns"],
                "hidden_sizes": list(hidden_sizes),
                "dropout": float(dropout),
                "batch_size": int(batch_size),
                "max_epochs": int(max_epochs),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "early_stopping_patience": int(early_stopping_patience),
                "device": device or "auto",
                "resolved_device_type": bundle["device_type"],
                "random_seed": int(random_seed),
                "num_workers": int(num_workers),
                "use_amp": bool(use_amp),
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
                "max_test_samples": max_test_samples,
                "best_epoch": bundle["best_epoch"],
                "best_val_loss": bundle["best_val_loss"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "model_bundle_path": str(model_bundle_path),
        "metrics_path": str(metrics_path),
        "training_config_path": str(training_config_path),
        "history_path": str(history_path),
        "training_curves_path": str(training_curves_path),
        "pred_vs_true_test_path": str(pred_vs_true_test_path),
        "residual_hist_test_path": str(residual_hist_test_path),
    }


def run_ablation_study(
    *,
    split_root: str | Path,
    output_dir: str | Path,
    variant_names: list[str] | None = None,
    base_feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    hidden_sizes: tuple[int, ...] = (256, 256),
    dropout: float = 0.0,
    batch_size: int = 4096,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 8,
    device: str | None = None,
    random_seed: int = 42,
    num_workers: int = 0,
    use_amp: bool = True,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_variants = resolve_ablation_variants(variant_names, base_feature_columns=base_feature_columns)
    summary_rows: list[dict[str, Any]] = []
    variant_outputs: dict[str, dict[str, str]] = {}

    for variant_name, feature_columns in resolved_variants.items():
        variant_output_dir = output_path / variant_name
        outputs = run_training_job(
            split_root=split_root,
            output_dir=variant_output_dir,
            feature_columns=feature_columns,
            target_columns=target_columns,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device,
            random_seed=random_seed,
            num_workers=num_workers,
            use_amp=use_amp,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )
        variant_outputs[variant_name] = outputs

        metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
        training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "variant_name": variant_name,
            "output_dir": str(variant_output_dir),
            "feature_count": len(feature_columns),
            "feature_columns": json.dumps(feature_columns),
            "best_epoch": int(training_config["best_epoch"]),
            "best_val_loss": float(training_config["best_val_loss"]),
        }
        for split_name in ["train", "val", "test"]:
            row.update(_flatten_split_metrics(split_name, metrics[split_name]))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_csv_path = output_path / "ablation_summary.csv"
    summary_json_path = output_path / "ablation_summary.json"
    summary_plot_path = output_path / "ablation_summary.png"

    summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    _save_ablation_summary_plot(summary, summary_plot_path)

    return {
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "summary_plot_path": str(summary_plot_path),
    }
