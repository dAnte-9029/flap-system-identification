"""Existing run, ablation, and baseline-comparison orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from system_identification.artifacts.io import (
    _save_pred_vs_true_plot,
    _save_residual_hist_plot,
    _save_training_curves,
)
from system_identification.evaluation.diagnostics import _training_audit_flags, evaluate_model_bundle
from system_identification.evaluation.metrics import _combine_disjoint_target_metrics
from system_identification.evaluation.reports import (
    _flatten_split_metrics,
    _history_frame,
    _target_groups_label,
)
from system_identification.models.bundles import (
    _is_rollout_model_type,
    _is_sequence_model_type,
    _normalized_model_type,
    _to_serializable_bundle,
)
from system_identification.models.features import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_FEATURE_SETS,
    NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS,
    resolve_feature_set_columns,
)
from system_identification.plotting.figures import _save_ablation_summary_plot, _save_baseline_comparison_plot
from system_identification.training.data_preparation import DEFAULT_TARGET_COLUMNS, _load_split_frame
from system_identification.training.losses import _target_loss_weights_as_dict, resolve_target_loss_weights
from system_identification.training.recipes import (
    fit_torch_regressor,
    fit_torch_rollout_regressor,
    fit_torch_sequence_regressor,
)

LEAKAGE_RESISTANT_BASELINE_PROTOCOL: dict[str, Any] = {
    "name": "leakage_resistant_mlp_baseline_v1",
    "split_policy": "whole_log",
    "feature_set_name": "paper_no_accel_v2",
    "model_type": "mlp",
    "loss_type": "huber",
    "huber_delta": 1.5,
    "window_mode": "single",
    "window_radius": 0,
    "window_feature_mode": "all",
    "selection_metric": "val_loss",
    "primary_reported_metrics": ["per_target_mae", "per_target_rmse", "per_target_r2"],
    "forbidden_feature_columns": NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS,
}


BASELINE_COMPARISON_RECIPES: dict[str, dict[str, Any]] = {
    "mlp_paper_no_accel_v2": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "mlp",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
    },
    "mlp_paper_pfnn_10": {
        "feature_set_name": "paper_pfnn_10",
        "model_type": "mlp",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
    },
    "pfnn_paper_pfnn_10": {
        "feature_set_name": "paper_pfnn_10",
        "model_type": "pfnn",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
    },
    "mlp_paper_no_accel_v2_causal_phase_actuator": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "mlp",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "causal",
        "window_radius": 6,
        "window_feature_mode": "phase_actuator",
    },
    "split_axis_mlp_paper_no_accel_v2": {
        "recipe_type": "split_axis",
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "split_axis_mlp",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "target_groups": {
            "longitudinal": ["fx_b", "fz_b", "my_b"],
            "lateral": ["fy_b", "mx_b", "mz_b"],
        },
    },
    "causal_gru_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_gru",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
    },
    "causal_gru_asl_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_gru_asl",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "asl_hidden_size": 128,
        "asl_dropout": 0.1,
        "asl_max_frequency_bins": None,
    },
    "causal_lstm_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_lstm",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
    },
    "causal_tcn_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_tcn",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "tcn_channels": 128,
        "tcn_num_blocks": 4,
        "tcn_kernel_size": 3,
    },
    "causal_transformer_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_transformer",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "transformer_d_model": 64,
        "transformer_num_layers": 1,
        "transformer_num_heads": 4,
        "transformer_dim_feedforward": 128,
    },
    "causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_transformer_head_film",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "transformer_d_model": 64,
        "transformer_num_layers": 1,
        "transformer_num_heads": 4,
        "transformer_dim_feedforward": 128,
    },
    "causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_transformer_input_film",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "transformer_d_model": 64,
        "transformer_num_layers": 1,
        "transformer_num_heads": 4,
        "transformer_dim_feedforward": 128,
    },
    "causal_transformer_paper_no_accel_v2_phase_harmonic_airdata": {
        "feature_set_name": "paper_no_accel_v2_phase_harmonic",
        "model_type": "causal_transformer",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_harmonic_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "transformer_d_model": 64,
        "transformer_num_layers": 1,
        "transformer_num_heads": 4,
        "transformer_dim_feedforward": 128,
    },
    "causal_transformer_paper_no_accel_v2_raw_phase_airdata": {
        "feature_set_name": "paper_no_accel_v2_raw_phase",
        "model_type": "causal_transformer",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "raw_phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "transformer_d_model": 64,
        "transformer_num_layers": 1,
        "transformer_num_heads": 4,
        "transformer_dim_feedforward": 128,
    },
    "causal_transformer_paper_no_accel_v2_no_phase_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_transformer",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "no_phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "transformer_d_model": 64,
        "transformer_num_layers": 1,
        "transformer_num_heads": 4,
        "transformer_dim_feedforward": 128,
    },
    "causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "causal_tcn_gru",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "tcn_channels": 128,
        "tcn_num_blocks": 3,
        "tcn_kernel_size": 3,
    },
    "subsection_gru_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "subsection_gru",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "rollout_size": 32,
        "rollout_stride": 32,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
    },
    "subnet_discrete_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "subnet_discrete",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "rollout_size": 32,
        "rollout_stride": 32,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "latent_size": 16,
    },
    "ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata": {
        "feature_set_name": "paper_no_accel_v2",
        "model_type": "ct_subnet_euler",
        "loss_type": "huber",
        "huber_delta": 1.5,
        "window_mode": "single",
        "window_radius": 0,
        "window_feature_mode": "all",
        "sequence_history_size": 64,
        "rollout_size": 32,
        "rollout_stride": 32,
        "sequence_feature_mode": "phase_actuator_airdata",
        "current_feature_mode": "remaining_current",
        "latent_size": 16,
        "dt_over_tau": 0.03,
        "ct_integrator": "euler",
    },
}


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


def run_training_job(
    *,
    split_root: str | Path,
    output_dir: str | Path,
    feature_set_name: str | None = None,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    model_type: str = "mlp",
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
    target_loss_weights: str | dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None = None,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    window_mode: str = "single",
    window_radius: int = 0,
    window_feature_mode: str = "all",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    pfnn_expanded_input_dim: int = 45,
    pfnn_phase_node_count: int = 5,
    pfnn_control_points: int = 6,
    sequence_history_size: int = 64,
    sequence_feature_mode: str = "phase_actuator_airdata",
    current_feature_mode: str = "remaining_current",
    rollout_size: int = 32,
    rollout_stride: int | None = None,
    gru_num_layers: int = 1,
    asl_hidden_size: int = 128,
    asl_dropout: float = 0.1,
    asl_max_frequency_bins: int | None = None,
    tcn_channels: int = 128,
    tcn_num_blocks: int = 4,
    tcn_kernel_size: int = 3,
    transformer_d_model: int = 64,
    transformer_num_layers: int = 1,
    transformer_num_heads: int = 4,
    transformer_dim_feedforward: int = 128,
    transformer_use_positional_encoding: bool = True,
    latent_size: int = 16,
    dt_over_tau: float = 0.03,
    ct_integrator: str = "euler",
    skip_test_eval: bool = False,
    lr_scheduler: str | None = None,
    lr_warmup_ratio: float = 0.0,
    gradient_clip_norm: float | None = None,
    ema_decay: float = 0.0,
) -> dict[str, str]:
    if feature_set_name is not None and feature_columns is not None:
        raise ValueError("feature_set_name and feature_columns cannot both be provided")

    resolved_feature_columns = resolve_feature_set_columns(feature_set_name) if feature_set_name is not None else feature_columns
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_frame = _load_split_frame(split_root, "train", max_train_samples, random_seed)
    val_frame = _load_split_frame(split_root, "val", max_val_samples, random_seed + 1)
    test_frame = None if skip_test_eval else _load_split_frame(split_root, "test", max_test_samples, random_seed + 2)

    resolved_model_type = _normalized_model_type(model_type)
    if _is_sequence_model_type(resolved_model_type):
        bundle = fit_torch_sequence_regressor(
            train_frame=train_frame,
            val_frame=val_frame,
            feature_columns=resolved_feature_columns,
            target_columns=target_columns,
            model_type=resolved_model_type,
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
            target_loss_weights=target_loss_weights,
            loss_type=loss_type,
            huber_delta=huber_delta,
            sequence_history_size=sequence_history_size,
            sequence_feature_mode=sequence_feature_mode,
            current_feature_mode=current_feature_mode,
            gru_num_layers=gru_num_layers,
            asl_hidden_size=asl_hidden_size,
            asl_dropout=asl_dropout,
            asl_max_frequency_bins=asl_max_frequency_bins,
            tcn_channels=tcn_channels,
            tcn_num_blocks=tcn_num_blocks,
            tcn_kernel_size=tcn_kernel_size,
            transformer_d_model=transformer_d_model,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_use_positional_encoding=transformer_use_positional_encoding,
            lr_scheduler=lr_scheduler,
            lr_warmup_ratio=lr_warmup_ratio,
            gradient_clip_norm=gradient_clip_norm,
            ema_decay=ema_decay,
        )
    elif _is_rollout_model_type(resolved_model_type):
        bundle = fit_torch_rollout_regressor(
            train_frame=train_frame,
            val_frame=val_frame,
            feature_columns=resolved_feature_columns,
            target_columns=target_columns,
            model_type=resolved_model_type,
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
            target_loss_weights=target_loss_weights,
            loss_type=loss_type,
            huber_delta=huber_delta,
            sequence_history_size=sequence_history_size,
            rollout_size=rollout_size,
            rollout_stride=rollout_stride,
            sequence_feature_mode=sequence_feature_mode,
            current_feature_mode=current_feature_mode,
            gru_num_layers=gru_num_layers,
            latent_size=latent_size,
            dt_over_tau=dt_over_tau,
            ct_integrator=ct_integrator,
        )
    else:
        bundle = fit_torch_regressor(
            train_frame=train_frame,
            val_frame=val_frame,
            feature_columns=resolved_feature_columns,
            target_columns=target_columns,
            model_type=resolved_model_type,
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
            target_loss_weights=target_loss_weights,
            loss_type=loss_type,
            huber_delta=huber_delta,
            window_mode=window_mode,
            window_radius=window_radius,
            window_feature_mode=window_feature_mode,
            pfnn_expanded_input_dim=pfnn_expanded_input_dim,
            pfnn_phase_node_count=pfnn_phase_node_count,
            pfnn_control_points=pfnn_control_points,
        )

    metrics = {
        "train": evaluate_model_bundle(bundle, train_frame, split_name="train", batch_size=batch_size, device=device),
        "val": evaluate_model_bundle(bundle, val_frame, split_name="val", batch_size=batch_size, device=device),
    }
    if test_frame is not None:
        metrics["test"] = evaluate_model_bundle(bundle, test_frame, split_name="test", batch_size=batch_size, device=device)

    model_bundle_path = output_path / "model_bundle.pt"
    metrics_path = output_path / "metrics.json"
    training_config_path = output_path / "training_config.json"
    history_path = output_path / "history.csv"
    training_curves_path = output_path / "training_curves.png"
    pred_vs_true_test_path = output_path / "pred_vs_true_test.png"
    residual_hist_test_path = output_path / "residual_hist_test.png"

    bundle["feature_set_name"] = feature_set_name or ("custom" if feature_columns is not None else "full")
    torch.save(_to_serializable_bundle(bundle), model_bundle_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    history = _history_frame(bundle["history"])
    history.to_csv(history_path, index=False)
    _save_training_curves(history, training_curves_path)
    if test_frame is not None:
        _save_pred_vs_true_plot(bundle, test_frame, pred_vs_true_test_path, batch_size=batch_size, device=device)
        _save_residual_hist_plot(bundle, test_frame, residual_hist_test_path, batch_size=batch_size, device=device)
    audit_flags = _training_audit_flags(bundle, split_root=split_root)
    training_config_path.write_text(
        json.dumps(
            {
                "split_root": str(split_root),
                "feature_set_name": feature_set_name or ("custom" if feature_columns is not None else "full"),
                "model_type": bundle["model_type"],
                "feature_columns": bundle["feature_columns"],
                "base_feature_columns": bundle["base_feature_columns"],
                "target_columns": bundle["target_columns"],
                "target_loss_weights": bundle["target_loss_weights_by_name"],
                "loss_type": bundle["loss_type"],
                "huber_delta": bundle["huber_delta"],
                "window_mode": bundle.get("window_mode"),
                "window_radius": bundle.get("window_radius"),
                "window_feature_mode": bundle.get("window_feature_mode"),
                "window_feature_columns": bundle.get("window_feature_columns"),
                "sequence_history_size": bundle.get("sequence_history_size"),
                "sequence_feature_mode": bundle.get("sequence_feature_mode"),
                "sequence_feature_columns": bundle.get("sequence_feature_columns"),
                "current_feature_mode": bundle.get("current_feature_mode"),
                "current_feature_columns": bundle.get("current_feature_columns"),
                "sequence_sample_count_train": bundle.get("sequence_sample_count_train"),
                "sequence_sample_count_val": bundle.get("sequence_sample_count_val"),
                "rollout_size": bundle.get("rollout_size"),
                "rollout_stride": bundle.get("rollout_stride"),
                "context_feature_columns": bundle.get("context_feature_columns"),
                "rollout_feature_columns": bundle.get("rollout_feature_columns"),
                "rollout_sample_count_train": bundle.get("rollout_sample_count_train"),
                "rollout_timestep_count_train": bundle.get("rollout_timestep_count_train"),
                "rollout_sample_count_val": bundle.get("rollout_sample_count_val"),
                "rollout_timestep_count_val": bundle.get("rollout_timestep_count_val"),
                "sequence_group_columns": bundle.get("sequence_group_columns"),
                "sequence_sort_column": bundle.get("sequence_sort_column"),
                "hidden_sizes": list(hidden_sizes),
                "dropout": float(dropout),
                "phase_feature_index": bundle.get("phase_feature_index"),
                "phase_feature_column": bundle.get("phase_feature_column"),
                "pfnn_expanded_input_dim": int(pfnn_expanded_input_dim),
                "pfnn_phase_node_count": int(pfnn_phase_node_count),
                "pfnn_control_points": int(pfnn_control_points),
                "gru_num_layers": bundle.get("gru_num_layers"),
                "asl_hidden_size": bundle.get("asl_hidden_size"),
                "asl_dropout": bundle.get("asl_dropout"),
                "asl_max_frequency_bins": bundle.get("asl_max_frequency_bins"),
                "tcn_channels": bundle.get("tcn_channels"),
                "tcn_num_blocks": bundle.get("tcn_num_blocks"),
                "tcn_kernel_size": bundle.get("tcn_kernel_size"),
                "transformer_d_model": bundle.get("transformer_d_model"),
                "transformer_num_layers": bundle.get("transformer_num_layers"),
                "transformer_num_heads": bundle.get("transformer_num_heads"),
                "transformer_dim_feedforward": bundle.get("transformer_dim_feedforward"),
                "transformer_use_positional_encoding": bundle.get("transformer_use_positional_encoding"),
                "film_mode": bundle.get("film_mode"),
                "phase_conditioning_columns": bundle.get("phase_conditioning_columns"),
                "phase_conditioning_indices": bundle.get("phase_conditioning_indices"),
                "film_hidden_size": bundle.get("film_hidden_size"),
                "film_scale": bundle.get("film_scale"),
                "lr_scheduler": bundle.get("lr_scheduler"),
                "lr_warmup_ratio": bundle.get("lr_warmup_ratio"),
                "lr_warmup_steps": bundle.get("lr_warmup_steps"),
                "gradient_clip_norm": bundle.get("gradient_clip_norm"),
                "ema_decay": bundle.get("ema_decay"),
                "latent_size": bundle.get("latent_size"),
                "dt_over_tau": bundle.get("dt_over_tau"),
                "ct_integrator": bundle.get("ct_integrator"),
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
                "skip_test_eval": bool(skip_test_eval),
                "best_epoch": bundle["best_epoch"],
                "best_val_loss": bundle["best_val_loss"],
                **audit_flags,
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


def _resolve_baseline_recipe_names(recipe_names: list[str] | None = None) -> list[str]:
    resolved = recipe_names or list(BASELINE_COMPARISON_RECIPES.keys())
    unknown = [name for name in resolved if name not in BASELINE_COMPARISON_RECIPES]
    if unknown:
        raise ValueError(f"Unknown baseline comparison recipes: {unknown}")
    return list(resolved)


def _run_single_baseline_recipe(
    *,
    recipe_name: str,
    recipe: dict[str, Any],
    recipe_output_dir: Path,
    split_root: str | Path,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    device: str | None,
    random_seed: int,
    num_workers: int,
    use_amp: bool,
    target_loss_weights: str | dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    pfnn_expanded_input_dim: int,
    pfnn_phase_node_count: int,
    pfnn_control_points: int,
    sequence_history_size: int,
    sequence_feature_mode: str | None,
    current_feature_mode: str,
    rollout_size: int,
    rollout_stride: int | None,
    gru_num_layers: int,
    asl_hidden_size: int,
    asl_dropout: float,
    asl_max_frequency_bins: int | None,
    tcn_channels: int,
    tcn_num_blocks: int,
    tcn_kernel_size: int,
    transformer_d_model: int,
    transformer_num_layers: int,
    transformer_num_heads: int,
    transformer_dim_feedforward: int,
    transformer_use_positional_encoding: bool,
    latent_size: int,
    dt_over_tau: float,
    ct_integrator: str,
    skip_test_eval: bool,
    lr_scheduler: str | None,
    lr_warmup_ratio: float,
    gradient_clip_norm: float | None,
    ema_decay: float,
) -> dict[str, Any]:
    resolved_sequence_history_size = int(sequence_history_size)
    resolved_sequence_feature_mode = str(sequence_feature_mode or recipe.get("sequence_feature_mode", "phase_actuator_airdata"))
    resolved_current_feature_mode = str(current_feature_mode)
    resolved_rollout_size = int(rollout_size)
    resolved_rollout_stride = rollout_stride
    if resolved_rollout_stride is not None:
        resolved_rollout_stride = int(resolved_rollout_stride)
    resolved_asl_hidden_size = int(asl_hidden_size)
    resolved_asl_dropout = float(asl_dropout)
    resolved_asl_max_frequency_bins = asl_max_frequency_bins
    resolved_tcn_channels = int(tcn_channels)
    resolved_tcn_num_blocks = int(tcn_num_blocks)
    resolved_tcn_kernel_size = int(tcn_kernel_size)
    resolved_transformer_d_model = int(transformer_d_model)
    resolved_transformer_num_layers = int(transformer_num_layers)
    resolved_transformer_num_heads = int(transformer_num_heads)
    resolved_transformer_dim_feedforward = int(transformer_dim_feedforward)
    resolved_transformer_use_positional_encoding = bool(transformer_use_positional_encoding)
    resolved_latent_size = int(latent_size)
    resolved_dt_over_tau = float(dt_over_tau)
    resolved_ct_integrator = str(ct_integrator)
    outputs = run_training_job(
        split_root=split_root,
        output_dir=recipe_output_dir,
        feature_set_name=recipe["feature_set_name"],
        model_type=recipe["model_type"],
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
        target_loss_weights=target_loss_weights,
        loss_type=recipe["loss_type"],
        huber_delta=float(recipe["huber_delta"]),
        window_mode=recipe["window_mode"],
        window_radius=int(recipe["window_radius"]),
        window_feature_mode=recipe["window_feature_mode"],
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        pfnn_expanded_input_dim=pfnn_expanded_input_dim,
        pfnn_phase_node_count=pfnn_phase_node_count,
        pfnn_control_points=pfnn_control_points,
        sequence_history_size=resolved_sequence_history_size,
        sequence_feature_mode=resolved_sequence_feature_mode,
        current_feature_mode=resolved_current_feature_mode,
        rollout_size=resolved_rollout_size,
        rollout_stride=resolved_rollout_stride,
        gru_num_layers=gru_num_layers,
        asl_hidden_size=resolved_asl_hidden_size,
        asl_dropout=resolved_asl_dropout,
        asl_max_frequency_bins=resolved_asl_max_frequency_bins,
        tcn_channels=resolved_tcn_channels,
        tcn_num_blocks=resolved_tcn_num_blocks,
        tcn_kernel_size=resolved_tcn_kernel_size,
        transformer_d_model=resolved_transformer_d_model,
        transformer_num_layers=resolved_transformer_num_layers,
        transformer_num_heads=resolved_transformer_num_heads,
        transformer_dim_feedforward=resolved_transformer_dim_feedforward,
        transformer_use_positional_encoding=resolved_transformer_use_positional_encoding,
        latent_size=resolved_latent_size,
        dt_over_tau=resolved_dt_over_tau,
        ct_integrator=resolved_ct_integrator,
        skip_test_eval=skip_test_eval,
        lr_scheduler=lr_scheduler,
        lr_warmup_ratio=lr_warmup_ratio,
        gradient_clip_norm=gradient_clip_norm,
        ema_decay=ema_decay,
    )
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    training_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))

    row: dict[str, Any] = {
        "recipe_name": recipe_name,
        "output_dir": str(recipe_output_dir),
        "feature_set_name": recipe["feature_set_name"],
        "model_type": recipe["model_type"],
        "loss_type": recipe["loss_type"],
        "huber_delta": float(recipe["huber_delta"]),
        "window_mode": recipe["window_mode"],
        "window_radius": int(recipe["window_radius"]),
        "window_feature_mode": recipe["window_feature_mode"],
        "feature_count": len(training_config["feature_columns"]),
        "sequence_history_size": training_config.get("sequence_history_size"),
        "sequence_feature_mode": training_config.get("sequence_feature_mode"),
        "current_feature_mode": training_config.get("current_feature_mode"),
        "rollout_size": training_config.get("rollout_size"),
        "rollout_stride": training_config.get("rollout_stride"),
        "latent_size": training_config.get("latent_size"),
        "dt_over_tau": training_config.get("dt_over_tau"),
        "ct_integrator": training_config.get("ct_integrator"),
        "tcn_channels": training_config.get("tcn_channels"),
        "tcn_num_blocks": training_config.get("tcn_num_blocks"),
        "tcn_kernel_size": training_config.get("tcn_kernel_size"),
        "transformer_d_model": training_config.get("transformer_d_model"),
        "transformer_num_layers": training_config.get("transformer_num_layers"),
        "transformer_num_heads": training_config.get("transformer_num_heads"),
        "transformer_dim_feedforward": training_config.get("transformer_dim_feedforward"),
        "transformer_use_positional_encoding": training_config.get("transformer_use_positional_encoding"),
        "lr_scheduler": training_config.get("lr_scheduler"),
        "lr_warmup_ratio": training_config.get("lr_warmup_ratio"),
        "lr_warmup_steps": training_config.get("lr_warmup_steps"),
        "gradient_clip_norm": training_config.get("gradient_clip_norm"),
        "ema_decay": training_config.get("ema_decay"),
        "best_epoch": int(training_config["best_epoch"]),
        "best_val_loss": float(training_config["best_val_loss"]),
    }
    for split_name in ["train", "val", "test"]:
        if split_name in metrics:
            row.update(_flatten_split_metrics(split_name, metrics[split_name]))
    return row


def _run_split_axis_baseline_recipe(
    *,
    recipe_name: str,
    recipe: dict[str, Any],
    recipe_output_dir: Path,
    split_root: str | Path,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    device: str | None,
    random_seed: int,
    num_workers: int,
    use_amp: bool,
    target_loss_weights: str | dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    pfnn_expanded_input_dim: int,
    pfnn_phase_node_count: int,
    pfnn_control_points: int,
    sequence_history_size: int,
    sequence_feature_mode: str,
    current_feature_mode: str,
    rollout_size: int,
    rollout_stride: int | None,
    gru_num_layers: int,
    asl_hidden_size: int,
    asl_dropout: float,
    asl_max_frequency_bins: int | None,
    tcn_channels: int,
    tcn_num_blocks: int,
    tcn_kernel_size: int,
    transformer_d_model: int,
    transformer_num_layers: int,
    transformer_num_heads: int,
    transformer_dim_feedforward: int,
    transformer_use_positional_encoding: bool,
    latent_size: int,
    dt_over_tau: float,
    ct_integrator: str,
    skip_test_eval: bool,
    lr_scheduler: str | None,
    lr_warmup_ratio: float,
    gradient_clip_norm: float | None,
    ema_decay: float,
) -> dict[str, Any]:
    if skip_test_eval:
        raise ValueError("skip_test_eval is not supported for split-axis baseline recipes")
    target_groups = {name: list(targets) for name, targets in recipe["target_groups"].items()}
    split_group_metrics: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "val": {}, "test": {}}
    group_configs: dict[str, dict[str, Any]] = {}

    for group_index, (group_name, targets) in enumerate(target_groups.items()):
        group_output_dir = recipe_output_dir / group_name
        group_target_loss_weights = target_loss_weights
        if target_loss_weights is not None:
            full_weights = resolve_target_loss_weights(DEFAULT_TARGET_COLUMNS, target_loss_weights)
            full_weight_map = _target_loss_weights_as_dict(DEFAULT_TARGET_COLUMNS, full_weights)
            group_target_loss_weights = {target: full_weight_map[target] for target in targets}
        outputs = run_training_job(
            split_root=split_root,
            output_dir=group_output_dir,
            feature_set_name=recipe["feature_set_name"],
            target_columns=targets,
            model_type="mlp",
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device,
            random_seed=random_seed + group_index,
            num_workers=num_workers,
            use_amp=use_amp,
            target_loss_weights=group_target_loss_weights,
            loss_type=recipe["loss_type"],
            huber_delta=float(recipe["huber_delta"]),
            window_mode=recipe["window_mode"],
            window_radius=int(recipe["window_radius"]),
            window_feature_mode=recipe["window_feature_mode"],
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            pfnn_expanded_input_dim=pfnn_expanded_input_dim,
            pfnn_phase_node_count=pfnn_phase_node_count,
            pfnn_control_points=pfnn_control_points,
        )
        group_metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
        group_config = json.loads(Path(outputs["training_config_path"]).read_text(encoding="utf-8"))
        group_configs[group_name] = group_config
        for split_name in ["train", "val", "test"]:
            split_group_metrics[split_name][group_name] = group_metrics[split_name]

    combined_metrics = {
        split_name: _combine_disjoint_target_metrics(split_name, group_metrics)
        for split_name, group_metrics in split_group_metrics.items()
    }
    feature_counts = {len(config["feature_columns"]) for config in group_configs.values()}
    best_val_losses = [float(config["best_val_loss"]) for config in group_configs.values()]
    best_epochs = [int(config["best_epoch"]) for config in group_configs.values()]

    row: dict[str, Any] = {
        "recipe_name": recipe_name,
        "output_dir": str(recipe_output_dir),
        "feature_set_name": recipe["feature_set_name"],
        "model_type": recipe["model_type"],
        "loss_type": recipe["loss_type"],
        "huber_delta": float(recipe["huber_delta"]),
        "window_mode": recipe["window_mode"],
        "window_radius": int(recipe["window_radius"]),
        "window_feature_mode": recipe["window_feature_mode"],
        "feature_count": int(feature_counts.pop()) if len(feature_counts) == 1 else -1,
        "best_epoch": int(max(best_epochs)) if best_epochs else 0,
        "best_val_loss": float(np.mean(best_val_losses)) if best_val_losses else math.nan,
        "target_groups": _target_groups_label(target_groups),
    }
    for group_name, config in group_configs.items():
        row[f"{group_name}_best_epoch"] = int(config["best_epoch"])
        row[f"{group_name}_best_val_loss"] = float(config["best_val_loss"])
    for split_name in ["train", "val", "test"]:
        row.update(_flatten_split_metrics(split_name, combined_metrics[split_name]))
    return row


def run_baseline_comparison(
    *,
    split_root: str | Path,
    output_dir: str | Path,
    recipe_names: list[str] | None = None,
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
    target_loss_weights: str | dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    pfnn_expanded_input_dim: int = 45,
    pfnn_phase_node_count: int = 5,
    pfnn_control_points: int = 6,
    sequence_history_size: int = 64,
    sequence_feature_mode: str | None = None,
    current_feature_mode: str = "remaining_current",
    rollout_size: int = 32,
    rollout_stride: int | None = None,
    gru_num_layers: int = 1,
    asl_hidden_size: int = 128,
    asl_dropout: float = 0.1,
    asl_max_frequency_bins: int | None = None,
    tcn_channels: int = 128,
    tcn_num_blocks: int = 4,
    tcn_kernel_size: int = 3,
    transformer_d_model: int = 64,
    transformer_num_layers: int = 1,
    transformer_num_heads: int = 4,
    transformer_dim_feedforward: int = 128,
    transformer_use_positional_encoding: bool = True,
    latent_size: int = 16,
    dt_over_tau: float = 0.03,
    ct_integrator: str = "euler",
    skip_test_eval: bool = False,
    lr_scheduler: str | None = None,
    lr_warmup_ratio: float = 0.0,
    gradient_clip_norm: float | None = None,
    ema_decay: float = 0.0,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    resolved_recipe_names = _resolve_baseline_recipe_names(recipe_names)

    for recipe_name in resolved_recipe_names:
        recipe = dict(BASELINE_COMPARISON_RECIPES[recipe_name])
        recipe_output_dir = output_path / recipe_name
        common_kwargs = {
            "recipe_name": recipe_name,
            "recipe": recipe,
            "recipe_output_dir": recipe_output_dir,
            "split_root": split_root,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "device": device,
            "random_seed": random_seed,
            "num_workers": num_workers,
            "use_amp": use_amp,
            "target_loss_weights": target_loss_weights,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "max_test_samples": max_test_samples,
            "pfnn_expanded_input_dim": pfnn_expanded_input_dim,
            "pfnn_phase_node_count": pfnn_phase_node_count,
            "pfnn_control_points": pfnn_control_points,
            "sequence_history_size": sequence_history_size,
            "sequence_feature_mode": sequence_feature_mode,
            "current_feature_mode": current_feature_mode,
            "rollout_size": rollout_size,
            "rollout_stride": rollout_stride,
            "gru_num_layers": gru_num_layers,
            "asl_hidden_size": asl_hidden_size,
            "asl_dropout": asl_dropout,
            "asl_max_frequency_bins": asl_max_frequency_bins,
            "tcn_channels": tcn_channels,
            "tcn_num_blocks": tcn_num_blocks,
            "tcn_kernel_size": tcn_kernel_size,
            "transformer_d_model": transformer_d_model,
            "transformer_num_layers": transformer_num_layers,
            "transformer_num_heads": transformer_num_heads,
            "transformer_dim_feedforward": transformer_dim_feedforward,
            "transformer_use_positional_encoding": transformer_use_positional_encoding,
            "latent_size": latent_size,
            "dt_over_tau": dt_over_tau,
            "ct_integrator": ct_integrator,
            "skip_test_eval": skip_test_eval,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_ratio": lr_warmup_ratio,
            "gradient_clip_norm": gradient_clip_norm,
            "ema_decay": ema_decay,
        }
        if recipe.get("recipe_type") == "split_axis":
            row = _run_split_axis_baseline_recipe(**common_kwargs)
        else:
            row = _run_single_baseline_recipe(**common_kwargs)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_csv_path = output_path / "baseline_comparison_summary.csv"
    summary_json_path = output_path / "baseline_comparison_summary.json"
    summary_plot_path = output_path / "baseline_comparison_summary.png"
    protocol_path = output_path / "baseline_protocol.json"

    summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    _save_baseline_comparison_plot(summary, summary_plot_path)
    protocol_path.write_text(
        json.dumps(LEAKAGE_RESISTANT_BASELINE_PROTOCOL, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "summary_plot_path": str(summary_plot_path),
        "protocol_path": str(protocol_path),
    }
