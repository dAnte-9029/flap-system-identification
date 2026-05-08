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
import torch.nn.functional as F
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

NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS = [
    "vehicle_local_position.ax",
    "vehicle_local_position.ay",
    "vehicle_local_position.az",
    "vehicle_angular_velocity.xyz_derivative[0]",
    "vehicle_angular_velocity.xyz_derivative[1]",
    "vehicle_angular_velocity.xyz_derivative[2]",
]

NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS = [
    column for column in DEFAULT_FEATURE_COLUMNS if column not in set(NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS)
]

PAPER_NO_ACCEL_V2_ADDED_FEATURE_COLUMNS = [
    "vehicle_local_position.heading",
    "airspeed_validated.indicated_airspeed_m_s",
    "airspeed_validated.calibrated_airspeed_m_s",
    "airspeed_validated.calibrated_ground_minus_wind_m_s",
    "airspeed_validated.true_ground_minus_wind_m_s",
    "airspeed_validated.pitch_filtered",
    "roll_rad",
    "pitch_rad",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "relative_air_velocity_b.x",
    "relative_air_velocity_b.y",
    "relative_air_velocity_b.z",
    "alpha_rad",
    "beta_rad",
    "dynamic_pressure_pa",
    "elevator_like",
    "aileron_like",
]

PAPER_NO_ACCEL_V2_FEATURE_COLUMNS = NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS + PAPER_NO_ACCEL_V2_ADDED_FEATURE_COLUMNS

PHASE_HARMONIC_FEATURE_COLUMNS = [
    "phase_corrected_h2_sin",
    "phase_corrected_h2_cos",
    "phase_corrected_h3_sin",
    "phase_corrected_h3_cos",
]

PAPER_NO_ACCEL_V2_RAW_PHASE_FEATURE_COLUMNS = PAPER_NO_ACCEL_V2_FEATURE_COLUMNS + ["phase_corrected_rad"]
PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS = PAPER_NO_ACCEL_V2_FEATURE_COLUMNS + PHASE_HARMONIC_FEATURE_COLUMNS

PAPER_PFNN_10_FEATURE_COLUMNS = [
    "phase_corrected_rad",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "pitch_rad",
    "roll_rad",
    "alpha_rad",
    "beta_rad",
    "cycle_flap_frequency_hz",
    "elevator_like",
    "servo_rudder",
]

DEFAULT_FEATURE_SETS: dict[str, list[str]] = {
    "full": DEFAULT_FEATURE_COLUMNS,
    "no_accel_no_alpha": NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS,
    "paper_no_accel_v2": PAPER_NO_ACCEL_V2_FEATURE_COLUMNS,
    "paper_no_accel_v2_raw_phase": PAPER_NO_ACCEL_V2_RAW_PHASE_FEATURE_COLUMNS,
    "paper_no_accel_v2_phase_harmonic": PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS,
    "paper_pfnn_10": PAPER_PFNN_10_FEATURE_COLUMNS,
}

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

DEFAULT_REGIME_BIN_SPECS: dict[str, list[float]] = {
    "airspeed_validated.true_airspeed_m_s": [0.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "cycle_flap_frequency_hz": [0.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "phase_corrected_rad": [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
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


def resolve_feature_set_columns(feature_set_name: str | None = None) -> list[str]:
    resolved_name = feature_set_name or "full"
    if resolved_name not in DEFAULT_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {resolved_name}")
    return list(DEFAULT_FEATURE_SETS[resolved_name])


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


class CausalGRURegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.sequence_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )

        head_layers: list[nn.Module] = []
        last_dim = self.hidden_size + self.current_input_dim
        for hidden_dim in head_hidden_sizes:
            if hidden_dim <= 0:
                raise ValueError("head_hidden_sizes entries must be positive")
            head_layers.append(nn.Linear(last_dim, int(hidden_dim)))
            head_layers.append(nn.ReLU())
            if dropout > 0.0:
                head_layers.append(nn.Dropout(float(dropout)))
            last_dim = int(hidden_dim)
        head_layers.append(nn.Linear(last_dim, self.output_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        _, hidden = self.gru(sequence_inputs)
        representation = hidden[-1]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class AdaptiveSpectrumLayer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
        max_frequency_bins: int | None = None,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if max_frequency_bins is not None and max_frequency_bins <= 0:
            raise ValueError("max_frequency_bins must be positive when provided")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.max_frequency_bins = None if max_frequency_bins is None else int(max_frequency_bins)
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim * 3, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, self.input_dim),
            nn.Sigmoid(),
        )

    def forward(self, sequence_inputs: torch.Tensor) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.input_dim:
            raise ValueError(f"Expected input feature dim {self.input_dim}, got {sequence_inputs.shape[2]}")

        history_size = sequence_inputs.shape[1]
        spectrum = torch.fft.rfft(sequence_inputs, dim=1)
        retained_bins = spectrum.shape[1]
        if self.max_frequency_bins is not None:
            retained_bins = min(retained_bins, self.max_frequency_bins)
        retained = spectrum[:, :retained_bins, :]

        magnitude = torch.abs(retained)
        phase = torch.angle(retained)
        gate_inputs = torch.cat([magnitude, torch.cos(phase), torch.sin(phase)], dim=2)
        weights = self.gate(gate_inputs)
        weighted = retained * weights.to(dtype=retained.dtype)

        if retained_bins < spectrum.shape[1]:
            padded = torch.zeros_like(spectrum)
            padded[:, :retained_bins, :] = weighted
            weighted = padded
        reconstructed = torch.fft.irfft(weighted, n=history_size, dim=1)
        return sequence_inputs + reconstructed


class CausalGRUASLRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        asl_hidden_size: int = 128,
        asl_dropout: float = 0.1,
        asl_max_frequency_bins: int | None = None,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        self.asl = AdaptiveSpectrumLayer(
            input_dim=sequence_input_dim,
            hidden_size=asl_hidden_size,
            dropout=asl_dropout,
            max_frequency_bins=asl_max_frequency_bins,
        )
        self.regressor = CausalGRURegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        return self.regressor(self.asl(sequence_inputs), current_inputs)


def _make_mlp_layers(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    *,
    dropout: float = 0.0,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        if hidden_dim <= 0:
            raise ValueError("hidden_sizes entries must be positive")
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class CausalLSTMRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.sequence_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = _make_mlp_layers(
            self.hidden_size + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        _, (hidden, _) = self.lstm(sequence_inputs)
        representation = hidden[-1]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class _CausalConvBlock(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            int(input_channels),
            int(output_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.residual = (
            nn.Identity()
            if int(input_channels) == int(output_channels)
            else nn.Conv1d(int(input_channels), int(output_channels), kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (self.left_padding, 0))
        convolved = self.conv(padded)
        convolved = self.dropout(self.activation(convolved))
        return self.activation(convolved + self.residual(inputs))


class CausalTCNRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        channels: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        blocks: list[nn.Module] = []
        input_channels = self.sequence_input_dim
        for block_idx in range(self.num_blocks):
            blocks.append(
                _CausalConvBlock(
                    input_channels=input_channels,
                    output_channels=self.channels,
                    kernel_size=self.kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
            )
            input_channels = self.channels
        self.network = nn.Sequential(*blocks)
        self.head = _make_mlp_layers(
            self.channels + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def encode_sequence(self, sequence_inputs: torch.Tensor) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        temporal = sequence_inputs.transpose(1, 2)
        encoded = self.network(temporal)
        return encoded.transpose(1, 2)

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode_sequence(sequence_inputs)
        representation = encoded[:, -1, :]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class CausalTransformerRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        d_model: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
        max_history_size: int = 512,
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if max_history_size <= 0:
            raise ValueError("max_history_size must be positive")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dim_feedforward = int(dim_feedforward)
        self.max_history_size = int(max_history_size)
        self.input_projection = nn.Linear(self.sequence_input_dim, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_history_size, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=float(dropout),
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.head = _make_mlp_layers(
            self.d_model + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        history_size = sequence_inputs.shape[1]
        if history_size > self.max_history_size:
            raise ValueError(f"history size {history_size} exceeds max_history_size {self.max_history_size}")
        embedded = self.input_projection(sequence_inputs)
        embedded = embedded + self.position_embedding[:, :history_size, :]
        mask = torch.triu(
            torch.ones(history_size, history_size, dtype=torch.bool, device=sequence_inputs.device),
            diagonal=1,
        )
        encoded = self.encoder(embedded, mask=mask)
        representation = encoded[:, -1, :]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class CausalTCNGRURegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        tcn_channels: int = 128,
        tcn_num_blocks: int = 3,
        tcn_kernel_size: int = 3,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if gru_hidden_size <= 0:
            raise ValueError("gru_hidden_size must be positive")
        if gru_num_layers <= 0:
            raise ValueError("gru_num_layers must be positive")
        self.tcn = CausalTCNRegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=0,
            output_dim=output_dim,
            channels=tcn_channels,
            num_blocks=tcn_num_blocks,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )
        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.tcn_channels = int(tcn_channels)
        self.tcn_num_blocks = int(tcn_num_blocks)
        self.tcn_kernel_size = int(tcn_kernel_size)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        gru_dropout = float(dropout) if gru_num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.tcn_channels,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = _make_mlp_layers(
            self.gru_hidden_size + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.tcn.encode_sequence(sequence_inputs)
        _, hidden = self.gru(encoded)
        representation = hidden[-1]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class SubsectionGRUWrenchRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if context_input_dim <= 0 or rollout_input_dim <= 0 or output_dim <= 0:
            raise ValueError("context_input_dim, rollout_input_dim, and output_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")

        self.context_input_dim = int(context_input_dim)
        self.rollout_input_dim = int(rollout_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.context_encoder = nn.GRU(
            input_size=self.context_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.rollout_decoder = nn.GRU(
            input_size=self.rollout_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = _make_mlp_layers(
            self.hidden_size + self.current_input_dim,
            self.output_dim,
            head_hidden_sizes,
            dropout=dropout,
        )

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_inputs.ndim != 3 or context_inputs.shape[2] != self.context_input_dim:
            raise ValueError(f"Expected context_inputs shape [batch, history, {self.context_input_dim}]")
        if rollout_inputs.ndim != 3 or rollout_inputs.shape[2] != self.rollout_input_dim:
            raise ValueError(f"Expected rollout_inputs shape [batch, rollout, {self.rollout_input_dim}]")
        _, hidden = self.context_encoder(context_inputs)
        decoded, _ = self.rollout_decoder(rollout_inputs, hidden)
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 3 or current_inputs.shape[:2] != rollout_inputs.shape[:2]:
                raise ValueError("current_inputs must have shape [batch, rollout, current_features]")
            if current_inputs.shape[2] != self.current_input_dim:
                raise ValueError(f"Expected current feature dim {self.current_input_dim}")
            decoded = torch.cat([decoded, current_inputs], dim=2)
        batch_size, rollout_size, feature_dim = decoded.shape
        flat = decoded.reshape(batch_size * rollout_size, feature_dim)
        return self.head(flat).reshape(batch_size, rollout_size, self.output_dim)


class DiscreteSUBNETWrenchRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        latent_size: int = 16,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        if context_input_dim <= 0 or rollout_input_dim <= 0 or output_dim <= 0:
            raise ValueError("context_input_dim, rollout_input_dim, and output_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        self.context_input_dim = int(context_input_dim)
        self.rollout_input_dim = int(rollout_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.latent_size = int(latent_size)
        encoder_hidden = int(hidden_sizes[0])
        self.context_encoder = nn.GRU(
            input_size=self.context_input_dim,
            hidden_size=encoder_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.encoder_to_latent = nn.Linear(encoder_hidden, self.latent_size)
        self.transition_net = _make_mlp_layers(
            self.latent_size + self.rollout_input_dim,
            self.latent_size,
            hidden_sizes,
            dropout=dropout,
        )
        self.output_net = _make_mlp_layers(
            self.latent_size + self.rollout_input_dim + self.current_input_dim,
            self.output_dim,
            hidden_sizes,
            dropout=dropout,
        )
        self.last_latent_rms = 0.0
        self.last_delta_latent_rms = 0.0

    def _initial_latent(self, context_inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.context_encoder(context_inputs)
        return self.encoder_to_latent(hidden[-1])

    def _step_delta(self, latent: torch.Tensor, rollout_input: torch.Tensor) -> torch.Tensor:
        return self.transition_net(torch.cat([latent, rollout_input], dim=1))

    def _output(
        self,
        latent: torch.Tensor,
        rollout_input: torch.Tensor,
        current_input: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [latent, rollout_input]
        if self.current_input_dim > 0:
            if current_input is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            parts.append(current_input)
        return self.output_net(torch.cat(parts, dim=1))

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_inputs.ndim != 3 or context_inputs.shape[2] != self.context_input_dim:
            raise ValueError(f"Expected context_inputs shape [batch, history, {self.context_input_dim}]")
        if rollout_inputs.ndim != 3 or rollout_inputs.shape[2] != self.rollout_input_dim:
            raise ValueError(f"Expected rollout_inputs shape [batch, rollout, {self.rollout_input_dim}]")
        if self.current_input_dim > 0:
            if current_inputs is None or current_inputs.ndim != 3 or current_inputs.shape[:2] != rollout_inputs.shape[:2]:
                raise ValueError("current_inputs must have shape [batch, rollout, current_features]")
            if current_inputs.shape[2] != self.current_input_dim:
                raise ValueError(f"Expected current feature dim {self.current_input_dim}")

        latent = self._initial_latent(context_inputs)
        outputs: list[torch.Tensor] = []
        latent_values: list[torch.Tensor] = []
        delta_values: list[torch.Tensor] = []
        for step in range(rollout_inputs.shape[1]):
            rollout_step = rollout_inputs[:, step, :]
            current_step = current_inputs[:, step, :] if current_inputs is not None and self.current_input_dim > 0 else None
            outputs.append(self._output(latent, rollout_step, current_step))
            delta = self._step_delta(latent, rollout_step)
            latent_values.append(latent)
            delta_values.append(delta)
            latent = latent + delta
        with torch.no_grad():
            self.last_latent_rms = float(torch.sqrt(torch.mean(torch.square(torch.stack(latent_values)))) .detach().cpu())
            self.last_delta_latent_rms = float(torch.sqrt(torch.mean(torch.square(torch.stack(delta_values)))) .detach().cpu())
        return torch.stack(outputs, dim=1)


class ContinuousTimeSUBNETWrenchRegressor(DiscreteSUBNETWrenchRegressor):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        latent_size: int = 16,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
        dt_over_tau: float = 0.03,
        integrator: str = "euler",
    ):
        if dt_over_tau <= 0.0 or not math.isfinite(float(dt_over_tau)):
            raise ValueError("dt_over_tau must be positive and finite")
        if integrator != "euler":
            raise ValueError("Only euler integrator is currently supported")
        super().__init__(
            context_input_dim=context_input_dim,
            rollout_input_dim=rollout_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.dt_over_tau = float(dt_over_tau)
        self.integrator = integrator
        self.last_latent_derivative_rms = 0.0

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_inputs.ndim != 3 or context_inputs.shape[2] != self.context_input_dim:
            raise ValueError(f"Expected context_inputs shape [batch, history, {self.context_input_dim}]")
        if rollout_inputs.ndim != 3 or rollout_inputs.shape[2] != self.rollout_input_dim:
            raise ValueError(f"Expected rollout_inputs shape [batch, rollout, {self.rollout_input_dim}]")
        if self.current_input_dim > 0:
            if current_inputs is None or current_inputs.ndim != 3 or current_inputs.shape[:2] != rollout_inputs.shape[:2]:
                raise ValueError("current_inputs must have shape [batch, rollout, current_features]")
            if current_inputs.shape[2] != self.current_input_dim:
                raise ValueError(f"Expected current feature dim {self.current_input_dim}")

        latent = self._initial_latent(context_inputs)
        outputs: list[torch.Tensor] = []
        latent_values: list[torch.Tensor] = []
        derivative_values: list[torch.Tensor] = []
        for step in range(rollout_inputs.shape[1]):
            rollout_step = rollout_inputs[:, step, :]
            current_step = current_inputs[:, step, :] if current_inputs is not None and self.current_input_dim > 0 else None
            outputs.append(self._output(latent, rollout_step, current_step))
            derivative = self._step_delta(latent, rollout_step)
            latent_values.append(latent)
            derivative_values.append(derivative)
            latent = latent + self.dt_over_tau * derivative
        with torch.no_grad():
            self.last_latent_rms = float(torch.sqrt(torch.mean(torch.square(torch.stack(latent_values)))) .detach().cpu())
            self.last_latent_derivative_rms = float(
                torch.sqrt(torch.mean(torch.square(torch.stack(derivative_values)))).detach().cpu()
            )
            self.last_delta_latent_rms = self.dt_over_tau * self.last_latent_derivative_rms
        return torch.stack(outputs, dim=1)


def cyclic_catmull_rom_weights(
    phase_radians: torch.Tensor,
    *,
    num_control_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_control_points < 4:
        raise ValueError("num_control_points must be at least 4 for Catmull-Rom interpolation")

    phase = torch.remainder(phase_radians, 2.0 * math.pi)
    position = phase * (float(num_control_points) / (2.0 * math.pi))
    base_index = torch.floor(position).to(torch.long)
    t = position - base_index.to(position.dtype)
    t2 = t * t
    t3 = t2 * t

    indices = torch.stack(
        [
            torch.remainder(base_index - 1, num_control_points),
            torch.remainder(base_index, num_control_points),
            torch.remainder(base_index + 1, num_control_points),
            torch.remainder(base_index + 2, num_control_points),
        ],
        dim=1,
    )
    weights = torch.stack(
        [
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ],
        dim=1,
    )
    return indices, weights


class PhaseFunctionedLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_control_points: int = 6):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if num_control_points < 4:
            raise ValueError("num_control_points must be at least 4")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_control_points = int(num_control_points)
        self.weight_control_points = nn.Parameter(torch.empty(num_control_points, output_dim, input_dim))
        self.bias_control_points = nn.Parameter(torch.empty(num_control_points, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for idx in range(self.num_control_points):
            nn.init.xavier_uniform_(self.weight_control_points[idx])
        nn.init.zeros_(self.bias_control_points)

    def forward(self, inputs: torch.Tensor, phase_radians: torch.Tensor) -> torch.Tensor:
        indices, weights = cyclic_catmull_rom_weights(
            phase_radians.to(dtype=inputs.dtype),
            num_control_points=self.num_control_points,
        )
        selected_weights = self.weight_control_points[indices]
        selected_biases = self.bias_control_points[indices]
        interpolated_weights = torch.sum(weights[:, :, None, None] * selected_weights, dim=1)
        interpolated_biases = torch.sum(weights[:, :, None] * selected_biases, dim=1)
        return torch.einsum("boi,bi->bo", interpolated_weights, inputs) + interpolated_biases


class HybridPFNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, ...] = (40, 40),
        *,
        phase_feature_index: int = 0,
        expanded_input_dim: int = 45,
        phase_node_count: int = 5,
        phase_control_points: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("HybridPFNNRegressor requires at least two hidden sizes")
        if input_dim <= 1:
            raise ValueError("input_dim must include phase plus at least one state feature")
        if phase_feature_index < 0 or phase_feature_index >= input_dim:
            raise ValueError("phase_feature_index is out of range")
        if expanded_input_dim <= 0 or phase_node_count < 0:
            raise ValueError("expanded_input_dim must be positive and phase_node_count must be non-negative")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.phase_feature_index = int(phase_feature_index)
        self.state_dim = int(input_dim - 1)
        self.expanded_input_dim = int(expanded_input_dim)
        self.phase_node_count = int(phase_node_count)
        self.phase_control_points = int(phase_control_points)

        first_hidden = int(hidden_sizes[0])
        second_hidden = int(hidden_sizes[1])
        self.input_expansion = nn.Sequential(nn.Linear(self.state_dim, self.expanded_input_dim), nn.ELU())
        self.phase_node_control_points = nn.Parameter(torch.empty(self.phase_control_points, self.phase_node_count))
        nn.init.normal_(self.phase_node_control_points, mean=0.0, std=0.02)

        first_input_dim = self.expanded_input_dim + self.phase_node_count + self.state_dim
        self.first_layer = nn.Linear(first_input_dim, first_hidden)
        self.phase_layer = PhaseFunctionedLinear(first_hidden + self.state_dim, second_hidden, self.phase_control_points)
        self.output_layer = nn.Linear(second_hidden + self.state_dim, output_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _split_phase_and_state(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = inputs[:, self.phase_feature_index]
        if self.phase_feature_index == 0:
            state = inputs[:, 1:]
        elif self.phase_feature_index == inputs.shape[1] - 1:
            state = inputs[:, :-1]
        else:
            state = torch.cat([inputs[:, : self.phase_feature_index], inputs[:, self.phase_feature_index + 1 :]], dim=1)
        return phase, state

    def _phase_nodes(self, phase_radians: torch.Tensor) -> torch.Tensor:
        if self.phase_node_count == 0:
            return phase_radians.new_empty((len(phase_radians), 0))
        indices, weights = cyclic_catmull_rom_weights(
            phase_radians,
            num_control_points=self.phase_control_points,
        )
        selected = self.phase_node_control_points[indices]
        return torch.sum(weights[:, :, None] * selected, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phase, state = self._split_phase_and_state(inputs)
        expanded = self.input_expansion(state)
        phase_nodes = self._phase_nodes(phase.to(dtype=inputs.dtype))
        hidden1_input = torch.cat([expanded, phase_nodes, state], dim=1)
        hidden1 = self.dropout(self.activation(self.first_layer(hidden1_input)))
        hidden2_input = torch.cat([hidden1, state], dim=1)
        hidden2 = self.dropout(self.activation(self.phase_layer(hidden2_input, phase)))
        output_input = torch.cat([hidden2, state], dim=1)
        return self.output_layer(output_input)


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
    for harmonic in (2, 3):
        derived[f"phase_corrected_h{harmonic}_sin"] = np.sin(float(harmonic) * phase)
        derived[f"phase_corrected_h{harmonic}_cos"] = np.cos(float(harmonic) * phase)

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

        derived["roll_rad"] = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_argument = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        derived["pitch_rad"] = np.arcsin(pitch_argument)

        rotation_body_to_ned = np.full((len(derived), 3, 3), np.nan, dtype=float)
        rotation_body_to_ned[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        rotation_body_to_ned[:, 0, 1] = 2.0 * (x * y - z * w)
        rotation_body_to_ned[:, 0, 2] = 2.0 * (x * z + y * w)
        rotation_body_to_ned[:, 1, 0] = 2.0 * (x * y + z * w)
        rotation_body_to_ned[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        rotation_body_to_ned[:, 1, 2] = 2.0 * (y * z - x * w)
        rotation_body_to_ned[:, 2, 0] = 2.0 * (x * z - y * w)
        rotation_body_to_ned[:, 2, 1] = 2.0 * (y * z + x * w)
        rotation_body_to_ned[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

        velocity_columns = [
            "vehicle_local_position.vx",
            "vehicle_local_position.vy",
            "vehicle_local_position.vz",
        ]
        if all(column in derived.columns for column in velocity_columns):
            velocity_n = derived.loc[:, velocity_columns].to_numpy(dtype=float, copy=True)
            velocity_b = np.einsum("nji,nj->ni", rotation_body_to_ned, velocity_n)
            derived["velocity_b.x"] = velocity_b[:, 0]
            derived["velocity_b.y"] = velocity_b[:, 1]
            derived["velocity_b.z"] = velocity_b[:, 2]

            wind_columns = ["wind.windspeed_north", "wind.windspeed_east"]
            if all(column in derived.columns for column in wind_columns):
                wind_n = np.zeros_like(velocity_n)
                wind_n[:, 0] = derived["wind.windspeed_north"].to_numpy(dtype=float)
                wind_n[:, 1] = derived["wind.windspeed_east"].to_numpy(dtype=float)
                relative_air_velocity_n = velocity_n - wind_n
                relative_air_velocity_b = np.einsum("nji,nj->ni", rotation_body_to_ned, relative_air_velocity_n)
                derived["relative_air_velocity_b.x"] = relative_air_velocity_b[:, 0]
                derived["relative_air_velocity_b.y"] = relative_air_velocity_b[:, 1]
                derived["relative_air_velocity_b.z"] = relative_air_velocity_b[:, 2]

                relative_air_speed = np.linalg.norm(relative_air_velocity_b, axis=1)
                valid_speed = relative_air_speed > 1e-8
                alpha_rad = np.full(len(derived), np.nan, dtype=float)
                beta_rad = np.full(len(derived), np.nan, dtype=float)
                alpha_rad[valid_speed] = np.arctan2(
                    relative_air_velocity_b[valid_speed, 2],
                    relative_air_velocity_b[valid_speed, 0],
                )
                beta_rad[valid_speed] = np.arcsin(
                    np.clip(relative_air_velocity_b[valid_speed, 1] / relative_air_speed[valid_speed], -1.0, 1.0)
                )
                derived["alpha_rad"] = alpha_rad
                derived["beta_rad"] = beta_rad

    if {"vehicle_air_data.rho", "airspeed_validated.true_airspeed_m_s"}.issubset(derived.columns):
        true_airspeed = derived["airspeed_validated.true_airspeed_m_s"].to_numpy(dtype=float)
        rho = derived["vehicle_air_data.rho"].to_numpy(dtype=float)
        derived["dynamic_pressure_pa"] = 0.5 * rho * true_airspeed * true_airspeed

    if {"servo_left_elevon", "servo_right_elevon"}.issubset(derived.columns):
        left = derived["servo_left_elevon"].to_numpy(dtype=float)
        right = derived["servo_right_elevon"].to_numpy(dtype=float)
        derived["elevator_like"] = 0.5 * (left + right)
        derived["aileron_like"] = 0.5 * (left - right)
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


def _normalized_window_mode(window_mode: str | None) -> str:
    normalized = (window_mode or "single").lower()
    if normalized not in {"single", "causal", "centered"}:
        raise ValueError(f"Unknown window_mode: {window_mode}")
    return normalized


def _window_offsets(window_mode: str, window_radius: int) -> list[int]:
    if window_radius < 0:
        raise ValueError("window_radius must be non-negative")
    resolved_mode = _normalized_window_mode(window_mode)
    if resolved_mode == "single" or window_radius == 0:
        return [0]
    if resolved_mode == "causal":
        return list(range(-window_radius, 1))
    return list(range(-window_radius, window_radius + 1))


def _window_feature_name(column: str, offset: int) -> str:
    return f"{column}@t{offset:+d}"


WINDOW_FEATURE_MODE_COLUMNS: dict[str, list[str]] = {
    "phase_actuator": [
        "phase_corrected_sin",
        "phase_corrected_cos",
        "phase_corrected_rad",
        "wing_stroke_angle_rad",
        "flap_frequency_hz",
        "cycle_flap_frequency_hz",
        "motor_cmd_0",
        "servo_left_elevon",
        "servo_right_elevon",
        "servo_rudder",
        "elevator_like",
        "aileron_like",
    ],
    "airdata": [
        "airspeed_validated.indicated_airspeed_m_s",
        "airspeed_validated.calibrated_airspeed_m_s",
        "airspeed_validated.true_airspeed_m_s",
        "airspeed_validated.calibrated_ground_minus_wind_m_s",
        "airspeed_validated.true_ground_minus_wind_m_s",
        "airspeed_validated.pitch_filtered",
        "vehicle_air_data.rho",
        "wind.windspeed_north",
        "wind.windspeed_east",
        "dynamic_pressure_pa",
    ],
}

KINEMATIC_WINDOW_EXCLUDED_COLUMNS = {
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "vehicle_local_position.heading",
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
    "gravity_b.x",
    "gravity_b.y",
    "gravity_b.z",
    "roll_rad",
    "pitch_rad",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "relative_air_velocity_b.x",
    "relative_air_velocity_b.y",
    "relative_air_velocity_b.z",
    "alpha_rad",
    "beta_rad",
}

SEQUENCE_FEATURE_MODE_COLUMNS: dict[str, list[str]] = {
    "phase_actuator": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"],
    "phase_actuator_airdata": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
    "phase_harmonic": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + PHASE_HARMONIC_FEATURE_COLUMNS,
    "phase_harmonic_actuator_airdata": (
        WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + PHASE_HARMONIC_FEATURE_COLUMNS + WINDOW_FEATURE_MODE_COLUMNS["airdata"]
    ),
    "raw_phase_actuator_airdata": [
        "phase_corrected_rad",
        "flap_frequency_hz",
        "cycle_flap_frequency_hz",
        "motor_cmd_0",
        "servo_left_elevon",
        "servo_right_elevon",
        "servo_rudder",
        "elevator_like",
        "aileron_like",
    ]
    + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
    "no_phase_actuator_airdata": [
        "flap_frequency_hz",
        "cycle_flap_frequency_hz",
        "motor_cmd_0",
        "servo_left_elevon",
        "servo_right_elevon",
        "servo_rudder",
        "elevator_like",
        "aileron_like",
    ]
    + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
}

SEQUENCE_HISTORY_DANGEROUS_COLUMNS = KINEMATIC_WINDOW_EXCLUDED_COLUMNS


def resolve_window_feature_columns(feature_columns: list[str], window_feature_mode: str = "all") -> list[str]:
    mode = (window_feature_mode or "all").lower()
    available = set(feature_columns)
    if mode == "all":
        return list(feature_columns)
    if mode == "none":
        return []
    if mode == "phase_actuator":
        return [column for column in feature_columns if column in WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"]]
    if mode == "phase_actuator_airdata":
        selected = set(WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"]) | set(WINDOW_FEATURE_MODE_COLUMNS["airdata"])
        return [column for column in feature_columns if column in selected]
    if mode == "no_kinematics":
        return [column for column in feature_columns if column not in KINEMATIC_WINDOW_EXCLUDED_COLUMNS]
    raise ValueError(f"Unknown window_feature_mode: {window_feature_mode}")


def resolve_sequence_feature_columns(feature_columns: list[str], sequence_feature_mode: str = "phase_actuator_airdata") -> list[str]:
    mode = (sequence_feature_mode or "phase_actuator_airdata").lower()
    if mode == "all":
        return list(feature_columns)
    if mode == "none":
        return []
    if mode in SEQUENCE_FEATURE_MODE_COLUMNS:
        selected = set(SEQUENCE_FEATURE_MODE_COLUMNS[mode])
        return [column for column in feature_columns if column in selected]
    raise ValueError(f"Unknown sequence_feature_mode: {sequence_feature_mode}")


def prepare_windowed_feature_target_frames(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    *,
    window_mode: str = "single",
    window_radius: int = 0,
    window_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, targets = prepare_feature_target_frames(frame, feature_columns, target_columns)
    offsets = _window_offsets(window_mode, window_radius)
    if offsets == [0]:
        return features, targets
    selected_window_features = set(window_feature_columns or list(features.columns))
    unknown_window_features = selected_window_features - set(features.columns)
    if unknown_window_features:
        raise ValueError(f"Unknown window feature columns: {sorted(unknown_window_features)}")

    group_columns = [column for column in ["log_id", "segment_id"] if column in frame.columns]
    if group_columns:
        groups = frame.groupby(group_columns, sort=False).indices.values()
    else:
        groups = [np.arange(len(frame))]

    windowed_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    for indices in groups:
        index = np.asarray(indices)
        group_features = features.iloc[index].reset_index(drop=True)
        group_targets = targets.iloc[index].reset_index(drop=True)
        if len(group_features) < len(offsets):
            continue

        shifted_columns: list[pd.DataFrame] = []
        valid = np.ones(len(group_features), dtype=bool)
        for offset in offsets:
            shifted = group_features.loc[:, [column for column in group_features.columns if column in selected_window_features]].shift(
                periods=-offset
            )
            shifted.columns = [_window_feature_name(column, offset) for column in shifted.columns]
            valid &= shifted.notna().all(axis=1).to_numpy()
            if len(shifted.columns) > 0:
                shifted_columns.append(shifted)

        current_only_columns = [column for column in group_features.columns if column not in selected_window_features]
        if current_only_columns:
            current = group_features.loc[:, current_only_columns].copy()
            current.columns = [_window_feature_name(column, 0) for column in current.columns]
            shifted_columns.append(current)

        if not np.any(valid):
            continue
        windowed_parts.append(pd.concat(shifted_columns, axis=1).loc[valid].reset_index(drop=True))
        target_parts.append(group_targets.loc[valid].reset_index(drop=True))

    if not windowed_parts:
        raise ValueError("No complete windowed samples were produced")
    return pd.concat(windowed_parts, ignore_index=True), pd.concat(target_parts, ignore_index=True)


def prepare_causal_sequence_feature_target_frames(
    frame: pd.DataFrame,
    sequence_feature_columns: list[str],
    current_feature_columns: list[str],
    target_columns: list[str],
    *,
    history_size: int,
    group_columns: tuple[str, ...] = ("log_id", "segment_id"),
    sort_column: str = "time_s",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if history_size < 1:
        raise ValueError("history_size must be at least 1")

    derived = _with_derived_columns(frame)
    required_columns = list(sequence_feature_columns) + list(current_feature_columns) + list(target_columns)
    missing_columns = [column for column in required_columns if column not in derived.columns]
    if missing_columns:
        raise ValueError(f"Missing required sequence training columns: {missing_columns}")

    available_group_columns = [column for column in group_columns if column in derived.columns]
    metadata_columns = [column for column in [*available_group_columns, sort_column] if column in derived.columns]
    if available_group_columns:
        groups = derived.groupby(available_group_columns, sort=False, dropna=False).indices.values()
    else:
        groups = [np.arange(len(derived))]

    sequence_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    target_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []

    for indices in groups:
        group = derived.iloc[np.asarray(indices)].copy()
        if sort_column in group.columns:
            group = group.sort_values(sort_column, kind="mergesort")
        group = group.reset_index(drop=True)
        if len(group) < history_size:
            continue

        group_sequence = group.loc[:, sequence_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_current = group.loc[:, current_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_targets = group.loc[:, target_columns].copy()
        group_metadata = group.loc[:, metadata_columns].copy()
        end_indices = np.arange(history_size - 1, len(group), dtype=np.int64)

        group_windows = np.lib.stride_tricks.sliding_window_view(group_sequence, history_size, axis=0)
        sequence_parts.append(np.swapaxes(group_windows, 1, 2))
        if current_feature_columns:
            current_parts.append(group_current[end_indices])
        target_parts.append(group_targets.iloc[end_indices].reset_index(drop=True))
        metadata_parts.append(group_metadata.iloc[end_indices].reset_index(drop=True))

    if not sequence_parts:
        raise ValueError("No complete causal sequence samples were produced")

    sequence_features = np.concatenate(sequence_parts, axis=0).astype(np.float32, copy=False)
    if current_feature_columns:
        current_features = np.concatenate(current_parts, axis=0).astype(np.float32, copy=False)
    else:
        current_features = np.empty((len(sequence_features), 0), dtype=np.float32)
    targets = pd.concat(target_parts, ignore_index=True)
    metadata = pd.concat(metadata_parts, ignore_index=True) if metadata_parts else pd.DataFrame(index=range(len(sequence_features)))
    return sequence_features, current_features, targets, metadata


def prepare_causal_rollout_feature_target_frames(
    frame: pd.DataFrame,
    context_feature_columns: list[str],
    rollout_feature_columns: list[str],
    current_feature_columns: list[str],
    target_columns: list[str],
    *,
    history_size: int,
    rollout_size: int,
    rollout_stride: int | None = None,
    group_columns: tuple[str, ...] = ("log_id", "segment_id"),
    sort_column: str = "time_s",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if history_size < 1:
        raise ValueError("history_size must be at least 1")
    if rollout_size < 1:
        raise ValueError("rollout_size must be at least 1")
    resolved_stride = rollout_size if rollout_stride is None else int(rollout_stride)
    if resolved_stride < 1:
        raise ValueError("rollout_stride must be at least 1")

    derived = _with_derived_columns(frame)
    required_columns = (
        list(context_feature_columns)
        + list(rollout_feature_columns)
        + list(current_feature_columns)
        + list(target_columns)
    )
    missing_columns = [column for column in required_columns if column not in derived.columns]
    if missing_columns:
        raise ValueError(f"Missing required rollout training columns: {missing_columns}")

    available_group_columns = [column for column in group_columns if column in derived.columns]
    metadata_columns = [column for column in [*available_group_columns, sort_column] if column in derived.columns]
    if available_group_columns:
        groups = derived.groupby(available_group_columns, sort=False, dropna=False).indices.values()
    else:
        groups = [np.arange(len(derived))]

    context_parts: list[np.ndarray] = []
    rollout_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []

    for indices in groups:
        group = derived.iloc[np.asarray(indices)].copy()
        if sort_column in group.columns:
            group = group.sort_values(sort_column, kind="mergesort")
        group = group.reset_index(drop=True)
        if len(group) < history_size + rollout_size:
            continue

        starts = np.arange(history_size, len(group) - rollout_size + 1, resolved_stride, dtype=np.int64)
        if len(starts) == 0:
            continue
        context_indices = starts[:, None] - history_size + np.arange(history_size, dtype=np.int64)[None, :]
        rollout_indices = starts[:, None] + np.arange(rollout_size, dtype=np.int64)[None, :]

        group_context = group.loc[:, context_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_rollout = group.loc[:, rollout_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_targets = group.loc[:, target_columns].to_numpy(dtype=np.float32, copy=True)

        context_parts.append(group_context[context_indices])
        rollout_parts.append(group_rollout[rollout_indices])
        target_parts.append(group_targets[rollout_indices])
        if current_feature_columns:
            group_current = group.loc[:, current_feature_columns].to_numpy(dtype=np.float32, copy=True)
            current_parts.append(group_current[rollout_indices])
        else:
            current_parts.append(np.empty((len(starts), rollout_size, 0), dtype=np.float32))
        if metadata_columns:
            metadata_parts.append(group.loc[:, metadata_columns].iloc[rollout_indices.reshape(-1)].reset_index(drop=True))

    if not context_parts:
        raise ValueError("No complete causal rollout subsections were produced")

    context_features = np.concatenate(context_parts, axis=0).astype(np.float32, copy=False)
    rollout_features = np.concatenate(rollout_parts, axis=0).astype(np.float32, copy=False)
    current_features = np.concatenate(current_parts, axis=0).astype(np.float32, copy=False)
    target_sequences = np.concatenate(target_parts, axis=0).astype(np.float32, copy=False)
    metadata = (
        pd.concat(metadata_parts, ignore_index=True)
        if metadata_parts
        else pd.DataFrame(index=range(context_features.shape[0] * rollout_size))
    )
    return context_features, rollout_features, current_features, target_sequences, metadata


def _fit_feature_stats(
    features: np.ndarray,
    *,
    raw_feature_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(features), features, medians)
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    if raw_feature_indices:
        means[raw_feature_indices] = 0.0
        stds[raw_feature_indices] = 1.0
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
    for key in [
        "feature_medians",
        "feature_means",
        "feature_stds",
        "sequence_feature_medians",
        "sequence_feature_means",
        "sequence_feature_stds",
        "context_feature_medians",
        "context_feature_means",
        "context_feature_stds",
        "rollout_feature_medians",
        "rollout_feature_means",
        "rollout_feature_stds",
        "current_feature_medians",
        "current_feature_means",
        "current_feature_stds",
        "target_means",
        "target_stds",
        "target_loss_weights",
    ]:
        if key in bundle:
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


def _make_sequence_loader(
    sequence_features: np.ndarray,
    current_features: np.ndarray,
    targets: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    sequence_tensor = torch.from_numpy(sequence_features.astype(np.float32, copy=False))
    current_tensor = torch.from_numpy(current_features.astype(np.float32, copy=False))
    if targets is None:
        dataset = TensorDataset(sequence_tensor, current_tensor)
    else:
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        dataset = TensorDataset(sequence_tensor, current_tensor, target_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _make_rollout_loader(
    context_features: np.ndarray,
    rollout_features: np.ndarray,
    current_features: np.ndarray,
    targets: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    context_tensor = torch.from_numpy(context_features.astype(np.float32, copy=False))
    rollout_tensor = torch.from_numpy(rollout_features.astype(np.float32, copy=False))
    current_tensor = torch.from_numpy(current_features.astype(np.float32, copy=False))
    if targets is None:
        dataset = TensorDataset(context_tensor, rollout_tensor, current_tensor)
    else:
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        dataset = TensorDataset(context_tensor, rollout_tensor, current_tensor, target_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def resolve_target_loss_weights(
    target_columns: list[str],
    target_loss_weights: str | dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    if target_loss_weights is None or target_loss_weights == "":
        return np.ones(len(target_columns), dtype=np.float32)

    if isinstance(target_loss_weights, str):
        parsed: dict[str, float] = {}
        for item in target_loss_weights.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Invalid target loss weight item: {item!r}")
            target_name, raw_weight = item.split("=", 1)
            parsed[target_name.strip()] = float(raw_weight.strip())
        missing = [target for target in target_columns if target not in parsed]
        unknown = [target for target in parsed if target not in target_columns]
        if missing or unknown:
            raise ValueError(f"Target loss weights mismatch; missing={missing}, unknown={unknown}")
        weights = np.array([parsed[target] for target in target_columns], dtype=np.float32)
    elif isinstance(target_loss_weights, dict):
        missing = [target for target in target_columns if target not in target_loss_weights]
        unknown = [target for target in target_loss_weights if target not in target_columns]
        if missing or unknown:
            raise ValueError(f"Target loss weights mismatch; missing={missing}, unknown={unknown}")
        weights = np.array([target_loss_weights[target] for target in target_columns], dtype=np.float32)
    else:
        weights = np.asarray(target_loss_weights, dtype=np.float32)
        if weights.shape != (len(target_columns),):
            raise ValueError(f"target_loss_weights must have shape ({len(target_columns)},), got {weights.shape}")

    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("target_loss_weights must be finite and non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("At least one target loss weight must be positive")
    return weights.astype(np.float32, copy=False)


def _target_loss_weights_as_dict(target_columns: list[str], weights: np.ndarray) -> dict[str, float]:
    return {target: float(weight) for target, weight in zip(target_columns, weights)}


def _normalized_loss_type(loss_type: str | None) -> str:
    normalized = (loss_type or "mse").lower()
    if normalized not in {"mse", "huber"}:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return normalized


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_loss_weights: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    resolved_loss_type = _normalized_loss_type(loss_type)
    if huber_delta <= 0.0 or not math.isfinite(float(huber_delta)):
        raise ValueError("huber_delta must be positive and finite")

    error = predictions - targets
    if resolved_loss_type == "mse":
        per_target_loss = torch.square(error)
    else:
        abs_error = torch.abs(error)
        delta = torch.as_tensor(huber_delta, device=predictions.device, dtype=predictions.dtype)
        quadratic = torch.minimum(abs_error, delta)
        linear = abs_error - quadratic
        per_target_loss = 0.5 * torch.square(quadratic) + delta * linear

    weights = target_loss_weights.to(device=predictions.device, dtype=predictions.dtype)
    weighted = per_target_loss * weights
    return weighted.sum() / (predictions.shape[0] * torch.clamp(weights.sum(), min=1e-8))


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


def _predict_sequence_scaled_batches(
    model: nn.Module,
    sequence_features_scaled: np.ndarray,
    current_features_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    loader = _make_sequence_loader(
        sequence_features_scaled,
        current_features_scaled,
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
        for batch_sequence, batch_current in loader:
            batch_sequence = batch_sequence.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[1] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_sequence, current_arg)
            outputs.append(batch_predictions.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _predict_rollout_scaled_batches(
    model: nn.Module,
    context_features_scaled: np.ndarray,
    rollout_features_scaled: np.ndarray,
    current_features_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    loader = _make_rollout_loader(
        context_features_scaled,
        rollout_features_scaled,
        current_features_scaled,
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
        for batch_context, batch_rollout, batch_current in loader:
            batch_context = batch_context.to(device, non_blocking=True)
            batch_rollout = batch_rollout.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[2] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_context, batch_rollout, current_arg)
            outputs.append(batch_predictions.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _history_frame(history: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(history)


def _normalized_lr_scheduler(name: str | None) -> str:
    resolved = (name or "none").strip().lower()
    aliases = {"constant": "none", "off": "none", "disabled": "none"}
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"none", "warmup_cosine"}:
        raise ValueError("lr_scheduler must be one of: none, warmup_cosine")
    return resolved


def _warmup_cosine_lr_lambda(*, warmup_steps: int, total_steps: int):
    resolved_total_steps = max(int(total_steps), 1)
    resolved_warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step_index: int) -> float:
        step = int(step_index) + 1
        if resolved_warmup_steps > 0 and step <= resolved_warmup_steps:
            return max(float(step) / float(resolved_warmup_steps), 1e-8)
        decay_steps = max(resolved_total_steps - resolved_warmup_steps, 1)
        progress = min(max((step - resolved_warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _ema_update_state(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if not torch.is_floating_point(value):
                ema_state[name] = value.detach().clone()
                continue
            ema_state[name].mul_(float(decay)).add_(value.detach(), alpha=1.0 - float(decay))


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
    targets_df = _targets_for_bundle(bundle, frame)
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
    targets_df = _targets_for_bundle(bundle, frame)
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
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
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
                loss = regression_loss(
                    predictions,
                    batch_targets,
                    target_loss_weights=target_loss_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
            batch_size = len(batch_features)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def _evaluate_sequence_scaled_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_sequence, batch_current, batch_targets in loader:
            batch_sequence = batch_sequence.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[1] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_sequence, current_arg)
                loss = regression_loss(
                    predictions,
                    batch_targets,
                    target_loss_weights=target_loss_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
            batch_size = len(batch_sequence)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def _evaluate_rollout_scaled_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_context, batch_rollout, batch_current, batch_targets in loader:
            batch_context = batch_context.to(device, non_blocking=True)
            batch_rollout = batch_rollout.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[2] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_context, batch_rollout, current_arg)
                loss = regression_loss(
                    predictions.reshape(-1, predictions.shape[-1]),
                    batch_targets.reshape(-1, batch_targets.shape[-1]),
                    target_loss_weights=target_loss_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
            batch_size = len(batch_context)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def _normalized_model_type(model_type: str | None) -> str:
    normalized = (model_type or "mlp").lower()
    if normalized not in {
        "mlp",
        "pfnn",
        "causal_gru",
        "causal_gru_asl",
        "causal_lstm",
        "causal_tcn",
        "causal_transformer",
        "causal_tcn_gru",
        "subsection_gru",
        "subnet_discrete",
        "ct_subnet_euler",
    }:
        raise ValueError(f"Unknown model_type: {model_type}")
    return normalized


def _is_sequence_model_type(model_type: str | None) -> bool:
    return _normalized_model_type(model_type) in {
        "causal_gru",
        "causal_gru_asl",
        "causal_lstm",
        "causal_tcn",
        "causal_transformer",
        "causal_tcn_gru",
    }


def _is_rollout_model_type(model_type: str | None) -> bool:
    return _normalized_model_type(model_type) in {"subsection_gru", "subnet_discrete", "ct_subnet_euler"}


def resolve_current_feature_columns(
    base_feature_columns: list[str],
    sequence_feature_columns: list[str],
    current_feature_mode: str = "remaining_current",
) -> list[str]:
    mode = (current_feature_mode or "remaining_current").lower()
    if mode == "remaining_current":
        sequence_set = set(sequence_feature_columns)
        return [column for column in base_feature_columns if column not in sequence_set]
    if mode == "all":
        return list(base_feature_columns)
    if mode == "none":
        return []
    raise ValueError(f"Unknown current_feature_mode: {current_feature_mode}")


ACCELERATION_INPUT_COLUMNS = set(NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS)
VELOCITY_HISTORY_COLUMNS = {
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "relative_air_velocity_b.x",
    "relative_air_velocity_b.y",
    "relative_air_velocity_b.z",
}
ANGULAR_VELOCITY_HISTORY_COLUMNS = {
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
}
ALPHA_BETA_HISTORY_COLUMNS = {"alpha_rad", "beta_rad"}


def _training_audit_flags(bundle: dict[str, Any], *, split_root: str | Path | None = None) -> dict[str, bool]:
    model_type = bundle.get("model_type", "mlp")
    if _is_sequence_model_type(model_type):
        history_columns = set(bundle.get("sequence_feature_columns", []))
        input_columns = history_columns | set(bundle.get("current_feature_columns", []))
        has_centered_window = False
    elif _is_rollout_model_type(model_type):
        history_columns = set(bundle.get("context_feature_columns", []))
        input_columns = (
            history_columns
            | set(bundle.get("rollout_feature_columns", []))
            | set(bundle.get("current_feature_columns", []))
        )
        has_centered_window = False
    else:
        history_columns = set(bundle.get("window_feature_columns", []))
        input_columns = set(bundle.get("base_feature_columns", bundle.get("feature_columns", [])))
        has_centered_window = bundle.get("window_mode") == "centered"

    uses_whole_log_split = False
    if split_root is not None:
        manifest_path = Path(split_root) / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                uses_whole_log_split = manifest.get("split_policy") == "whole_log"
            except json.JSONDecodeError:
                uses_whole_log_split = False

    return {
        "has_acceleration_inputs": bool(input_columns & ACCELERATION_INPUT_COLUMNS),
        "has_velocity_history": bool(history_columns & VELOCITY_HISTORY_COLUMNS),
        "has_angular_velocity_history": bool(history_columns & ANGULAR_VELOCITY_HISTORY_COLUMNS),
        "has_alpha_beta_history": bool(history_columns & ALPHA_BETA_HISTORY_COLUMNS),
        "has_centered_window": bool(has_centered_window),
        "uses_whole_log_split": bool(uses_whole_log_split),
    }


def _phase_feature_index_for_model(model_type: str, feature_columns: list[str]) -> int | None:
    if model_type != "pfnn":
        return None
    phase_column = "phase_corrected_rad"
    if phase_column not in feature_columns:
        raise ValueError(f"PFNN model_type requires feature column: {phase_column}")
    return feature_columns.index(phase_column)


def _build_regressor(
    *,
    model_type: str,
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    phase_feature_index: int | None = None,
    pfnn_expanded_input_dim: int = 45,
    pfnn_phase_node_count: int = 5,
    pfnn_control_points: int = 6,
) -> nn.Module:
    if model_type == "mlp":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
    if phase_feature_index is None:
        raise ValueError("PFNN requires phase_feature_index")
    return HybridPFNNRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        phase_feature_index=phase_feature_index,
        expanded_input_dim=pfnn_expanded_input_dim,
        phase_node_count=pfnn_phase_node_count,
        phase_control_points=pfnn_control_points,
        dropout=dropout,
    )


def _fit_sequence_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    return _fit_feature_stats(flat)


def _transform_sequence_features(
    features: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    transformed = _transform_features(flat, medians, means, stds)
    return transformed.reshape(features.shape).astype(np.float32, copy=False)


def _fit_rollout_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    return _fit_feature_stats(flat)


def _transform_rollout_features(
    features: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    transformed = _transform_features(flat, medians, means, stds)
    return transformed.reshape(features.shape).astype(np.float32, copy=False)


def _build_sequence_regressor(
    *,
    model_type: str,
    sequence_input_dim: int,
    current_input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    gru_num_layers: int,
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
) -> nn.Module:
    if not hidden_sizes:
        raise ValueError("hidden_sizes must not be empty for sequence models")
    base_hidden_size = int(hidden_sizes[0])
    head_hidden_sizes = tuple(int(v) for v in hidden_sizes[1:]) or (base_hidden_size,)

    if model_type == "causal_gru":
        return CausalGRURegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            hidden_size=base_hidden_size,
            num_layers=int(gru_num_layers),
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )
    if model_type == "causal_gru_asl":
        return CausalGRUASLRegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            gru_hidden_size=base_hidden_size,
            gru_num_layers=int(gru_num_layers),
            asl_hidden_size=int(asl_hidden_size),
            asl_dropout=float(asl_dropout),
            asl_max_frequency_bins=asl_max_frequency_bins,
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )
    if model_type == "causal_lstm":
        return CausalLSTMRegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            hidden_size=base_hidden_size,
            num_layers=int(gru_num_layers),
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )
    if model_type == "causal_tcn":
        return CausalTCNRegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            channels=int(tcn_channels),
            num_blocks=int(tcn_num_blocks),
            kernel_size=int(tcn_kernel_size),
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )
    if model_type == "causal_transformer":
        transformer_head_sizes = tuple(int(v) for v in hidden_sizes[1:]) or (int(transformer_d_model),)
        return CausalTransformerRegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            d_model=int(transformer_d_model),
            num_layers=int(transformer_num_layers),
            num_heads=int(transformer_num_heads),
            dim_feedforward=int(transformer_dim_feedforward),
            dropout=dropout,
            head_hidden_sizes=transformer_head_sizes,
        )
    if model_type == "causal_tcn_gru":
        return CausalTCNGRURegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            tcn_channels=int(tcn_channels),
            tcn_num_blocks=int(tcn_num_blocks),
            tcn_kernel_size=int(tcn_kernel_size),
            gru_hidden_size=base_hidden_size,
            gru_num_layers=int(gru_num_layers),
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )
    raise ValueError(f"Unknown sequence model_type: {model_type}")


def fit_torch_sequence_regressor(
    *,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    model_type: str = "causal_gru",
    hidden_sizes: tuple[int, ...] = (128,),
    dropout: float = 0.0,
    batch_size: int = 512,
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
    sequence_history_size: int = 64,
    sequence_feature_mode: str | None = None,
    current_feature_mode: str = "remaining_current",
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
    lr_scheduler: str | None = None,
    lr_warmup_ratio: float = 0.0,
    gradient_clip_norm: float | None = None,
    ema_decay: float = 0.0,
) -> dict[str, Any]:
    _set_random_seed(random_seed)
    resolved_model_type = _normalized_model_type(model_type)
    if not _is_sequence_model_type(resolved_model_type):
        raise ValueError(f"Sequence training requires a sequence model_type, got {model_type}")
    resolved_loss_type = _normalized_loss_type(loss_type)
    if huber_delta <= 0.0 or not math.isfinite(float(huber_delta)):
        raise ValueError("huber_delta must be positive and finite")
    if sequence_history_size < 1:
        raise ValueError("sequence_history_size must be at least 1")
    if not hidden_sizes:
        raise ValueError("hidden_sizes must not be empty for sequence models")
    resolved_lr_scheduler = _normalized_lr_scheduler(lr_scheduler)
    if not 0.0 <= float(lr_warmup_ratio) < 1.0:
        raise ValueError("lr_warmup_ratio must be in [0, 1)")
    resolved_gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
    if resolved_gradient_clip_norm is not None and resolved_gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided")
    resolved_ema_decay = float(ema_decay)
    if not 0.0 <= resolved_ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")

    resolved_device = _resolve_device(device)
    pin_memory = resolved_device.type == "cuda"
    base_feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    resolved_target_columns = target_columns or DEFAULT_TARGET_COLUMNS
    sequence_feature_columns = resolve_sequence_feature_columns(list(base_feature_columns), sequence_feature_mode)
    if not sequence_feature_columns:
        raise ValueError("Sequence models require at least one sequence feature")
    current_feature_columns = resolve_current_feature_columns(
        list(base_feature_columns),
        sequence_feature_columns,
        current_feature_mode,
    )

    train_sequence, train_current, train_targets_df, _ = prepare_causal_sequence_feature_target_frames(
        train_frame,
        sequence_feature_columns,
        current_feature_columns,
        resolved_target_columns,
        history_size=sequence_history_size,
    )
    val_sequence, val_current, val_targets_df, _ = prepare_causal_sequence_feature_target_frames(
        val_frame,
        sequence_feature_columns,
        current_feature_columns,
        resolved_target_columns,
        history_size=sequence_history_size,
    )

    train_targets = train_targets_df.to_numpy(dtype=np.float32, copy=True)
    val_targets = val_targets_df.to_numpy(dtype=np.float32, copy=True)
    sequence_feature_medians, sequence_feature_means, sequence_feature_stds = _fit_sequence_feature_stats(train_sequence)
    if current_feature_columns:
        current_feature_medians, current_feature_means, current_feature_stds = _fit_feature_stats(train_current)
        train_current_scaled = _transform_features(train_current, current_feature_medians, current_feature_means, current_feature_stds)
        val_current_scaled = _transform_features(val_current, current_feature_medians, current_feature_means, current_feature_stds)
    else:
        current_feature_medians = np.empty((0,), dtype=np.float32)
        current_feature_means = np.empty((0,), dtype=np.float32)
        current_feature_stds = np.empty((0,), dtype=np.float32)
        train_current_scaled = train_current
        val_current_scaled = val_current

    target_means, target_stds = _fit_target_stats(train_targets)
    train_sequence_scaled = _transform_sequence_features(
        train_sequence,
        sequence_feature_medians,
        sequence_feature_means,
        sequence_feature_stds,
    )
    val_sequence_scaled = _transform_sequence_features(
        val_sequence,
        sequence_feature_medians,
        sequence_feature_means,
        sequence_feature_stds,
    )
    train_targets_scaled = _transform_targets(train_targets, target_means, target_stds)
    val_targets_scaled = _transform_targets(val_targets, target_means, target_stds)
    target_loss_weights_array = resolve_target_loss_weights(list(train_targets_df.columns), target_loss_weights)
    target_loss_weights_tensor = torch.as_tensor(target_loss_weights_array, dtype=torch.float32, device=resolved_device)

    train_loader = _make_sequence_loader(
        train_sequence_scaled,
        train_current_scaled,
        train_targets_scaled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_sequence_loader(
        val_sequence_scaled,
        val_current_scaled,
        val_targets_scaled,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = _build_sequence_regressor(
        model_type=resolved_model_type,
        sequence_input_dim=train_sequence_scaled.shape[2],
        current_input_dim=train_current_scaled.shape[1],
        output_dim=train_targets_scaled.shape[1],
        hidden_sizes=tuple(int(v) for v in hidden_sizes),
        dropout=dropout,
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
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_training_steps = max(len(train_loader) * int(max_epochs), 1)
    warmup_steps = int(round(total_training_steps * float(lr_warmup_ratio)))
    scheduler = None
    if resolved_lr_scheduler == "warmup_cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine_lr_lambda(warmup_steps=warmup_steps, total_steps=total_training_steps),
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and resolved_device.type == "cuda")

    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    amp_enabled = use_amp and resolved_device.type == "cuda"
    ema_state = None
    if resolved_ema_decay > 0.0:
        ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_sequence, batch_current, batch_targets in train_loader:
            batch_sequence = batch_sequence.to(resolved_device, non_blocking=True)
            batch_current = batch_current.to(resolved_device, non_blocking=True)
            batch_targets = batch_targets.to(resolved_device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[1] > 0 else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=resolved_device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_sequence, current_arg)
                loss = regression_loss(
                    predictions,
                    batch_targets,
                    target_loss_weights=target_loss_weights_tensor,
                    loss_type=resolved_loss_type,
                    huber_delta=huber_delta,
                )

            scaler.scale(loss).backward()
            if resolved_gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), resolved_gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if ema_state is not None:
                _ema_update_state(ema_state, model, resolved_ema_decay)

            batch_count = len(batch_sequence)
            train_loss_sum += float(loss.item()) * batch_count
            train_sample_count += batch_count

        train_loss = train_loss_sum / max(train_sample_count, 1)
        evaluation_state = None
        if ema_state is not None:
            evaluation_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_state)
        val_loss = _evaluate_sequence_scaled_loss(
            model,
            val_loader,
            resolved_device,
            use_amp=use_amp,
            target_loss_weights=target_loss_weights_tensor,
            loss_type=resolved_loss_type,
            huber_delta=huber_delta,
        )
        val_predictions_scaled = _predict_sequence_scaled_batches(
            model,
            val_sequence_scaled,
            val_current_scaled,
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
        if evaluation_state is not None:
            model.load_state_dict(evaluation_state)
        history_row: dict[str, float] = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
            best_state_dict = copy.deepcopy(ema_state if ema_state is not None else model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    model.load_state_dict(best_state_dict)

    return {
        "model_state_dict": best_state_dict,
        "model_type": resolved_model_type,
        "feature_columns": list(base_feature_columns),
        "base_feature_columns": list(base_feature_columns),
        "target_columns": list(train_targets_df.columns),
        "sequence_feature_columns": list(sequence_feature_columns),
        "current_feature_columns": list(current_feature_columns),
        "sequence_feature_medians": sequence_feature_medians,
        "sequence_feature_means": sequence_feature_means,
        "sequence_feature_stds": sequence_feature_stds,
        "current_feature_medians": current_feature_medians,
        "current_feature_means": current_feature_means,
        "current_feature_stds": current_feature_stds,
        "target_means": target_means,
        "target_stds": target_stds,
        "target_loss_weights": target_loss_weights_array,
        "target_loss_weights_by_name": _target_loss_weights_as_dict(list(train_targets_df.columns), target_loss_weights_array),
        "loss_type": resolved_loss_type,
        "huber_delta": float(huber_delta),
        "sequence_history_size": int(sequence_history_size),
        "sequence_feature_mode": sequence_feature_mode,
        "current_feature_mode": current_feature_mode,
        "sequence_group_columns": ["log_id", "segment_id"],
        "sequence_sort_column": "time_s",
        "sequence_sample_count_train": int(len(train_sequence_scaled)),
        "sequence_sample_count_val": int(len(val_sequence_scaled)),
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(dropout),
        "gru_hidden_size": int(hidden_sizes[0]),
        "gru_num_layers": int(gru_num_layers),
        "asl_hidden_size": int(asl_hidden_size),
        "asl_dropout": float(asl_dropout),
        "asl_max_frequency_bins": asl_max_frequency_bins,
        "tcn_channels": int(tcn_channels),
        "tcn_num_blocks": int(tcn_num_blocks),
        "tcn_kernel_size": int(tcn_kernel_size),
        "transformer_d_model": int(transformer_d_model),
        "transformer_num_layers": int(transformer_num_layers),
        "transformer_num_heads": int(transformer_num_heads),
        "transformer_dim_feedforward": int(transformer_dim_feedforward),
        "lr_scheduler": resolved_lr_scheduler,
        "lr_warmup_ratio": float(lr_warmup_ratio),
        "lr_warmup_steps": int(warmup_steps),
        "gradient_clip_norm": resolved_gradient_clip_norm,
        "ema_decay": float(resolved_ema_decay),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "device_type": resolved_device.type,
        "use_amp": bool(amp_enabled),
        "random_seed": int(random_seed),
    }


def _build_rollout_regressor(
    *,
    model_type: str,
    context_input_dim: int,
    rollout_input_dim: int,
    current_input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    gru_num_layers: int,
    latent_size: int,
    dt_over_tau: float,
    ct_integrator: str,
) -> nn.Module:
    if model_type == "subsection_gru":
        return SubsectionGRUWrenchRegressor(
            context_input_dim=context_input_dim,
            rollout_input_dim=rollout_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            hidden_size=int(hidden_sizes[0]),
            num_layers=int(gru_num_layers),
            dropout=dropout,
            head_hidden_sizes=tuple(int(v) for v in hidden_sizes[1:]) or (int(hidden_sizes[0]),),
        )
    if model_type == "subnet_discrete":
        return DiscreteSUBNETWrenchRegressor(
            context_input_dim=context_input_dim,
            rollout_input_dim=rollout_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            latent_size=int(latent_size),
            hidden_sizes=tuple(int(v) for v in hidden_sizes),
            dropout=dropout,
        )
    if model_type == "ct_subnet_euler":
        return ContinuousTimeSUBNETWrenchRegressor(
            context_input_dim=context_input_dim,
            rollout_input_dim=rollout_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            latent_size=int(latent_size),
            hidden_sizes=tuple(int(v) for v in hidden_sizes),
            dropout=dropout,
            dt_over_tau=float(dt_over_tau),
            integrator=ct_integrator,
        )
    raise ValueError(f"Unknown rollout model_type: {model_type}")


def fit_torch_rollout_regressor(
    *,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    model_type: str = "subsection_gru",
    hidden_sizes: tuple[int, ...] = (128, 128),
    dropout: float = 0.0,
    batch_size: int = 512,
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
    sequence_history_size: int = 64,
    rollout_size: int = 32,
    rollout_stride: int | None = None,
    sequence_feature_mode: str | None = None,
    current_feature_mode: str = "remaining_current",
    gru_num_layers: int = 1,
    latent_size: int = 16,
    dt_over_tau: float = 0.03,
    ct_integrator: str = "euler",
) -> dict[str, Any]:
    _set_random_seed(random_seed)
    resolved_model_type = _normalized_model_type(model_type)
    if not _is_rollout_model_type(resolved_model_type):
        raise ValueError(f"Rollout training requires a rollout model_type, got {model_type}")
    resolved_loss_type = _normalized_loss_type(loss_type)
    if huber_delta <= 0.0 or not math.isfinite(float(huber_delta)):
        raise ValueError("huber_delta must be positive and finite")
    if sequence_history_size < 1:
        raise ValueError("sequence_history_size must be at least 1")
    if rollout_size < 1:
        raise ValueError("rollout_size must be at least 1")
    resolved_rollout_stride = rollout_size if rollout_stride is None else int(rollout_stride)
    if resolved_rollout_stride < 1:
        raise ValueError("rollout_stride must be at least 1")
    if not hidden_sizes:
        raise ValueError("hidden_sizes must not be empty for rollout models")

    resolved_device = _resolve_device(device)
    pin_memory = resolved_device.type == "cuda"
    base_feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    resolved_target_columns = target_columns or DEFAULT_TARGET_COLUMNS
    context_feature_columns = resolve_sequence_feature_columns(list(base_feature_columns), sequence_feature_mode)
    rollout_feature_columns = list(context_feature_columns)
    if not context_feature_columns:
        raise ValueError("Rollout models require at least one context feature")
    current_feature_columns = resolve_current_feature_columns(
        list(base_feature_columns),
        context_feature_columns,
        current_feature_mode,
    )

    train_context, train_rollout, train_current, train_targets, _ = prepare_causal_rollout_feature_target_frames(
        train_frame,
        context_feature_columns,
        rollout_feature_columns,
        current_feature_columns,
        resolved_target_columns,
        history_size=sequence_history_size,
        rollout_size=rollout_size,
        rollout_stride=resolved_rollout_stride,
    )
    val_context, val_rollout, val_current, val_targets, _ = prepare_causal_rollout_feature_target_frames(
        val_frame,
        context_feature_columns,
        rollout_feature_columns,
        current_feature_columns,
        resolved_target_columns,
        history_size=sequence_history_size,
        rollout_size=rollout_size,
        rollout_stride=resolved_rollout_stride,
    )

    context_feature_medians, context_feature_means, context_feature_stds = _fit_rollout_feature_stats(train_context)
    rollout_feature_medians, rollout_feature_means, rollout_feature_stds = _fit_rollout_feature_stats(train_rollout)
    if current_feature_columns:
        current_feature_medians, current_feature_means, current_feature_stds = _fit_rollout_feature_stats(train_current)
        train_current_scaled = _transform_rollout_features(
            train_current,
            current_feature_medians,
            current_feature_means,
            current_feature_stds,
        )
        val_current_scaled = _transform_rollout_features(
            val_current,
            current_feature_medians,
            current_feature_means,
            current_feature_stds,
        )
    else:
        current_feature_medians = np.empty((0,), dtype=np.float32)
        current_feature_means = np.empty((0,), dtype=np.float32)
        current_feature_stds = np.empty((0,), dtype=np.float32)
        train_current_scaled = train_current
        val_current_scaled = val_current

    flat_train_targets = train_targets.reshape(-1, train_targets.shape[-1])
    flat_val_targets = val_targets.reshape(-1, val_targets.shape[-1])
    target_means, target_stds = _fit_target_stats(flat_train_targets)
    train_context_scaled = _transform_rollout_features(
        train_context,
        context_feature_medians,
        context_feature_means,
        context_feature_stds,
    )
    val_context_scaled = _transform_rollout_features(
        val_context,
        context_feature_medians,
        context_feature_means,
        context_feature_stds,
    )
    train_rollout_scaled = _transform_rollout_features(
        train_rollout,
        rollout_feature_medians,
        rollout_feature_means,
        rollout_feature_stds,
    )
    val_rollout_scaled = _transform_rollout_features(
        val_rollout,
        rollout_feature_medians,
        rollout_feature_means,
        rollout_feature_stds,
    )
    train_targets_scaled = _transform_targets(flat_train_targets, target_means, target_stds).reshape(train_targets.shape)
    val_targets_scaled = _transform_targets(flat_val_targets, target_means, target_stds).reshape(val_targets.shape)
    target_loss_weights_array = resolve_target_loss_weights(list(resolved_target_columns), target_loss_weights)
    target_loss_weights_tensor = torch.as_tensor(target_loss_weights_array, dtype=torch.float32, device=resolved_device)

    train_loader = _make_rollout_loader(
        train_context_scaled,
        train_rollout_scaled,
        train_current_scaled,
        train_targets_scaled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_rollout_loader(
        val_context_scaled,
        val_rollout_scaled,
        val_current_scaled,
        val_targets_scaled,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = _build_rollout_regressor(
        model_type=resolved_model_type,
        context_input_dim=train_context_scaled.shape[2],
        rollout_input_dim=train_rollout_scaled.shape[2],
        current_input_dim=train_current_scaled.shape[2],
        output_dim=train_targets_scaled.shape[2],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        gru_num_layers=gru_num_layers,
        latent_size=latent_size,
        dt_over_tau=dt_over_tau,
        ct_integrator=ct_integrator,
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

        for batch_context, batch_rollout, batch_current, batch_targets in train_loader:
            batch_context = batch_context.to(resolved_device, non_blocking=True)
            batch_rollout = batch_rollout.to(resolved_device, non_blocking=True)
            batch_current = batch_current.to(resolved_device, non_blocking=True)
            batch_targets = batch_targets.to(resolved_device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[2] > 0 else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=resolved_device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_context, batch_rollout, current_arg)
                loss = regression_loss(
                    predictions.reshape(-1, predictions.shape[-1]),
                    batch_targets.reshape(-1, batch_targets.shape[-1]),
                    target_loss_weights=target_loss_weights_tensor,
                    loss_type=resolved_loss_type,
                    huber_delta=huber_delta,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = len(batch_context)
            train_loss_sum += float(loss.item()) * batch_count
            train_sample_count += batch_count

        train_loss = train_loss_sum / max(train_sample_count, 1)
        val_loss = _evaluate_rollout_scaled_loss(
            model,
            val_loader,
            resolved_device,
            use_amp=use_amp,
            target_loss_weights=target_loss_weights_tensor,
            loss_type=resolved_loss_type,
            huber_delta=huber_delta,
        )
        val_predictions_scaled = _predict_rollout_scaled_batches(
            model,
            val_context_scaled,
            val_rollout_scaled,
            val_current_scaled,
            batch_size=batch_size,
            device=resolved_device,
            use_amp=use_amp,
        )
        val_predictions = _inverse_transform_targets(
            val_predictions_scaled.reshape(-1, val_predictions_scaled.shape[-1]),
            target_means,
            target_stds,
        )
        val_metrics = _metrics_from_arrays(
            val_targets.reshape(-1, val_targets.shape[-1]).astype(np.float32, copy=False),
            val_predictions.astype(np.float32, copy=False),
            target_columns=list(resolved_target_columns),
            split_name="val",
        )
        history_row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_overall_mae": float(val_metrics["overall_mae"]),
            "val_overall_rmse": float(val_metrics["overall_rmse"]),
            "val_overall_r2": float(val_metrics["overall_r2"]),
            "latent_rms": float(getattr(model, "last_latent_rms", 0.0)),
            "delta_latent_rms": float(getattr(model, "last_delta_latent_rms", 0.0)),
            "latent_derivative_rms": float(getattr(model, "last_latent_derivative_rms", 0.0)),
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
        "model_type": resolved_model_type,
        "feature_columns": list(base_feature_columns),
        "base_feature_columns": list(base_feature_columns),
        "target_columns": list(resolved_target_columns),
        "context_feature_columns": list(context_feature_columns),
        "rollout_feature_columns": list(rollout_feature_columns),
        "current_feature_columns": list(current_feature_columns),
        "context_feature_medians": context_feature_medians,
        "context_feature_means": context_feature_means,
        "context_feature_stds": context_feature_stds,
        "rollout_feature_medians": rollout_feature_medians,
        "rollout_feature_means": rollout_feature_means,
        "rollout_feature_stds": rollout_feature_stds,
        "current_feature_medians": current_feature_medians,
        "current_feature_means": current_feature_means,
        "current_feature_stds": current_feature_stds,
        "target_means": target_means,
        "target_stds": target_stds,
        "target_loss_weights": target_loss_weights_array,
        "target_loss_weights_by_name": _target_loss_weights_as_dict(list(resolved_target_columns), target_loss_weights_array),
        "loss_type": resolved_loss_type,
        "huber_delta": float(huber_delta),
        "sequence_history_size": int(sequence_history_size),
        "rollout_size": int(rollout_size),
        "rollout_stride": int(resolved_rollout_stride),
        "sequence_feature_mode": sequence_feature_mode,
        "current_feature_mode": current_feature_mode,
        "sequence_group_columns": ["log_id", "segment_id"],
        "sequence_sort_column": "time_s",
        "rollout_sample_count_train": int(len(train_context_scaled)),
        "rollout_timestep_count_train": int(train_targets.shape[0] * train_targets.shape[1]),
        "rollout_sample_count_val": int(len(val_context_scaled)),
        "rollout_timestep_count_val": int(val_targets.shape[0] * val_targets.shape[1]),
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(dropout),
        "gru_hidden_size": int(hidden_sizes[0]),
        "gru_num_layers": int(gru_num_layers),
        "latent_size": int(latent_size),
        "dt_over_tau": float(dt_over_tau),
        "ct_integrator": ct_integrator,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "device_type": resolved_device.type,
        "use_amp": bool(amp_enabled),
        "random_seed": int(random_seed),
    }


def fit_torch_regressor(
    *,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
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
    pfnn_expanded_input_dim: int = 45,
    pfnn_phase_node_count: int = 5,
    pfnn_control_points: int = 6,
) -> dict[str, Any]:
    _set_random_seed(random_seed)
    resolved_model_type = _normalized_model_type(model_type)
    resolved_loss_type = _normalized_loss_type(loss_type)
    resolved_window_mode = _normalized_window_mode(window_mode)
    if resolved_model_type == "pfnn" and (resolved_window_mode != "single" or window_radius != 0):
        raise ValueError("Windowed training is currently supported for MLP models only")
    if huber_delta <= 0.0 or not math.isfinite(float(huber_delta)):
        raise ValueError("huber_delta must be positive and finite")
    resolved_device = _resolve_device(device)
    pin_memory = resolved_device.type == "cuda"

    base_feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    resolved_target_columns = target_columns or DEFAULT_TARGET_COLUMNS
    train_features_df, train_targets_df = prepare_windowed_feature_target_frames(
        train_frame,
        base_feature_columns,
        resolved_target_columns,
        window_mode=resolved_window_mode,
        window_radius=window_radius,
        window_feature_columns=resolve_window_feature_columns(list(base_feature_columns), window_feature_mode),
    )
    val_features_df, val_targets_df = prepare_windowed_feature_target_frames(
        val_frame,
        base_feature_columns,
        resolved_target_columns,
        window_mode=resolved_window_mode,
        window_radius=window_radius,
        window_feature_columns=resolve_window_feature_columns(list(base_feature_columns), window_feature_mode),
    )
    phase_feature_index = _phase_feature_index_for_model(resolved_model_type, list(train_features_df.columns))
    raw_feature_indices = [] if phase_feature_index is None else [phase_feature_index]

    train_features = train_features_df.to_numpy(dtype=np.float32, copy=True)
    train_targets = train_targets_df.to_numpy(dtype=np.float32, copy=True)
    val_features = val_features_df.to_numpy(dtype=np.float32, copy=True)
    val_targets = val_targets_df.to_numpy(dtype=np.float32, copy=True)

    feature_medians, feature_means, feature_stds = _fit_feature_stats(
        train_features,
        raw_feature_indices=raw_feature_indices,
    )
    target_means, target_stds = _fit_target_stats(train_targets)

    train_features_scaled = _transform_features(train_features, feature_medians, feature_means, feature_stds)
    val_features_scaled = _transform_features(val_features, feature_medians, feature_means, feature_stds)
    train_targets_scaled = _transform_targets(train_targets, target_means, target_stds)
    val_targets_scaled = _transform_targets(val_targets, target_means, target_stds)
    target_loss_weights_array = resolve_target_loss_weights(list(train_targets_df.columns), target_loss_weights)
    target_loss_weights_tensor = torch.as_tensor(target_loss_weights_array, dtype=torch.float32, device=resolved_device)

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

    model = _build_regressor(
        model_type=resolved_model_type,
        input_dim=train_features_scaled.shape[1],
        output_dim=train_targets_scaled.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        phase_feature_index=phase_feature_index,
        pfnn_expanded_input_dim=pfnn_expanded_input_dim,
        pfnn_phase_node_count=pfnn_phase_node_count,
        pfnn_control_points=pfnn_control_points,
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
                loss = regression_loss(
                    predictions,
                    batch_targets,
                    target_loss_weights=target_loss_weights_tensor,
                    loss_type=resolved_loss_type,
                    huber_delta=huber_delta,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = len(batch_features)
            train_loss_sum += float(loss.item()) * batch_count
            train_sample_count += batch_count

        train_loss = train_loss_sum / max(train_sample_count, 1)
        val_loss = _evaluate_scaled_loss(
            model,
            val_loader,
            resolved_device,
            use_amp=use_amp,
            target_loss_weights=target_loss_weights_tensor,
            loss_type=resolved_loss_type,
            huber_delta=huber_delta,
        )
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
        "model_type": resolved_model_type,
        "feature_columns": list(train_features_df.columns),
        "base_feature_columns": list(base_feature_columns),
        "target_columns": list(train_targets_df.columns),
        "feature_medians": feature_medians,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "target_means": target_means,
        "target_stds": target_stds,
        "target_loss_weights": target_loss_weights_array,
        "target_loss_weights_by_name": _target_loss_weights_as_dict(list(train_targets_df.columns), target_loss_weights_array),
        "loss_type": resolved_loss_type,
        "huber_delta": float(huber_delta),
        "window_mode": resolved_window_mode,
        "window_radius": int(window_radius),
        "window_feature_mode": window_feature_mode,
        "window_feature_columns": resolve_window_feature_columns(list(base_feature_columns), window_feature_mode),
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(dropout),
        "phase_feature_index": phase_feature_index,
        "phase_feature_column": train_features_df.columns[phase_feature_index] if phase_feature_index is not None else None,
        "pfnn_expanded_input_dim": int(pfnn_expanded_input_dim),
        "pfnn_phase_node_count": int(pfnn_phase_node_count),
        "pfnn_control_points": int(pfnn_control_points),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "device_type": resolved_device.type,
        "use_amp": bool(amp_enabled),
        "random_seed": int(random_seed),
    }


def _build_model_from_bundle(bundle: dict[str, Any], device: torch.device) -> nn.Module:
    model_type = _normalized_model_type(bundle.get("model_type", "mlp"))
    if _is_sequence_model_type(model_type):
        return _build_sequence_model_from_bundle(bundle, device)
    if _is_rollout_model_type(model_type):
        return _build_rollout_model_from_bundle(bundle, device)
    phase_feature_index = bundle.get("phase_feature_index")
    if phase_feature_index is not None:
        phase_feature_index = int(phase_feature_index)
    model = _build_regressor(
        model_type=model_type,
        input_dim=len(bundle["feature_columns"]),
        output_dim=len(bundle["target_columns"]),
        hidden_sizes=tuple(int(v) for v in bundle["hidden_sizes"]),
        dropout=float(bundle["dropout"]),
        phase_feature_index=phase_feature_index,
        pfnn_expanded_input_dim=int(bundle.get("pfnn_expanded_input_dim", 45)),
        pfnn_phase_node_count=int(bundle.get("pfnn_phase_node_count", 5)),
        pfnn_control_points=int(bundle.get("pfnn_control_points", 6)),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model


def _build_rollout_model_from_bundle(bundle: dict[str, Any], device: torch.device) -> nn.Module:
    model_type = _normalized_model_type(bundle.get("model_type", "subsection_gru"))
    model = _build_rollout_regressor(
        model_type=model_type,
        context_input_dim=len(bundle["context_feature_columns"]),
        rollout_input_dim=len(bundle["rollout_feature_columns"]),
        current_input_dim=len(bundle["current_feature_columns"]),
        output_dim=len(bundle["target_columns"]),
        hidden_sizes=tuple(int(v) for v in bundle.get("hidden_sizes", [128, 128])),
        dropout=float(bundle.get("dropout", 0.0)),
        gru_num_layers=int(bundle.get("gru_num_layers", 1)),
        latent_size=int(bundle.get("latent_size", 16)),
        dt_over_tau=float(bundle.get("dt_over_tau", 0.03)),
        ct_integrator=str(bundle.get("ct_integrator", "euler")),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model


def _build_sequence_model_from_bundle(bundle: dict[str, Any], device: torch.device) -> nn.Module:
    model_type = _normalized_model_type(bundle.get("model_type", "causal_gru"))
    model = _build_sequence_regressor(
        model_type=model_type,
        sequence_input_dim=len(bundle["sequence_feature_columns"]),
        current_input_dim=len(bundle["current_feature_columns"]),
        output_dim=len(bundle["target_columns"]),
        hidden_sizes=tuple(int(v) for v in bundle.get("hidden_sizes", [128])),
        dropout=float(bundle.get("dropout", 0.0)),
        gru_num_layers=int(bundle.get("gru_num_layers", 1)),
        asl_hidden_size=int(bundle.get("asl_hidden_size", 128)),
        asl_dropout=float(bundle.get("asl_dropout", 0.1)),
        asl_max_frequency_bins=bundle.get("asl_max_frequency_bins"),
        tcn_channels=int(bundle.get("tcn_channels", 128)),
        tcn_num_blocks=int(bundle.get("tcn_num_blocks", 4)),
        tcn_kernel_size=int(bundle.get("tcn_kernel_size", 3)),
        transformer_d_model=int(bundle.get("transformer_d_model", 64)),
        transformer_num_layers=int(bundle.get("transformer_num_layers", 1)),
        transformer_num_heads=int(bundle.get("transformer_num_heads", 4)),
        transformer_dim_feedforward=int(bundle.get("transformer_dim_feedforward", 128)),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model


def _sequence_arrays_for_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequence_features, current_features, targets_df, _ = _sequence_arrays_with_metadata_for_bundle(bundle, frame)
    return sequence_features, current_features, targets_df


def _sequence_arrays_with_metadata_for_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    sequence_features, current_features, targets_df, metadata = prepare_causal_sequence_feature_target_frames(
        frame,
        list(bundle["sequence_feature_columns"]),
        list(bundle["current_feature_columns"]),
        list(bundle["target_columns"]),
        history_size=int(bundle["sequence_history_size"]),
        group_columns=tuple(bundle.get("sequence_group_columns", ("log_id", "segment_id"))),
        sort_column=str(bundle.get("sequence_sort_column", "time_s")),
    )
    return sequence_features, current_features, targets_df, metadata


def _rollout_arrays_for_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    return prepare_causal_rollout_feature_target_frames(
        frame,
        list(bundle["context_feature_columns"]),
        list(bundle["rollout_feature_columns"]),
        list(bundle["current_feature_columns"]),
        list(bundle["target_columns"]),
        history_size=int(bundle["sequence_history_size"]),
        rollout_size=int(bundle["rollout_size"]),
        rollout_stride=int(bundle.get("rollout_stride", bundle["rollout_size"])),
        group_columns=tuple(bundle.get("sequence_group_columns", ("log_id", "segment_id"))),
        sort_column=str(bundle.get("sequence_sort_column", "time_s")),
    )


def _targets_for_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        _, _, targets_df = _sequence_arrays_for_bundle(bundle, frame)
        return targets_df
    if _is_rollout_model_type(bundle.get("model_type", "mlp")):
        _, _, _, targets, _ = _rollout_arrays_for_bundle(bundle, frame)
        return pd.DataFrame(targets.reshape(-1, targets.shape[-1]), columns=bundle["target_columns"])
    _, targets_df = prepare_windowed_feature_target_frames(
        frame,
        bundle.get("base_feature_columns", bundle["feature_columns"]),
        bundle["target_columns"],
        window_mode=bundle.get("window_mode", "single"),
        window_radius=int(bundle.get("window_radius", 0)),
        window_feature_columns=bundle.get("window_feature_columns"),
    )
    return targets_df


def prediction_metadata_frame_for_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    """Align model predictions, true targets, and row metadata for diagnostics."""
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)
    targets_df = _targets_for_bundle(bundle, frame)
    derived = _with_derived_columns(frame)

    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        _, _, _, metadata = _sequence_arrays_with_metadata_for_bundle(bundle, frame)
        join_keys = [column for column in ["log_id", "segment_id", "time_s"] if column in metadata.columns and column in derived.columns]
        if join_keys:
            extra_columns = [column for column in derived.columns if column not in metadata.columns]
            metadata = metadata.merge(
                derived.loc[:, [*join_keys, *extra_columns]].drop_duplicates(join_keys),
                on=join_keys,
                how="left",
            )
    elif _is_rollout_model_type(bundle.get("model_type", "mlp")):
        raise NotImplementedError("prediction metadata alignment is not implemented for rollout model bundles")
    else:
        features_df, _ = prepare_windowed_feature_target_frames(
            derived,
            bundle.get("base_feature_columns", bundle["feature_columns"]),
            bundle["target_columns"],
            window_mode=bundle.get("window_mode", "single"),
            window_radius=int(bundle.get("window_radius", 0)),
            window_feature_columns=bundle.get("window_feature_columns"),
        )
        metadata = derived.loc[features_df.index].reset_index(drop=True)

    aligned = metadata.reset_index(drop=True).copy()
    for target in bundle["target_columns"]:
        true_values = targets_df[target].to_numpy()
        pred_values = predictions_df[target].to_numpy()
        aligned[f"true_{target}"] = true_values
        aligned[f"pred_{target}"] = pred_values
        aligned[f"resid_{target}"] = true_values - pred_values
    aligned["split"] = split_name
    return aligned


def predict_model_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    resolved_device = _resolve_device(device or bundle.get("device_type"))
    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        sequence_features, current_features, _ = _sequence_arrays_for_bundle(bundle, frame)
        sequence_scaled = _transform_sequence_features(
            sequence_features,
            _as_numpy_array(bundle["sequence_feature_medians"]),
            _as_numpy_array(bundle["sequence_feature_means"]),
            _as_numpy_array(bundle["sequence_feature_stds"]),
        )
        if current_features.shape[1] > 0:
            current_scaled = _transform_features(
                current_features,
                _as_numpy_array(bundle["current_feature_medians"]),
                _as_numpy_array(bundle["current_feature_means"]),
                _as_numpy_array(bundle["current_feature_stds"]),
            )
        else:
            current_scaled = current_features.astype(np.float32, copy=False)
        model = _build_sequence_model_from_bundle(bundle, resolved_device)
        predictions_scaled_arr = _predict_sequence_scaled_batches(
            model,
            sequence_scaled,
            current_scaled,
            batch_size=batch_size,
            device=resolved_device,
            use_amp=bool(bundle.get("use_amp", False)),
        )
        predictions = _inverse_transform_targets(
            predictions_scaled_arr,
            _as_numpy_array(bundle["target_means"]),
            _as_numpy_array(bundle["target_stds"]),
        )
        return pd.DataFrame(predictions, columns=bundle["target_columns"])

    if _is_rollout_model_type(bundle.get("model_type", "mlp")):
        context_features, rollout_features, current_features, _, _ = _rollout_arrays_for_bundle(bundle, frame)
        context_scaled = _transform_rollout_features(
            context_features,
            _as_numpy_array(bundle["context_feature_medians"]),
            _as_numpy_array(bundle["context_feature_means"]),
            _as_numpy_array(bundle["context_feature_stds"]),
        )
        rollout_scaled = _transform_rollout_features(
            rollout_features,
            _as_numpy_array(bundle["rollout_feature_medians"]),
            _as_numpy_array(bundle["rollout_feature_means"]),
            _as_numpy_array(bundle["rollout_feature_stds"]),
        )
        if current_features.shape[2] > 0:
            current_scaled = _transform_rollout_features(
                current_features,
                _as_numpy_array(bundle["current_feature_medians"]),
                _as_numpy_array(bundle["current_feature_means"]),
                _as_numpy_array(bundle["current_feature_stds"]),
            )
        else:
            current_scaled = current_features.astype(np.float32, copy=False)
        model = _build_rollout_model_from_bundle(bundle, resolved_device)
        predictions_scaled_arr = _predict_rollout_scaled_batches(
            model,
            context_scaled,
            rollout_scaled,
            current_scaled,
            batch_size=batch_size,
            device=resolved_device,
            use_amp=bool(bundle.get("use_amp", False)),
        )
        predictions = _inverse_transform_targets(
            predictions_scaled_arr.reshape(-1, predictions_scaled_arr.shape[-1]),
            _as_numpy_array(bundle["target_means"]),
            _as_numpy_array(bundle["target_stds"]),
        )
        return pd.DataFrame(predictions, columns=bundle["target_columns"])

    features_df, _ = prepare_windowed_feature_target_frames(
        frame,
        bundle.get("base_feature_columns", bundle["feature_columns"]),
        bundle["target_columns"],
        window_mode=bundle.get("window_mode", "single"),
        window_radius=int(bundle.get("window_radius", 0)),
        window_feature_columns=bundle.get("window_feature_columns"),
    )
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
    targets_df = _targets_for_bundle(bundle, frame)
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)

    y_true = targets_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = predictions_df.to_numpy(dtype=np.float64, copy=False)
    return _metrics_from_arrays(y_true, y_pred, target_columns=bundle["target_columns"], split_name=split_name)


def _metrics_table_row(
    metrics: dict[str, Any],
    *,
    split_name: str,
    diagnostic_type: str,
    group_column: str,
    group_value: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split_name,
        "diagnostic_type": diagnostic_type,
        "group_column": group_column,
        "group_value": group_value,
    }
    row.update(_flatten_split_metrics(split_name, metrics))
    return row


def evaluate_model_bundle_by_log(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    log_column: str = "log_id",
    min_samples: int = 1,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    if log_column not in frame.columns:
        metrics = evaluate_model_bundle(bundle, frame, split_name=split_name, batch_size=batch_size, device=device)
        row = _metrics_table_row(
            metrics,
            split_name=split_name,
            diagnostic_type="per_log",
            group_column=log_column,
            group_value="__missing_log_id__",
        )
        row["log_id"] = "__missing_log_id__"
        return pd.DataFrame([row])

    rows: list[dict[str, Any]] = []
    for log_id, group in frame.groupby(log_column, sort=True):
        if len(group) < min_samples:
            continue
        metrics = evaluate_model_bundle(bundle, group.copy(), split_name=split_name, batch_size=batch_size, device=device)
        row = _metrics_table_row(
            metrics,
            split_name=split_name,
            diagnostic_type="per_log",
            group_column=log_column,
            group_value=str(log_id),
        )
        row["log_id"] = str(log_id)
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_bin_edges(column: str, edges: list[float]) -> list[float]:
    resolved = [float(edge) for edge in edges]
    if len(resolved) < 2:
        raise ValueError(f"Bin spec for {column} must contain at least two edges")
    if any(not math.isfinite(edge) for edge in resolved):
        raise ValueError(f"Bin spec for {column} must contain finite edges")
    if any(right <= left for left, right in zip(resolved, resolved[1:])):
        raise ValueError(f"Bin edges for {column} must be strictly increasing")
    return resolved


def evaluate_model_bundle_by_regime_bins(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    bin_specs: dict[str, list[float]] | None = None,
    min_samples: int = 1,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    resolved_bin_specs = bin_specs or DEFAULT_REGIME_BIN_SPECS
    derived = _with_derived_columns(frame)
    rows: list[dict[str, Any]] = []

    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        _, _, targets_df, metadata = _sequence_arrays_with_metadata_for_bundle(bundle, frame)
        predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)
        join_keys = [column for column in ["log_id", "segment_id", "time_s"] if column in metadata.columns and column in derived.columns]
        if join_keys:
            regime_columns = [column for column in resolved_bin_specs if column in derived.columns and column not in join_keys]
            meta_with_regimes = metadata.merge(
                derived.loc[:, [*join_keys, *regime_columns]].drop_duplicates(join_keys),
                on=join_keys,
                how="left",
            )
        else:
            meta_with_regimes = metadata.copy()
            for column in resolved_bin_specs:
                if column in derived.columns and len(derived) == len(meta_with_regimes):
                    meta_with_regimes[column] = derived[column].to_numpy()

        for column, raw_edges in resolved_bin_specs.items():
            if column not in meta_with_regimes.columns:
                continue
            edges = _validate_bin_edges(column, raw_edges)
            values = pd.to_numeric(meta_with_regimes[column], errors="coerce")
            binned = pd.cut(values, bins=edges, include_lowest=True)
            for interval in binned.cat.categories:
                mask = (binned == interval).to_numpy()
                if int(mask.sum()) < min_samples:
                    continue
                metrics = _metrics_from_arrays(
                    targets_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    predictions_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    target_columns=bundle["target_columns"],
                    split_name=split_name,
                )
                row = _metrics_table_row(
                    metrics,
                    split_name=split_name,
                    diagnostic_type="regime_bin",
                    group_column=column,
                    group_value=str(interval),
                )
                row["regime_column"] = column
                row["bin_label"] = str(interval)
                row["bin_left"] = float(interval.left)
                row["bin_right"] = float(interval.right)
                rows.append(row)
        return pd.DataFrame(rows)

    if _is_rollout_model_type(bundle.get("model_type", "mlp")):
        _, _, _, target_sequences, metadata = _rollout_arrays_for_bundle(bundle, frame)
        targets_df = pd.DataFrame(
            target_sequences.reshape(-1, target_sequences.shape[-1]),
            columns=bundle["target_columns"],
        )
        predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)
        join_keys = [column for column in ["log_id", "segment_id", "time_s"] if column in metadata.columns and column in derived.columns]
        if join_keys:
            regime_columns = [column for column in resolved_bin_specs if column in derived.columns and column not in join_keys]
            meta_with_regimes = metadata.merge(
                derived.loc[:, [*join_keys, *regime_columns]].drop_duplicates(join_keys),
                on=join_keys,
                how="left",
            )
        else:
            meta_with_regimes = metadata.copy()
            for column in resolved_bin_specs:
                if column in derived.columns and len(derived) == len(meta_with_regimes):
                    meta_with_regimes[column] = derived[column].to_numpy()

        for column, raw_edges in resolved_bin_specs.items():
            if column not in meta_with_regimes.columns:
                continue
            edges = _validate_bin_edges(column, raw_edges)
            values = pd.to_numeric(meta_with_regimes[column], errors="coerce")
            binned = pd.cut(values, bins=edges, include_lowest=True)
            for interval in binned.cat.categories:
                mask = (binned == interval).to_numpy()
                if int(mask.sum()) < min_samples:
                    continue
                metrics = _metrics_from_arrays(
                    targets_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    predictions_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    target_columns=bundle["target_columns"],
                    split_name=split_name,
                )
                row = _metrics_table_row(
                    metrics,
                    split_name=split_name,
                    diagnostic_type="regime_bin",
                    group_column=column,
                    group_value=str(interval),
                )
                row["regime_column"] = column
                row["bin_label"] = str(interval)
                row["bin_left"] = float(interval.left)
                row["bin_right"] = float(interval.right)
                rows.append(row)
        return pd.DataFrame(rows)

    for column, raw_edges in resolved_bin_specs.items():
        if column not in derived.columns:
            continue
        edges = _validate_bin_edges(column, raw_edges)
        values = pd.to_numeric(derived[column], errors="coerce")
        binned = pd.cut(values, bins=edges, include_lowest=True)
        for interval in binned.cat.categories:
            mask = binned == interval
            if int(mask.sum()) < min_samples:
                continue
            group = frame.loc[mask.to_numpy()].copy()
            try:
                metrics = evaluate_model_bundle(bundle, group, split_name=split_name, batch_size=batch_size, device=device)
            except ValueError as exc:
                if "No complete causal sequence samples were produced" in str(exc):
                    continue
                raise
            row = _metrics_table_row(
                metrics,
                split_name=split_name,
                diagnostic_type="regime_bin",
                group_column=column,
                group_value=str(interval),
            )
            row["regime_column"] = column
            row["bin_label"] = str(interval)
            row["bin_left"] = float(interval.left)
            row["bin_right"] = float(interval.right)
            rows.append(row)

    return pd.DataFrame(rows)


def run_diagnostic_evaluation(
    *,
    model_bundle_path: str | Path,
    split_root: str | Path,
    output_dir: str | Path,
    split_names: tuple[str, ...] = ("test",),
    bin_specs: dict[str, list[float]] | None = None,
    min_samples: int = 16,
    batch_size: int = 8192,
    device: str | None = None,
) -> dict[str, str]:
    if not split_names:
        raise ValueError("split_names must not be empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(model_bundle_path, map_location="cpu", weights_only=False)

    per_log_parts: list[pd.DataFrame] = []
    per_regime_parts: list[pd.DataFrame] = []
    for split_name in split_names:
        frame = _load_split_frame(split_root, split_name, None, 0)
        per_log_parts.append(
            evaluate_model_bundle_by_log(
                bundle,
                frame,
                split_name=split_name,
                min_samples=min_samples,
                batch_size=batch_size,
                device=device,
            )
        )
        per_regime_parts.append(
            evaluate_model_bundle_by_regime_bins(
                bundle,
                frame,
                split_name=split_name,
                bin_specs=bin_specs,
                min_samples=min_samples,
                batch_size=batch_size,
                device=device,
            )
        )

    per_log = pd.concat(per_log_parts, ignore_index=True) if per_log_parts else pd.DataFrame()
    per_regime = pd.concat(per_regime_parts, ignore_index=True) if per_regime_parts else pd.DataFrame()

    per_log_metrics_path = output_path / "per_log_metrics.csv"
    per_regime_metrics_path = output_path / "per_regime_metrics.csv"
    diagnostics_config_path = output_path / "diagnostics_config.json"

    per_log.to_csv(per_log_metrics_path, index=False)
    per_regime.to_csv(per_regime_metrics_path, index=False)
    diagnostics_config_path.write_text(
        json.dumps(
            {
                "model_bundle_path": str(model_bundle_path),
                "split_root": str(split_root),
                "split_names": list(split_names),
                "bin_specs": bin_specs or DEFAULT_REGIME_BIN_SPECS,
                "min_samples": int(min_samples),
                "batch_size": int(batch_size),
                "device": device or "auto",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "per_log_metrics_path": str(per_log_metrics_path),
        "per_regime_metrics_path": str(per_regime_metrics_path),
        "diagnostics_config_path": str(diagnostics_config_path),
    }


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


def _save_baseline_comparison_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    if summary.empty:
        return
    fig_width = max(8.0, 1.8 * len(summary))
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    x = np.arange(len(summary))
    if "test_overall_r2" in summary.columns:
        width = 0.36
        ax.bar(x - width / 2, summary["val_overall_r2"], width=width, label="val_overall_r2")
        ax.bar(x + width / 2, summary["test_overall_r2"], width=width, label="test_overall_r2")
    else:
        ax.bar(x, summary["val_overall_r2"], width=0.48, label="val_overall_r2")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["recipe_name"], rotation=20, ha="right")
    ax.set_ylabel("R^2")
    ax.set_title("Baseline Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _target_groups_label(target_groups: dict[str, list[str]]) -> str:
    return ";".join(f"{group_name}:{'|'.join(targets)}" for group_name, targets in target_groups.items())


def _combine_disjoint_target_metrics(split_name: str, group_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_target: dict[str, dict[str, float]] = {}
    sample_counts: list[int] = []
    for metrics in group_metrics.values():
        sample_counts.append(int(metrics["sample_count"]))
        for target_name, target_metrics in metrics["per_target"].items():
            per_target[target_name] = {
                "mae": float(target_metrics["mae"]),
                "rmse": float(target_metrics["rmse"]),
                "r2": float(target_metrics["r2"]),
            }

    if not per_target:
        raise ValueError("Cannot combine split-axis metrics without per-target metrics")

    mae_values = np.array([metrics["mae"] for metrics in per_target.values()], dtype=float)
    rmse_values = np.array([metrics["rmse"] for metrics in per_target.values()], dtype=float)
    r2_values = np.array([metrics["r2"] for metrics in per_target.values()], dtype=float)

    return {
        "split": split_name,
        "sample_count": int(min(sample_counts)) if sample_counts else 0,
        "overall_mae": float(np.mean(mae_values)),
        "overall_rmse": float(np.sqrt(np.mean(np.square(rmse_values)))),
        "overall_r2": float(np.mean(r2_values)),
        "per_target": per_target,
    }


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
