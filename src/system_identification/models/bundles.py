"""Existing model-bundle schema conversion and inference model factories."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from system_identification.models.neural import (
    CausalGRUASLRegressor,
    CausalGRURegressor,
    CausalLSTMRegressor,
    CausalTCNGRURegressor,
    CausalTCNRegressor,
    CausalTransformerRegressor,
    ContinuousTimeSUBNETWrenchRegressor,
    DiscreteSUBNETWrenchRegressor,
    HybridPFNNRegressor,
    MLPRegressor,
    SubsectionGRUWrenchRegressor,
)

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
        "causal_transformer_head_film",
        "causal_transformer_input_film",
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
        "causal_transformer_head_film",
        "causal_transformer_input_film",
        "causal_tcn_gru",
    }


def _is_rollout_model_type(model_type: str | None) -> bool:
    return _normalized_model_type(model_type) in {"subsection_gru", "subnet_discrete", "ct_subnet_euler"}


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
    phase_conditioning_indices: tuple[int, ...] | None = None,
    film_hidden_size: int = 32,
    film_scale: float = 0.1,
    transformer_use_positional_encoding: bool = True,
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
    if model_type in {"causal_transformer", "causal_transformer_head_film", "causal_transformer_input_film"}:
        transformer_head_sizes = tuple(int(v) for v in hidden_sizes[1:]) or (int(transformer_d_model),)
        film_mode = "none"
        if model_type == "causal_transformer_head_film":
            film_mode = "head"
        elif model_type == "causal_transformer_input_film":
            film_mode = "input"
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
            film_mode=film_mode,
            phase_conditioning_indices=phase_conditioning_indices,
            film_hidden_size=film_hidden_size,
            film_scale=film_scale,
            use_positional_encoding=transformer_use_positional_encoding,
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
        phase_conditioning_indices=tuple(int(v) for v in bundle.get("phase_conditioning_indices", [])),
        film_hidden_size=int(bundle.get("film_hidden_size", 32)),
        film_scale=float(bundle.get("film_scale", 0.1)),
        transformer_use_positional_encoding=bool(bundle.get("transformer_use_positional_encoding", True)),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model
