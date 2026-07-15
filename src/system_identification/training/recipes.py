"""Existing standard, sequence, and rollout training recipes."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from system_identification.evaluation.metrics import _metrics_from_arrays
from system_identification.evaluation.prediction import (
    _inverse_transform_targets,
    _predict_rollout_scaled_batches,
    _predict_scaled_batches,
    _predict_sequence_scaled_batches,
    _resolve_device,
)
from system_identification.models.bundles import (
    _is_rollout_model_type,
    _is_sequence_model_type,
    _normalized_model_type,
    _phase_feature_index_for_model,
)
from system_identification.models.features import (
    DEFAULT_FEATURE_COLUMNS,
    resolve_current_feature_columns,
    resolve_phase_conditioning_indices,
    resolve_sequence_feature_columns,
    resolve_window_feature_columns,
)
from system_identification.training.bundle_assembly import (
    assemble_rollout_training_bundle,
    assemble_sequence_training_bundle,
    assemble_training_bundle,
)
from system_identification.training.data_preparation import DEFAULT_TARGET_COLUMNS, _set_random_seed
from system_identification.training.early_stopping import EarlyStoppingState, update_early_stopping
from system_identification.training.history import (
    build_rollout_validation_history_row,
    build_sequence_validation_history_row,
    build_validation_history_row,
)
from system_identification.training.loaders import _make_loader, _make_rollout_loader, _make_sequence_loader
from system_identification.training.loop import (
    _evaluate_rollout_scaled_loss,
    _evaluate_scaled_loss,
    _evaluate_sequence_scaled_loss,
    _normalized_lr_scheduler,
    _train_rollout_scaled_epoch,
    _train_scaled_epoch,
    _train_sequence_scaled_epoch,
)
from system_identification.training.losses import _normalized_loss_type, resolve_target_loss_weights
from system_identification.training.model_factory import (
    _build_regressor,
    _build_rollout_regressor,
    _build_sequence_regressor,
)
from system_identification.training.normalization import (
    _fit_feature_stats,
    _fit_rollout_feature_stats,
    _fit_sequence_feature_stats,
    _fit_target_stats,
    _transform_features,
    _transform_rollout_features,
    _transform_sequence_features,
    _transform_targets,
)
from system_identification.training.optimizer_factory import build_adamw_optimizer, build_training_scheduler
from system_identification.training.selection import BestEpochSelection, update_best_epoch_selection
from system_identification.training.windows import (
    _normalized_window_mode,
    prepare_causal_rollout_feature_target_frames,
    prepare_causal_sequence_feature_target_frames,
    prepare_windowed_feature_target_frames,
)

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
    prior_target_columns: list[str] | None = None,
    prior_loss_weight: float = 0.0,
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
    transformer_use_positional_encoding: bool = True,
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
    resolved_prior_loss_weight = float(prior_loss_weight)
    if not math.isfinite(resolved_prior_loss_weight) or resolved_prior_loss_weight < 0.0:
        raise ValueError("prior_loss_weight must be finite and nonnegative")
    resolved_prior_target_columns = list(prior_target_columns or [])
    if resolved_prior_loss_weight > 0.0 and not resolved_prior_target_columns:
        raise ValueError("prior_target_columns are required when prior_loss_weight is positive")
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
    if resolved_prior_target_columns and len(resolved_prior_target_columns) != len(resolved_target_columns):
        raise ValueError("prior_target_columns must have the same length as target_columns")
    sequence_feature_columns = resolve_sequence_feature_columns(list(base_feature_columns), sequence_feature_mode)
    if not sequence_feature_columns:
        raise ValueError("Sequence models require at least one sequence feature")
    current_feature_columns = resolve_current_feature_columns(
        list(base_feature_columns),
        sequence_feature_columns,
        current_feature_mode,
    )
    film_mode = "none"
    if resolved_model_type == "causal_transformer_head_film":
        film_mode = "head"
    elif resolved_model_type == "causal_transformer_input_film":
        film_mode = "input"
    phase_conditioning_indices: tuple[int, ...] | None = None
    phase_conditioning_columns: list[str] = []
    film_hidden_size = 32
    film_scale = 0.1
    if film_mode != "none":
        phase_conditioning_indices = resolve_phase_conditioning_indices(sequence_feature_columns)
        phase_conditioning_columns = [sequence_feature_columns[index] for index in phase_conditioning_indices]

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
    train_prior_targets_df: pd.DataFrame | None = None
    if resolved_prior_target_columns:
        _, _, train_prior_targets_df, _ = prepare_causal_sequence_feature_target_frames(
            train_frame,
            sequence_feature_columns,
            current_feature_columns,
            resolved_prior_target_columns,
            history_size=sequence_history_size,
        )
        if len(train_prior_targets_df) != len(train_targets_df):
            raise ValueError("prior target rows do not align with supervised target rows")

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
    train_prior_targets_scaled = (
        _transform_targets(train_prior_targets_df.to_numpy(dtype=np.float32, copy=True), target_means, target_stds)
        if train_prior_targets_df is not None
        else None
    )
    target_loss_weights_array = resolve_target_loss_weights(list(train_targets_df.columns), target_loss_weights)
    target_loss_weights_tensor = torch.as_tensor(target_loss_weights_array, dtype=torch.float32, device=resolved_device)

    train_loader = _make_sequence_loader(
        train_sequence_scaled,
        train_current_scaled,
        train_targets_scaled,
        prior_targets=train_prior_targets_scaled,
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
        phase_conditioning_indices=phase_conditioning_indices,
        film_hidden_size=film_hidden_size,
        film_scale=film_scale,
        transformer_use_positional_encoding=transformer_use_positional_encoding,
    ).to(resolved_device)
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    total_training_steps = max(len(train_loader) * int(max_epochs), 1)
    warmup_steps = int(round(total_training_steps * float(lr_warmup_ratio)))
    scheduler = build_training_scheduler(
        optimizer,
        scheduler_name=resolved_lr_scheduler,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and resolved_device.type == "cuda")

    best_selection = BestEpochSelection(
        state_dict=copy.deepcopy(model.state_dict()),
        val_loss=math.inf,
        epoch=0,
    )
    early_stopping_state = EarlyStoppingState()
    history: list[dict[str, float]] = []
    amp_enabled = use_amp and resolved_device.type == "cuda"
    ema_state = None
    if resolved_ema_decay > 0.0:
        ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_supervised_loss, train_prior_loss = _train_sequence_scaled_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            resolved_device,
            use_amp=use_amp,
            target_loss_weights=target_loss_weights_tensor,
            loss_type=resolved_loss_type,
            huber_delta=huber_delta,
            prior_loss_weight=resolved_prior_loss_weight,
            gradient_clip_norm=resolved_gradient_clip_norm,
            scheduler=scheduler,
            ema_state=ema_state,
            ema_decay=resolved_ema_decay,
        )
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
        history_row = build_sequence_validation_history_row(
            epoch=epoch,
            learning_rate=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            train_supervised_loss=train_supervised_loss,
            train_prior_loss=train_prior_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        history.append(history_row)

        best_selection, improved = update_best_epoch_selection(
            best_selection,
            val_loss=val_loss,
            epoch=epoch,
            state_dict=ema_state if ema_state is not None else model.state_dict(),
        )
        early_stopping_state, should_stop = update_early_stopping(
            early_stopping_state,
            improved=improved,
            patience=early_stopping_patience,
        )
        if should_stop:
            break

    best_state_dict = best_selection.state_dict
    best_val_loss = best_selection.val_loss
    best_epoch = best_selection.epoch
    model.load_state_dict(best_state_dict)

    return assemble_sequence_training_bundle(
        amp_enabled=amp_enabled,
        asl_dropout=asl_dropout,
        asl_hidden_size=asl_hidden_size,
        asl_max_frequency_bins=asl_max_frequency_bins,
        base_feature_columns=base_feature_columns,
        best_epoch=best_epoch,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        current_feature_columns=current_feature_columns,
        current_feature_means=current_feature_means,
        current_feature_medians=current_feature_medians,
        current_feature_mode=current_feature_mode,
        current_feature_stds=current_feature_stds,
        dropout=dropout,
        film_hidden_size=film_hidden_size,
        film_mode=film_mode,
        film_scale=film_scale,
        gru_num_layers=gru_num_layers,
        hidden_sizes=hidden_sizes,
        history=history,
        huber_delta=huber_delta,
        lr_warmup_ratio=lr_warmup_ratio,
        phase_conditioning_columns=phase_conditioning_columns,
        phase_conditioning_indices=phase_conditioning_indices,
        random_seed=random_seed,
        resolved_device=resolved_device,
        resolved_ema_decay=resolved_ema_decay,
        resolved_gradient_clip_norm=resolved_gradient_clip_norm,
        resolved_loss_type=resolved_loss_type,
        resolved_lr_scheduler=resolved_lr_scheduler,
        resolved_model_type=resolved_model_type,
        resolved_prior_loss_weight=resolved_prior_loss_weight,
        resolved_prior_target_columns=resolved_prior_target_columns,
        sequence_feature_columns=sequence_feature_columns,
        sequence_feature_means=sequence_feature_means,
        sequence_feature_medians=sequence_feature_medians,
        sequence_feature_mode=sequence_feature_mode,
        sequence_feature_stds=sequence_feature_stds,
        sequence_history_size=sequence_history_size,
        target_loss_weights_array=target_loss_weights_array,
        target_means=target_means,
        target_stds=target_stds,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_num_blocks=tcn_num_blocks,
        train_sequence_scaled=train_sequence_scaled,
        train_targets_df=train_targets_df,
        transformer_d_model=transformer_d_model,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_num_heads=transformer_num_heads,
        transformer_num_layers=transformer_num_layers,
        transformer_use_positional_encoding=transformer_use_positional_encoding,
        val_sequence_scaled=val_sequence_scaled,
        warmup_steps=warmup_steps,
    )


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
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and resolved_device.type == "cuda")

    best_selection = BestEpochSelection(
        state_dict=copy.deepcopy(model.state_dict()),
        val_loss=math.inf,
        epoch=0,
    )
    early_stopping_state = EarlyStoppingState()
    history: list[dict[str, float]] = []
    amp_enabled = use_amp and resolved_device.type == "cuda"

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_rollout_scaled_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            resolved_device,
            use_amp=use_amp,
            target_loss_weights=target_loss_weights_tensor,
            loss_type=resolved_loss_type,
            huber_delta=huber_delta,
        )
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
        history_row = build_rollout_validation_history_row(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            latent_rms=getattr(model, "last_latent_rms", 0.0),
            delta_latent_rms=getattr(model, "last_delta_latent_rms", 0.0),
            latent_derivative_rms=getattr(model, "last_latent_derivative_rms", 0.0),
        )
        history.append(history_row)

        best_selection, improved = update_best_epoch_selection(
            best_selection,
            val_loss=val_loss,
            epoch=epoch,
            state_dict=model.state_dict(),
        )
        early_stopping_state, should_stop = update_early_stopping(
            early_stopping_state,
            improved=improved,
            patience=early_stopping_patience,
        )
        if should_stop:
            break

    best_state_dict = best_selection.state_dict
    best_val_loss = best_selection.val_loss
    best_epoch = best_selection.epoch
    model.load_state_dict(best_state_dict)

    return assemble_rollout_training_bundle(
        amp_enabled=amp_enabled,
        base_feature_columns=base_feature_columns,
        best_epoch=best_epoch,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        context_feature_columns=context_feature_columns,
        context_feature_means=context_feature_means,
        context_feature_medians=context_feature_medians,
        context_feature_stds=context_feature_stds,
        ct_integrator=ct_integrator,
        current_feature_columns=current_feature_columns,
        current_feature_means=current_feature_means,
        current_feature_medians=current_feature_medians,
        current_feature_mode=current_feature_mode,
        current_feature_stds=current_feature_stds,
        dropout=dropout,
        dt_over_tau=dt_over_tau,
        gru_num_layers=gru_num_layers,
        hidden_sizes=hidden_sizes,
        history=history,
        huber_delta=huber_delta,
        latent_size=latent_size,
        random_seed=random_seed,
        resolved_device=resolved_device,
        resolved_loss_type=resolved_loss_type,
        resolved_model_type=resolved_model_type,
        resolved_rollout_stride=resolved_rollout_stride,
        resolved_target_columns=resolved_target_columns,
        rollout_feature_columns=rollout_feature_columns,
        rollout_feature_means=rollout_feature_means,
        rollout_feature_medians=rollout_feature_medians,
        rollout_feature_stds=rollout_feature_stds,
        rollout_size=rollout_size,
        sequence_feature_mode=sequence_feature_mode,
        sequence_history_size=sequence_history_size,
        target_loss_weights_array=target_loss_weights_array,
        target_means=target_means,
        target_stds=target_stds,
        train_context_scaled=train_context_scaled,
        train_targets=train_targets,
        val_context_scaled=val_context_scaled,
        val_targets=val_targets,
    )


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
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and resolved_device.type == "cuda")

    best_selection = BestEpochSelection(
        state_dict=copy.deepcopy(model.state_dict()),
        val_loss=math.inf,
        epoch=0,
    )
    early_stopping_state = EarlyStoppingState()
    history: list[dict[str, float]] = []
    amp_enabled = use_amp and resolved_device.type == "cuda"

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_scaled_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            resolved_device,
            use_amp=use_amp,
            target_loss_weights=target_loss_weights_tensor,
            loss_type=resolved_loss_type,
            huber_delta=huber_delta,
        )
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
        history_row = build_validation_history_row(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        history.append(history_row)

        best_selection, improved = update_best_epoch_selection(
            best_selection,
            val_loss=val_loss,
            epoch=epoch,
            state_dict=model.state_dict(),
        )
        early_stopping_state, should_stop = update_early_stopping(
            early_stopping_state,
            improved=improved,
            patience=early_stopping_patience,
        )
        if should_stop:
            break

    best_state_dict = best_selection.state_dict
    best_val_loss = best_selection.val_loss
    best_epoch = best_selection.epoch
    model.load_state_dict(best_state_dict)

    return assemble_training_bundle(
        amp_enabled=amp_enabled,
        base_feature_columns=base_feature_columns,
        best_epoch=best_epoch,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        dropout=dropout,
        feature_means=feature_means,
        feature_medians=feature_medians,
        feature_stds=feature_stds,
        hidden_sizes=hidden_sizes,
        history=history,
        huber_delta=huber_delta,
        pfnn_control_points=pfnn_control_points,
        pfnn_expanded_input_dim=pfnn_expanded_input_dim,
        pfnn_phase_node_count=pfnn_phase_node_count,
        phase_feature_index=phase_feature_index,
        random_seed=random_seed,
        resolved_device=resolved_device,
        resolved_loss_type=resolved_loss_type,
        resolved_model_type=resolved_model_type,
        resolved_window_mode=resolved_window_mode,
        target_loss_weights_array=target_loss_weights_array,
        target_means=target_means,
        target_stds=target_stds,
        train_features_df=train_features_df,
        train_targets_df=train_targets_df,
        window_feature_mode=window_feature_mode,
        window_radius=window_radius,
    )
