from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

# Keep this legacy module as the public entry point while allowing canonical
# foundation modules to live below ``system_identification.training``.
__path__ = [str(Path(__file__).with_suffix(""))]

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from system_identification.models.features import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_FEATURE_SETS,
    KINEMATIC_WINDOW_EXCLUDED_COLUMNS,
    NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS,
    NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS,
    PAPER_NO_ACCEL_V2_ADDED_FEATURE_COLUMNS,
    PAPER_NO_ACCEL_V2_FEATURE_COLUMNS,
    PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS,
    PAPER_NO_ACCEL_V2_RAW_PHASE_FEATURE_COLUMNS,
    PAPER_PFNN_10_FEATURE_COLUMNS,
    PHASE_CONDITIONING_COLUMNS,
    PHASE_HARMONIC_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_MODE_COLUMNS,
    SEQUENCE_HISTORY_DANGEROUS_COLUMNS,
    WINDOW_FEATURE_MODE_COLUMNS,
    _with_derived_columns,
    apply_sequence_order_ablation,
    resolve_current_feature_columns,
    resolve_feature_set_columns,
    resolve_phase_conditioning_indices,
    resolve_sequence_feature_columns,
    resolve_window_feature_columns,
)
from system_identification.models.neural import (
    AdaptiveSpectrumLayer,
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
    PhaseFunctionedLinear,
    SubsectionGRUWrenchRegressor,
    _CausalConvBlock,
    _PhaseFiLM,
    _make_mlp_layers,
    cyclic_catmull_rom_weights,
)
from system_identification.models.bundles import (
    _build_model_from_bundle,
    _build_rollout_model_from_bundle,
    _build_sequence_model_from_bundle,
    _is_rollout_model_type,
    _is_sequence_model_type,
    _normalized_model_type,
    _phase_feature_index_for_model,
    _to_serializable_bundle,
)
from system_identification.evaluation.metrics import (
    _combine_disjoint_target_metrics,
    _metrics_from_arrays,
    _validate_bin_edges,
)
from system_identification.evaluation.prediction import (
    _as_numpy_array,
    _predict_rollout_scaled_batches,
    _predict_scaled_batches,
    _predict_sequence_scaled_batches,
    _resolve_device,
)
from system_identification.evaluation.reports import (
    _flatten_split_metrics,
    _history_frame,
    _metrics_table_row,
    _target_groups_label,
)
from system_identification.plotting.figures import (
    _save_ablation_summary_plot,
    _save_baseline_comparison_plot,
)
from system_identification.training.loaders import (
    _make_loader,
    _make_rollout_loader,
    _make_sequence_loader,
)
from system_identification.training.early_stopping import (
    EarlyStoppingState,
    update_early_stopping,
)
from system_identification.training.bundle_assembly import (
    assemble_rollout_training_bundle,
    assemble_sequence_training_bundle,
    assemble_training_bundle,
)
from system_identification.training.history import (
    build_rollout_validation_history_row,
    build_sequence_validation_history_row,
    build_validation_history_row,
)
from system_identification.training.loop import (
    _ema_update_state,
    _evaluate_rollout_scaled_loss,
    _evaluate_scaled_loss,
    _evaluate_sequence_scaled_loss,
    _normalized_lr_scheduler,
    _train_rollout_scaled_epoch,
    _train_scaled_epoch,
    _train_sequence_scaled_epoch,
    _warmup_cosine_lr_lambda,
)
from system_identification.training.losses import (
    _normalized_loss_type,
    _target_loss_weights_as_dict,
    regression_loss,
    resolve_target_loss_weights,
)
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
    _inverse_transform_targets,
    _transform_features,
    _transform_rollout_features,
    _transform_sequence_features,
    _transform_targets,
)
from system_identification.training.optimizer_factory import (
    build_adamw_optimizer,
    build_training_scheduler,
)
from system_identification.training.selection import (
    BestEpochSelection,
    update_best_epoch_selection,
)
from system_identification.training.windows import (
    _normalized_window_mode,
    _window_feature_name,
    _window_offsets,
    prepare_causal_rollout_feature_target_frames,
    prepare_causal_sequence_feature_target_frames,
    prepare_feature_target_frames,
    prepare_windowed_feature_target_frames,
)
from system_identification.training.data_preparation import (
    DEFAULT_TARGET_COLUMNS,
    _load_split_frame,
    _set_random_seed,
)
from system_identification.training.recipes import (
    fit_torch_regressor,
    fit_torch_rollout_regressor,
    fit_torch_sequence_regressor,
)
from system_identification.evaluation.diagnostics import (
    ACCELERATION_INPUT_COLUMNS,
    ALPHA_BETA_HISTORY_COLUMNS,
    ANGULAR_VELOCITY_HISTORY_COLUMNS,
    DEFAULT_REGIME_BIN_SPECS,
    VELOCITY_HISTORY_COLUMNS,
    _rollout_arrays_for_bundle,
    _sequence_arrays_for_bundle,
    _sequence_arrays_with_metadata_for_bundle,
    _targets_for_bundle,
    _training_audit_flags,
    evaluate_model_bundle,
    evaluate_model_bundle_by_log,
    evaluate_model_bundle_by_regime_bins,
    predict_model_bundle,
    prediction_metadata_frame_for_bundle,
    run_diagnostic_evaluation,
)
from system_identification.artifacts.io import (
    _save_pred_vs_true_plot,
    _save_residual_hist_plot,
    _save_training_curves,
)
from system_identification.training.orchestration import (
    BASELINE_COMPARISON_RECIPES,
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_FEATURE_GROUPS,
    LEAKAGE_RESISTANT_BASELINE_PROTOCOL,
    _ordered_unique_columns,
    _resolve_baseline_recipe_names,
    _run_single_baseline_recipe,
    _run_split_axis_baseline_recipe,
    resolve_ablation_variants,
    run_ablation_study,
    run_baseline_comparison,
    run_training_job,
)
