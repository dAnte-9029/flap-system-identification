from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import system_identification.models.features as features
import system_identification.models.neural as neural
import system_identification.training as training


FEATURE_SYMBOLS = (
    "DEFAULT_FEATURE_COLUMNS",
    "NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS",
    "NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS",
    "PAPER_NO_ACCEL_V2_ADDED_FEATURE_COLUMNS",
    "PAPER_NO_ACCEL_V2_FEATURE_COLUMNS",
    "PHASE_HARMONIC_FEATURE_COLUMNS",
    "PHASE_CONDITIONING_COLUMNS",
    "PAPER_NO_ACCEL_V2_RAW_PHASE_FEATURE_COLUMNS",
    "PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS",
    "PAPER_PFNN_10_FEATURE_COLUMNS",
    "DEFAULT_FEATURE_SETS",
    "resolve_feature_set_columns",
    "_with_derived_columns",
    "WINDOW_FEATURE_MODE_COLUMNS",
    "KINEMATIC_WINDOW_EXCLUDED_COLUMNS",
    "SEQUENCE_FEATURE_MODE_COLUMNS",
    "SEQUENCE_HISTORY_DANGEROUS_COLUMNS",
    "resolve_window_feature_columns",
    "resolve_sequence_feature_columns",
    "resolve_current_feature_columns",
    "resolve_phase_conditioning_indices",
    "apply_sequence_order_ablation",
)

NEURAL_SYMBOLS = (
    "MLPRegressor",
    "CausalGRURegressor",
    "AdaptiveSpectrumLayer",
    "CausalGRUASLRegressor",
    "_make_mlp_layers",
    "CausalLSTMRegressor",
    "_CausalConvBlock",
    "CausalTCNRegressor",
    "_PhaseFiLM",
    "CausalTransformerRegressor",
    "CausalTCNGRURegressor",
    "SubsectionGRUWrenchRegressor",
    "DiscreteSUBNETWrenchRegressor",
    "ContinuousTimeSUBNETWrenchRegressor",
    "cyclic_catmull_rom_weights",
    "PhaseFunctionedLinear",
    "HybridPFNNRegressor",
)


def test_training_reexports_canonical_feature_objects():
    for symbol in FEATURE_SYMBOLS:
        assert hasattr(training, symbol)
        assert hasattr(features, symbol)
        assert getattr(training, symbol) is getattr(features, symbol)


def test_training_reexports_canonical_neural_objects():
    for symbol in NEURAL_SYMBOLS:
        assert hasattr(training, symbol)
        assert hasattr(neural, symbol)
        assert getattr(training, symbol) is getattr(neural, symbol)


def test_legacy_and_canonical_feature_results_match():
    frame = pd.DataFrame({"phase_corrected_rad": [0.0, np.pi / 2.0, np.pi]})

    legacy = training._with_derived_columns(frame)
    canonical = features._with_derived_columns(frame)

    pd.testing.assert_frame_equal(legacy, canonical)
    assert training.resolve_feature_set_columns("paper_pfnn_10") == features.resolve_feature_set_columns(
        "paper_pfnn_10"
    )


def test_legacy_and_canonical_model_outputs_match_with_fixed_seed():
    inputs = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)

    torch.manual_seed(17)
    legacy_model = training.MLPRegressor(input_dim=3, output_dim=2, hidden_sizes=(4,), dropout=0.0)
    torch.manual_seed(17)
    canonical_model = neural.MLPRegressor(input_dim=3, output_dim=2, hidden_sizes=(4,), dropout=0.0)

    legacy_model.eval()
    canonical_model.eval()
    torch.testing.assert_close(legacy_model(inputs), canonical_model(inputs), rtol=0.0, atol=0.0)
