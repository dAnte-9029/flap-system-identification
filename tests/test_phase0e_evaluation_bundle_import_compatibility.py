from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import system_identification.evaluation.metrics as metrics
import system_identification.evaluation.prediction as prediction
import system_identification.evaluation.reports as reports
import system_identification.models.bundles as bundles
import system_identification.plotting.figures as figures
import system_identification.training as training


MODULE_SYMBOLS = {
    bundles: (
        "_to_serializable_bundle",
        "_normalized_model_type",
        "_is_sequence_model_type",
        "_is_rollout_model_type",
        "_phase_feature_index_for_model",
        "_build_regressor",
        "_build_sequence_regressor",
        "_build_rollout_regressor",
        "_build_model_from_bundle",
        "_build_rollout_model_from_bundle",
        "_build_sequence_model_from_bundle",
    ),
    prediction: (
        "_resolve_device",
        "_transform_features",
        "_inverse_transform_targets",
        "_as_numpy_array",
        "_make_loader",
        "_make_sequence_loader",
        "_make_rollout_loader",
        "_predict_scaled_batches",
        "_predict_sequence_scaled_batches",
        "_predict_rollout_scaled_batches",
        "_transform_sequence_features",
        "_transform_rollout_features",
    ),
    metrics: (
        "_metrics_from_arrays",
        "_validate_bin_edges",
        "_combine_disjoint_target_metrics",
    ),
    reports: (
        "_history_frame",
        "_flatten_split_metrics",
        "_metrics_table_row",
        "_target_groups_label",
    ),
    figures: (
        "_save_ablation_summary_plot",
        "_save_baseline_comparison_plot",
    ),
}


def test_training_reexports_all_canonical_phase0e_objects():
    assert sum(len(symbols) for symbols in MODULE_SYMBOLS.values()) == 32
    for module, symbols in MODULE_SYMBOLS.items():
        for symbol in symbols:
            assert hasattr(training, symbol)
            assert getattr(training, symbol) is getattr(module, symbol)


def test_legacy_and_canonical_metric_results_match():
    y_true = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    y_pred = np.array([[0.5, 0.5], [1.5, 4.0]], dtype=float)
    kwargs = {"target_columns": ["fx_b", "fz_b"], "split_name": "validation"}

    assert training._metrics_from_arrays(y_true, y_pred, **kwargs) == metrics._metrics_from_arrays(
        y_true, y_pred, **kwargs
    )


def test_legacy_and_canonical_batch_prediction_results_match():
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.bias.copy_(torch.tensor([0.25]))
    inputs = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    kwargs = {"batch_size": 1, "device": torch.device("cpu"), "use_amp": False}

    legacy = training._predict_scaled_batches(model, inputs, **kwargs)
    canonical = prediction._predict_scaled_batches(model, inputs, **kwargs)

    np.testing.assert_array_equal(legacy, canonical)


def test_bundle_serialization_schema_matches_legacy_path():
    bundle = {
        "model_type": "mlp",
        "feature_medians": np.array([1.0, 2.0], dtype=np.float64),
        "target_means": [3.0],
        "untouched_metadata": {"name": "fixture"},
    }

    legacy = training._to_serializable_bundle(bundle)
    canonical = bundles._to_serializable_bundle(bundle)

    assert legacy.keys() == canonical.keys() == bundle.keys()
    assert legacy["feature_medians"].dtype == canonical["feature_medians"].dtype == torch.float32
    torch.testing.assert_close(legacy["feature_medians"], canonical["feature_medians"])
    torch.testing.assert_close(legacy["target_means"], canonical["target_means"])
    assert legacy["untouched_metadata"] == canonical["untouched_metadata"]


def test_plotting_helper_writes_equivalent_figures(tmp_path):
    summary = pd.DataFrame(
        {
            "variant_name": ["a", "b"],
            "val_overall_r2": [0.1, 0.2],
            "test_overall_r2": [0.0, 0.3],
        }
    )
    legacy_path = tmp_path / "legacy.png"
    canonical_path = tmp_path / "canonical.png"

    training._save_ablation_summary_plot(summary, legacy_path)
    figures._save_ablation_summary_plot(summary, canonical_path)

    assert legacy_path.read_bytes() == canonical_path.read_bytes()
