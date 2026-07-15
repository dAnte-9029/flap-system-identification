from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import system_identification.evaluation.prediction as prediction
import system_identification.models.bundles as bundles
import system_identification.training as training
import system_identification.training.loaders as loaders
import system_identification.training.losses as losses
import system_identification.training.normalization as normalization
import system_identification.training.windows as windows


MODULE_SYMBOLS = {
    windows: (
        "prepare_feature_target_frames",
        "_normalized_window_mode",
        "_window_offsets",
        "_window_feature_name",
        "prepare_windowed_feature_target_frames",
        "prepare_causal_sequence_feature_target_frames",
        "prepare_causal_rollout_feature_target_frames",
    ),
    normalization: (
        "_fit_feature_stats",
        "_fit_target_stats",
        "_transform_targets",
        "_fit_sequence_feature_stats",
        "_fit_rollout_feature_stats",
        "_transform_features",
        "_inverse_transform_targets",
        "_transform_sequence_features",
        "_transform_rollout_features",
    ),
    losses: (
        "resolve_target_loss_weights",
        "_target_loss_weights_as_dict",
        "_normalized_loss_type",
        "regression_loss",
    ),
    loaders: ("_make_loader", "_make_sequence_loader", "_make_rollout_loader"),
}


def test_training_reexports_all_canonical_phase0f_objects():
    assert training.__file__.endswith("system_identification/training.py")
    assert sum(len(symbols) for symbols in MODULE_SYMBOLS.values()) == 23
    for module, symbols in MODULE_SYMBOLS.items():
        for symbol in symbols:
            assert getattr(training, symbol) is getattr(module, symbol)

    assert prediction._make_loader is loaders._make_loader
    assert prediction._transform_features is normalization._transform_features


def test_legacy_and_canonical_window_results_match():
    frame = pd.DataFrame(
        {
            "log_id": ["a", "a", "a", "b", "b", "b"],
            "segment_id": [0, 0, 0, 0, 0, 0],
            "phase_corrected_rad": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            "feature": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "fx_b": [2.0, 4.0, 6.0, 20.0, 22.0, 24.0],
        }
    )
    kwargs = {
        "feature_columns": ["feature"],
        "target_columns": ["fx_b"],
        "window_mode": "causal",
        "window_radius": 1,
    }

    legacy_features, legacy_targets = training.prepare_windowed_feature_target_frames(frame, **kwargs)
    canonical_features, canonical_targets = windows.prepare_windowed_feature_target_frames(frame, **kwargs)

    pd.testing.assert_frame_equal(legacy_features, canonical_features)
    pd.testing.assert_frame_equal(legacy_targets, canonical_targets)


def test_normalization_fit_transform_and_inverse_match():
    features = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]], dtype=np.float64)
    targets = np.array([[2.0, -1.0], [4.0, 1.0], [6.0, 3.0]], dtype=np.float64)

    legacy_stats = training._fit_feature_stats(features)
    canonical_stats = normalization._fit_feature_stats(features)
    for legacy_value, canonical_value in zip(legacy_stats, canonical_stats):
        np.testing.assert_array_equal(legacy_value, canonical_value)

    target_means, target_stds = normalization._fit_target_stats(targets)
    legacy_scaled = training._transform_targets(targets, target_means, target_stds)
    canonical_scaled = normalization._transform_targets(targets, target_means, target_stds)
    np.testing.assert_array_equal(legacy_scaled, canonical_scaled)
    np.testing.assert_allclose(
        normalization._inverse_transform_targets(canonical_scaled, target_means, target_stds),
        targets,
        rtol=1e-6,
        atol=1e-6,
    )


def test_loss_and_loader_results_match():
    predictions = torch.tensor([[1.0, -1.0], [3.0, 2.0]])
    targets = torch.tensor([[0.0, 1.0], [2.0, 4.0]])
    weights = torch.tensor([1.0, 2.0])
    legacy_loss = training.regression_loss(
        predictions, targets, target_loss_weights=weights, loss_type="huber", huber_delta=1.5
    )
    canonical_loss = losses.regression_loss(
        predictions, targets, target_loss_weights=weights, loss_type="huber", huber_delta=1.5
    )
    torch.testing.assert_close(legacy_loss, canonical_loss, rtol=0.0, atol=0.0)

    feature_array = np.arange(12, dtype=np.float32).reshape(6, 2)
    target_array = np.arange(6, dtype=np.float32).reshape(6, 1)
    kwargs = {"batch_size": 4, "shuffle": False, "num_workers": 0, "pin_memory": False}
    legacy_batches = list(training._make_loader(feature_array, target_array, **kwargs))
    canonical_batches = list(loaders._make_loader(feature_array, target_array, **kwargs))
    assert len(legacy_batches) == len(canonical_batches) == 2
    for legacy_batch, canonical_batch in zip(legacy_batches, canonical_batches):
        for legacy_tensor, canonical_tensor in zip(legacy_batch, canonical_batch):
            torch.testing.assert_close(legacy_tensor, canonical_tensor, rtol=0.0, atol=0.0)


def test_existing_bundle_checkpoint_schema_round_trip_is_unchanged(tmp_path):
    bundle = training._to_serializable_bundle(
        {"model_type": "mlp", "feature_means": np.array([1.0, 2.0]), "metadata": {"fold": 0}}
    )
    checkpoint_path = tmp_path / "bundle.pt"
    torch.save(bundle, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert training._to_serializable_bundle is bundles._to_serializable_bundle
    assert loaded.keys() == bundle.keys()
    torch.testing.assert_close(loaded["feature_means"], bundle["feature_means"])
    assert loaded["metadata"] == bundle["metadata"]
