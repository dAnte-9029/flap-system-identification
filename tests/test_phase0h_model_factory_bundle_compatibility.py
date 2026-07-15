from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

import system_identification.models.bundles as model_bundles
import system_identification.training as training
import system_identification.training.bundle_assembly as bundle_assembly
import system_identification.training.history as history
import system_identification.training.model_factory as model_factory
import system_identification.training.optimizer_factory as optimizer_factory


MODULE_SYMBOLS = {
    model_factory: ("_build_regressor", "_build_sequence_regressor", "_build_rollout_regressor"),
    optimizer_factory: ("build_adamw_optimizer", "build_training_scheduler"),
    history: (
        "build_validation_history_row",
        "build_sequence_validation_history_row",
        "build_rollout_validation_history_row",
    ),
    bundle_assembly: (
        "assemble_training_bundle",
        "assemble_sequence_training_bundle",
        "assemble_rollout_training_bundle",
    ),
}


def _validation_metrics():
    return {
        "overall_mae": 1.25,
        "overall_rmse": 1.5,
        "overall_r2": 0.75,
        "per_target": {
            "fx_b": {"mae": 1.0, "rmse": 1.2, "r2": 0.8},
            "fz_b": {"mae": 1.5, "rmse": 1.8, "r2": 0.7},
        },
    }


def test_training_reexports_all_phase0h_objects():
    assert sum(len(symbols) for symbols in MODULE_SYMBOLS.values()) == 11
    for module, symbols in MODULE_SYMBOLS.items():
        for symbol in symbols:
            assert getattr(training, symbol) is getattr(module, symbol)

    assert model_factory._build_regressor is model_bundles._build_regressor
    assert model_factory._build_sequence_regressor is model_bundles._build_sequence_regressor
    assert model_factory._build_rollout_regressor is model_bundles._build_rollout_regressor


def test_model_factory_preserves_type_parameter_schema_and_initial_state():
    kwargs = {
        "model_type": "mlp",
        "input_dim": 3,
        "output_dim": 2,
        "hidden_sizes": (5, 4),
        "dropout": 0.0,
    }
    torch.manual_seed(31)
    legacy_model = training._build_regressor(**kwargs)
    torch.manual_seed(31)
    canonical_model = model_factory._build_regressor(**kwargs)

    assert type(legacy_model) is type(canonical_model)
    assert list(legacy_model.state_dict()) == list(canonical_model.state_dict())
    for name, value in legacy_model.state_dict().items():
        assert value.shape == canonical_model.state_dict()[name].shape
        torch.testing.assert_close(value, canonical_model.state_dict()[name], rtol=0.0, atol=0.0)


def test_optimizer_and_scheduler_factory_preserve_configuration_and_state():
    torch.manual_seed(41)
    reference_model = torch.nn.Linear(2, 1)
    canonical_model = copy.deepcopy(reference_model)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=2e-3, weight_decay=3e-5)
    canonical_optimizer = training.build_adamw_optimizer(
        canonical_model, learning_rate=2e-3, weight_decay=3e-5
    )

    reference_group = reference_optimizer.state_dict()["param_groups"][0]
    canonical_group = canonical_optimizer.state_dict()["param_groups"][0]
    for key in reference_group:
        assert canonical_group[key] == reference_group[key]

    reference_scheduler = torch.optim.lr_scheduler.LambdaLR(
        reference_optimizer,
        lr_lambda=training._warmup_cosine_lr_lambda(warmup_steps=2, total_steps=8),
    )
    canonical_scheduler = optimizer_factory.build_training_scheduler(
        canonical_optimizer,
        scheduler_name="warmup_cosine",
        warmup_steps=2,
        total_steps=8,
    )
    assert canonical_scheduler is not None
    assert canonical_scheduler.state_dict() == reference_scheduler.state_dict()
    assert optimizer_factory.build_training_scheduler(
        canonical_optimizer,
        scheduler_name="none",
        warmup_steps=0,
        total_steps=8,
    ) is None


def test_validation_history_helpers_preserve_schema_values_and_order():
    metrics = _validation_metrics()
    standard = training.build_validation_history_row(
        epoch=2, train_loss=0.5, val_loss=0.75, val_metrics=metrics
    )
    assert list(standard) == [
        "epoch",
        "train_loss",
        "val_loss",
        "val_overall_mae",
        "val_overall_rmse",
        "val_overall_r2",
        "val_fx_b_mae",
        "val_fx_b_rmse",
        "val_fx_b_r2",
        "val_fz_b_mae",
        "val_fz_b_rmse",
        "val_fz_b_r2",
    ]
    assert standard["epoch"] == 2.0
    assert standard["val_fx_b_rmse"] == 1.2

    sequence = history.build_sequence_validation_history_row(
        epoch=2,
        learning_rate=1e-3,
        train_loss=0.5,
        train_supervised_loss=0.4,
        train_prior_loss=0.1,
        val_loss=0.75,
        val_metrics=metrics,
    )
    assert sequence["train_total_loss"] == sequence["train_loss"] == 0.5
    assert list(sequence)[:10] == [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_total_loss",
        "train_supervised_loss",
        "train_prior_loss",
        "val_loss",
        "val_overall_mae",
        "val_overall_rmse",
        "val_overall_r2",
    ]

    rollout = history.build_rollout_validation_history_row(
        epoch=2,
        train_loss=0.5,
        val_loss=0.75,
        val_metrics=metrics,
        latent_rms=0.3,
        delta_latent_rms=0.2,
        latent_derivative_rms=0.1,
    )
    assert list(rollout)[6:9] == ["latent_rms", "delta_latent_rms", "latent_derivative_rms"]


def test_bundle_assembly_preserves_fields_dtype_schema_and_state_dict():
    state_dict = {"layers.0.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
    feature_medians = np.array([1.0, 2.0], dtype=np.float32)
    feature_means = np.array([1.5, 2.5], dtype=np.float32)
    feature_stds = np.array([0.5, 0.75], dtype=np.float32)
    target_means = np.array([3.0], dtype=np.float32)
    target_stds = np.array([2.0], dtype=np.float32)
    target_weights = np.array([1.0], dtype=np.float32)
    train_features = pd.DataFrame({"a": [0.0], "b": [1.0]})
    train_targets = pd.DataFrame({"fx_b": [2.0]})

    bundle = training.assemble_training_bundle(
        amp_enabled=False,
        base_feature_columns=["a", "b"],
        best_epoch=3,
        best_state_dict=state_dict,
        best_val_loss=0.25,
        dropout=0.0,
        feature_means=feature_means,
        feature_medians=feature_medians,
        feature_stds=feature_stds,
        hidden_sizes=(4,),
        history=[{"epoch": 1.0}],
        huber_delta=1.0,
        pfnn_control_points=6,
        pfnn_expanded_input_dim=45,
        pfnn_phase_node_count=5,
        phase_feature_index=None,
        random_seed=7,
        resolved_device=torch.device("cpu"),
        resolved_loss_type="mse",
        resolved_model_type="mlp",
        resolved_window_mode="single",
        target_loss_weights_array=target_weights,
        target_means=target_means,
        target_stds=target_stds,
        train_features_df=train_features,
        train_targets_df=train_targets,
        window_feature_mode="all",
        window_radius=0,
    )

    assert list(bundle) == [
        "model_state_dict",
        "model_type",
        "feature_columns",
        "base_feature_columns",
        "target_columns",
        "feature_medians",
        "feature_means",
        "feature_stds",
        "target_means",
        "target_stds",
        "target_loss_weights",
        "target_loss_weights_by_name",
        "loss_type",
        "huber_delta",
        "window_mode",
        "window_radius",
        "window_feature_mode",
        "window_feature_columns",
        "hidden_sizes",
        "dropout",
        "phase_feature_index",
        "phase_feature_column",
        "pfnn_expanded_input_dim",
        "pfnn_phase_node_count",
        "pfnn_control_points",
        "best_epoch",
        "best_val_loss",
        "history",
        "device_type",
        "use_amp",
        "random_seed",
    ]
    assert bundle["feature_columns"] == ["a", "b"]
    assert bundle["target_columns"] == ["fx_b"]
    assert bundle["feature_means"].dtype == np.float32
    assert bundle["target_stds"].dtype == np.float32
    assert bundle["target_loss_weights_by_name"] == {"fx_b": 1.0}
    assert bundle["model_state_dict"] is state_dict
    torch.testing.assert_close(bundle["model_state_dict"]["layers.0.weight"], state_dict["layers.0.weight"])
