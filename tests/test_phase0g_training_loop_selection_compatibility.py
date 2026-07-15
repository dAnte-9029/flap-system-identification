from __future__ import annotations

import copy
import math

import numpy as np
import torch

import system_identification.training as training
import system_identification.training.early_stopping as early_stopping
import system_identification.training.loop as loop
import system_identification.training.selection as selection


MODULE_SYMBOLS = {
    loop: (
        "_normalized_lr_scheduler",
        "_warmup_cosine_lr_lambda",
        "_ema_update_state",
        "_train_scaled_epoch",
        "_train_sequence_scaled_epoch",
        "_train_rollout_scaled_epoch",
        "_evaluate_scaled_loss",
        "_evaluate_sequence_scaled_loss",
        "_evaluate_rollout_scaled_loss",
    ),
    early_stopping: ("EarlyStoppingState", "update_early_stopping"),
    selection: ("BestEpochSelection", "update_best_epoch_selection"),
}


def _legacy_reference_train_epoch(model, loader, optimizer, scaler, weights):
    model.train()
    train_loss_sum = 0.0
    train_sample_count = 0
    for batch_features, batch_targets in loader:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch_features)
        loss = training.regression_loss(
            predictions,
            batch_targets,
            target_loss_weights=weights,
            loss_type="mse",
            huber_delta=1.0,
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_count = len(batch_features)
        train_loss_sum += float(loss.item()) * batch_count
        train_sample_count += batch_count
    return train_loss_sum / max(train_sample_count, 1)


def test_training_reexports_all_canonical_phase0g_objects():
    assert sum(len(symbols) for symbols in MODULE_SYMBOLS.values()) == 13
    for module, symbols in MODULE_SYMBOLS.items():
        for symbol in symbols:
            assert getattr(training, symbol) is getattr(module, symbol)


def test_early_stopping_and_validation_selection_match_legacy_rule():
    scores = [1.0, 1.0 - 5e-9, 0.9, 0.91, 0.92]
    state_dict = {"weight": torch.tensor([0.0])}
    best = selection.BestEpochSelection(copy.deepcopy(state_dict), math.inf, 0)
    patience = early_stopping.EarlyStoppingState()

    legacy_best_loss = math.inf
    legacy_best_epoch = 0
    legacy_epochs_without_improvement = 0
    legacy_stop_epoch = None
    stop_epoch = None

    for epoch, val_loss in enumerate(scores, start=1):
        state_dict["weight"].fill_(float(epoch))
        best, improved = selection.update_best_epoch_selection(
            best, val_loss=val_loss, epoch=epoch, state_dict=state_dict
        )
        patience, should_stop = early_stopping.update_early_stopping(
            patience, improved=improved, patience=2
        )

        if val_loss < legacy_best_loss - 1e-8:
            legacy_best_loss = val_loss
            legacy_best_epoch = epoch
            legacy_epochs_without_improvement = 0
        else:
            legacy_epochs_without_improvement += 1
            if legacy_epochs_without_improvement >= 2:
                legacy_stop_epoch = epoch

        if should_stop:
            stop_epoch = epoch
            break

    assert best.val_loss == legacy_best_loss
    assert best.epoch == legacy_best_epoch == 3
    assert stop_epoch == legacy_stop_epoch == 5
    assert best.state_dict["weight"].item() == 3.0


def test_tiny_cpu_train_epoch_matches_pre_migration_loop():
    features = np.array([[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0], [1.5, 0.0]], dtype=np.float32)
    targets = np.array([[0.5], [1.0], [-1.0], [0.25]], dtype=np.float32)
    loader = training._make_loader(
        features,
        targets,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    weights = torch.ones(1)

    torch.manual_seed(17)
    reference_model = torch.nn.Linear(2, 1)
    canonical_model = copy.deepcopy(reference_model)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-2, weight_decay=1e-5)
    canonical_optimizer = torch.optim.AdamW(canonical_model.parameters(), lr=1e-2, weight_decay=1e-5)
    reference_scaler = torch.amp.GradScaler("cuda", enabled=False)
    canonical_scaler = torch.amp.GradScaler("cuda", enabled=False)

    reference_loss = _legacy_reference_train_epoch(
        reference_model, loader, reference_optimizer, reference_scaler, weights
    )
    canonical_loss = loop._train_scaled_epoch(
        canonical_model,
        loader,
        canonical_optimizer,
        canonical_scaler,
        torch.device("cpu"),
        use_amp=False,
        target_loss_weights=weights,
        loss_type="mse",
        huber_delta=1.0,
    )

    assert canonical_loss == reference_loss
    for reference_parameter, canonical_parameter in zip(reference_model.parameters(), canonical_model.parameters()):
        torch.testing.assert_close(reference_parameter, canonical_parameter, rtol=0.0, atol=0.0)
    assert canonical_optimizer.state_dict()["state"].keys() == reference_optimizer.state_dict()["state"].keys()


def test_tiny_cpu_validation_step_preserves_loss_and_parameters():
    features = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    targets = np.array([[0.25], [-0.75]], dtype=np.float32)
    loader = training._make_loader(
        features,
        targets,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = torch.nn.Linear(2, 1)
    before = copy.deepcopy(model.state_dict())
    weights = torch.ones(1)

    expected_sum = 0.0
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            expected_sum += float(
                training.regression_loss(
                    model(batch_features),
                    batch_targets,
                    target_loss_weights=weights,
                    loss_type="mse",
                    huber_delta=1.0,
                ).item()
            )
    actual = training._evaluate_scaled_loss(
        model,
        loader,
        torch.device("cpu"),
        use_amp=False,
        target_loss_weights=weights,
        loss_type="mse",
        huber_delta=1.0,
    )

    assert actual == expected_sum / len(features)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_warmup_cosine_and_ema_legacy_paths_share_canonical_behavior():
    legacy_schedule = training._warmup_cosine_lr_lambda(warmup_steps=2, total_steps=6)
    canonical_schedule = loop._warmup_cosine_lr_lambda(warmup_steps=2, total_steps=6)
    assert [legacy_schedule(step) for step in range(6)] == [canonical_schedule(step) for step in range(6)]

    model = torch.nn.Linear(1, 1)
    ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(1.0)
    training._ema_update_state(ema_state, model, 0.5)
    for name, value in model.state_dict().items():
        expected = value if not torch.is_floating_point(value) else value - 0.5
        torch.testing.assert_close(ema_state[name], expected)
