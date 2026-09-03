from __future__ import annotations

import numpy as np
import torch

from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import (
    ActuatorAwareTrajectoryModel,
    causal_first_order_filter,
    transform_tail_controls,
)


def _base_model() -> CausalHistoryTrajectoryModel:
    torch.manual_seed(4)
    return CausalHistoryTrajectoryModel(
        hidden_size=8,
        use_controls=False,
        feature_mean=np.zeros(12),
        feature_std=np.ones(12),
        control_mean=np.zeros(4),
        control_std=np.ones(4),
        derivative_mean=np.zeros(7),
        derivative_std=np.ones(7),
    ).eval()


def _inputs(batch_size: int = 2, steps: int = 4) -> dict[str, torch.Tensor]:
    return {
        "history_state_features": torch.zeros((batch_size, 3, 12)),
        "history_controls": torch.zeros((batch_size, 3, 4)),
        "history_mask": torch.ones((batch_size, 3), dtype=torch.bool),
        "position_n": torch.zeros((batch_size, 3)),
        "velocity_n": torch.tensor([[1.0, 0.0, 0.0]]).repeat(batch_size, 1),
        "quaternion_nb": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch_size, 1),
        "angular_velocity_b": torch.zeros((batch_size, 3)),
        "relative_phase_rad": torch.zeros(batch_size),
        "flap_frequency_hz": torch.full((batch_size,), 4.0),
        "future_controls": torch.zeros((batch_size, steps, 4)),
        "dt_s": torch.full((batch_size, steps), 0.02),
    }


def test_tail_transform_uses_symmetric_differential_and_rudder_channels() -> None:
    controls = torch.tensor([[0.4, 0.6, -0.2, 0.3]])

    transformed = transform_tail_controls(controls)

    torch.testing.assert_close(transformed, torch.tensor([[0.2, 0.4, 0.3]]))


def test_first_order_filter_is_causal_and_respects_mask() -> None:
    commands = torch.tensor([[9.0, 0.0, 1.0, 1.0]])
    mask = torch.tensor([[False, True, True, True]])

    filtered = causal_first_order_filter(commands, mask, dt_s=0.1, tau_s=0.1)

    alpha = 1.0 - np.exp(-1.0)
    expected = alpha + (1.0 - alpha) * alpha
    torch.testing.assert_close(filtered, torch.tensor([expected], dtype=torch.float32))


def test_disabled_actuator_paths_exactly_reproduce_frozen_base() -> None:
    base = _base_model()
    model = ActuatorAwareTrajectoryModel(
        base_model=base,
        use_drive=False,
        use_tail=False,
        gated_tail=False,
        tail_mean=np.zeros(3),
        tail_std=np.ones(3),
    ).eval()
    inputs = _inputs()

    expected = base(**inputs)
    actual = model(**inputs)

    for expected_value, actual_value in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_base_is_frozen_and_gates_begin_near_fallback() -> None:
    model = ActuatorAwareTrajectoryModel(
        base_model=_base_model(),
        use_drive=True,
        use_tail=True,
        gated_tail=True,
        tail_mean=np.zeros(3),
        tail_std=np.ones(3),
        initial_tail_gate=0.05,
    )

    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    gates = model.tail_gate_values()
    torch.testing.assert_close(gates, torch.full((3,), 0.05))
    assert any(parameter.requires_grad for parameter in model.tail_heads.parameters())


def test_actuator_rollout_is_finite_and_does_not_consume_future_state() -> None:
    model = ActuatorAwareTrajectoryModel(
        base_model=_base_model(),
        use_drive=True,
        use_tail=True,
        gated_tail=True,
        tail_mean=np.zeros(3),
        tail_std=np.ones(3),
    ).eval()
    inputs = _inputs()
    inputs["future_controls"][:, :, 0] = 0.8

    prediction = model(**inputs)

    assert prediction.position_n.shape == (2, 5, 3)
    assert torch.isfinite(prediction.position_n).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction.quaternion_nb, dim=-1), torch.ones((2, 5))
    )
