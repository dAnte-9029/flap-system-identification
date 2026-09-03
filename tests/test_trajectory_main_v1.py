from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from system_identification.models.trajectory import InitialTrajectoryState
from system_identification.models.trajectory_main_v1 import (
    CausalHistoryTrajectoryModel,
    TorchTrajectoryPrediction,
)
from system_identification.training.trajectory_main_v1 import (
    assemble_history_trajectory_windows,
    trajectory_rollout_loss,
)


def _samples() -> pd.DataFrame:
    rows = []
    for segment_id, velocity_offset in ((0, 0.0), (1, 10.0)):
        for sample in range(6):
            phase = 0.25 * sample + 1.7 * segment_id
            rows.append(
                {
                    "log_id": "log_a",
                    "segment_id": segment_id,
                    "sample_in_segment": sample,
                    "timestamp_us": (segment_id * 100 + sample) * 20_000,
                    "valid_core": True,
                    "position_ned_m_x": float(sample),
                    "position_ned_m_y": 0.0,
                    "position_ned_m_z": 0.0,
                    "velocity_ned_m_s_x": velocity_offset + sample,
                    "velocity_ned_m_s_y": 0.0,
                    "velocity_ned_m_s_z": 0.0,
                    "attitude_q_w": 1.0,
                    "attitude_q_x": 0.0,
                    "attitude_q_y": 0.0,
                    "attitude_q_z": 0.0,
                    "angular_velocity_body_rad_s_x": 0.0,
                    "angular_velocity_body_rad_s_y": 0.0,
                    "angular_velocity_body_rad_s_z": 0.1 * sample,
                    "relative_flap_phase_rad": phase,
                    "flap_frequency_hz": 4.0,
                    "control_flap_motor_normalized": 0.5 + 0.01 * sample,
                    "control_left_elevon_normalized": 0.1,
                    "control_right_elevon_normalized": -0.1,
                    "control_rudder_normalized": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _windows() -> pd.DataFrame:
    rows = []
    for segment_id, start in ((0, 0), (1, 1)):
        rows.append(
            {
                "window_id": f"w{segment_id}",
                "log_id": "log_a",
                "segment_id": segment_id,
                "start_sample_in_segment": start,
                "state_sample_count": 3,
            }
        )
    return pd.DataFrame(rows)


def test_history_windows_are_causal_offset_invariant_and_mask_left_padding() -> None:
    batch = assemble_history_trajectory_windows(_samples(), _windows(), history_steps=3)

    assert batch.history_state_features.shape == (2, 3, 12)
    assert batch.history_controls.shape == (2, 3, 4)
    np.testing.assert_array_equal(batch.history_mask, [[False, False, True], [False, True, True]])
    np.testing.assert_allclose(batch.history_state_features[:, -1, 9:11], [[0.0, 1.0], [0.0, 1.0]])
    assert batch.history_state_features[1, -1, 0] == 11.0
    assert batch.trajectory.window_ids.tolist() == ["w0", "w1"]


def _initial(batch_size: int = 2) -> InitialTrajectoryState:
    return InitialTrajectoryState(
        position_n=np.zeros((batch_size, 3)),
        velocity_n=np.tile([1.0, 0.0, 0.0], (batch_size, 1)),
        quaternion_nb=np.tile([1.0, 0.0, 0.0, 0.0], (batch_size, 1)),
        angular_velocity_b=np.zeros((batch_size, 3)),
        relative_phase_rad=np.zeros(batch_size),
        flap_frequency_hz=np.full(batch_size, 4.0),
    )


def _model(*, use_controls: bool) -> CausalHistoryTrajectoryModel:
    torch.manual_seed(3)
    return CausalHistoryTrajectoryModel(
        hidden_size=8,
        use_controls=use_controls,
        feature_mean=np.zeros(12),
        feature_std=np.ones(12),
        control_mean=np.zeros(4),
        control_std=np.ones(4),
        derivative_mean=np.zeros(7),
        derivative_std=np.ones(7),
    )


def _rollout(model: CausalHistoryTrajectoryModel, controls: torch.Tensor):
    batch_size, steps, _ = controls.shape
    initial = _initial(batch_size)
    return model(
        history_state_features=torch.zeros((batch_size, 3, 12)),
        history_controls=torch.zeros((batch_size, 3, 4)),
        history_mask=torch.ones((batch_size, 3), dtype=torch.bool),
        position_n=torch.as_tensor(initial.position_n, dtype=torch.float32),
        velocity_n=torch.as_tensor(initial.velocity_n, dtype=torch.float32),
        quaternion_nb=torch.as_tensor(initial.quaternion_nb, dtype=torch.float32),
        angular_velocity_b=torch.as_tensor(initial.angular_velocity_b, dtype=torch.float32),
        relative_phase_rad=torch.as_tensor(initial.relative_phase_rad, dtype=torch.float32),
        flap_frequency_hz=torch.as_tensor(initial.flap_frequency_hz, dtype=torch.float32),
        future_controls=controls,
        dt_s=torch.full((batch_size, steps), 0.02),
    )


def test_no_control_model_is_invariant_to_future_control_tape() -> None:
    model = _model(use_controls=False).eval()
    zeros = _rollout(model, torch.zeros((2, 4, 4)))
    changed = _rollout(model, torch.ones((2, 4, 4)))

    for first, second in zip(zeros, changed, strict=True):
        torch.testing.assert_close(first, second)


def test_main_model_rollout_is_finite_normalized_and_has_expected_shape() -> None:
    prediction = _rollout(_model(use_controls=True).eval(), torch.zeros((2, 4, 4)))

    assert prediction.position_n.shape == (2, 5, 3)
    assert prediction.quaternion_nb.shape == (2, 5, 4)
    assert torch.isfinite(prediction.position_n).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction.quaternion_nb, dim=-1), torch.ones((2, 5))
    )


def test_multistep_objective_penalizes_late_rollout_error() -> None:
    zeros3 = torch.zeros((1, 4, 3))
    zeros4 = torch.zeros((1, 4, 4))
    zeros1 = torch.zeros((1, 4))
    truth = TorchTrajectoryPrediction(
        position_n=zeros3.clone(),
        velocity_n=zeros3.clone(),
        quaternion_nb=zeros4.clone(),
        angular_velocity_b=zeros3.clone(),
        relative_phase_rad=zeros1.clone(),
        flap_frequency_hz=zeros1.clone(),
    )
    truth.quaternion_nb[..., 0] = 1.0
    prediction = TorchTrajectoryPrediction(**{name: value.clone() for name, value in vars(truth).items()})
    prediction.position_n[:, 2:, 0] = 1.0

    local = trajectory_rollout_loss(prediction, truth, objective_steps=1)
    multistep = trajectory_rollout_loss(prediction, truth, objective_steps=3)

    assert local.item() == 0.0
    assert multistep.item() > 0.0
