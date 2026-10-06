"""Four-command inference interface with the frozen 40070-parameter GRU."""
import torch

from .reference_gru_unit_gravity import ReferenceGRUUnitGravity


class ActuatorCommandGRU(ReferenceGRUUnitGravity):
    """Post-allocation motor0/servo0/servo1/servo2 conditional dynamics.

    Legacy reference_* buffers normalize commands in this model. The fifth
    slot is structurally zero, including after normalization. History commands
    align with observed states; history_actuator keeps the original one-step
    lag. No controller, allocator, PWM conversion or measured servo state is
    implied by this interface.
    """

    def __init__(self, stats, hidden=64):
        if stats['reference_mean'][4] != 0 or stats['reference_scale'][4] != 1:
            raise ValueError('padding normalization must be mean zero, scale one')
        super().__init__(stats, hidden)

    @staticmethod
    def padded(command):
        if command.shape[-1] != 4:
            raise ValueError('four normalized actuator commands required')
        return torch.cat([command, torch.zeros_like(command[..., :1])], dim=-1)

    def forward(self, history_state, history_command, history_actuator,
                future_command, initial_quaternion, initial_velocity_n,
                initial_rates_b, current_mode, dt_s):
        return super().forward(
            history_state, self.padded(history_command), history_actuator,
            self.padded(future_command), initial_quaternion, initial_velocity_n,
            initial_rates_b, current_mode, dt_s)
