"""V2 slow branch: a control-driven latent state with a linear acceleration readout."""
import argparse
import math
from pathlib import Path

import torch
from torch import nn

from hybrid import HISTORY, TAPES, TransformerHarmonics, conditional_data, run, shared_loss


class LatentTransformerHarmonics(TransformerHarmonics):
    latent_width = 24

    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        width = self.latent_width
        self.latent_init = nn.Linear(self.backbone.width, width)
        self.latent_context = nn.Linear(self.backbone.width + 6, width)
        self.latent_command = nn.Linear(4, width, bias=False)
        self.latent_feedback = nn.Linear(width, width, bias=False)
        self.latent_target = nn.Linear(width, width)
        # Initial response times range from one tenth of a second to one second.
        self.log_rate = nn.Parameter(torch.linspace(0., math.log(10.), width))
        self.trend = nn.Linear(width, 6)

    def initialize_dynamics(self, normalized_history, steps):
        history, cache = super().initialize_dynamics(normalized_history, steps)
        return history, (cache, self.latent_init(history).tanh())

    def step_dynamics(self, context, phase_basis, memory):
        cache, z = memory
        hidden, cache = super().step_dynamics(context, phase_basis, cache)
        return hidden, (cache, z)

    def advance_latent(self, hidden, body, commands, dt, memory):
        """One Euler latent step per native ZOH interval, including identity on padding."""
        cache, z = memory
        context = self.latent_context(torch.cat((hidden, body), -1))[:, None]
        drive = context + self.latent_command(commands)
        gain = self.log_rate.exp()
        starts, slopes = [], []
        for forcing, elapsed in zip(drive.unbind(1), dt.unbind(1)):
            target = self.latent_target(torch.nn.functional.silu(forcing + self.latent_feedback(z)))
            dz = gain * (target - z)
            starts.append(z)
            slopes.append(dz)
            z = z + elapsed * dz
        return torch.stack(starts, 1), torch.stack(slopes, 1), (cache, z)

    def segment_slow_rates(self, hidden, body, commands, change, dt, memory):
        starts, slopes, memory = self.advance_latent(hidden, body, commands, dt, memory)
        # Within an interval z(t)=z_start+t*dz. Linear readout gives an affine
        # acceleration, so an interior query can use its own elapsed time exactly.
        base = self.trend(starts) + self.control_rate(change)
        jerk = torch.nn.functional.linear(slopes, self.trend.weight)
        return base, jerk, memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.set_defaults(model='transformer_slowfast_latent_v2', seed=17, epochs=24, batch=1024, cache=HISTORY)
    args = parser.parse_args()
    run(args, load_data=lambda cache, split: conditional_data(cache, split, TAPES),
        build_model=LatentTransformerHarmonics, loss_fn=shared_loss,
        extra_config=dict(macro_dt=.02, aux_weight=0, control_tapes=str(TAPES),
                          latent_width=LatentTransformerHarmonics.latent_width,
                          latent_update='Euler per native command segment with linear acceleration readout',
                          fast_harmonics=[1, 2], envelope_rate_per_s=2.,
                          task='logged ZOH commands; autonomous physical state and phase'))
