"""20 ms causal Transformer with slow-motion feedback and two phase harmonics."""
import argparse
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'controlled_long_horizon_stage1'))
from common import HISTORY, TAPES, conditional_data, trajectory_losses
from harmonic import SegmentHarmonics
from stage_query import query
from train import run


class TransformerHarmonics(SegmentHarmonics):
    def __init__(self, mean, scale):
        # Initialize all shared weights exactly as the archived Transformer does.
        super().__init__(mean, scale, structured=False, backbone='transformer')
        self.structured = True
        width = self.backbone.width
        self.initial_amplitude = nn.Linear(width, 24)
        self.amplitude = nn.Linear(width + 24, 24)
        self.control_amplitude = nn.Linear(4, 24, bias=False)
        for head in (self.initial_amplitude, self.amplitude, self.control_amplitude):
            nn.init.normal_(head.weight, std=.001)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def initialize_dynamics(self, normalized_history, steps):
        last, memory = self.backbone.encode_history(normalized_history, steps)
        return self.backbone.final_norm(last), memory

    def step_dynamics(self, context, phase_basis, memory):
        # Same 22 token channels as the Transformer. Motion channels carry the slow state.
        return self.backbone.step(torch.cat((context, phase_basis), dim=-1), memory)


def shared_loss(model, batch, units):
    history, initial, target, times, valid, beat, endpoints, tape = batch
    roll = model(history, initial, tape)
    pv = query(roll, times[:, :, 0], 'pv')
    q = query(roll, times[:, :, 1], 'q')
    w = query(roll, times[:, :, 2], 'w')
    return trajectory_losses(pv, q, w, target, valid, beat, endpoints, units)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.set_defaults(model='transformer_slowfast_v1', seed=17, epochs=24, batch=1024, cache=HISTORY)
    args = parser.parse_args()
    run(args, load_data=lambda cache, split: conditional_data(cache, split, TAPES),
        build_model=TransformerHarmonics, loss_fn=shared_loss,
        extra_config=dict(macro_dt=.02, aux_weight=0, control_tapes=str(TAPES),
                          fast_harmonics=[1, 2], envelope_rate_per_s=2.,
                          task='logged ZOH commands; autonomous physical state and phase'))
