"""Inference adapter for v5's explicit internal grid and output time contracts."""
import torch
from torch import nn
from .reference_antialias_readout import antialias_rate_readout


class ReferenceBandwidthPredictor(nn.Module):
    """Wrap an already-loaded latent model; never read files or fit statistics.

    raw100: all outputs at100Hz, rate is the latent instantaneous hypothesis.
    aa50: pose/velocity at50Hz current forecast time, rate is the causal FIR
    observation with100ms group delay. The raw rate is never silently exposed
    as validated truth or replaced inside the latent attitude integrator.
    """
    def __init__(self,latent_model,readout):
        super().__init__()
        if readout not in ['raw100','aa50']:raise ValueError('unknown bandwidth readout')
        self.latent_model=latent_model
        self.readout=readout

    def forward(self,**inputs):
        history=inputs['history_state']
        if history.shape[1]!=51:raise ValueError('51 history samples at100Hz required')
        if inputs['future_reference'].shape[1]%2:raise ValueError('even internal forecast steps required')
        if not torch.allclose(inputs['dt_s'],torch.full_like(inputs['dt_s'],.01),atol=1e-7,rtol=0):
            raise ValueError('internal prediction dt must be0.01s')
        raw=self.latent_model(**inputs)
        if self.readout=='raw100':return raw
        return {'rates_b':antialias_rate_readout(history[:,:,3:6],raw['rates_b']),
                'quaternion':raw['quaternion'][:,1::2],
                'velocity_n':raw['velocity_n'][:,1::2]}
