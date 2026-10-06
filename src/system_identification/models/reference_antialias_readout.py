"""Causal 100Hz latent-rate -> 50Hz observation readout, with known history.

21-tap Hamming-window sinc, 15Hz nominal cutoff, 100ms linear-phase delay.
The latent physical rate still drives current attitude integration. This delayed
readout must never be substituted for that instantaneous latent rate.
"""
import math
import torch


def antialias_taps(dtype=torch.float32,device=None):
    k=torch.arange(21,dtype=dtype,device=device)
    centered=k-10
    taps=.3*torch.sinc(.3*centered)*(.54-.46*torch.cos(2*math.pi*k/20))
    return taps/taps.sum()


def antialias_rate_readout(history_rates,future_rates,stride=2):
    if history_rates.ndim!=3 or future_rates.ndim!=3 or history_rates.shape[1]<20:
        raise ValueError('20 observed 100Hz rate samples required for filter state')
    if stride not in [1,2]:raise ValueError('readout stride must be1 or2')
    values=torch.cat([history_rates[:,-20:],future_rates],dim=1)
    taps=antialias_taps(values.dtype,values.device).flip(0)
    filtered=(values.unfold(1,21,1)*taps).sum(-1)
    return filtered[:,stride-1::stride]
