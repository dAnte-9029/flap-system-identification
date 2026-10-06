"""Causal rate readout with an explicit observed filter history.

At 50 Hz, width 10 averages samples t-0.18,...,t: a 0.09 s linear-phase
lag. This readout is not an instantaneous rate estimate or deconvolution.
"""
import torch


def causal_average_rates(history_rates, future_rates, width=10):
    if width < 1 or history_rates.shape[1] < width-1:
        raise ValueError('insufficient observed history for causal rate filter')
    if width == 1:
        return future_rates
    joined=torch.cat([history_rates[:,-(width-1):],future_rates],dim=1)
    return joined.unfold(1,width,1).mean(-1)
