"""Exact interval features from two causal exponential prefix integrals of native ZOH commands."""
import torch


def prefix_grid(pooled, time):
    return (torch.cat((torch.zeros_like(pooled[:, :1]), pooled), dim=1),
            torch.cat((torch.zeros_like(time[:, :1]), time), dim=1))


def interval(pooled, time, u0, start, end):
    """Return actual duration and 20 control features, one timeline for each native observation stream."""
    dt = time[:, end] - time[:, start]
    tau = dt.new_tensor([0.2, 1.0])
    decay = torch.exp(-dt.unsqueeze(-1) / tau)
    mass = -torch.expm1(-dt.unsqueeze(-1) / tau)
    p_end = pooled[:, end].unflatten(-1, (2, 8))
    p_start = pooled[:, start].unflatten(-1, (2, 8))
    avg = ((p_end - decay.unsqueeze(-1) * p_start) / mass.unsqueeze(-1)).flatten(-2)
    held = torch.cat((u0, torch.tanh(2 * u0)), dim=-1).repeat(1, 2)
    return dt.unsqueeze(-1), torch.cat((u0[:, None].expand(-1, 2, -1), avg - held[:, None]), dim=-1)
