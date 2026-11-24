import torch
from torch import nn


class FlightPredictor(nn.Module):
    """History-dependent nominal motion and a separate forced, relaxing latent response."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean',mean)
        self.register_buffer('std',std)
        self.encoder = nn.Sequential(nn.Linear(len(mean),64),nn.SiLU(),nn.Linear(64,48),nn.SiLU())
        self.nominal = nn.Linear(48,6*3)
        self.coupling = nn.Linear(48,6*16)
        self.register_buffer('tau',torch.tensor([.2,1.]))

    def command_state(self, pooled, u0, time):
        return pooled

    def forward(self,hist,pooled,u0,time):
        context = self.encoder((hist-self.mean)/self.std)
        n = len(hist)
        nominal = self.nominal(context).reshape(n,2,3,3)
        coupling = self.coupling(context).reshape(n,2,3,16)
        # Queries alternate native velocity and native rate timestamps at each horizon.
        t = time.reshape(n,3,2)
        basis = torch.stack((t, -torch.expm1(-t/.2), -torch.expm1(-t)),dim=-1)
        slow = torch.einsum('nhcb,ncdb->nhcd',basis,nominal)
        z = self.command_state(pooled,u0,time).reshape(n,3,2,16)
        forced = torch.einsum('nhcb,ncdb->nhcd',z,coupling)
        return (slow+forced).reshape(n,6,3)
