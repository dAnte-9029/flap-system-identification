"""Input-conditioned continuous latent dynamics with a triangular nonlinear vector field."""
import torch
from torch import nn


class ControlledODE(nn.Module):
    def __init__(self, mean, scale, latent=24, width=48):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.encoder = nn.Sequential(nn.Linear(len(mean), 96), nn.SiLU(), nn.Linear(96, 2*latent+2*width))
        self.command = nn.Linear(4, width)
        self.drive_first = nn.Linear(width, latent)
        self.drive_second = nn.Linear(width, latent)
        self.state_field = nn.Linear(latent, width, bias=False)
        self.field_out = nn.Linear(width, latent)
        self.rate_logits = nn.Parameter(torch.zeros(2,latent))
        self.readout_linear = nn.Linear(2*latent, 6, bias=False)
        self.readout = nn.Sequential(nn.Linear(2*latent, 48), nn.Tanh(), nn.Linear(48, 6, bias=False))
        self.sizes = (latent, latent, width, width)

    @staticmethod
    def constant_drive(initial, drive, end, dt, rates):
        # Integrate exp(-rate*(t-s))*drive(s). Exp is bounded by exp(3*2.1).
        decay = torch.exp(-end*rates)
        integral = torch.exp((end-dt)*rates) * torch.expm1(dt*rates) / rates
        return decay*(initial[:,None] + torch.cumsum(integral*drive, dim=1))

    def observe(self, state):
        return self.readout_linear(state) + self.readout(state)

    def forward(self, hist, command, end, dt, query):
        z0,y0,bias,gate=self.encoder((hist-self.mean)/self.scale).split(self.sizes,dim=-1)
        command_feature=self.command(command)*(1+torch.tanh(gate[:,None]))+bias[:,None]
        command_feature=torch.tanh(command_feature)
        rate_z,rate_y=(.08+2.92*torch.sigmoid(self.rate_logits)).unbind()
        dz=self.drive_first(command_feature)
        end=end.unsqueeze(-1); dt=dt.unsqueeze(-1)
        z=self.constant_drive(z0,dz,end,dt,rate_z)
        start=torch.cat((z0[:,None],z[:,:-1]),1)
        half=dt*.5
        mid=torch.exp(-half*rate_z)*start+(-torch.expm1(-half*rate_z))/rate_z*dz
        dy=self.drive_second(command_feature)+self.field_out(torch.tanh(self.state_field(mid)+command_feature))
        y=self.constant_drive(y0,dy,end,dt,rate_y)
        all_states=torch.cat((z,y),-1)
        rows=torch.arange(len(hist),device=hist.device)[:,None,None]
        queried=all_states[rows,query]
        delta=self.observe(queried)-self.observe(torch.cat((z0,y0),-1))[:,None,None]
        return torch.stack((delta[:,:,0,:3],delta[:,:,1,3:]),dim=2)
