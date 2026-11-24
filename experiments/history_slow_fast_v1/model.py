"""History-only multi-rate GRU. All rollout clocks and inputs are independent of labels."""
import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from system_identification.models.velocity_rate_gru import geodesic, integrate, qexp, qmul, unit


def inverse(q):
    return torch.cat([q[...,:1],-q[...,1:]],-1)


def rotate(q, v):
    qv,v = torch.broadcast_tensors(q[...,1:],v)
    cross = 2*torch.linalg.cross(qv,v)
    return v+q[...,:1]*cross+torch.linalg.cross(qv,cross)


def product5(q):
    return qmul(qmul(qmul(q[:,0],q[:,1]),qmul(q[:,2],q[:,3])),q[:,4])


def prefix_rotations(q):
    # Inclusive, time-ordered parallel scan of body-frame rotation increments.
    stride = 1
    while stride<q.shape[1]:
        q = torch.cat([q[:,:stride],qmul(q[:,:-stride],q[:,stride:])],1)
        stride *= 2
    return unit(q)


class Predictor(nn.Module):
    def __init__(self, structured, mean, scale):
        super().__init__()
        self.structured = structured
        self.encoder = nn.GRU(len(mean),64,batch_first=True)
        self.init_hidden = nn.Linear(64,48)
        self.cell = nn.GRUCell(6+3+4+1,48)
        if structured:
            self.initial_amplitude = nn.Linear(64,24)
            self.amplitude = nn.Linear(48+24,24)
            self.trend = nn.Linear(48,6)
            outputs = [self.trend,self.initial_amplitude,self.amplitude]
        else:
            self.readout = nn.Sequential(nn.Linear(48+4,61),nn.SiLU(),nn.Linear(61,6))
            outputs = [self.readout[-1]]
        for head in outputs:
            nn.init.normal_(head.weight,std=.001)
            nn.init.zeros_(head.bias)
        self.register_buffer('mean',mean)
        self.register_buffer('scale',scale)
        self.register_buffer('substeps',torch.arange(1,6,dtype=mean.dtype).view(1,5,1))
        self.register_buffer('down',torch.tensor([0.,0.,1.]))
        self.register_buffer('axis_scale',scale[:6].clamp_min(.1))

    def forward(self, history, initial):
        _,encoded = self.encoder((history-self.mean)/self.scale)
        encoded = encoded[0]
        h = torch.tanh(self.init_hidden(encoded))
        last = history[:,-1]
        controls = last[:,9:13]
        freq = last[:,15:16]
        phase_ok = last[:,21:22]*last[:,22:23]
        quat = initial[:,6:10]
        w = initial[:,10:13]
        total = torch.cat([initial[:,3:6],w],-1)
        angle = 2*math.pi*.02*freq[:,None,:]*torch.arange(106,device=h.device).view(1,-1,1)
        basis = phase_ok[:,None,:]*torch.cat([angle.cos(),angle.sin(),(2*angle).cos(),(2*angle).sin()],-1)

        def residual(coeff,carrier,q):
            body = (coeff.view(-1,1,6,4)*carrier[:,:,None,:]).sum(-1)*self.axis_scale
            return torch.cat([rotate(q[:,None,:],body[...,:3]),body[...,3:]],-1)

        if self.structured:
            coeff = self.initial_amplitude(encoded)
            slow = total-residual(coeff,basis[:,:1],quat)[:,0]
        else:
            slow = total
        total_path,slow_path,rotations = [total[:,None]],[slow[:,None]],[]
        for k in range(21):
            body = rotate(inverse(quat),slow[:,:3])
            cell_input = torch.cat([torch.cat([body,slow[:,3:]],-1)/self.axis_scale,
                                    rotate(inverse(quat),self.down),controls,freq/5],-1)
            h = self.cell(cell_input,h)
            carrier = basis[:,1+k*5:6+k*5]
            if self.structured:
                delta = self.trend(h)*self.axis_scale
                delta = torch.cat([rotate(quat,delta[:,:3]),delta[:,3:]],-1)
                trend = slow[:,None]+.02*self.substeps*delta[:,None]
                next_coeff = .8*coeff+.2*self.amplitude(torch.cat([h,coeff],-1))
                # Continuous envelope at macro boundaries, interpolated over five substeps.
                b0 = residual(coeff,carrier,quat)
                b1 = residual(next_coeff,carrier,quat)
                path = trend + b0 + .2*self.substeps*(b1-b0)
                coeff = next_coeff
            else:
                ff = torch.cat([h[:,None].expand(-1,5,-1),carrier],-1)
                delta = self.readout(ff)*self.axis_scale
                delta = torch.cat([rotate(quat[:,None],delta[...,:3]),delta[...,3:]],-1)
                path = slow[:,None]+.02*delta.cumsum(1)
                trend = path
            rates = path[:,:,3:]
            prev = torch.cat([w[:,None],rates[:,:-1]],1)
            dq = qexp(.01*(prev+rates))
            quat = unit(qmul(quat,product5(dq)))
            w,slow = rates[:,-1],trend[:,-1]
            total_path.append(path)
            slow_path.append(trend)
            rotations.append(dq)
        motion = torch.cat(total_path,1)
        rel = prefix_rotations(torch.cat(rotations,1))
        q0 = initial[:,None,6:10]
        attitudes = torch.cat([q0,unit(qmul(q0,rel))],1)
        pos = .01*(motion[:,:-1,:3]+motion[:,1:,:3]).cumsum(1)
        position = initial[:,None,:3]+torch.cat([torch.zeros_like(pos[:,:1]),pos],1)
        return torch.cat([position,motion[...,:3],attitudes,motion[...,3:]],-1),torch.cat(slow_path,1)


class StandardGRU(nn.Module):
    """Single recurrent state, one feedback update and one linear readout per 20 ms."""
    structured = False

    def __init__(self, mean, scale):
        super().__init__()
        self.encoder = nn.GRU(len(mean), 64, batch_first=True)
        # 54 makes the active parameter budget close to the two 34k reference models.
        self.init_hidden = nn.Linear(64, 54)
        self.cell = nn.GRUCell(6 + 3 + 4 + 1 + 4, 54)
        self.readout = nn.Linear(54, 6)
        nn.init.normal_(self.readout.weight, std=.001)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('axis_scale', scale[:6].clamp_min(.1))
        self.register_buffer('down', torch.tensor([0., 0., 1.]))
        self.register_buffer('query_times', .02 * torch.arange(1, 106, dtype=mean.dtype))

    def forward(self, history, initial):
        _, encoded = self.encoder((history - self.mean) / self.scale)
        hidden = torch.tanh(self.init_hidden(encoded[0]))
        last = history[:, -1]
        controls, freq = last[:, 9:13], last[:, 15:16]
        phase_valid = last[:, 21:22] * last[:, 22:23]
        angle = 2 * math.pi * freq * self.query_times
        phase = torch.stack([angle.cos(), angle.sin(), (2*angle).cos(), (2*angle).sin()], -1)
        phase = phase * phase_valid.unsqueeze(-1)
        state = initial
        path = [initial]
        dt = initial.new_full((len(initial),), .02)
        for phase_at_step in phase.unbind(1):
            quat = state[:, 6:10]
            velocity, rate = state[:, 3:6], state[:, 10:]
            body_state = torch.cat([rotate(inverse(quat), velocity), rate], -1)
            features = torch.cat([body_state / self.axis_scale, rotate(inverse(quat), self.down),
                                  controls, freq / 5., phase_at_step], -1)
            hidden = self.cell(features, hidden)
            increment = self.readout(hidden) * self.axis_scale * .02
            state = integrate(state, velocity + rotate(quat, increment[:, :3]),
                              rate + increment[:, 3:], dt)
            path.append(state)
        return torch.stack(path, 1), None


def at_time(series,t):
    step = (t/.02).clamp(0,series.shape[1]-1)
    i = step.long().clamp(max=series.shape[1]-2)
    def take(j):
        return series.gather(1,j[...,None].expand(-1,-1,series.shape[-1]))
    return torch.lerp(take(i),take(i+1),(step-i)[...,None])


def losses(model,batch,units):
    history,initial,y,t,valid,beat,endpoint = batch
    result,trend = model(history,initial)
    pv = at_time(result[:,:,:6],t[:,:,0])
    q = unit(at_time(result[:,:,6:10],t[:,:,1]))
    w = at_time(result[:,:,10:],t[:,:,2])
    slow_v = slow_w = None
    if model.structured:
        slow_v = at_time(trend[:,:,:3],t[:,:,0])
        slow_w = at_time(trend[:,:,3:],t[:,:,2])
    return trajectory_losses(pv,q,w,y,valid,beat,endpoint,units,slow_v,slow_w)


def trajectory_losses(pv,q,w,y,valid,beat,endpoint,units,slow_v=None,slow_w=None):
    def mse(a):
        return a.square().sum(-1)
    sq = torch.stack([mse(pv[:,:,:3]-y[:,:,:3]),mse(pv[:,:,3:]-y[:,:,3:6]),
                      geodesic(q,y[:,:,6:10]).square(),mse(w-y[:,:,10:])],-1)
    mask = torch.cat([valid[:,:,:1],valid],-1)
    point = ((sq/units.square())*mask).sum(1)/mask.sum(1).clamp_min(1)
    # endpoint contains a native target index per horizon and physical metric, -1 if absent.
    endpoint_sq = sq.gather(1,endpoint.clamp_min(0))
    ea = endpoint>=0
    ep = (endpoint_sq*ea/units.square()).sum(1)/ea.sum(1).clamp_min(1)
    shared = (point[:,1]+point[:,3]+ep[:,0]+ep[:,2])*.25
    loss = shared.mean()
    if slow_v is not None:
        diffs = torch.stack([slow_v-y[:,:,3:6],slow_w-y[:,:,10:]],-2)
        duration = beat.sum(1)
        avg = (diffs*beat[...,None]).sum(1)/duration.clamp_min(1e-6)[...,None]
        eligible = duration>.05
        aux = (avg.square().sum(-1)/units[1::2].square()*eligible).sum()/eligible.sum().clamp_min(1)
        loss = loss+.25*aux
    return loss,shared,endpoint_sq,ea
