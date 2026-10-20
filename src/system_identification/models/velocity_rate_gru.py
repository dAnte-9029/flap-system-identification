"""Autonomous NED velocity / FRD rate GRU; Hamilton wxyz body-to-NED attitude."""
import torch
from torch import nn
import torch.nn.functional as F


def unit(q):
    return F.normalize(q, dim=-1)


def qmul(a, b):
    return torch.cat((a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True),
                      a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]
                      +torch.cross(a[..., 1:], b[..., 1:], dim=-1)), -1)


def qexp(rot):
    theta = torch.linalg.vector_norm(rot, dim=-1, keepdim=True)
    return torch.cat((torch.cos(theta/2), .5*torch.sinc(theta/(2*torch.pi))*rot), -1)


def geodesic(a, b):
    # atan2 avoids acos' singular derivative at identity; invariant to q -> -q.
    conjugate = torch.cat((unit(a)[..., :1], -unit(a)[..., 1:]), -1)
    rel = qmul(conjugate, unit(b))
    return 2*torch.atan2(torch.linalg.vector_norm(rel[..., 1:], dim=-1), rel[..., 0].abs())


def rotation6(q):
    w, x, y, z = unit(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)


def integrate(state, velocity, rate, dt):
    p, v, q, w = state[..., :3], state[..., 3:6], state[..., 6:10], state[..., 10:13]
    return torch.cat((p+.5*(v+velocity)*dt[..., None], velocity,
                      unit(qmul(unit(q), qexp(.5*(w+rate)*dt[..., None]))), rate), -1)


def baseline(initial, dt):
    s, out = initial, []
    for k in range(dt.shape[1]):
        s = integrate(s, s[:, 3:6], s[:, 10:13], dt[:, k])
        out.append(s)
    return torch.stack(out, 1)


def errors(prediction, target):
    return torch.stack((torch.linalg.vector_norm(prediction[..., :3]-target[..., :3], dim=-1),
                        torch.linalg.vector_norm(prediction[..., 3:6]-target[..., 3:6], dim=-1),
                        geodesic(prediction[..., 6:10], target[..., 6:10]),
                        torch.linalg.vector_norm(prediction[..., 10:13]-target[..., 10:13], dim=-1)), -1)


class VelocityRateGRU(nn.Module):
    def __init__(self, normalization, control=False, hidden=64, event_hidden=16):
        super().__init__()
        self.control = control
        self.register_buffer('mean', torch.tensor(normalization['mean'], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(normalization['scale'], dtype=torch.float32))
        self.register_buffer('output_scale', torch.tensor(normalization['output_scale'], dtype=torch.float32))
        # Both variants allocate identical tensors. NoControl never reads either control input.
        self.history_gru = nn.GRU(22, hidden, batch_first=True)
        self.event_gru = nn.GRU(5, event_hidden, batch_first=True)
        self.step_gru = nn.GRUCell(18+event_hidden, hidden)
        self.head = nn.Linear(hidden, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, state, phase, phase_mask, dt):
        dynamic = torch.cat((state[..., 3:6], state[..., 10:13]), -1)
        dynamic = (dynamic-self.mean[:6])/self.scale[:6]
        freq = (phase[..., 2:3]-self.mean[6])/self.scale[6]
        flap = torch.cat((phase[..., :2]*phase_mask[..., :1], freq*phase_mask[..., 1:2]), -1)
        return torch.cat((dynamic, rotation6(state[..., 6:10]), flap, phase_mask, dt[..., None]/.02), -1)

    def forward(self, inputs):
        h, hm = inputs['history'], inputs['history_mask']
        masks = torch.stack((hm[..., 17]&hm[..., 18], hm[..., 19]), -1).float()
        feature = self.features(h[..., :13], h[..., 17:20], masks, inputs['history_dt'])
        control = ((h[..., 13:17]-self.mean[7:11])/self.scale[7:11]) if self.control else torch.zeros_like(h[..., 13:17])
        sequence, _ = self.history_gru(torch.cat((feature, control), -1))
        batch = torch.arange(len(h), device=h.device)
        hidden = sequence[batch, inputs['history_length']-1]
        last = h[batch, inputs['history_length']-1]
        last_mask = masks[batch, inputs['history_length']-1]
        phase0 = torch.atan2(last[:, 17], last[:, 18])
        freq = last[:, 19]*last_mask[:, 1]
        future_mask = torch.stack((last_mask.prod(-1), last_mask[:, 1]), -1)
        s = inputs['initial']
        dt = inputs['dt']
        if self.control:
            events = inputs['events']
            b, k, e, _ = events.shape
            ec = (events[..., :4]-self.mean[7:11])/self.scale[7:11]
            es, _ = self.event_gru(torch.cat((ec, events[..., 4:5]/.02), -1).reshape(b*k, e, 5))
            lengths = inputs['event_length'].reshape(-1)
            encoded = es[torch.arange(b*k, device=h.device), lengths-1].reshape(b, k, -1)
        else:
            encoded = h.new_zeros((len(h), dt.shape[1], self.event_gru.hidden_size))
        out, elapsed = [], torch.zeros_like(freq)
        for k in range(dt.shape[1]):
            phase = phase0+2*torch.pi*freq*elapsed
            flap = torch.stack((torch.sin(phase), torch.cos(phase), freq), -1)
            f = self.features(s, flap, future_mask, dt[:, k])
            hidden = self.step_gru(torch.cat((f, encoded[:, k]), -1), hidden)
            # Direct discrete v/w prediction with an identity skip; no acceleration target.
            vw = torch.cat((s[:, 3:6], s[:, 10:13]), -1)+self.head(hidden)*self.output_scale
            s = integrate(s, vw[:, :3], vw[:, 3:], dt[:, k])
            out.append(s)
            elapsed = elapsed+dt[:, k]
        return torch.stack(out, 1)
