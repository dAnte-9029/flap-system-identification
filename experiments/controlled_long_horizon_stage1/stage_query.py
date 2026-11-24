"""Native-time prefix queries, using the archived implementation for earlier models."""
import torch
from dynamics import qmul, qexp, unit, query as archived_query


def query(roll,t,component):
    """Integrate only the causal prefix of the containing constant-command segment."""
    if len(roll) == 4:
        return archived_query(roll,t,component)
    end,state,_,trend = roll[:4]
    ticks = (t*1e6).round().clamp_min(0).contiguous()
    # left search places an exact boundary at the end of the preceding segment.
    ix = torch.searchsorted(end.contiguous(),ticks).clamp(max=end.shape[1]-1)
    prev = torch.cat([torch.zeros_like(end[:,:1]),end[:,:-1]],1).gather(1,ix)
    dt = (ticks-prev).unsqueeze(-1)*1e-6
    def take(x):
        return x.gather(1,ix[...,None].expand(-1,-1,x.size(-1)))
    if component in ('slow_v','slow_w'):
        sl = slice(0,3) if component=='slow_v' else slice(3,6)
        return take(trend[0][...,sl])+dt*take(trend[1][...,sl])
    acceleration = roll[4](dt,*(take(x) for x in roll[5]))
    if component=='pv':
        v,a = take(state[...,3:6]),acceleration[...,:3]
        return torch.cat([take(state[...,:3])+dt*(v+.5*dt*a),v+dt*a],-1)
    w,alpha = take(state[...,10:]),acceleration[...,3:]
    if component=='q':
        return unit(qmul(take(state[...,6:10]),qexp(dt*(w+.5*dt*alpha))))
    if component!='w':
        raise ValueError(component)
    return w+dt*alpha
