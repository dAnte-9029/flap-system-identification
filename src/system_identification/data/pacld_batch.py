"""CPU compilation of independent native-clock execution plans; no resampling."""
from dataclasses import dataclass
import bisect
import math
import torch
from system_identification.models.pacld.types import initial_state
from system_identification.models.pacld.batched_types import BatchedState
State = BatchedState
from system_identification.training.pacld import valid_pairs, SOURCES

FIELDS = ('x','z_s','c','f','phi','actuator','variance','phase_quality','phase_covariance')

@dataclass
class BatchedWindow:
    initial: State
    history: list
    future: list
    targets: dict
    pairs: dict
    metadata: list
    times: list
    starts: tuple = ()
    history_timestamps: tuple = ()

    def to(self, device, dtype=torch.float64):
        def move(x):
            if isinstance(x, torch.Tensor): return x.to(device=device, dtype=dtype if x.is_floating_point() else x.dtype)
            if isinstance(x, State): return x.with_(**{k:move(getattr(x,k)) for k in FIELDS})
            if isinstance(x, dict): return {k:move(v) for k,v in x.items()}
            if isinstance(x, list): return [move(v) for v in x]
            if isinstance(x, tuple): return tuple(move(v) for v in x)
            return x
        return BatchedWindow(move(self.initial),move(self.history),move(self.future),move(self.targets),move(self.pairs),self.metadata,self.times,self.starts,self.history_timestamps)


def steps(t, end, tape, max_step):
    if end < t or not all(math.isfinite(v) for v in (t,end,*tape.times)): raise ValueError('invalid propagation times')
    i=bisect.bisect_right(tape.times,t)-1
    if i<0: raise ValueError('no causal control')
    result=[]
    while t<end:
        boundary=min(end,tape.times[i+1] if i+1<len(tape.times) else end)
        span=boundary-t; n=max(1,math.ceil(span/max_step))
        result.extend([(span/n,tape.values[i])]*n)
        t=boundary
        if i+1<len(tape.times) and t>=tape.times[i+1]: i+=1
    return result


def pad_steps(rows):
    B=len(rows); N=max(map(len,rows),default=0)
    dt=torch.zeros(N,B,1,dtype=torch.float64); u=torch.zeros(N,B,4,dtype=torch.float64)
    for b,row in enumerate(rows):
        for j,(delta,control) in enumerate(row): dt[j,b]=delta; u[j,b]=control
    return dt,u


def compile_windows(windows,cfg):
    if not windows: raise ValueError('empty batch')
    if any(w.initial.x.device.type!='cpu' for w in windows): raise ValueError('compile on CPU before transfer')
    B=len(windows); tapes=[w.controls.canonical() for w in windows]
    histories=[]
    for w in windows:
        if not math.isfinite(w.start) or w.start>0 or -w.start>cfg.replay_seconds+1e-6: raise ValueError('history interval')
        published=[e for e in w.history if w.start<=e.availability<=0]
        if any(not math.isfinite(e.measurement) or e.measurement>e.availability for e in published): raise ValueError('history time')
        es=sorted([e for e in published if e.measurement>=w.start and e.availability-e.measurement<=cfg.replay_seconds],key=lambda e:(e.measurement,e.availability,e.source))
        for e in es:
            if e.frame!='NED/FRD/Hamilton-wxyz' or any(not torch.isfinite(v).all() for v in (e.value,e.mask,e.quality,e.variance)): raise ValueError('observation contract')
        histories.append(es)
    ss=[initial_state(w,cfg) for w in windows]
    initial=State(**{k:torch.stack([getattr(s,k) if getattr(s,k) is not None else torch.diag_embed(s.variance[12:14]) for s in ss]) for k in FIELDS})
    history=[]; clocks=[w.start for w in windows]; last=clocks.copy()
    for j in range(max(map(len,histories))):
        rows=[]; value=torch.zeros(B,15,dtype=torch.float64); value[:,6]=1
        obs=dict(value=value,mask=torch.zeros(B,14,dtype=torch.float64),quality=torch.zeros(B,14,dtype=torch.float64),variance=torch.ones(B,14,dtype=torch.float64),age=torch.zeros(B,1,dtype=torch.float64),elapsed=torch.zeros(B,1,dtype=torch.float64),hall=torch.zeros(B,1,dtype=torch.bool),valid=torch.zeros(B,1,dtype=torch.bool),control=torch.zeros(B,4,dtype=torch.float64))
        for b,es in enumerate(histories):
            if j>=len(es): rows.append([]); continue
            e=es[j]; rows.append(steps(clocks[b],e.measurement,tapes[b],cfg.max_step)); clocks[b]=e.measurement
            for k in ('value','mask','quality','variance'): obs[k][b]=getattr(e,k)
            obs['age'][b]=e.availability-e.measurement; obs['elapsed'][b]=e.measurement-last[b]; last[b]=e.measurement
            obs['hall'][b]=e.source=='wing_phase'; obs['valid'][b]=True
            obs['control'][b]=tapes[b].values[bisect.bisect_right(tapes[b].times,e.measurement)-1]
        history.append((pad_steps(rows),obs))
    history.append((pad_steps([steps(clocks[b],0,tapes[b],cfg.max_step) for b in range(B)]),None))
    times=[sorted({e.measurement for e in w.targets}) for w in windows]; clocks=[0.]*B; future=[]
    for j in range(max(map(len,times))):
        rows=[]
        for b,ts in enumerate(times):
            end=ts[j] if j<len(ts) else clocks[b]
            rows.append(steps(clocks[b],end,tapes[b],cfg.max_step)); clocks[b]=end
        future.append(pad_steps(rows))
    # Flatten native targets/pairs once. Reduction weights encode equal family/source means.
    target=dict(batch=[],query=[],value=[],weight=[],noise=[],reduction=[])
    pair=dict(batch=[],a=[],b=[],ya=[],yb=[],weight=[],noise=[],reduction=[])
    valid_macro=[]
    for b,w in enumerate(windows):
        lookup={t:i for i,t in enumerate(times[b])}; families={}
        for e in w.targets:
            if e.measurement<=0 or e.source not in SOURCES: raise ValueError('physical future target required')
            if (e.mask[:12]*e.quality[:12]>0).any(): families.setdefault(int(e.source=='vehicle_angular_velocity'),[]).append(e)
        if not families: raise ValueError('no valid target')
        for es in families.values():
            for e in es:
                for k,v in dict(batch=b,query=lookup[e.measurement],value=e.value[:13],weight=e.mask[:12]*e.quality[:12],noise=e.variance[:12].clamp_min(1e-4),reduction=1/(len(families)*len(es))).items(): target[k].append(v)
        pairs,_=valid_pairs(w); sources={}
        for i,j in pairs: sources.setdefault(w.targets[i].source,[]).append((i,j))
        valid_macro.append(bool(pairs))
        for ps in sources.values():
            for i,j in ps:
                a,c=w.targets[i],w.targets[j]
                for k,v in dict(batch=b,a=lookup[a.measurement],b=lookup[c.measurement],ya=a.value[:13],yb=c.value[:13],weight=a.mask[:12]*c.mask[:12]*torch.minimum(a.quality[:12],c.quality[:12]),noise=(a.variance[:12]+c.variance[:12]).clamp_min(1e-4),reduction=1/(len(sources)*len(ps))).items(): pair[k].append(v)
    def pack(d):
        return {k:torch.stack(v) if v and isinstance(v[0],torch.Tensor) else torch.tensor(v,dtype=torch.long if k in ('batch','query','a','b') else torch.float64) for k,v in d.items()}
    pair=pack(pair); pair['valid_windows']=torch.tensor(valid_macro)
    return BatchedWindow(initial,history,future,pack(target),pair,[w.metadata for w in windows],times,tuple(w.start for w in windows),tuple(tuple((e.measurement,e.availability,e.source) for e in es) for es in histories))
