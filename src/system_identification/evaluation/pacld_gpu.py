"""Read-only GPU evaluation; explicit BASE and arbitrary factual query clocks."""
import torch
from system_identification.models.pacld.batch import FIELDS
from system_identification.models.pacld.types import State
from system_identification.data.pacld_batch import compile_windows, steps, pad_steps

@torch.no_grad()
def predict(model, windows, queries=None, base=True):
    if any(w.metadata['split'] not in ('train','validation') for w in windows):
        raise ValueError('sealed split forbidden')
    batch=compile_windows(windows,model.config).to(next(model.parameters()).device,torch.float64)
    times=batch.times if queries is None else queries
    if len(times)!=len(windows): raise ValueError('query batch size')
    for w,ts in zip(windows,times):
        if ts!=sorted(set(ts)) or any(t<0 or t>w.metadata['actual_horizon_s']+1e-9 for t in ts):
            raise ValueError('query outside support or not unique sorted')
    if queries is not None:
        clocks=[0.]*len(windows); batch.future=[]
        for j in range(max(map(len,times),default=0)):
            rows=[]
            for b,(w,ts) in enumerate(zip(windows,times)):
                end=ts[j] if j<len(ts) else clocks[b]
                rows.append(steps(clocks[b],end,w.controls.canonical(),model.config.max_step));clocks[b]=end
            dt,u=pad_steps(rows);batch.future.append((dt.to(batch.initial.x),u.to(batch.initial.x)))
    belief=model.encode_batch(batch);B=len(windows)
    state=belief
    if base:
        state=belief.with_(**{k:torch.cat((getattr(belief,k),getattr(belief,k)),0) for k in FIELDS})
        state=state.with_(phase_quality=torch.cat((belief.phase_quality,torch.zeros_like(belief.phase_quality)),0))
    def single(s,i,t):
        return State(**{k:getattr(s,k)[i].detach().cpu().double() for k in FIELDS},time=t)
    out=[dict(belief=single(belief,i,0),times=ts,predictions={m:[] for m in (('FULL','BASE') if base else ('FULL',))}) for i,ts in enumerate(times)]
    for j,(dt,u) in enumerate(batch.future):
        if base:dt,u=torch.cat((dt,dt),1),torch.cat((u,u),1)
        state=model.propagate_plan(state,(dt,u))
        for i,ts in enumerate(times):
            if j<len(ts):
                for mode,offset in [('FULL',0)]+([('BASE',B)] if base else []):
                    out[i]['predictions'][mode].append(single(state,i+offset,ts[j]))
    return out


def factual_predictor(model,windows):
    lookup={w:i for i,w in enumerate(windows.index.window_id)}
    def call(wid,times):
        w=windows[lookup[wid]]; unique=sorted(set(times))
        r=predict(model,[w],[unique],base=False)[0]
        states=dict(zip(unique,r['predictions']['FULL']))
        import numpy as np
        return r['belief'].x.numpy(),np.stack([states[t].x.numpy() for t in times]) if times else np.empty((0,13))
    return call
