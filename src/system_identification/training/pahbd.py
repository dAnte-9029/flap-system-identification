"""Noise-aware truncated autonomous simulation, checkpoint and validation workflows."""
from dataclasses import asdict
from pathlib import Path
import torch
from system_identification.models.pahbd import PAHBD,Config
from system_identification.models.pahbd import geometry as so3


def trajectory_loss(predictions,targets,times,variance_weight=.01,horizon_weights=None,cycles=(),cycle_weight=.1):
    """EKF p/v/R share ONE family reduction, gyro another; no independent-truth NLL claim.

    Main state loss uses fixed source noise. The optional bounded predictive variance
    score cannot suppress that loss. Every target is evaluated at its native time.
    cycles contains target-index pairs from the same source/reference section.
    """
    if not targets:raise ValueError('no target observations')
    lookup={t:s for t,s in zip(times,predictions)}
    families={};uncertainty=[]
    horizons={float(k):float(v) for k,v in (horizon_weights or { .2:1., .5:1., 1.:1., 2.:1. }).items()}
    for e in targets:
        if e.measurement<=0:raise ValueError('future target must follow decision')
        s=lookup[e.measurement];r=so3.state_residual(s.x,e.value[:13])
        weights=e.mask[:12]*e.quality[:12]
        noise=e.variance[:12].clamp_min(1e-4)
        norm=weights.sum().clamp_min(1)
        horizon_weight=next((v for t,v in sorted(horizons.items()) if e.measurement<=t+1e-9),1.)
        score=(r.square()/noise*weights).sum()/norm*horizon_weight
        family='gyro' if e.source=='vehicle_angular_velocity' else 'ekf'
        families.setdefault(family,[]).append(score)
        total=noise+s.variance[:12]
        uncertainty.append(((r.square()/total+torch.log(total/noise))*weights).sum()/norm)
    state=torch.stack([torch.stack(v).mean() for v in families.values()]).mean()
    calibration=torch.stack(uncertainty).mean()
    cycle_terms=[]
    for i,j in cycles:
        a,b=targets[i],targets[j]
        pa,pb=lookup[a.measurement].x,lookup[b.measurement].x
        ya,yb=a.value[:13],b.value[:13]
        # Relative attitude error compares group increments, not differences of rotvecs.
        pred_delta=so3.multiply(so3.conjugate(pa[6:10]),pb[6:10])
        obs_delta=so3.multiply(so3.conjugate(ya[6:10]),yb[6:10])
        r=torch.cat(((pb[:6]-pa[:6])-(yb[:6]-ya[:6]),so3.residual(obs_delta,pred_delta),
                     (pb[10:]-pa[10:])-(yb[10:]-ya[10:])))
        weight=a.mask[:12]*b.mask[:12]*torch.minimum(a.quality[:12],b.quality[:12])
        noise=(a.variance[:12]+b.variance[:12]).clamp_min(1e-4)
        cycle_terms.append((r.square()/noise*weight).sum()/weight.sum().clamp_min(1))
    cycle=torch.stack(cycle_terms).mean() if cycle_terms else state*0
    total=state+variance_weight*calibration+cycle_weight*cycle
    return total,dict(state=state,variance=calibration,cycle=cycle,total=total)


def window_loss(model,window,**kwargs):
    times=sorted(set(e.measurement for e in window.targets))
    belief,predictions=model(window,times)
    loss,parts=trajectory_loss(predictions,window.targets,times,cycles=window.metadata.get('cycle_pairs',()),**kwargs)
    # h consistency is optional in future protocols; no per-window free optimization here.
    return loss,parts,belief,predictions,times


def fit_normalization(model,windows):
    """Train-only causal physical-context statistics. No validation/target values used."""
    values=[]
    for w in windows:
        if w.metadata['split']!='train':raise ValueError('normalization requires train')
        values.append(model.flow.slow.features(w.initial).detach())
    if not values:raise ValueError('no train windows for normalization')
    v=torch.stack(values)
    with torch.no_grad():
        model.flow.slow.input_mean.copy_(v.mean(0))
        model.flow.slow.input_scale.copy_(v.std(0,unbiased=False).clamp_min(.1))


def save_checkpoint(path,model,optimizer,metadata,step):
    path=Path(path)
    if path.exists():raise FileExistsError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='pahbd_v1',config=asdict(model.config),model=model.state_dict(),
                    optimizer=optimizer.state_dict() if optimizer else None,
                    metadata=metadata,step=step,dtype=str(next(model.parameters()).dtype)),path)


def load_checkpoint(path,device='cpu',optimizer=False):
    bundle=torch.load(path,map_location=device,weights_only=True)
    if bundle['schema']!='pahbd_v1':raise ValueError('unsupported checkpoint')
    dtype=getattr(torch,bundle['dtype'].split('.')[-1])
    model=PAHBD(Config(**bundle['config'])).to(device=device,dtype=dtype)
    model.load_state_dict(bundle['model'])
    opt=torch.optim.Adam(model.parameters()) if optimizer else None
    if opt is not None and bundle['optimizer'] is not None:opt.load_state_dict(bundle['optimizer'])
    return model,opt,bundle


def train_steps(model,windows,optimizer,steps,loss_config=None):
    if not windows:raise ValueError('empty training set')
    model.train();records=[]
    for step in range(steps):
        window=windows[step%len(windows)]
        if window.metadata['split']!='train':raise ValueError('parameter fitting requires train')
        optimizer.zero_grad(set_to_none=True)
        loss,parts,*_=window_loss(model,window,**(loss_config or {}))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('invalid gradients')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        records.append(dict(step=step+1,**{k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm)))
    return records


@torch.no_grad()
def validate(model,windows):
    from system_identification.evaluation.pahbd import native_horizon_errors,aggregate_flights
    model.eval();records=[]
    for w in windows:
        if w.metadata['split'] not in ('train','validation'):raise ValueError('sealed split')
        _,_,_,pred,times=window_loss(model,w)
        records.extend(native_horizon_errors(pred,times,w.targets,w.metadata))
    return dict(windows=records,flight_equal=aggregate_flights(records))
