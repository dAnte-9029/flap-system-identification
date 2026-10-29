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
    cycle_terms={}
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
        family='gyro' if a.source=='vehicle_angular_velocity' else 'ekf'
        cycle_terms.setdefault(family,[]).append((r.square()/noise*weight).sum()/weight.sum().clamp_min(1))
    cycle=torch.stack([torch.stack(v).mean() for v in cycle_terms.values()]).mean() if cycle_terms else state*0
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


def save_checkpoint(path,model,optimizer,metadata,step,training_state=None):
    path=Path(path)
    if path.exists():raise FileExistsError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='pahbd_v2',config=asdict(model.config),model=model.state_dict(),
                    optimizer=optimizer.state_dict() if optimizer else None,
                    metadata=metadata,step=step,training_state=training_state,dtype=str(next(model.parameters()).dtype)),path)


def load_checkpoint(path,device='cpu',optimizer=False):
    bundle=torch.load(path,map_location=device,weights_only=True)
    if bundle['schema']!='pahbd_v2':raise ValueError('requires pahbd_v2 checkpoint; v1 used different delayed-observation and uncertainty semantics')
    dtype=getattr(torch,bundle['dtype'].split('.')[-1])
    model=PAHBD(Config(**bundle['config'])).to(device=device,dtype=dtype)
    model.load_state_dict(bundle['model'])
    opt=torch.optim.Adam(model.parameters()) if optimizer else None
    if opt is not None and bundle['optimizer'] is not None:opt.load_state_dict(bundle['optimizer'])
    return model,opt,bundle


def train_steps(model,windows,optimizer,steps,loss_config=None,sampler=None,seed=17):
    if not windows:raise ValueError('empty training set')
    index=window_index(windows)
    if 'phase_policy' in index and (index.phase_policy=='relative_single').any() and index.phase_reference_id.nunique()!=1:
        raise ValueError('unknown independent log origins cannot share periodic coefficients')
    sampler=sampler or FlightBalancedSampler(index,seed)
    model.train();records=[]
    for step in range(steps):
        index=sampler.draw();window=windows[index]
        if window.metadata['split']!='train':raise ValueError('parameter fitting requires train')
        optimizer.zero_grad(set_to_none=True)
        loss,parts,*_=window_loss(model,window,**(loss_config or {}))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('invalid gradients')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        records.append(dict(step=sampler.draws,window_id=window.metadata['window_id'],flight_id=window.metadata['flight_id'],horizon_us=int(sampler.index.iloc[index].horizon_us),actual_horizon_s=max(e.measurement for e in window.targets),**{k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm)))
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


class FlightBalancedSampler:
    """Shuffle flight rounds, then horizon rounds within each flight, then a window.

    The index is CPU metadata only. Sampling never materializes native tables.
    A/B use identical index+seed, independent of torch RNG and model errors.
    """
    def __init__(self,index,seed=17):
        import numpy as np
        import hashlib
        self.index=index.reset_index(drop=True).copy()
        if len(index)==0 or set(index.split)!={'train'}:raise ValueError('nonempty train index required')
        self.identity=hashlib.sha256(self.index[['window_id','flight_id','horizon_us']].to_csv(index=False).encode()).hexdigest()
        self.groups={}
        for (flight,horizon),g in self.index.groupby(['flight_id','horizon_us'],sort=True):
            self.groups.setdefault(flight,{})[int(horizon)]=g.index.to_numpy()
        self.rng=np.random.default_rng(seed);self.seed=seed;self.flights=[];self.horizons={}
        self.draws=0;self.counts={};self.visited=set()

    def draw(self):
        if not self.flights:self.flights=[str(f) for f in self.rng.permutation(sorted(self.groups))]
        f=str(self.flights.pop())
        if not self.horizons.get(f):self.horizons[f]=[int(h) for h in self.rng.permutation(sorted(self.groups[f]))]
        h=self.horizons[f].pop();g=self.groups[f][h];i=int(g[self.rng.integers(len(g))])
        key=f'{f}|{h}';self.counts[key]=self.counts.get(key,0)+1
        self.draws+=1;self.visited.add(i)
        return i

    def state_dict(self):
        return dict(identity=self.identity,seed=self.seed,rng=self.rng.bit_generator.state,
                    flights=self.flights.copy(),horizons={k:v.copy() for k,v in self.horizons.items()},
                    draws=self.draws,counts=self.counts.copy(),visited=sorted(self.visited))

    def load_state_dict(self,state):
        if state['identity']!=self.identity or state['seed']!=self.seed:raise ValueError('sampling contract changed')
        self.rng.bit_generator.state=state['rng'];self.flights=state['flights'].copy()
        self.horizons={k:v.copy() for k,v in state['horizons'].items()}
        self.draws=state['draws'];self.counts=state['counts'].copy();self.visited=set(state['visited'])

    def coverage(self):
        return dict(seed=self.seed,draws=self.draws,unique_windows=len(self.visited),
                    sampled_flights=len({self.index.iloc[i].flight_id for i in self.visited}),
                    groups=[dict(flight_id=f,horizon_us=h,eligible_windows=len(g),
                                 draws=self.counts.get(f'{f}|{h}',0))
                            for f,hs in self.groups.items() for h,g in hs.items()])


def window_index(windows):
    import pandas as pd
    if hasattr(windows,'index') and not callable(windows.index):return windows.index
    return pd.DataFrame([dict(w.metadata,horizon_us=w.metadata.get('horizon_us',round(max(e.measurement for e in w.targets)*1e6))) for w in windows])


@torch.no_grad()
def validation_selection_score(model,windows,loss_config=None):
    """Fixed source-scaled objective: average windows per flight/horizon, then horizons,
    then independent flights. No pooling of high-rate target counts across windows.
    """
    from system_identification.evaluation.pahbd import native_horizon_errors,aggregate_flights
    groups={};metrics=[];model.eval()
    for w in windows:
        if w.metadata['split']!='validation':raise ValueError('selection requires validation')
        loss,_,_,pred,times=window_loss(model,w,**(loss_config or {}))
        metrics.extend(native_horizon_errors(pred,times,w.targets,w.metadata))
        key=(w.metadata['flight_id'],w.metadata['horizon_us'])
        groups.setdefault(key,[]).append(float(loss))
    if not groups:raise ValueError('empty validation cohort')
    flights={}
    for (f,h),values in groups.items():flights.setdefault(f,[]).append(sum(values)/len(values))
    scores={f:sum(v)/len(v) for f,v in flights.items()}
    return dict(score=sum(scores.values())/len(scores),per_flight=scores,
                native_flight_equal=aggregate_flights(metrics),
                native_per_flight={f:aggregate_flights([r for r in metrics if r['flight_id']==f]) for f in scores},
                per_flight_horizon=[dict(flight_id=f,horizon_us=h,loss=sum(v)/len(v),windows=len(v)) for (f,h),v in groups.items()])


def formal_fit(model,train,validation,optimizer,steps,output,metadata,seed=17,
               validation_interval=1000,patience=10,loss_config=None,resume=None):
    """Explicit caller-owned budget; no search or automatic follow-on runs."""
    import json
    if min(steps,validation_interval,patience)<1:raise ValueError('positive explicit schedule required')
    if 'training' in metadata.get('config',{}):
        require_training_admission(metadata['config'],model.config,window_index(train),window_index(validation))
    if resume and not resume.get('training_state'):raise ValueError('checkpoint has no resumable formal schedule')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sampler=FlightBalancedSampler(window_index(train),seed)
    schedule=dict(step=0,best_score=float('inf'),bad_validations=0,validation_interval=validation_interval,patience=patience)
    if resume:
        if resume['metadata']['config']!=metadata['config']:raise ValueError('resume configuration differs')
        if resume['metadata'].get('validation_window_ids')!=metadata.get('validation_window_ids'):raise ValueError('validation cohort changed')
        sampler.load_state_dict(resume['training_state']['sampler'])
        schedule=resume['training_state']['schedule'].copy()
        if schedule['validation_interval']!=validation_interval or schedule['patience']!=patience:raise ValueError('schedule changed')
        torch.set_rng_state(resume['training_state']['torch_rng'].cpu())
        if torch.cuda.is_available() and resume['training_state'].get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(resume['training_state']['cuda_rng'])
    records=[]
    if schedule['bad_validations']>=patience:raise ValueError('checkpoint already early-stopped')
    while schedule['step']<steps:
        count=min(validation_interval-schedule['step']%validation_interval,steps-schedule['step'])
        draws=train_steps(model,train,optimizer,count,loss_config,sampler=sampler)
        with (output/'training_draws.jsonl').open('a') as stream:
            for draw in draws:stream.write(json.dumps(draw)+'\n')
        schedule['step']+=count
        result=validation_selection_score(model,validation,loss_config)
        if not torch.isfinite(torch.tensor(result['score'])):raise FloatingPointError('nonfinite validation score')
        improved=result['score']<schedule['best_score']
        if improved:schedule.update(best_score=result['score'],bad_validations=0)
        else:schedule['bad_validations']+=1
        name=f"step_{schedule['step']:08d}.pt"
        if improved:schedule['best_checkpoint']=str((output/name).resolve())
        state=dict(sampler=sampler.state_dict(),schedule=schedule.copy(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        save_checkpoint(output/name,model,optimizer,metadata,schedule['step'],training_state=state)
        (output/'best.json').write_text(json.dumps(dict(checkpoint=schedule['best_checkpoint'],score=schedule['best_score'])))
        records.append(dict(step=schedule['step'],checkpoint=name,improved=improved,validation=result))
        (output/'progress.json').write_text(json.dumps(dict(records=records,coverage=sampler.coverage()),indent=2))
        if schedule['bad_validations']>=patience:break
    return dict(schedule=schedule,records=records,coverage=sampler.coverage())


def require_training_admission(config,model_config,index,validation_index=None):
    """Fail closed on a mislabeled mean-only A/B experiment or mixed reference groups."""
    objective=config['training']['objective']
    if objective=='average_dynamics':
        if config['data']['phase_policy']!='average':
            raise ValueError('average_dynamics requires explicit data.phase_policy=average')
        return
    if objective!='shared_periodic_AB':raise ValueError('unknown training objective')
    if config['data']['phase_policy']!='shared_or_average' or model_config.phase_mode!='relative':
        raise ValueError('shared periodic A/B requires active certified phase admission')
    references=set(index.loc[index.shared_periodic_eligible,'phase_reference_id'])
    if len(references)!=1:raise ValueError('shared periodic A/B requires exactly one certified shared sensor reference')
    if validation_index is not None:
        val=set(validation_index.loc[validation_index.shared_periodic_eligible,'phase_reference_id'])
        if references!=val:raise ValueError('validation must contain the same certified shared sensor reference')
