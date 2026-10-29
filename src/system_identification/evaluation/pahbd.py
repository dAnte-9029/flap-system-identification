"""Observation-consistency and model-internal response interfaces, not causal proof."""
import math
import torch
from system_identification.models.pahbd import geometry as so3


def native_horizon_errors(predictions,times,targets,metadata,horizons=(.2,.5,1.,2.),tolerance=.03):
    lookup=dict(zip(times,predictions));rows=[]
    for horizon in horizons:
        for name,sl in [('p',slice(0,3)),('v',slice(3,6)),('R',slice(6,9)),('omega',slice(9,12))]:
            candidates=[e for e in targets if bool(e.mask[sl].all()) and abs(e.measurement-horizon)<=tolerance]
            if not candidates:continue
            e=min(candidates,key=lambda e:abs(e.measurement-horizon))
            residual=so3.state_residual(lookup[e.measurement].x,e.value[:13])[sl]
            rows.append(dict(flight_id=metadata['flight_id'],window_id=metadata['window_id'],
                             split=metadata['split'],horizon_s=horizon,actual_time_s=e.measurement,
                             component=name,squared_error=float(residual.square().sum()),
                             interpretation='native_observation_consistency_conditional_control_replay'))
    return rows


def aggregate_flights(rows):
    groups={}
    for r in rows:groups.setdefault((r['component'],r['horizon_s']),{}).setdefault(r['flight_id'],[]).append(r['squared_error'])
    return [dict(component=k[0],horizon_s=k[1],flights=len(v),
                 rmse=math.sqrt(sum(sum(x)/len(x) for x in v.values())/len(v))) for k,v in groups.items()]


def net_transition(a,b):
    """p,v,body relative rotation,omega increment at a declared pair of sections."""
    return so3.state_residual(a.x,b.x)


def section_transitions(states,phase_tolerance=.05):
    """Pairs near equal unwrapped phase modulo a complete turn; returns actual times' indices.

    Query a dense autonomous grid or use section_times below; never reset the flow.
    """
    pairs=[]
    for i,a in enumerate(states):
        if not bool(a.phase_quality.all()):continue
        j=next((j for j in range(i+1,len(states)) if float(states[j].phi-a.phi)>=2*math.pi-phase_tolerance),None)
        if j is not None and abs(float(states[j].phi-a.phi)-2*math.pi)<=phase_tolerance:
            pairs.append((i,j,net_transition(a,states[j])))
    return pairs


def periodic_amplitude_phase(phases,values,harmonics=3):
    """Offline diagnostic Fourier fit with intercept, never an encoder input or offset fit."""
    k=torch.arange(1,harmonics+1,device=values.device,dtype=values.dtype)
    angle=phases.reshape(-1,1)*k
    design=torch.cat((torch.ones_like(angle[:,:1]),torch.cos(angle),torch.sin(angle)),1)
    if len(values)<design.shape[1] or torch.linalg.matrix_rank(design)<design.shape[1]:
        raise ValueError('insufficient phase coverage')
    coef=torch.linalg.lstsq(design,values).solution
    c,s=coef[1:harmonics+1],coef[harmonics+1:]
    return dict(mean=coef[0],amplitude=torch.sqrt(c.square()+s.square()),phase=torch.atan2(s,c))


def matched_control_response(model,belief,times,control_b,control_d,observed_b=None,observed_d=None):
    """B/D same-history control exchange. Observed matched pairs must be supplied externally.

    Returns model contrast and, if supplied, observed contrast. Matching quality and
    closed-loop confounding remain external scientific requirements.
    """
    b=model.flow.rollout(belief,times,control_b);d=model.flow.rollout(belief,times,control_d)
    result={'model_contrast':torch.stack([so3.state_residual(x.x,y.x) for x,y in zip(b,d)]),
            'interpretation':'model_internal_conditional_response_not_counterfactual_truth'}
    if (observed_b is None)!=(observed_d is None):raise ValueError('both observed matched trajectories required')
    if observed_b is not None:result['observed_contrast']=so3.state_residual(observed_b,observed_d)
    return result


def control_sensitivity(model,state):
    """Jacobian of internal effective derivative w.r.t. actuator proxy, fixed x,h,f,phi."""
    c,_=model.flow.slow(state)
    return torch.autograd.functional.jacobian(lambda a:model.flow.mixer(c,a,state.phi,state.phase_quality,model.flow.phase_covariance(state)[...,1,1:2])[0],state.actuator,
                                               create_graph=True)


@torch.no_grad()
def wingbeat_transitions(model,belief,controls,count=2,max_time=2.,phase_tolerance=1e-9):
    """Autonomous equal-phase sections, located with the analytic frequency model.

    Evaluation-only event location: no claim of differentiating crossing times.
    State/control rollout remains the same Lie flow; no measured phase resets.
    """
    import bisect
    if belief.f.numel()!=1 or not bool(belief.phase_quality.all()):
        raise ValueError('section location needs one valid phase reference')
    if count<1 or max_time<=0:raise ValueError('positive section count and horizon required')
    tape=controls.canonical();i=bisect.bisect_right(tape.times,0.)-1
    if i<0:raise ValueError('no control at section start')
    t=0.;f=belief.f;phi=belief.phi;origin=float(phi);times=[]
    while len(times)<count and t<max_time-1e-12:
        boundary=min(max_time,tape.times[i+1] if i+1<len(tape.times) else max_time)
        u=tape.values[i,:1]
        fend,pend=model.flow.phase(f,phi,u,boundary-t)
        if min(float(f),float(fend))<=0:raise ValueError('nonmonotonic phase: positive frequency required')
        target=origin+2*math.pi*(len(times)+1)
        if float(pend)>=target:
            lo,hi=0.,boundary-t
            for _ in range(60):
                mid=(lo+hi)/2
                _,p=model.flow.phase(f,phi,u,mid)
                if float(p)<target:lo=mid
                else:hi=mid
                if hi-lo<phase_tolerance/(2*math.pi*max(float(f),float(fend))):break
            dt=(lo+hi)/2
            f,phi=model.flow.phase(f,phi,u,dt);t+=dt;times.append(t)
        else:f,phi=fend,pend;t=boundary
        if i+1<len(tape.times) and t>=tape.times[i+1]-1e-12:i+=1
    if len(times)<count:raise ValueError('requested sections outside supported horizon')
    states=model.flow.rollout(belief,times,tape)
    starts=[belief]+states[:-1]
    return dict(times=times,phase=torch.stack([s.phi for s in states]),
                transitions=torch.stack([net_transition(a,b) for a,b in zip(starts,states)]),
                reference='model_predicted_same_phase_not_verified_wing_position')


def phase_scenarios(model,belief,times,controls):
    """Four positive-weight joint initial sigma points, same state flow/control tape.

    Initial (f,unwrapped phi) covariance is represented by scenarios. Each scenario
    starts with zero residual joint covariance; uncalibrated process diffusion
    grows inside each flow and damps harmonics in Gaussian mode. This is not a
    calibrated probability ensemble or the exact nonlinear ensemble expectation.
    """
    cov=model.flow.phase_covariance(belief)
    if cov.shape!=(2,2):raise ValueError('one belief required for scenario interface')
    values,vectors=torch.linalg.eigh(cov)
    if float(values.min()) < -1e-10:raise ValueError('non-PSD phase covariance')
    root=vectors@torch.diag(values.clamp_min(0).sqrt())
    offsets=torch.cat((math.sqrt(2)*root.T,-math.sqrt(2)*root.T),0)
    states=[]
    for delta in offsets:
        s=belief.with_(f=belief.f+delta[:1],phi=belief.phi+delta[1:],phase_covariance=torch.zeros_like(cov))
        states.append(model.flow.rollout(s,times,controls))
    return dict(weights=belief.x.new_full((4,),.25),initial_offsets=offsets,states=states,
                mean_phase=model.flow.rollout(belief,times,controls),
                interpretation='joint_initial_sigma_points_with_conditional_Gaussian_process_diffusion_uncalibrated')


class FrozenRollResponse:
    """Read-only adapter to the existing frozen B/D observational roll cohort.

    Retains original selection, labels, matching quality, flight and dependency
    identities. It does not select pairs or score internal exchanged controls.
    """
    def __init__(self,source):
        from pathlib import Path
        import json
        import pandas as pd
        import importlib.util
        self.source=Path(source)
        self.config=json.loads((self.source/'config.json').read_text())
        # Reuse the original component algorithm and metric definitions verbatim.
        spec=importlib.util.spec_from_file_location('frozen_roll_analysis',self.source/'analyze.py')
        self.analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.analysis)
        root=self.source/'results'
        self.windows=pd.read_csv(root/'windows.csv')
        pairs=pd.read_csv(root/'pairs.csv');cohorts=pd.read_csv(root/'cohorts.csv')
        if not set(self.windows.split)<= {'train','validation'} or not set(pairs.split)<= {'train','validation'}:
            raise ValueError('sealed partitions forbidden')
        self.pairs=pairs[pairs.pair_uid.isin(cohorts.loc[cohorts.cohort=='main','pair_uid'])].copy()
        self.pairs['dependency_component']=-1
        for split,g in self.pairs.groupby('split',sort=False):
            labels=self.analysis.labels(g)
            self.pairs.loc[g.index,'dependency_component']=g.pair_uid.map(labels)
        tr=self.pairs[self.pairs.split=='train']
        if len(tr)!=48 or tr.dependency_component.nunique()!=11:raise ValueError('frozen 48/11 cohort changed')

    def load_truth(self):
        import numpy as np
        # Never materialize any historical model outputs in the archive.
        with np.load(self.source/'results/native_rollouts.npz',allow_pickle=False) as z:
            return {k:z[k] for k in ('truth','truth_initial','initial','future_count','native_time_us')}

    def evaluate(self,predict,split='train',times=(.1,.2,.5)):
        """predict(window_id, decision_relative_times)->(causal_anchor13, states[N,13]).

        Callable adapters permit trained PA-HBD A/B and a future C. Factual logged
        controls only. Targets retain the original aligned observation clock and
        origin; model trajectories use their own causal encoded decision anchor.
        """
        import numpy as np
        import pandas as pd
        if split not in ('train','validation'):raise ValueError('sealed split')
        z=self.load_truth();rows=[];cache={}
        columns=[3,4,5,10,11,12];variables=['vx','vy','vz','p','q','r']
        for pair in self.pairs[self.pairs.split==split].itertuples():
            members=[]
            for idx,wid in [(int(pair.a),pair.window_a),(int(pair.b),pair.window_b)]:
                n=int(z['future_count'][idx]);t=z['native_time_us'][idx,:n+1]*1e-6
                y=np.vstack((z['truth_initial'][idx],z['truth'][idx,:n]))[:,columns]
                meta=self.windows.iloc[idx]
                if meta.window_id!=wid:raise ValueError('frozen window ordering changed')
                members.append((idx,wid,t,y,meta))
            endpoint=min(.5,members[0][2][-1],members[1][2][-1])
            at=np.array(list(times)+[endpoint]);kind=['requested']*len(times)+['supported_endpoint']
            observed=[];predicted=[];anchors=[];valid=np.ones(len(at),bool)
            for idx,wid,t,y,meta in members:
                real,ok=self.analysis.interpolate(t,y,at);valid &= ok
                shift=(int(meta.initial_measurement_us)-int(meta.decision_us))*1e-6
                query=at+shift;ok &= query>=0
                valid &= ok
                key=(wid,tuple(query[ok]))
                if key not in cache:cache[key]=predict(wid,query[ok].tolist())
                anchor,states=cache[key];anchor=np.asarray(anchor)[columns]
                pred=np.full_like(real,np.nan);pred[ok]=np.asarray(states)[:,columns]
                observed.append(real);predicted.append(pred);anchors.append(anchor)
            real_a,real_b=observed;pa,pb=predicted;aa,ab=anchors
            ya0,yb0=members[0][3][0],members[1][3][0]
            da,db=pa-aa,pb-ab;ra,rb=real_a-ya0,real_b-yb0
            for k,t in enumerate(at):
                for j,var in enumerate(variables):
                    rows.append(dict(pair_uid=pair.pair_uid,split=split,flight_a=pair.flight_a,flight_b=pair.flight_b,
                        dependency_component=int(pair.dependency_component),score=pair.score,strict=bool(pair.strict),
                        variable=var,time_s=float(t),time_kind=kind[k],supported=bool(valid[k]),
                        real=ra[k,j]-rb[k,j],pred=da[k,j]-db[k,j],
                        increment_error_a=da[k,j]-ra[k,j],increment_error_b=db[k,j]-rb[k,j],
                        absolute_error_a=pa[k,j]-real_a[k,j],absolute_error_b=pb[k,j]-real_b[k,j],
                        causal_real=(real_a[k,j]-aa[j])-(real_b[k,j]-ab[j]),
                        interpretation='factual_prediction_difference_vs_observational_increment_difference_not_counterfactual'))
        frame=pd.DataFrame(rows);summary=[]
        pairs=self.pairs[self.pairs.split==split]
        # Dependencies remain those of the full frozen cohort even if support is missing.
        for (var,kind),g in frame.groupby(['variable','time_kind']):
            groups=[('endpoint',g)] if kind=='supported_endpoint' else list(g.groupby('time_s'))
            for time,gg in groups:
                for weighting in ('pair','flight','component'):
                    supported=pairs[pairs.pair_uid.isin(gg.loc[gg.supported,'pair_uid'])]
                    if len(supported)==0:weights={}
                    elif weighting=='component':
                        counts=supported.dependency_component.value_counts()
                        weights={r.pair_uid:1/counts[r.dependency_component] for r in supported.itertuples()}
                    else:weights=self.analysis.weights(supported,weighting)
                    summary.append(dict(variable=var,time=time,kind=kind,weighting=weighting,
                        **self.analysis.metrics(gg,gg.pair_uid.map(weights).fillna(0).to_numpy(),self.config['direction_floors'][var])))
        component=[]
        for (var,comp),g in frame[frame.time_kind=='supported_endpoint'].groupby(['variable','dependency_component']):
            component.append(dict(variable=var,dependency_component=int(comp),**self.analysis.metrics(g,np.ones(len(g)),self.config['direction_floors'][var])))
        return dict(rows=frame,summary=summary,components=component)


def factual_predictor(model,windows):
    """Lazy identity adapter for FrozenRollResponse; accepts explicit trained model."""
    index=windows.index
    lookup={wid:i for i,wid in enumerate(index.window_id)}
    @torch.no_grad()
    def predict(window_id,times):
        w=windows[lookup[window_id]]
        if not times:return model.encode(w).x.cpu().numpy(),torch.empty(0,13).numpy()
        if min(times)<0 or max(times)>w.metadata['actual_horizon_s']+1e-9:raise ValueError('query outside factual support')
        unique=sorted(set(times));belief,states=model(w,unique);bytime=dict(zip(unique,states))
        return belief.x.cpu().numpy(),torch.stack([bytime[t].x for t in times]).cpu().numpy()
    return predict
