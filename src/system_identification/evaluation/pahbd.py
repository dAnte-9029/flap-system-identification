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
    return torch.autograd.functional.jacobian(lambda a:model.flow.mixer(c,a,state.phi,state.phase_quality)[0],state.actuator,
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
