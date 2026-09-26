"""Fixed PX4 pitch component, guarded model rollouts and cross-model tape replay."""
from __future__ import annotations
import copy
from dataclasses import fields,replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

import run_paper_rollout_horizon_ablation as source
from system_identification.integration.control_dynamics_candidate import (
    load_candidate,JointSupport,support_vector,command_coordinates,FEATURE_NAMES)
from system_identification.models.control_ensemble import PHYSICAL,ControlDerivativeEnsemble
from system_identification.models.trajectory_main_v1 import _state_features
from system_identification.control.px4_pitch import PitchConfig,PitchInput,PX4PitchRateController
from system_identification.control.px4_flight_chain import FlightAllocator

m=source.m
OUT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1/px4_guarded_closed_loop'


def state_features(s):
    return _state_features(s.velocity_n,s.quaternion_nb,s.angular_velocity_b,s.relative_phase_rad,s.phase_anchor,s.flap_frequency_hz).numpy()


def freeze_inactive(previous,current,active):
    mask=torch.as_tensor(active)
    def choose(a,b):return torch.where(mask.reshape((-1,)+(1,)*(a.ndim-1)),b,a)
    return replace(current,**{f.name:(tuple(choose(a,b) for a,b in zip(previous.hidden,current.hidden)) if f.name=='hidden' else choose(getattr(previous,f.name),getattr(current,f.name))) for f in fields(current)})


def fit_support(batch,folds):
    x=support_vector(batch.history_state_features[:,-1],batch.trajectory.controls[:,0],batch.history_controls[:,-6])
    ids=batch.trajectory.log_ids
    mean=x.mean(0);scale=x.std(0).clip(1e-6);z=(x-mean)/scale
    distance=np.full(len(x),np.nan)
    assigned=np.array([folds[g] for g in ids])
    for j in range(5):
        held=assigned==j
        assert not set(ids[held])&set(ids[~held])
        distance[held]=cKDTree(z[~held]).query(z[held])[0]
    assert np.isfinite(distance).all()
    lo,hi=np.quantile(x,[.01,.99],axis=0)
    limit=float(np.quantile(distance,.99))
    np.savez_compressed(OUT/'support.npz',reference=x,mean=mean,scale=scale,lower=lo,upper=hi,distance_limit=limit,
        log_ids=ids.astype(str),oof_distance=distance)
    pd.DataFrame(dict(feature=FEATURE_NAMES,lower=lo,upper=hi,mean=mean,scale=scale)).to_csv(OUT/'feature_ranges.csv',index=False)
    return JointSupport(x,mean,scale,lo,hi,limit),limit


def warm_inputs(batch,indices):
    def tensor(x):return torch.as_tensor(x[indices],dtype=torch.float32)
    return dict(history_state_features=tensor(batch.history_state_features),history_controls=tensor(batch.history_controls),
        history_mask=torch.as_tensor(batch.history_mask[indices]),
        **{k:tensor(getattr(batch.trajectory.truth,k)[:,0]) for k in PHYSICAL})


def simulate(model,initial,u0,past,support,cfg,allocator,common_bounds,offset,mode,replay=None):
    n=len(u0);state=model.reset(**initial);past=past.copy()
    target=initial['angular_velocity_b'][:,1].numpy()+offset
    initial_check=support.query(support_vector(state_features(state),u0,past[:,-5]))
    active=initial_check['accepted']&(common_bounds[:,0]<=common_bounds[:,1])
    admitted=active.copy();termination=np.where(active,50,0)
    controllers=[PX4PitchRateController(cfg) for _ in range(n)]
    for controller in controllers:controller.warm_start(0.,cfg.airspeed_trim)
    rate=state.angular_velocity_b[:,1].numpy()
    previous=initial['history_state_features'][:,-2,4].numpy()
    feedback=np.zeros(n);tape=[];trace=[];last_command=u0.copy()
    for k in range(50):
        prior_active=active.copy()
        q=state.angular_velocity_b[:,1].numpy();acc=(q-previous)/.02;previous=q.copy()
        requested=u0.copy();guard=np.zeros(n,bool);torque=np.zeros(n)
        if replay is not None:
            requested=replay[:,k].copy()
        elif mode=='px4':
            for i in range(n):
                if not active[i]:continue
                for substep in range(8):
                    o=controllers[i].step(PitchInput(float(q[i]),float(target[i]),float(acc[i]),.0025,
                        float('nan'),float('inf'),saturation_positive=feedback[i]>np.finfo(np.float32).eps,
                        saturation_negative=feedback[i]<-np.finfo(np.float32).eps))
                    demand=allocator.effectiveness@u0[i,1:];demand[1]=o.torque
                    allocation=allocator.allocate(demand,float(u0[i,0]));raw=allocation.command
                    common=(raw[1]+raw[2])/2;diff=(u0[i,1]-u0[i,2])/2
                    projected=np.clip(common,*common_bounds[i])
                    requested[i]=[u0[i,0],projected+diff,projected-diff,u0[i,3]]
                    guard[i]|=abs(projected-common)>1e-8 or o.clipped or np.any(np.abs(allocation.raw_surfaces)>1)
                    feedback[i]=o.torque-float(allocator.effectiveness[1]@requested[i,1:])
                    torque[i]=o.torque
        request_check=support.query(support_vector(state_features(state),requested,past[:,-5]))
        reject=active&~request_check['accepted'];termination[reject]=k;active&=request_check['accepted']
        command=np.where(active[:,None],requested,last_command)
        nxt,diagnostic=model.step(state,torch.as_tensor(command,dtype=torch.float32),torch.full((n,),.02))
        finite=np.ones(n,bool)
        for name in PHYSICAL:finite&=np.isfinite(getattr(nxt,name).numpy()).reshape(n,-1).all(1)
        # Never silently substitute invalid prediction, even in an inactive lane.
        assert finite.all(),'nonfinite simulation state'
        nxt=freeze_inactive(state,nxt,active)
        new_past=np.concatenate([past[:,1:],command[:,None]],axis=1)
        next_check=support.query(support_vector(state_features(nxt),command,new_past[:,-5]))
        clipped=diagnostic['member_clipped'].any(0).numpy()|diagnostic['frequency_clipped'].numpy()
        leave=active&(~next_check['accepted']|clipped)
        termination[leave]=k+1
        valid=active&next_check['accepted']&~clipped
        for i in range(n):
            trace.append(dict(origin=i,step=k+1,time_s=(k+1)*.02,admitted=bool(admitted[i]),
                active_before=bool(prior_active[i]),accepted_step=bool(valid[i]),request_rejected=bool(reject[i]),
                state_left_support=bool(leave[i]),derivative_clipped=bool(clipped[i]),guarded=bool(guard[i]),
                q=float(nxt.angular_velocity_b[i,1]),target=float(target[i]),error=float(nxt.angular_velocity_b[i,1]-target[i]),
                common=float((command[i,1]+command[i,2])/2),requested_pitch_torque=float(torque[i]),
                distance=float(next_check['distance'][i]),derivative_q_std=float(diagnostic['derivative_std'][i,4])))
        tape.append(command.copy());state=nxt;past=new_past;last_command=command.copy();active=valid
    return pd.DataFrame(trace),np.stack(tape,axis=1),termination


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    candidate=OUT.parent/'ensemble_candidate'
    ep=m.read_json(candidate/'protocol.json');m.verify_pins(ep['pins'])
    done=m.read_json(candidate/'completion.json')
    model_path=candidate/'model.pt';hash_expected=done['outputs'][m.rel(model_path)]
    ensemble=load_candidate(model_path,hash_expected)
    models={f'seed{s}':ControlDerivativeEnsemble([copy.deepcopy(x)]).eval() for s,x in zip(m.SEEDS,ensemble.members)}
    models['ensemble']=ensemble
    sp,batches,stats=source.inputs();batch=batches['validation']
    param=m.ROOT/'docs/analysis/results/flight_chain_alignment/flight_parameters.json'
    alignment=m.read_json(param.parent/'manifest.json')
    assert m.file_hash(param)==alignment['parameter_sha256']
    parameters=m.read_json(param);cfg=PitchConfig.from_parameters(parameters);allocator=FlightAllocator(parameters)
    # Check preserved nonpitch torque is the existing physical allocation, not a new sign convention.
    np.testing.assert_allclose(allocator.effectiveness,np.array([[-.55,.55,0],[1,1,0],[0,0,1]]),atol=1e-7,rtol=0)
    ids=batch.trajectory.log_ids
    indices=np.concatenate([np.flatnonzero(ids==log)[np.linspace(0,(ids==log).sum()-1,3,dtype=int)] for log in sorted(set(ids))])
    identity=pd.DataFrame(dict(origin=np.arange(len(indices)),window_id=batch.trajectory.window_ids[indices],log_id=ids[indices],source_index=indices))
    identity.to_csv(OUT/'origins.csv',index=False)
    folds=m.read_json(OUT.parent/'local_identification/protocol.json')['folds']
    pins=dict(ep['pins'])
    for path in [Path(__file__),param,param.parent/'manifest.json',model_path,
        m.ROOT/'src/system_identification/integration/control_dynamics_candidate.py',
        m.ROOT/'src/system_identification/control/px4_pitch.py',m.ROOT/'src/system_identification/control/px4_flight_chain.py']:
        pins[m.rel(path)]=m.file_hash(path)
    m.write_json(OUT/'protocol.json',dict(experiment='guarded_px4_pitch_cross_model',pins=pins,
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(),heldout_accessed=False,training=False,parameter_source=alignment,
        origin_rule='three identity-spaced original validation origins per flight, no error selection',
        models=list(models),offsets_rad_s=[-.2,0.,.2],target='initial measured q + offset, all other channels held',
        loop='PX4 e624 pitch rate component and FlightAllocator; not complete flight firmware/outer-loop simulation',
        clock='8 controller updates at2.5ms per model20ms; rate held within substeps; causal backward20ms q derivative; initial from last2 history rows',
        initialization='I=0; airspeed filter=trim; airspeed unavailable uses PX4 trim fallback; not flight controller state reconstruction',
        allocator='preserve initial roll/yaw torque and motor; replace pitch demand from controller; common projected, differential/rudder held; previous actual unallocated pitch feeds antiwindup',
        guard='common change<=0.1 train origin-command common std; raw command q01..q99; no gain/physical/parameter retuning',
        support='18D body velocity/rate/gravity/frequency, command coordinates and command delta5; train q01..q99 marginal AND nearest standardized train distance<=train whole-flight5fold OOFq99',
        support_limit='coverage screen, not identified causal support; global train mean/std fixed before crossfit distances; no validation calibration',
        termination='reject proposed command outside joint support before step; terminate next state on support exit/derivative or frequency clipping; keep failures and no survivor-only success claim',
        replay='all four source PX4 tapes replayed to other three models without feedback recomputation; compare only common accepted intervals, report source/destination survival',
        evaluation='100/200ms primary,500ms/1s stress; tracking vs same-model open_hold and source vs transferred tape; no controller optimization or model selection from closure',
        claims='within-model component test only, independent initializations share data/architecture, no real causal validation'))
    support,limit=fit_support(batches['train'],folds)
    raw=batches['train'].trajectory.controls[:,0]
    rlo,rhi=np.quantile(raw,[.01,.99],axis=0)
    epsilon=.1*command_coordinates(raw)[:,1].std()
    u0=batch.trajectory.controls[indices,0].copy();common=(u0[:,1]+u0[:,2])/2;diff=(u0[:,1]-u0[:,2])/2
    bounds=np.stack([np.maximum.reduce([common-epsilon,rlo[1]-diff,rlo[2]+diff]),
                     np.minimum.reduce([common+epsilon,rhi[1]-diff,rhi[2]+diff])],axis=1)
    m.write_json(OUT/'guard.json',dict(raw_lower=rlo.tolist(),raw_upper=rhi.tolist(),common_delta_limit=epsilon,joint_distance_limit=limit))
    initial=warm_inputs(batch,indices);past=batch.history_controls[indices,:-1].copy()
    alltraces=[];tapes={}
    with torch.inference_mode():
        for name,model in models.items():
            for offset in [-.2,0.,.2]:
                for mode in ['open_hold','px4']:
                    f,t,stop=simulate(model,initial,u0,past,support,cfg,allocator,bounds,offset,mode)
                    alltraces.append(f.assign(model=name,source_model=name,offset=offset,mode=mode))
                    if mode=='px4':tapes[(name,offset)]=t
                print('feedback',name,offset,flush=True)
        for (src,offset),tape in tapes.items():
            for dst,model in models.items():
                if src==dst:continue
                f,_,_=simulate(model,initial,u0,past,support,cfg,allocator,bounds,offset,'replay',replay=tape)
                alltraces.append(f.assign(model=dst,source_model=src,offset=offset,mode='replay'))
            print('replayed',src,offset,flush=True)
    traces=pd.concat(alltraces,ignore_index=True).merge(identity[['origin','window_id','log_id']],on='origin',validate='many_to_one')
    traces.to_csv(OUT/'traces.csv',index=False)
    np.savez_compressed(OUT/'command_tapes.npz',**{f'{n}_{o}':t for (n,o),t in tapes.items()})
    keys=['model','source_model','mode','offset','origin','window_id','log_id']
    rows=[]
    for key,g in traces.groupby(keys,sort=False):
        g=g.sort_values('step')
        for steps in [5,10,25,50]:
            prefix=g[g.step<=steps];valid=prefix.accepted_step.to_numpy()
            rows.append(dict(zip(keys,key),steps=steps,admitted=bool(prefix.admitted.iloc[0]),
                completed=bool(valid.all()),accepted_steps=int(valid.sum()),
                rmse=float(np.sqrt(np.mean(prefix.error.to_numpy()[valid]**2))) if valid.any() else None,
                guard_fraction=float(prefix.guarded.mean()),terminal_error=float(prefix.error.iloc[-1]) if valid.all() else None))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'per_origin.csv',index=False)
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,control_ready=False,
        models=list(models),origins=len(indices),rollout_groups=len(alltraces),outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
