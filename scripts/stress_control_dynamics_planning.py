"""One-shot bounded command search; expose model-dependent apparent improvement."""
from dataclasses import fields,is_dataclass,replace
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch

import run_paper_rollout_horizon_ablation as source
from identify_control_dynamics_local import design,predict
from run_control_dynamics_closed_loop import warm_inputs,state_features
from system_identification.integration.control_dynamics_candidate import load_candidate,JointSupport,support_vector
from system_identification.models.control_ensemble import ControlDerivativeEnsemble

m=source.m
ROOT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=ROOT/'planning_stress'


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    ep=m.read_json(ROOT/'ensemble_candidate/protocol.json');m.verify_pins(ep['pins'])
    for name in ['ensemble_candidate','local_identification','px4_guarded_closed_loop']:
        m.verify_pins(m.read_json(ROOT/name/'completion.json')['outputs'])
    _,batches,_=source.inputs();batch=batches['validation']
    origin=pd.read_csv(ROOT/'px4_guarded_closed_loop/origins.csv');ix=origin.source_index.to_numpy()
    def take(x):
        if is_dataclass(x):return replace(x,**{f.name:take(getattr(x,f.name)) for f in fields(x)})
        return x[ix].copy() if isinstance(x,np.ndarray) else x
    b=take(batch);n=len(ix)
    bundle=ROOT/'ensemble_candidate/model.pt'
    candidate=load_candidate(bundle,m.read_json(ROOT/'ensemble_candidate/completion.json')['outputs'][m.rel(bundle)])
    models={f'seed{s}':ControlDerivativeEnsemble([copy.deepcopy(v)]) for s,v in zip(m.SEEDS,candidate.members)}
    models['ensemble']=candidate
    guard=m.read_json(ROOT/'px4_guarded_closed_loop/guard.json')
    with np.load(ROOT/'px4_guarded_closed_loop/support.npz',allow_pickle=False) as z:
        support=JointSupport(**{k:z[k] for k in ['reference','mean','scale','lower','upper','distance_limit']})
    pins=dict(ep['pins'])
    for path in [Path(__file__),ROOT/'px4_guarded_closed_loop/guard.json',ROOT/'px4_guarded_closed_loop/support.npz',
        ROOT/'px4_guarded_closed_loop/origins.csv',ROOT/'local_identification/planned_k5.npz',ROOT/'local_identification/planned_k10.npz',
        m.ROOT/'scripts/identify_control_dynamics_local.py',m.ROOT/'scripts/run_control_dynamics_closed_loop.py']:
        pins[m.rel(path)]=m.file_hash(path)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        experiment='bounded_single_shooting_model_exploitation_stress',heldout_accessed=False,training=False,
        origins='same51 identity-selected origins, initial histories only',
        actions='hold common initial +[0,-epsilon,+epsilon], epsilon from previous train-only guard; clamp same raw bounds; preserve motor/differential/rudder',
        planning='one-shot constant-command100/200ms; not receding-horizon deployed MPC',
        targets='q0+[-.2,0,.2]; no future true state consulted',
        policies=['nominal_seed17','ensemble','worst_of_three_seeds'],
        objective='terminal squared pitch-rate target error, no tuning; ties prefer unchanged command; discard source-unsupported actions; robust requires all3 supported',
        evaluation='compare selected action vs unchanged action on each frozen model; require both paths supported; includes rejected origins',
        alternative='train-only local planned-command ridge forecast; poorer predictor and NOT causal truth, included to reveal shared-family reliance',
        decision='no controller promotion from gains in any model; benefit reversal flags model dependence, never real harm estimate'))
    u0=b.trajectory.controls[:,0];c=(u0[:,1]+u0[:,2])/2;d=(u0[:,1]-u0[:,2])/2
    lo=np.maximum.reduce([c-guard['common_delta_limit'],guard['raw_lower'][1]-d,guard['raw_lower'][2]+d])
    hi=np.minimum.reduce([c+guard['common_delta_limit'],guard['raw_upper'][1]-d,guard['raw_upper'][2]+d])
    actions=[]
    for factor in [0,-1,1]:
        u=u0.copy();common=np.clip(c+factor*guard['common_delta_limit'],lo,hi)
        u[:,1]=common+d;u[:,2]=common-d;actions.append(u)
    # Original command (index0) is exact for admissible origins.
    names=list(models);qhat={};ok={};ridge={};initial=warm_inputs(b,np.arange(n));q0=initial['angular_velocity_b'][:,1].numpy()
    with torch.inference_mode():
        for name,model in models.items():
            for a,command in enumerate(actions):
                state=model.reset(**initial);past=b.history_controls[:,:-1].copy();accepted=lo<=hi
                for k in range(10):
                    pre=support.query(support_vector(state_features(state),command,past[:,-5]))
                    state,diag=model.step(state,torch.tensor(command,dtype=torch.float32),torch.full((n,),.02))
                    past=np.concatenate([past[:,1:],command[:,None]],axis=1)
                    post=support.query(support_vector(state_features(state),command,past[:,-5]))
                    accepted&=pre['accepted']&post['accepted']&~diag['member_clipped'].any(0).numpy()&~diag['frequency_clipped'].numpy()
                    if k+1 in (5,10):qhat[(name,a,k+1)]=state.angular_velocity_b[:,1].numpy().copy();ok[(name,a,k+1)]=accepted.copy()
        for steps in [5,10]:
            with np.load(ROOT/f'local_identification/planned_k{steps}.npz',allow_pickle=False) as z:
                linear={k:z[k] for k in ['mean','scale','coef','intercept']}
            for a,command in enumerate(actions):
                altered=copy.deepcopy(b);altered.trajectory.controls[:]=command[:,None]
                ridge[(a,steps)]=q0+predict(linear,design(altered,steps)[1])[:,4]
    records=[];selections=[]
    for steps in (5,10):
        for offset in [-.2,0.,.2]:
            target=q0+offset
            for policy in ['nominal_seed17','ensemble','worst_of_three_seeds']:
                ref=['seed17'] if policy=='nominal_seed17' else ['ensemble'] if policy=='ensemble' else ['seed17','seed23','seed42']
                score=np.stack([np.max(np.stack([(qhat[(r,a,steps)]-target)**2 for r in ref]),axis=0) for a in range(3)])
                admitted=np.stack([np.stack([ok[(r,a,steps)] for r in ref]).all(0) for a in range(3)])
                score=np.where(admitted,score,np.inf);choose=score.argmin(0);valid=np.isfinite(score.min(0))
                for i in range(n):
                    a=int(choose[i]);selected=actions[a][i]
                    selections.append(dict(policy=policy,steps=steps,offset=offset,origin=i,window_id=origin.window_id.iloc[i],
                        admitted=bool(valid[i]),action_index=a,common=float((selected[1]+selected[2])/2),common_delta=float((selected[1]+selected[2])/2-c[i])))
                    for dest in [*names,'observational_ridge']:
                        assess=bool(valid[i] and (True if dest=='observational_ridge' else ok[(dest,a,steps)][i] and ok[(dest,0,steps)][i]))
                        q=(ridge[(a,steps)][i] if dest=='observational_ridge' else qhat[(dest,a,steps)][i])
                        hold=(ridge[(0,steps)][i] if dest=='observational_ridge' else qhat[(dest,0,steps)][i])
                        effect=abs(q-target[i])-abs(hold-target[i])
                        records.append(dict(policy=policy,destination=dest,steps=steps,offset=offset,origin=i,
                            log_id=origin.log_id.iloc[i],window_id=origin.window_id.iloc[i],assessable=assess,
                            action_index=a,selected_abs_error=float(abs(q-target[i])),hold_abs_error=float(abs(hold-target[i])),
                            effect=float(effect) if assess else None,beneficial=bool(assess and effect< -1e-6),harmful=bool(assess and effect>1e-6)))
    pd.DataFrame(selections).to_csv(OUT/'selected_actions.csv',index=False)
    f=pd.DataFrame(records);f.to_csv(OUT/'per_origin.csv',index=False)
    f.groupby(['policy','destination','steps','offset']).agg(requested=('origin','size'),assessable=('assessable','sum'),
        beneficial=('beneficial','sum'),harmful=('harmful','sum')).to_csv(OUT/'counts.csv')
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,controller_approved=False,
        outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
