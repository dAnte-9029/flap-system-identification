"""Separate training-budget gains from matched-budget output gains."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT,model_from,metric_frames
from control_windows_v3 import predict
from evaluate_control_windows_v2 import METRICS


def compare(frame,candidate,baseline,name):
    a=frame[(frame.cohort=='validation')&frame.horizon_s.isin([.5,1.])];flights=sorted(a.log_id.unique());cube=np.full((2,3,len(flights),4),np.nan)
    for ai,arm in enumerate([baseline,candidate]):
        for si,seed in enumerate([17,23,42]):
            for fi,flight in enumerate(flights):
                for ki,(subset,axis,h) in enumerate([(s,ax,h) for s,ax in [('roll_active','p'),('pitch_active','q')] for h in [.5,1.]]):
                    z=a[(a.arm==arm)&(a.seed==seed)&(a.log_id==flight)&(a.subset==subset)&(a.horizon_s==h)]
                    if len(z):cube[ai,si,fi,ki]=z.iloc[0][axis]
    scores=np.nanmean(np.nanmean(cube,axis=2),axis=2);gain=1-scores[1]/scores[0];mean_gain=float(1-scores[1].mean()/scores[0].mean())
    rng=np.random.default_rng(906);boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(flights),len(flights));s=np.nanmean(np.nanmean(cube[:,:,ix,:],axis=2),axis=(1,2));boot.append(1-s[1]/s[0])
    allm=a[a.subset=='all'].groupby(['arm','seed','horizon_s'])[METRICS].mean();regression=allm.loc[candidate]/allm.loc[baseline]-1
    regression.to_csv(OUT/f'{name}_regressions.csv')
    return dict(comparison=name,candidate=candidate,baseline=baseline,primary_gain_fraction=mean_gain,seed_gain_fraction=gain.tolist(),
        baseline_seed_scores=scores[0].tolist(),candidate_seed_scores=scores[1].tolist(),paired_flight_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        maximum_all_metric_regression=float(regression.to_numpy().max()),relative_pass=bool(mean_gain>=.05 and (gain>0).all() and (regression<=.05).all().all()))


def main():
    torch.set_num_threads(1);status=json.loads((OUT/'formal_launch/status.json').read_text())
    if status['status']=='not_launched':base.write(OUT/'formal_decision.json',dict(status='not_launched',control_validated=False));return
    assert status['status']=='complete';frames=[];waves=[]
    for run in status['completed']:
        path=OUT/'formal'/f"{run['arm']}_{run['seed']}";s=json.loads((path/'status.json').read_text());assert s['status']=='complete'
        for ck in s['completed']:
            step=ck['step'];assert base.sha(path/f'step_{step}.pt')==ck['checkpoint_sha256']
            f=pd.read_csv(path/f'{step}_per_flight.csv');wa=pd.read_csv(path/f'{step}_waveform.csv');f['arm']=f'{run["arm"]}{step}';wa['arm']=f'{run["arm"]}{step}';frames.append(f);waves.append(wa)
    frame=pd.concat(frames);pd.concat(waves).to_csv(OUT/'formal_waveforms.csv',index=False);frame.to_csv(OUT/'formal_per_flight.csv',index=False)
    frame.groupby(['arm','cohort','subset','horizon_s'])[METRICS].mean().to_csv(OUT/'formal_summary.csv')
    comparisons=[compare(frame,'integrated6400','integrated1600','budget')]
    if 'direct6400' in set(frame.arm):
        comparisons.extend([compare(frame,'direct1600','integrated1600','output1600'),compare(frame,'direct6400','integrated6400','output6400'),compare(frame,'direct6400','integrated1600','combined')])
    absolute=[]
    for arm in sorted(frame.arm.unique()):
        for seed in [17,23,42]:
            z=frame[(frame.arm==arm)&(frame.seed==seed)&(frame.cohort=='validation')&(frame.horizon_s==.5)]
            p=float(z[z.subset=='roll_active'].p.mean());q=float(z[z.subset=='pitch_active'].q.mean());other=z[z.subset=='all'][['attitude','velocity']].mean()
            absolute.append(dict(arm=arm,seed=seed,active_p=p,active_q=q,attitude_deg=float(np.rad2deg(other.attitude)),velocity=float(other.velocity),
                passed=bool(p<=.15 and q<=.15 and other.attitude<=np.deg2rad(3) and other.velocity<=.5)))
    pd.DataFrame(absolute).to_csv(OUT/'absolute_gates.csv',index=False)
    # Frozen final-checkpoint reference ablation, no optimizer or future measurement feedback.
    x,y,_,_=None,None,None,None
    x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');meta=w[w.partition=='validation'];ix=meta.array_index.to_numpy();response=[]
    for run in status['completed']:
        arm=run['arm'];seed=run['seed'];path=OUT/'formal'/f'{arm}_{seed}'/'step_6400.pt';model=model_from(path)
        pp=predict(model,x,ix,hold=True);f,_=metric_frames(pp,{k:v[ix].numpy() for k,v in y.items()},meta,f'{arm}6400',seed);f['intervention']='held';response.append(f)
        original=frame[(frame.arm==f'{arm}6400')&(frame.seed==seed)&(frame.cohort=='validation')].copy();original['intervention']='observed';response.append(original)
    rr=pd.concat(response);rr.to_csv(OUT/'reference_hold_per_flight.csv',index=False);rr.groupby(['arm','intervention','subset','horizon_s'])[METRICS].mean().to_csv(OUT/'reference_hold_summary.csv')
    base.write(OUT/'formal_decision.json',dict(status='complete',comparisons=comparisons,absolute_by_seed=absolute,
        independent_causal_control_response_validated=False,controller_optimization_started=False,control_validated=False,
        response_scope='reference ablation demonstrates model reliance, not measured counterfactual truth',sealed_test_accessed=False))
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
