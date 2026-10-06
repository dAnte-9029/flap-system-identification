"""Freeze train-flight OOF decision, then audit baseline next-day generalization."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import OUT,SEEDS,model_from,predict,load_train
from reference_capacity_v6 import metric_frames
from evaluate_control_windows_v2 import METRICS


def compare(frame, cohort='oof', prefix='cv', minimum_positive_flights=4):
    f=frame[(frame.cohort==cohort)&frame.horizon_s.isin([.5,1.])]
    flights=sorted(f.log_id.unique());cube=np.full((2,3,len(flights),4),np.nan)
    for ai,arm in enumerate(['hold','phase']):
        for si,seed in enumerate(SEEDS):
            for fi,log in enumerate(flights):
                for ki,(subset,metric,h) in enumerate([(s,m,t) for s,m in [('roll_active','p'),('pitch_active','q')] for t in [.5,1.]]):
                    row=f[(f.arm==arm)&(f.seed==seed)&(f.log_id==log)&(f.subset==subset)&(f.horizon_s==h)]
                    if len(row):cube[ai,si,fi,ki]=row.iloc[0][metric]
    scores=np.nanmean(np.nanmean(cube,axis=2),axis=2);gains=1-scores[1]/scores[0]
    flight_scores=np.nanmean(cube,axis=(1,3));flight_gains=1-flight_scores[1]/flight_scores[0]
    mean_gain=float(1-scores[1].mean()/scores[0].mean())
    allm=f[f.subset=='all'].groupby(['arm','seed','horizon_s'])[METRICS].mean()
    reg=allm.loc['phase']/allm.loc['hold']-1;reg.to_csv(OUT/f'{prefix}_metric_regressions.csv')
    rng=np.random.default_rng(908);boot=[]
    for _ in range(2000):
        ix=rng.integers(len(flights),size=len(flights));s=np.nanmean(np.nanmean(cube[:,:,ix,:],axis=2),axis=(1,2));boot.append(1-s[1]/s[0])
    passed=bool(mean_gain>=.05 and (gains>0).all() and (flight_gains>0).sum()>=minimum_positive_flights and (reg<=.05).all().all())
    absolute=[]
    for arm in ['hold','phase']:
        for seed in SEEDS:
            a=f[(f.arm==arm)&(f.seed==seed)&(f.horizon_s==.5)]
            p=float(a[a.subset=='roll_active'].p.mean());q=float(a[a.subset=='pitch_active'].q.mean());o=a[a.subset=='all'][['velocity','attitude']].mean()
            absolute.append(dict(arm=arm,seed=seed,p=p,q=q,velocity=float(o.velocity),attitude_deg=float(np.rad2deg(o.attitude)),
                passed=bool(p<=.15 and q<=.15 and o.velocity<=.5 and o.attitude<=np.deg2rad(3))))
    return dict(evaluation_cohort=cohort,phase_primary_gain=mean_gain,seed_gains=gains.tolist(),baseline_seed_scores=scores[0].tolist(),phase_seed_scores=scores[1].tolist(),
        flight_gains=[dict(log_id=log,gain=float(g)) for log,g in zip(flights,flight_gains)],bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        maximum_allmetric_regression=float(reg.to_numpy().max()),cv_gate=passed,absolute=absolute,
        full_phase_training_authorized_by_protocol=passed,selection_uses_development_validation=False,control_validated=False)


def main():
    torch.set_num_threads(1)
    status=json.loads((OUT/'launch/status.json').read_text());assert status['status']=='complete' and len(status['completed'])==30
    assert not (OUT/'cv_decision.json').exists(),'decision immutable; use separate followup audit'
    frames=[];waves=[];checks=[];x,y,w=load_train()
    for fold in range(5):
        for seed in SEEDS:
            pair=[]
            for arm in ['hold','phase']:
                path=OUT/f'fold{fold}'/f'{arm}_{seed}';run=json.loads((path/'run_protocol.json').read_text());s=json.loads((path/'status.json').read_text())
                assert s['status']=='complete' and base.sha(path/'final.pt')==s['checkpoint_sha256']
                assert base.sha(path/'sample_draws.npz')==s['draws_sha256']
                frames.append(pd.read_csv(path/'per_flight.csv'));waves.append(pd.read_csv(path/'waveforms.csv'))
                pair.append((run,s));query=np.array(run['query_indices']);schedule=None
                if arm=='phase':
                    with np.load(path.parent/'phase_schedules.npz') as z:schedule=torch.tensor(z['phase_start'])
                model=model_from(path/'final.pt');pp=predict(model,x,query[:4],schedule)
                with np.load(path/'oof_predictions.npz') as z:maximum=max(float(np.max(abs(pp[k]-z[k][:4]))) for k in y)
                assert maximum<1e-5
                # Future reference changes must not alter the first five predictions.
                ids=query[:2];batch={k:v[ids].clone() for k,v in x.items()}
                if schedule is not None:batch['phase_start']=schedule[ids].clone()
                with torch.no_grad():
                    before=model(**batch);batch['future_reference'][:,5:]+=10;after=model(**batch)
                assert all(torch.equal(before[k][:,:5],after[k][:,:5]) for k in before)
                checks.append(dict(fold=fold,arm=arm,seed=seed,reload_max_abs=maximum,reference_prefix_causal=True))
            assert pair[0][0]['initial_parameter_sha256']==pair[1][0]['initial_parameter_sha256']
            # Compressed archive timestamps differ; compare the actual sample IDs.
            with np.load(OUT/f'fold{fold}'/f'hold_{seed}/sample_draws.npz') as a,np.load(OUT/f'fold{fold}'/f'phase_{seed}/sample_draws.npz') as b:
                np.testing.assert_array_equal(a['array_index'],b['array_index'])
    frame=pd.concat(frames);frame.to_csv(OUT/'cv_per_flight.csv',index=False);pd.concat(waves).to_csv(OUT/'cv_waveforms.csv',index=False)
    # Fit-flight errors occur in four models: reduce repeated source flights first.
    frame.groupby(['arm','cohort','seed','log_id','subset','horizon_s'])[METRICS].mean().groupby(['arm','cohort','subset','horizon_s']).mean().to_csv(OUT/'cv_summary.csv')
    decision=compare(frame);base.write(OUT/'cv_decision.json',decision);print(json.dumps(decision,indent=2),flush=True)
    base.write(OUT/'verification.json',dict(checkpoints=checks,paired_initialization_and_draws_identical=True,source_flight_count=5,sealed_test_accessed=False))
    # Decision is now frozen. Matched-model, same-training-size next-day baseline audit.
    xall,yall,_=base.load();wall=pd.read_csv(base.DATA/'windows_v2.csv');meta=wall[wall.partition=='validation'];ids=meta.array_index.to_numpy();vf=[]
    for fold in range(5):
        for seed in SEEDS:
            model=model_from(OUT/f'fold{fold}'/f'hold_{seed}/final.pt');p=predict(model,xall,ids)
            path=OUT/f'fold{fold}'/f'hold_{seed}/development_predictions.npz';np.savez_compressed(path,array_index=ids,**p)
            f,_=metric_frames(p,{k:v[ids].numpy() for k,v in yall.items()},meta,'hold',seed);f['cohort']='development';f['fold']=fold;vf.append(f)
    val=pd.concat(vf);val.to_csv(OUT/'development_per_flight.csv',index=False)
    combined=pd.concat([frame[frame.arm=='hold'],val]);combined.groupby(['cohort','seed','log_id','subset','horizon_s'])[METRICS].mean().groupby(['cohort','subset','horizon_s']).mean().to_csv(OUT/'baseline_generalization_summary.csv')
    base.write(OUT/'evaluation_status.json',dict(status='complete',decision_sha256=base.sha(OUT/'cv_decision.json'),validation_evaluated_after_decision=True))

if __name__=='__main__':main()
