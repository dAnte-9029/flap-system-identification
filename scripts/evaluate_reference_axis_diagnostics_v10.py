"""Freeze yaw OOF gate before any conditional development evaluation."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import SEEDS
from reference_error_structure_v9 import history_features,fit_predict
from reference_capacity_v6 import metric_frames
from evaluate_control_windows_v2 import METRICS
from run_reference_axis_diagnostics_v10 import OUT,V8,build_model,predict,read_prediction,rate_rows
from run_reference_generalization_v7 import verify_manifest


def gate(rate,state,cohort,name):
    r=rate[rate.cohort==cohort].copy();flights=sorted(r.log_id.unique())
    f=r.groupby(['arm','seed','log_id','horizon_s'])[['r_mean_mse','r_wave_mse']].mean().pow(.5).reset_index()
    f.to_csv(OUT/f'{name}_yaw_per_flight.csv',index=False)
    a=f[f.horizon_s==1.];cube=np.zeros((2,3,len(flights),2))
    for ai,arm in enumerate(['baseline','yaw']):
        for si,seed in enumerate(SEEDS):
            for fi,flight in enumerate(flights):cube[ai,si,fi]=a[(a.arm==arm)&(a.seed==seed)&(a.log_id==flight)][['r_mean_mse','r_wave_mse']].iloc[0]
    scores=cube.mean(2);seed_gain=1-scores[1]/scores[0];gains=1-scores[1].mean(0)/scores[0].mean(0)
    flight_gain=1-cube[1].mean(0)/cube[0].mean(0)
    s=state[(state.cohort==cohort)&(state.subset=='all')&state.horizon_s.isin([.5,1.])].groupby(['arm','seed','horizon_s'])[METRICS].mean()
    regression=s.loc['yaw']/s.loc['baseline']-1;regression.to_csv(OUT/f'{name}_regressions.csv')
    rng=np.random.default_rng(910);boot=[]
    for _ in range(2000):
        ids=rng.integers(len(flights),size=len(flights));mean=cube[:,:,ids].mean(axis=(1,2));boot.append(1-mean[1]/mean[0])
    protected=bool((regression<=.05).all().all())
    passed=bool(gains[0]>=.10 and gains[1]>=.05 and (seed_gain>0).all() and (flight_gain[:,1]>0).sum()>=min(4,len(flights)) and protected)
    absolute=[]
    for seed in SEEDS:
        row=s.loc[('yaw',seed,.5)];absolute.append(dict(seed=seed,p=float(row.p),q=float(row.q),r=float(row.r),velocity=float(row.velocity),attitude_deg=float(np.rad2deg(row.attitude)),
            passed=bool((row[['p','q','r']]<=.15).all() and row.velocity<=.5 and row.attitude<=np.deg2rad(3))))
    return dict(cohort=cohort,mean_gain=float(gains[0]),waveform_gain=float(gains[1]),seed_gains_mean_waveform=seed_gain.tolist(),
        bootstrap95_mean_waveform=np.quantile(boot,[.025,.975],axis=0).tolist(),
        flight_gains=[dict(log_id=log,mean_gain=float(v[0]),waveform_gain=float(v[1])) for log,v in zip(flights,flight_gain)],
        maximum_protected_regression=float(regression.to_numpy().max()),protected_metrics_pass=protected,relative_gate=passed,absolute=absolute,control_validated=False)


def main():
    torch.set_num_threads(1);assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    assert not (OUT/'oof_decision.json').exists()
    rate=pd.read_csv(OUT/'rate_per_window.csv');state=pd.read_csv(OUT/'state_per_flight.csv')
    decision=gate(rate,state,'oof','oof');base.write(OUT/'oof_decision.json',decision);print(json.dumps(decision,indent=2),flush=True)
    state.groupby(['arm','cohort','seed','log_id','subset','horizon_s'])[METRICS].mean().groupby(['arm','cohort','subset','horizon_s']).mean().to_csv(OUT/'state_summary.csv')
    if not decision['relative_gate']:
        base.write(OUT/'development_status.json',dict(status='not_evaluated',reason='frozen single-axis OOF gate failed',decision_sha256=base.sha(OUT/'oof_decision.json')));return
    source=OUT.parent/'control_windows_v6';verify_manifest(source/'delivery_manifest.json',['artifacts','code','source_pins'])
    x,y,_=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');fit=w[w.partition=='train'].array_index.to_numpy();query=w[w.partition=='validation'].array_index.to_numpy()
    feature=history_features(x,'full');target=np.cumsum(y['rates_b'][fit,:,2].numpy().astype(float),1)/np.arange(1,51)
    _,params=fit_predict(feature[fit],target,feature[query],w.iloc[fit].log_id.to_numpy(),'ridge')
    dest=OUT/'development';dest.mkdir();np.savez_compressed(dest/'mean_predictor.npz',fit_indices=fit,query_indices=query,**params)
    frames=[];rates=[];meta=w.iloc[query];truth={k:v[query].numpy() for k,v in y.items()}
    for seed in SEEDS:
        parent=source/'formal'/f'integrated_{seed}';model=build_model(parent/'step_1600.pt',params)
        torch.save(dict(state_dict=model.state_dict(),seed=seed,enabled=True,architecture='ReferenceGRUYawMean'),dest/f'candidate_{seed}.pt')
        candidate,bias=predict(model,x,query);old=read_prediction(parent/'1600_validation_predictions.npz',query)
        np.savez_compressed(dest/f'predictions_{seed}.npz',array_index=query,yaw_correction=bias,**candidate)
        for arm,p in [('baseline',old),('yaw',candidate)]:
            f,_=metric_frames(p,truth,meta,arm,seed);f['cohort']='development';frames.append(f)
            rates.append(rate_rows(p,truth,meta,arm,seed,-1,'development'))
    sf=pd.concat(frames);rf=pd.concat(rates);sf.to_csv(dest/'state_per_flight.csv',index=False);rf.to_csv(dest/'rate_per_window.csv',index=False)
    base.write(OUT/'development_decision.json',gate(rf,sf,'development','development'))
    base.write(OUT/'development_status.json',dict(status='complete',decision_sha256=base.sha(OUT/'oof_decision.json'),evaluation_after_oof_decision=True))

if __name__=='__main__':main()
