"""Frozen formal comparison on common targets, horizon and flight weighting."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT,load,causal_expand_50hz
from train_reference_bandwidth_v5 import metrics
from evaluate_control_windows_v2 import METRICS


def main():
    torch.set_num_threads(1);status=json.loads((OUT/'formal_launch/status.json').read_text());assert status['status']=='complete'
    x,y,stats,w=load();parts=[];waves=[];pins={}
    for run in status['completed']:
        path=OUT/'formal'/f"{run['arm']}_{run['seed']}";manifest=json.loads((path/'status.json').read_text());assert manifest['status']=='complete'
        assert base.sha(path/'final.pt')==manifest['checkpoint_sha256'];pins[str(path/'final.pt')]=manifest['checkpoint_sha256']
        parts.append(pd.read_csv(path/'per_flight.csv'));waves.append(pd.read_csv(path/'waveform_errors.csv'))
    # Reuse verified frozen A1600 forecasts, not another training/evaluation selection.
    v4=base.ROOT/'docs/analysis/results/control_windows_v4';oldmanifest=json.loads((v4/'delivery_manifest.json').read_text())
    for seed in [17,23,42]:
        for partition in ['train','validation']:
            meta=w[w.partition==partition];ix=meta.array_index.to_numpy();p=v4/'slow_objective'/f'A1600_{seed}_{partition}_predictions.npz'
            assert base.sha(p)==oldmanifest['artifacts'][str(p.relative_to(base.ROOT))]
            with np.load(p) as z:
                np.testing.assert_array_equal(z['array_index'],ix);pred={k:torch.tensor(z[k]) for k in y}
            initial={'rates_b':x['initial_rates_b'][ix],'velocity_n':x['initial_velocity_n'][ix],'quaternion':x['initial_quaternion'][ix]}
            pp={k:v.numpy() for k,v in causal_expand_50hz(pred,initial).items()}
            f,wa=metrics(pp,{k:v[ix] for k,v in y.items()},x['history_state'][ix,:,3:6],meta,'A1600_50hz',seed);f['cohort']=partition;wa['cohort']=partition;parts.append(f);waves.append(wa)
    allf=pd.concat(parts);allw=pd.concat(waves);allf.to_csv(OUT/'formal_per_flight.csv',index=False);allw.to_csv(OUT/'formal_waveform_errors.csv',index=False)
    allf.groupby(['arm','cohort','readout','subset','horizon_s'])[METRICS].mean().to_csv(OUT/'formal_summary.csv')
    records=[];scores=[];absrows=[]
    for candidate,baseline,readout in [('raw100','A1600_50hz','raw50'),('aa50','raw100','aa50')]:
        a=allf[(allf.cohort=='validation')&(allf.readout==readout)&allf.horizon_s.isin([.5,1.])]
        flights=sorted(a.log_id.unique());cube=np.full((2,3,len(flights),4),np.nan)
        for ai,arm in enumerate([baseline,candidate]):
            for si,seed in enumerate([17,23,42]):
                for fi,flight in enumerate(flights):
                    for ki,(subset,axis,hor) in enumerate([(s,ax,h) for s,ax in [('roll_active','p'),('pitch_active','q')] for h in [.5,1.]]):
                        z=a[(a.arm==arm)&(a.seed==seed)&(a.log_id==flight)&(a.subset==subset)&(a.horizon_s==hor)]
                        if len(z):cube[ai,si,fi,ki]=z.iloc[0][axis]
        score=np.nanmean(np.nanmean(cube,axis=2),axis=2);gain=1-score[1]/score[0];mean_gain=float(1-score[1].mean()/score[0].mean())
        for si,seed in enumerate([17,23,42]):scores.append(dict(candidate=candidate,baseline=baseline,readout=readout,seed=seed,baseline_score=score[0,si],candidate_score=score[1,si],gain_fraction=gain[si]))
        rng=np.random.default_rng(905);boot=[]
        for _ in range(2000):
            ids=rng.integers(0,len(flights),len(flights));b=np.nanmean(np.nanmean(cube[:,:,ids,:],axis=2),axis=(1,2));boot.append(1-b[1]/b[0])
        ci=np.quantile(boot,[.025,.975]).tolist()
        allmetrics=a[a.subset=='all'].groupby(['arm','seed','horizon_s'])[METRICS].mean();regression=allmetrics.loc[candidate]/allmetrics.loc[baseline]-1
        regression.to_csv(OUT/f'{candidate}_relative_regressions.csv')
        abs_pass=[]
        for seed in [17,23,42]:
            part=a[(a.arm==candidate)&(a.seed==seed)&(a.horizon_s==.5)]
            p=float(part[part.subset=='roll_active'].p.mean());q=float(part[part.subset=='pitch_active'].q.mean());other=part[part.subset=='all'][['attitude','velocity']].mean()
            passed=p<=.15 and q<=.15 and other.attitude<=np.deg2rad(3) and other.velocity<=.5;abs_pass.append(passed)
            absrows.append(dict(candidate=candidate,readout=readout,seed=seed,active_p=p,active_q=q,attitude_deg=float(np.rad2deg(other.attitude)),velocity=float(other.velocity),passed=bool(passed)))
        records.append(dict(candidate=candidate,baseline=baseline,readout=readout,primary_gain_fraction=mean_gain,paired_flight_bootstrap_gain_95percent=ci,
            all_seeds_improve=bool((gain>0).all()),maximum_all_metric_regression_fraction=float(regression.to_numpy().max()),
            relative_gate_passed=bool(mean_gain>=.05 and (gain>0).all() and (regression<=.05).all().all()),absolute_gate_passed=bool(all(abs_pass))))
    pd.DataFrame(scores).to_csv(OUT/'formal_primary_scores.csv',index=False);pd.DataFrame(absrows).to_csv(OUT/'formal_absolute_gates.csv',index=False)
    base.write(OUT/'formal_decision.json',dict(comparisons=records,absolute_by_seed=absrows,source_checkpoint_pins=pins,
        control_response_independently_validated=False,controller_optimization_started=False,
        accepted_for_control=False,reason='Prediction gates and independent control-response evidence are separate; this bandwidth experiment does not certify causal control effects.',
        inference_note='aa50 has100ms lagged rate readout plus current-time pose/velocity; internal raw100 latent rate remains unvalidated as instantaneous state',sealed_test_accessed=False))
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
