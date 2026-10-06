"""Evaluate every frozen strategy and budget arm on the same original validation."""
import json
import numpy as np
import pandas as pd
import torch
from control_windows_v3 import OUT,OLD,ROOT,ART,SEEDS,base
from evaluate_control_windows_v2 import error_rows,summarize

ARMS=['A','B','C','D','A_long']


def scores(cube):
    return np.nanmean(np.nanmean(cube,axis=2),axis=(2,3))


def main(expected_models=12,baseline_folder=None):
    status=json.loads((OUT/'training/status.json').read_text())
    assert status['status']=='complete' and len(status['completed'])==expected_models
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    x,y,stats=base.load();meta=pd.read_csv(OUT/'evaluation_windows.csv');ix=meta.array_index.to_numpy()
    truth={k:v[ix].numpy() for k,v in y.items()};rows=[]
    for arm in ARMS:
        folder=(baseline_folder or ROOT/'artifacts/control_windows_v2') if arm=='A' else ART
        for seed in SEEDS:
            manifest=json.loads((folder/f'{arm}_{seed}/manifest.json').read_text())
            file=folder/f'{arm}_{seed}/validation_predictions.npz'
            assert base.sha(file)==manifest['predictions_sha256']
            assert base.sha(ROOT/manifest['checkpoint'])==manifest['checkpoint_sha256']
            with np.load(file) as z:
                assert np.array_equal(z['array_index'],ix)
                prediction={k:z[k] for k in truth}
            rows.append(error_rows(prediction,truth,meta,arm,seed))
    windows=pd.concat(rows,ignore_index=True)
    windows.to_csv(dest/'per_window_squared_errors.csv',index=False)
    flights=summarize(windows);flights.to_csv(dest/'per_flight_rmse.csv',index=False)
    agg=flights.groupby(['arm','seed','subset','horizon_s'])[['p','q','r','velocity','attitude']].mean().reset_index()
    agg.to_csv(dest/'flight_equal_rmse.csv',index=False)
    logs=sorted(meta.log_id.unique());cube=np.full((5,3,len(logs),2,3),np.nan)
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            for fi,log in enumerate(logs):
                for axis,(subset,metric) in enumerate([('roll_active','p'),('pitch_active','q')]):
                    for hi,horizon in enumerate([.1,.2,.5]):
                        f=flights[(flights.arm==arm)&(flights.seed==seed)&(flights.log_id==log)&(flights.subset==subset)&np.isclose(flights.horizon_s,horizon)]
                        if len(f):cube[ai,si,fi,axis,hi]=f[metric].iloc[0]
    score=scores(cube);rng=np.random.default_rng(1309)
    boot=np.stack([scores(cube[:,:,rng.integers(0,len(logs),len(logs)),:,:]).mean(1) for _ in range(2000)])
    comparisons={}
    for arm,baselines in [('C',['A','B']),('D',['A','B']),('A_long',['A'])]:
        ai=ARMS.index(arm);result={}
        for baseline in baselines:
            bi=ARMS.index(baseline);gain=1-score[ai].mean()/score[bi].mean()
            interval=np.quantile(1-boot[:,ai]/boot[:,bi],[.025,.975])
            result[baseline]=dict(improvement_fraction=float(gain),flight_bootstrap_95ci=interval.tolist(),
                paired_seed_improvements=(1-score[ai]/score[bi]).tolist())
        guards={}
        for subset,tolerance in [('all',.03),('quiet',.05),('hard',.05)]:
            m=agg[(agg.subset==subset)&(agg.horizon_s<1)].groupby('arm')[['p','q','r','velocity','attitude']].mean()
            degradation=m.loc[arm]/m.loc['A']-1
            guards[subset]=dict(relative_degradation=degradation.to_dict(),passed=bool((degradation<=tolerance).all()))
        improving=np.logical_and.reduce([score[ai]<score[ARMS.index(b)] for b in baselines]).sum()
        passed=all(r['improvement_fraction']>=.05 and r['flight_bootstrap_95ci'][0]>0 for r in result.values()) and improving>=2 and all(g['passed'] for g in guards.values())
        comparisons[arm]=dict(versus=result,improving_seeds=int(improving),safeguards=guards,relative_gate_passed=bool(passed))
    absolute={}
    for arm in ARMS:
        records=[]
        for seed in SEEDS:
            current=agg[(agg.arm==arm)&(agg.seed==seed)&np.isclose(agg.horizon_s,.5)]
            p=float(current[current.subset=='roll_active'].p.iloc[0]);q=float(current[current.subset=='pitch_active'].q.iloc[0])
            attitude=float(current[current.subset=='all'].attitude.iloc[0]);velocity=float(current[current.subset=='all'].velocity.iloc[0])
            records.append(dict(seed=seed,p=p,q=q,attitude_rad=attitude,velocity=velocity,
                passed=bool(p<=.15 and q<=.15 and attitude<=np.deg2rad(3) and velocity<=.5)))
        absolute[arm]=dict(seeds=records,passed=all(r['passed'] for r in records))
    base.write(dest/'summary.json',dict(scores_by_arm={arm:score[i].tolist() for i,arm in enumerate(ARMS)},
        mean_by_arm={arm:float(score[i].mean()) for i,arm in enumerate(ARMS)},comparisons=comparisons,
        absolute_research_gate=absolute,absolute_control_candidate_passed=absolute['A_long']['passed'],
        thresholds_status='predeclared research tolerances, not validated flight-safety bounds',
        source_protocol_sha256=base.sha(OUT/'training/protocol.json'),sealed_test_accessed=False))
    print(json.dumps(json.loads((dest/'summary.json').read_text()),indent=2))


if __name__=='__main__':
    torch.set_num_threads(1);main()
