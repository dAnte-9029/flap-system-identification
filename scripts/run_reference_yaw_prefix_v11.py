"""Frozen same-fold comparison of history-only and planned-prefix yaw means."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,SEEDS,STAT_KEYS
from reference_yaw_prefix_v11 import prefix_features,fit_prefix
from run_reference_axis_diagnostics_v10 import predict,read_prediction,rate_rows,KEYS
from run_reference_axis_diagnostics_v10 import build_model as history_model
from reference_capacity_v6 import metric_frames
from run_reference_generalization_v7 import verify_manifest
from system_identification.models.reference_gru_yaw_reference import ReferenceGRUYawReference

OUT=base.ROOT/'docs/analysis/results/control_windows_v11'
V8=OUT.parent/'control_windows_v8';V10=OUT.parent/'control_windows_v10'


def build_model(checkpoint,params,enabled=True):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ReferenceGRUYawReference({k:ck['state_dict'][k] for k in STAT_KEYS},params,enabled=enabled)
    missing,extra=model.load_state_dict(ck['state_dict'],strict=False)
    assert set(missing)=={'yaw_mean','yaw_scale','yaw_coefficient','yaw_target_mean'} and not extra
    return model.eval()


def freeze():
    OUT.mkdir(exist_ok=False)
    source=[V8/'delivery_manifest.json',V10/'delivery_manifest.json',base.DATA/'manifest.json',base.DATA/'windows_v2.csv',
        base.ROOT/'dataset/control_windows_v2/windows.npz',base.ROOT/'dataset/control_windows_v2/normalization.npz',
        Path(__file__),base.ROOT/'scripts/reference_yaw_prefix_v11.py',base.ROOT/'scripts/evaluate_reference_yaw_prefix_v11.py',
        base.ROOT/'src/system_identification/models/reference_gru_yaw_reference.py',base.ROOT/'tests/test_reference_yaw_prefix_v11.py']
    m=json.loads((base.DATA/'manifest.json').read_text())
    base.write(OUT/'protocol.json',dict(stage='v11 planned-reference-prefix yaw cumulative mean',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in source},
        data='407 original train windows; same5 outerfolds x3 frozen hold GRU seeds17/23/42; no screening, new GRU optimization or sealed/reserved access',
        phase_contract=m['phase_contract'],frequency_contract=m['frequency_contract'],
        development_boundary='OOF flights repeatedly used for research; not blind. 739 development windows only evaluated after frozen OOF gate passes.',
        features='94 historical statistics plus5 current plannedreference minus last historical reference plus5 prefix-average plannedreference minus last historical reference;104 total. Step k uses reference indices0..k only.',
        fit='50 independent flight-balanced ridgealpha100 models of true cumulative r mean; fit and normalization on same4outerfit flights only, no pooled crossfold residual targets',
        mechanics='v10 correction loop unchanged:100ms smoothstep, magnitude<=trainrstd,slew<=2trainrstd/s, corrected rate integrated into quaternion and fed back to GRU; empirical research limits, not safety envelope',
        comparators=['frozen baseline','v10 history-only correction','v11 prefix-aware correction'],
        gate='retainv10:1s r mean gain>=10%,waveformgain>=5% vsbaseline;all3seedspositiveboth,>=4/5flightwavepositive;allpqr/attitude/velocityendpointregression<=5% at.5/1s perseed. Additional:1s mean andwaveflight-equal scores no worse thanv10;1s r held-reference effect RMS ratio tobaseline in[0.9,1.1] for eachseed.',
        sensitivity='hold all5future references at last history value and recompute BOTH predictor and decoder. Modeloutput contrast only, no counterfactual flighttruth. Report p/q/r .5/1s;[.9,1.1] is a conservative research comparison band, not physical-effect tolerance.',
        absolute_gate='eachseed allstate.5s p/q/r<=.15rad/s,attitude<=3deg,velocity<=.5m/s; controlresponse/closedloop separate',
        conditional_development='only if fullOOFgate passes: fitprefix and historymeans on original5train flights, attach3frozenA1600GRUs; evaluate739dev once, same3arm diagnostics. No tuning after results.',
        bootstrap='2000 pairedwholeflight resamples seed910;exploratory5flight intervals, not independent selection-adjusted confidence',sealed_test_accessed=False))


def response_rows(meta,arm,seed,factual,held):
    rows=[]
    for k in [25,50]:
        delta=factual['rates_b'][:,:k]-held['rates_b'][:,:k]
        for j,axis in enumerate(['p','q','r']):
            df=meta[['array_index','log_id']].copy();df['arm']=arm;df['seed']=seed;df['horizon_s']=k*.02;df['axis']=axis
            df['reference_effect_mse']=np.mean(delta[:,:,j]**2,1);rows.append(df)
    return rows


def main():
    torch.set_num_threads(1);freeze()
    for source in [V8,V10]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();features=prefix_features(x);logs=w.log_id.to_numpy();target=np.cumsum(y['rates_b'][:,:,2].numpy().astype(float),1)/np.arange(1,51)
    frames=[];rates=[];responses=[];checks=[]
    base.write(OUT/'status.json',dict(status='running'))
    for fold in range(5):
        source=V8/f'fold{fold}';info=json.loads((source/'manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
        expected,params=fit_prefix(features,target,fit,query,logs)
        dest=OUT/f'fold{fold}';dest.mkdir();np.savez_compressed(dest/'mean_predictor.npz',fit_indices=fit,query_indices=query,query_mean_prediction=expected,**params)
        with np.load(V10/f'fold{fold}/mean_predictor.npz') as z:hp={k:z[k] for k in ['mean','scale','coefficient','target_mean']}
        for seed in SEEDS:
            ck=source/f'hold_{seed}/final.pt';model=build_model(ck,params);run=dest/f'seed{seed}';run.mkdir()
            torch.save(dict(state_dict=model.state_dict(),architecture='ReferenceGRUYawReference',fold=fold,seed=seed,enabled=True,protocol_sha256=base.sha(OUT/'protocol.json')),run/'candidate.pt')
            for cohort,ids in [('fit',fit),('oof',query)]:
                meta=w.iloc[ids];truth={k:v[ids].numpy() for k,v in y.items()}
                old=read_prediction(source/f'hold_{seed}'/f'{cohort}_predictions.npz',ids)
                hist=read_prediction(V10/f'fold{fold}/seed{seed}'/f'{cohort}_predictions.npz',ids)
                new,bias=predict(model,x,ids);np.savez_compressed(run/f'{cohort}_predictions.npz',array_index=ids,yaw_correction=bias,**new)
                limit=float(model.rate_scale[2]);delta=np.diff(np.column_stack([np.zeros(len(ids)),bias]),axis=1)
                assert np.max(abs(bias))<=limit+1e-6 and np.max(abs(delta))<=2*limit*.02+1e-6
                for arm,p in [('baseline',old),('history',hist),('yaw',new)]:
                    f,_=metric_frames(p,truth,meta,arm,seed);f['cohort']=cohort;f['fold']=fold;frames.append(f)
                    rates.append(rate_rows(p,truth,meta,arm,seed,fold,cohort))
                if cohort=='oof':
                    for arm,p,mm in [('baseline',old,build_model(ck,params,False)),('history',hist,history_model(ck,hp)),('yaw',new,model)]:
                        held,_=predict(mm,x,ids,hold=True);responses.extend(response_rows(meta,arm,seed,p,held))
                        if arm=='yaw':np.savez_compressed(run/'held_predictions.npz',array_index=ids,**held)
                    ids2=ids[:4];batch={k:v[ids2].clone() for k,v in x.items()}
                    with torch.no_grad():a=model(**batch);batch['future_reference'][:,5:]+=10;b=model(**batch)
                    assert all(torch.equal(a[k][:,:5],b[k][:,:5]) for k in KEYS)
                    replay,_=predict(build_model(ck,params,False),x,ids2,diagnostics=False)
                    error=max(float(np.max(abs(replay[k]-old[k][:4]))) for k in KEYS);assert error<1e-5
                    checks.append(dict(fold=fold,seed=seed,prefix_causality=True,disabled_replay_max_abs=error,
                        magnitude_boundary_fraction=float((abs(bias)>=limit-1e-6).mean()),slew_boundary_fraction=float((abs(delta)>=2*limit*.02-1e-6).mean())))
        base.write(dest/'manifest.json',dict(fit_indices=fit.tolist(),query_indices=query.tolist(),predictor_sha256=base.sha(dest/'mean_predictor.npz'),source_fold_sha256=base.sha(source/'manifest.json')))
        print(f'Completed fold {fold}',flush=True)
    pd.concat(frames).to_csv(OUT/'state_per_flight.csv',index=False);pd.concat(rates).to_csv(OUT/'rate_per_window.csv',index=False)
    pd.concat(responses).to_csv(OUT/'reference_sensitivity.csv',index=False)
    base.write(OUT/'verification.json',dict(runs=checks,sealed_test_accessed=False))
    base.write(OUT/'status.json',dict(status='complete'))

if __name__=='__main__':main()
