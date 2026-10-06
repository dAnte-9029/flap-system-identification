"""Frozen p/q waveform diagnosis and prefix-causal single-axis yaw candidate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,SEEDS,STAT_KEYS
from reference_error_structure_v9 import history_features,fit_predict
from reference_axis_diagnostics_v10 import align_diagnostic
from system_identification.models.reference_gru_yaw_mean import ReferenceGRUYawMean
from reference_capacity_v6 import metric_frames
from run_reference_generalization_v7 import verify_manifest

OUT=base.ROOT/'docs/analysis/results/control_windows_v10'
V8=OUT.parent/'control_windows_v8';V9=OUT.parent/'control_windows_v9'
KEYS=['rates_b','quaternion','velocity_n']


def freeze():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((base.DATA/'manifest.json').read_text())
    sources=[V8/'delivery_manifest.json',V9/'delivery_manifest.json',base.DATA/'manifest.json',
        base.ROOT/'dataset/control_windows_v2/windows.npz',Path(__file__),base.ROOT/'scripts/reference_axis_diagnostics_v10.py',
        base.ROOT/'src/system_identification/models/reference_gru_yaw_mean.py',base.ROOT/'tests/test_reference_axis_diagnostics_v10.py']
    base.write(OUT/'protocol.json',dict(task='p/q waveform localization and constrained prefix-causal r mean research',
        source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in sources},phase_contract=manifest['phase_contract'],frequency_contract=manifest['frequency_contract'],
        data='same407originaltrainingwindows,5outerflight folds,3frozenholdGRUseeds; no new neural training, data screening, architecture search or sealed/reserved access',
        development_status='v9 r route selected after prior training OOF analysis; repeated development evidence, not blind test',
        pq_diagnostic='0.5/1s windows; fixed truth core excludes5samples eachend; lag -5..+5samples and positive least-squares gain clipped0..3; truthintercept removed percore; compare zero-lag gain and best-lag gain on identical support',
        pq_limit='oracle reads futuretruth, lag may reflect periodic alignment, not identified actuator delay; not deployable or an error lowerbound; no rescreening by oracle result',
        yaw_target='94past-onlyfeatures,flightbalancedridgealpha100 predict cumulative mean r forsteps1..50; outerquery excluded; finalcolumn must reproducev9 directmean1s parameters',
        yaw_update='subtract previousbias from rawGRU proposedr; accumulate virtualuncorrectedprefix; desired bias=smoothstep(t/.1)*clip(predictedcumulative_mean-prefixmean,+/-trainingrstd). Slew limit2*trainingrstd persecond; update bias difference, integrate quaternion midpoint and feed correctedstate into GRU. Currentfrequencyheld.',
        empirical_limits='train-derived amplitude/slew limits are only research constraints, not aircraft safety validation',
        causality='targetmean uses observedhistory only; correction sees simulatedprefix only, not fullfutureGRUmean. Future reference suffix must not alter earlierpredictions. Bias startszero and uses finite slew; pose integrated with correctedrates.',
        oof_gate='r1s fullwaveformRMSE gain>=5percent andr1swindowmeanRMSEgain>=10percent; everyseedpositive forboth, >=4of5flightpositive forwaveform; no allmetricp/q/r/velocity/attitude endpointregression>5percent at0.5/1s foranyseed',
        conditional_validation='only if OOFgate passes: fitridge on all5trainingflights, attach to frozen3A1600 models, evaluate739developmentwindows once; never future measuredinputs. Otherwise no developmentcandidate evaluation.',
        controls='always report reference-held sensitivity on trainingOOF; change in model reliance is not real counterfactual truth. No controller optimization or deployment.',
        absolute_gate='p/q/r0.5s<=.15rad/s,attitude<=3deg,velocity<=.5m/s; controlresponse and closedloop evidence still separate',
        bootstrap='2000 pairedwholeflight resamples seed910; fixedmodels only,5trainingflights, exploratory',sealed_test_accessed=False))


def build_model(checkpoint,params,enabled=True):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ReferenceGRUYawMean({k:ck['state_dict'][k] for k in STAT_KEYS},params,enabled=enabled)
    missing,extra=model.load_state_dict(ck['state_dict'],strict=False)
    assert set(missing)=={'yaw_mean','yaw_scale','yaw_coefficient','yaw_target_mean'} and not extra
    return model.eval()


def predict(model,x,ids,hold=False,diagnostics=True):
    result={k:[] for k in KEYS};bias=[]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            ix=ids[start:start+64];batch={k:v[ix] for k,v in x.items()}
            if hold:batch['future_reference']=batch['history_reference'][:,-1:,:].expand_as(batch['future_reference'])
            output=model(**batch,diagnostics=diagnostics)
            for k in KEYS:assert torch.isfinite(output[k]).all();result[k].append(output[k].numpy())
            if 'yaw_correction' in output:bias.append(output['yaw_correction'].numpy())
    return {k:np.concatenate(v) for k,v in result.items()},np.concatenate(bias) if bias else None


def read_prediction(path,ids):
    with np.load(path) as z:
        np.testing.assert_array_equal(z['array_index'],ids)
        return {k:z[k] for k in KEYS}


def rate_rows(pred,truth,meta,arm,seed,fold,cohort):
    rows=[]
    for k in [25,50]:
        error=pred['rates_b'][:,:k,2]-truth['rates_b'][:,:k,2]
        df=meta[['array_index','log_id']].copy();df['arm']=arm;df['seed']=seed;df['fold']=fold;df['cohort']=cohort;df['horizon_s']=k*.02
        df['r_mean_mse']=error.mean(1)**2;df['r_wave_mse']=np.mean(error**2,1);rows.append(df)
    return pd.concat(rows)


def main():
    torch.set_num_threads(1);freeze();verify_manifest(V8/'delivery_manifest.json',['artifacts','code']);verify_manifest(V9/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();history=history_features(x,'full');logs=w.log_id.to_numpy();truth=y['rates_b'].numpy().astype(float)
    target=np.cumsum(truth[:,:,2],axis=1)/np.arange(1,51)
    pq=[];frames=[];rates=[];checks=[];responses=[]
    base.write(OUT/'status.json',dict(status='running'))
    for fold in range(5):
        source=V8/f'fold{fold}';info=json.loads((source/'manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices']);dest=OUT/f'fold{fold}';dest.mkdir()
        assert not set(logs[fit])&set(logs[query])
        target_prediction,params=fit_predict(history[fit],target[fit],history[query],logs[fit],'ridge')
        np.savez_compressed(dest/'mean_predictor.npz',fit_indices=fit,query_indices=query,**params)
        with np.load(V9/f'fold{fold}/mean_full_17_k50.npz') as z:
            error=max(float(np.max(abs(params['coefficient'][:,-1]-z['coefficient'][:,2]))),float(abs(params['target_mean'][-1]-z['target_mean'][2])))
            assert error<1e-10,error
        for seed in SEEDS:
            model=build_model(source/f'hold_{seed}/final.pt',params);run=dest/f'seed{seed}';run.mkdir()
            torch.save(dict(state_dict=model.state_dict(),fold=fold,seed=seed,architecture='ReferenceGRUYawMean',enabled=True,protocol_sha256=base.sha(OUT/'protocol.json')),run/'candidate.pt')
            for cohort,ids in [('fit',fit),('oof',query)]:
                meta=w.iloc[ids];yy={k:v[ids].numpy() for k,v in y.items()};old=read_prediction(source/f'hold_{seed}'/f'{cohort}_predictions.npz',ids)
                for k in [25,50]:
                    for axis,index in [('p',0),('q',1)]:
                        a=align_diagnostic(old['rates_b'][:,:k,index],truth[ids,:k,index]);df=meta[['array_index','log_id']].copy();df['seed']=seed;df['fold']=fold;df['cohort']=cohort;df['horizon_s']=k*.02;df['axis']=axis
                        for name,value in a.items():df[name]=value
                        pq.append(df)
                new,bias=predict(model,x,ids);np.savez_compressed(run/f'{cohort}_predictions.npz',array_index=ids,yaw_correction=bias,**new)
                limit=float(model.rate_scale[2]);assert np.max(abs(bias))<=limit+1e-6
                assert np.max(abs(np.diff(np.column_stack([np.zeros(len(ids)),bias]),axis=1)))<=2*limit*.02+1e-6
                for arm,p in [('baseline',old),('yaw',new)]:
                    f,_=metric_frames(p,yy,meta,arm,seed);f['cohort']=cohort;f['fold']=fold;frames.append(f)
                    rates.append(rate_rows(p,yy,meta,arm,seed,fold,cohort))
                if cohort=='oof':
                    # Keep factual and held-reference comparisons paired for both models.
                    held,_=predict(model,x,ids,hold=True)
                    baseline_model=build_model(source/f'hold_{seed}/final.pt',params,enabled=False)
                    baseline_held,_=predict(baseline_model,x,ids,hold=True)
                    for arm,factual,counter in [('baseline',old,baseline_held),('yaw',new,held)]:
                        for k in [25,50]:
                            delta=factual['rates_b'][:,:k]-counter['rates_b'][:,:k]
                            for axis,j in [('p',0),('q',1),('r',2)]:
                                response=meta[['array_index','log_id']].copy();response['arm']=arm;response['seed']=seed;response['horizon_s']=k*.02;response['axis']=axis;response['reference_effect_mse']=np.mean(delta[:,:,j]**2,1);responses.append(response)
                    ids2=ids[:4];batch={k:v[ids2].clone() for k,v in x.items()}
                    with torch.no_grad():a=model(**batch);batch['future_reference'][:,5:]+=10;b=model(**batch)
                    assert all(torch.equal(a[k][:,:5],b[k][:,:5]) for k in KEYS)
                    reproduction,_=predict(baseline_model,x,ids2,diagnostics=False)
                    exact=max(float(np.max(abs(reproduction[k]-old[k][:4]))) for k in KEYS);assert exact<1e-5
                    checks.append(dict(fold=fold,seed=seed,disabled_replay_max_abs=exact,prefix_causality=True,mean_parameter_reproduction=error,
                        maximum_abs_bias=float(np.max(abs(bias))),r_scale=limit,amplitude_boundary_fraction=float(np.mean(abs(bias)>=limit-1e-6))))
        base.write(dest/'manifest.json',dict(fit_indices=fit.tolist(),query_indices=query.tolist(),predictor_sha256=base.sha(dest/'mean_predictor.npz'),source_fold_manifest_sha256=base.sha(source/'manifest.json')))
        print(f'Completed fold{fold}',flush=True)
    pd.concat(pq).to_csv(OUT/'pq_per_window.csv',index=False);pd.concat(frames).to_csv(OUT/'state_per_flight.csv',index=False)
    pd.concat(rates).to_csv(OUT/'rate_per_window.csv',index=False);pd.concat(responses).to_csv(OUT/'reference_sensitivity.csv',index=False)
    base.write(OUT/'verification.json',dict(runs=checks,sealed_test_accessed=False))
    base.write(OUT/'status.json',dict(status='complete'))

if __name__=='__main__':main()
