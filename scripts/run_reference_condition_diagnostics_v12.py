"""Frozen train-flight OOF p/q conditioning audit, without retraining or rescreening."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,SEEDS
from reference_condition_diagnostics_v12 import FEATURES,covariates,weighted_terciles,bins,waveform,aggregate,repeatability
from run_reference_axis_diagnostics_v10 import build_model,predict,read_prediction,KEYS
from run_reference_generalization_v7 import verify_manifest
OUT=base.ROOT/'docs/analysis/results/control_windows_v12'
V8=OUT.parent/'control_windows_v8';V10=OUT.parent/'control_windows_v10';V11=OUT.parent/'control_windows_v11'


def freeze():
    OUT.mkdir(exist_ok=False)
    paths=[V8/'delivery_manifest.json',V10/'delivery_manifest.json',V11/'delivery_manifest.json',base.DATA/'manifest.json',base.DATA/'windows_v2.csv',base.ROOT/'dataset/control_windows_v2/windows.npz',
        Path(__file__),base.ROOT/'scripts/reference_condition_diagnostics_v12.py',base.ROOT/'tests/test_reference_condition_diagnostics_v12.py']
    m=json.loads((base.DATA/'manifest.json').read_text())
    base.write(OUT/'protocol.json',dict(stage='v12 p/q history-conditional waveform mismatch localization',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        data='407 originaltrain windows,5sameouterfolds x3frozenv8holdseeds; no development evaluation, sealed/reserved access, modeltraining or sample removal',
        phase_contract=m['phase_contract'],frequency_contract=m['frequency_contract'],
        strata=FEATURES+['mode'],features='all features use history only. axis-specific rate and roll/pitch reference statistics. Actuator change RMS over all4history channels. Body speed is body velocity norm, not airspeed.',
        thresholds='fit flight-balanced empirical terciles on4outerfitflights; apply unchanged to samefold fit/query. Values<=lower inlow, <=upper inmid, remaininghigh; ties may empty bins. Mode is origin mode.',
        waveform='0.5 and1s fullprefix: waveformRMSE, centeredRMSE divided by RMS truthstd, RMS predstd/truthstd, mean perwindow correlation (undefined excluded and counted), exact bias/amplitude/shape MSE decomposition',
        reference='paired sameGRU replay observedfuture reference vs all5referencechannels held at last historical reference. Report outputcontrastRMS and 1-factualRMSE/heldRMSE against same observed truth. This is predictive usefulness of recorded references, not aircraft counterfactual response or causal effect.',
        aggregation='windowMSE->flightRMSE->equalflight/equalseed summary. Fit cohort repeated acrossfolds is descriptive training fit, not independent sample count.',
        screen='prespecified exploratory1s high/low normalizedcenterederror: at least8windows perextreme perflight; >=4pairedflights; geometricmean ratio>=1.2 or<=1/1.2; same direction in>=4flights andall3seeds; mean correlation worsens in matching direction for>=4flights. Display .5s separately; no automatic model choice frommany comparisons.',
        limits='8continuouscovariates x2axes, repeated-development flights, correlated features/mode/motion; no p-values or causal/root-cause claim, no filtering using errors. If support insufficient report it. No promotion or controller work.',sealed_test_accessed=False))


def main():
    torch.set_num_threads(1);freeze()
    for source in [V8,V10,V11]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();logs=w.log_id.to_numpy();features={axis:covariates(x,j) for j,axis in enumerate(['p','q'])}
    cv=pd.concat([v.assign(axis=axis,array_index=w.array_index.to_numpy(),log_id=logs) for axis,v in features.items()]);cv.to_csv(OUT/'history_covariates.csv',index=False)
    allrows=[];strata=[];thresholds=[];checks=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
        assert not set(logs[fit])&set(logs[query]);dest=OUT/f'fold{fold}';dest.mkdir()
        for axis in ['p','q']:
            for feature in FEATURES:
                cut=weighted_terciles(features[axis].iloc[fit][feature].to_numpy(),logs[fit])
                thresholds.append(dict(fold=fold,axis=axis,condition=feature,low=float(cut[0]),high=float(cut[1]),tied=bool(cut[0]==cut[1])))
                for cohort,ids in [('fit',fit),('oof',query)]:
                    df=w.iloc[ids][['array_index','log_id']].copy();df['fold']=fold;df['axis']=axis;df['cohort']=cohort;df['condition']=feature;df['stratum']=bins(features[axis].iloc[ids][feature].to_numpy(),cut);strata.append(df)
            for cohort,ids in [('fit',fit),('oof',query)]:
                df=w.iloc[ids][['array_index','log_id']].copy();df['fold']=fold;df['axis']=axis;df['cohort']=cohort;df['condition']='mode';df['stratum']=w.iloc[ids]['mode'].astype(str);strata.append(df)
        with np.load(V10/f'fold{fold}/mean_predictor.npz') as z:params={k:z[k] for k in ['mean','scale','coefficient','target_mean']}
        for seed in SEEDS:
            model=build_model(V8/f'fold{fold}/hold_{seed}/final.pt',params,enabled=False)
            for cohort,ids in [('fit',fit),('oof',query)]:
                old=read_prediction(V8/f'fold{fold}/hold_{seed}/{cohort}_predictions.npz',ids)
                held,_=predict(model,x,ids,hold=True,diagnostics=False)
                np.savez_compressed(dest/f'{cohort}_held_{seed}.npz',array_index=ids,rates_b=held['rates_b'])
                replay,_=predict(model,x,ids[:4],diagnostics=False)
                error=max(float(np.max(abs(replay[k]-old[k][:4]))) for k in KEYS);assert error<1e-5
                checks.append(dict(fold=fold,seed=seed,cohort=cohort,replay_max_abs=error))
                for axis,j in [('p',0),('q',1)]:
                    for k in [25,50]:
                        d=waveform(old['rates_b'][:,:k,j].astype(float),y['rates_b'][ids,:k,j].numpy().astype(float),held['rates_b'][:,:k,j].astype(float))
                        np.testing.assert_allclose(d['total_mse'],d['bias_mse']+d['amplitude_mse']+d['shape_mse'],rtol=1e-9,atol=1e-10)
                        df=w.iloc[ids][['array_index','log_id']].copy();df['fold']=fold;df['seed']=seed;df['axis']=axis;df['cohort']=cohort;df['horizon_s']=k*.02
                        for key,value in d.items():df[key]=value
                        allrows.append(df)
        base.write(dest/'manifest.json',dict(fit_indices=fit.tolist(),query_indices=query.tolist(),source_manifest_sha256=base.sha(V8/f'fold{fold}/manifest.json')))
        print(f'Completed fold {fold}',flush=True)
    rows=pd.concat(allrows);assign=pd.concat(strata);rows.to_csv(OUT/'waveform_per_window.csv',index=False);assign.to_csv(OUT/'strata.csv',index=False);pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
    keys=['fold','axis','cohort','array_index','log_id'];expanded=rows.merge(assign,on=keys,validate='many_to_many')
    assert len(expanded)==len(rows)*9
    grouped=expanded.groupby(['fold','seed','cohort','axis','horizon_s','condition','stratum','log_id'],observed=True).apply(aggregate,include_groups=False).reset_index()
    grouped.to_csv(OUT/'conditions_per_flight.csv',index=False)
    summary=grouped.groupby(['cohort','axis','horizon_s','condition','stratum']).mean(numeric_only=True).reset_index();summary.to_csv(OUT/'conditions_summary.csv',index=False)
    counts=assign[assign.cohort=='oof'].groupby(['axis','condition','stratum']).agg(windows=('array_index','size'),flights=('log_id','nunique')).reset_index();counts.to_csv(OUT/'condition_support.csv',index=False)
    overall=rows.groupby(['fold','seed','cohort','axis','horizon_s','log_id']).apply(aggregate,include_groups=False).reset_index();overall.to_csv(OUT/'overall_per_flight.csv',index=False)
    overall.groupby(['cohort','axis','horizon_s']).mean(numeric_only=True).to_csv(OUT/'overall_summary.csv')
    screen,pairs=repeatability(grouped);screen.to_csv(OUT/'repeatability_screen.csv',index=False);pairs.to_csv(OUT/'paired_extremes.csv',index=False)
    base.write(OUT/'verification.json',dict(replays=checks,history_only_strata=True,fold_fit_query_disjoint=True,oof_unique_windows=int(rows[rows.cohort=='oof'].array_index.nunique()),expanded_count=len(expanded),sealed_test_accessed=False))
    base.write(OUT/'decision.json',dict(status='diagnostic_complete',screened_associations=screen.to_dict('records'),repeatable=screen[screen.repeatable_screen].to_dict('records'),model_promoted=False,development_evaluated=False,sealed_test_accessed=False))
    print(screen.to_string(index=False),flush=True)

if __name__=='__main__':main()
