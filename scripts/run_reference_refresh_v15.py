"""Frozen 1s open rollout vs observed-history resets at100/200ms."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,model_from,predict
from reference_refresh_v15 import refresh_inputs,stitch,persistence,squared_errors,paired_local_age
from run_reference_generalization_v7 import verify_manifest
OUT=base.ROOT/'docs/analysis/results/control_windows_v15'
V8=OUT.parent/'control_windows_v8';V12=OUT.parent/'control_windows_v12';V14=OUT.parent/'control_windows_v14'
METRICS=['p','q','r','velocity','attitude']


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False);x,y,w=load_train()
    authority=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json';contract=json.loads(authority.read_text())
    paths=[V8/'delivery_manifest.json',V12/'delivery_manifest.json',V14/'delivery_manifest.json',base.DATA/'manifest.json',authority,Path(__file__),
        base.ROOT/'scripts/reference_refresh_v15.py',base.ROOT/'tests/test_reference_refresh_v15.py',base.ROOT/'src/system_identification/data/reference_windows.py']
    for cache in w.cache.unique():
        assert base.sha(base.ROOT/cache)==contract['cache_sha256'][cache];paths.append(base.ROOT/cache)
    base.write(OUT/'protocol.json',dict(stage='v15 observedhistory refresh diagnosis',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],
        scope='407originaltrain parents, same5outerfolds x3v8hold GRUseeds,fit+OOF. No retraining, new selection, development or sealed/reserved access.',
        arms=['free50','refresh5','refresh10','persistence50','persistence5','persistence10'],
        observations='For reset5 or10, rebuild complete26sample observedhistory at origin+0,5,..45 or origin+0,10,..40; actuator ends one sample before current origin; reencodeGRU, reset physicalstate, relativephase,freq,airspeed. Fixed parent starts/strata retained. No new labels outsideparentorigin+50.',
        inputs='each local forecast uses measurements only up to its own origin, actualcommands only through origin-1 and plannedreference prefix from neworigin. Later originalforecast observations are allowed ONLY after resettime, never called autonomous1s forecast.',
        fairness='all arms scored against identical50 originaltargettimes. Local output startsone step AFTER reset, never inserting observation as zeroerror prediction. First5/10steps must match originalfree rollout. Persistence holds observedrates/velocity/quaternion constant withsame reset cadence; simplebaseline, not physicalquaternion integration.',
        metrics='perflight RMSE then equalflights/seeds forfull .1/.2/.5/1sprefix; endpointprofiles vs globalelapsedtime; localage20..100/200ms withfree predictions scored onidenticalglobaltargetpositions for eachage. Fixedv12 originalhistoryfrequency strata only.',
        limitations='reset jointly changes state, hiddenhistory, phase/frequency, measurement freshness and forecastage; benefit not attributable only to integrator drift. Fullstate persistence may be unphysical. No controlreadiness or absolute1s gate applied to refreshed outputs.',
        diagnostic_only=True,sealed_test_accessed=False))
    for src in [V8,V12,V14]:verify_manifest(src/'delivery_manifest.json',['artifacts','code'])
    prepared={5:[],10:[]};source_checks=[]
    for cache,g in w.groupby('cache'):
        with np.load(base.ROOT/cache) as z:a={k:z[k] for k in z.files}
        for row in g.itertuples():
            for period in [5,10]:
                inp=refresh_inputs(a,row.origin_index,period)
                prepared[period].append((row.array_index,inp))
            source_checks.append(dict(array_index=row.array_index,minimum_history_index=row.origin_index-26,maximum_target_index=row.origin_index+50,all_segments_within_parent=True))
    blocks={};flat={}
    for period,items in prepared.items():
        items.sort(key=lambda z:z[0]);assert [i for i,_ in items]==list(range(407))
        blocks[period]={k:np.stack([d[k] for _,d in items]) for k in items[0][1]}
        flat[period]={k:torch.tensor(v.reshape(-1,*v.shape[2:]),dtype=torch.float32) for k,v in blocks[period].items()}
    strata=pd.read_csv(V12/'strata.csv');strata=strata[(strata.axis=='q')&(strata.condition=='frequency_mean')][['array_index','fold','cohort','stratum']]
    frames=[];profiles=[];ages=[];checks=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices']);assert not set(w.iloc[fit].log_id)&set(w.iloc[query].log_id)
        dest=OUT/f'fold{fold}';dest.mkdir()
        for seed in [17,23,42]:
            model=model_from(V8/f'fold{fold}/hold_{seed}/final.pt')
            for cohort,ids in [('fit',fit),('oof',query)]:
                meta=w.iloc[ids][['array_index','log_id']].copy();meta['fold']=fold;meta['seed']=seed;meta['cohort']=cohort
                meta=meta.merge(strata,on=['array_index','fold','cohort'],validate='one_to_one');truth={k:v[ids].numpy() for k,v in y.items()}
                with np.load(V8/f'fold{fold}/hold_{seed}/{cohort}_predictions.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ids);preds={'free50':{k:z[k] for k in ['rates_b','velocity_n','quaternion']}}
                preds['persistence50']=persistence({k:v[ids].numpy() for k,v in x.items()})
                for period in [5,10]:
                    nblocks=50//period;local_ids=(ids[:,None]*nblocks+np.arange(nblocks)).reshape(-1)
                    prediction=predict(model,flat[period],local_ids)
                    preds[f'refresh{period}']={k:stitch(v,len(ids),period) for k,v in prediction.items()}
                    pers=persistence({k:v[local_ids].numpy() for k,v in flat[period].items()});preds[f'persistence{period}']={k:stitch(v,len(ids),period) for k,v in pers.items()}
                    initial=max(float(np.max(abs(preds[f'refresh{period}'][k][:,:period]-preds['free50'][k][:,:period]))) for k in prediction);assert initial<1e-5
                    np.savez_compressed(dest/f'{cohort}_refresh{period}_seed{seed}.npz',array_index=ids,**preds[f'refresh{period}'])
                    checks.append(dict(fold=fold,seed=seed,cohort=cohort,period=period,initial_prefix_max_abs=initial))
                errors={arm:squared_errors(p,truth) for arm,p in preds.items()}
                for arm,e in errors.items():
                    for horizon in [5,10,25,50]:
                        df=meta.copy();df['arm']=arm;df['horizon_s']=horizon*.02
                        for j,key in enumerate(METRICS):df[key+'_mse']=e[:,:horizon,j].mean(1)
                        frames.append(df)
                    for stratum in ['all','low','mid','high']:
                        mask=np.ones(len(meta),bool) if stratum=='all' else meta.stratum.eq(stratum).to_numpy()
                        for log in meta.loc[mask,'log_id'].unique():
                            choose=mask&meta.log_id.eq(log).to_numpy();mean=e[choose].mean(0)
                            df=pd.DataFrame({key:np.sqrt(mean[:,j]) for j,key in enumerate(METRICS)})
                            df=df.assign(arm=arm,fold=fold,seed=seed,cohort=cohort,log_id=log,stratum=stratum,global_age_s=np.arange(1,51)*.02,windows=int(choose.sum()));profiles.append(df)
                for period in [5,10]:
                    for arm in ['free50',f'refresh{period}',f'persistence{period}']:
                        local=paired_local_age(errors[arm],period)
                        for stratum in ['all','low','mid','high']:
                            mask=np.ones(len(meta),bool) if stratum=='all' else meta.stratum.eq(stratum).to_numpy()
                            for log in meta.loc[mask,'log_id'].unique():
                                choose=mask&meta.log_id.eq(log).to_numpy();mean=local[choose].mean(0)
                                df=pd.DataFrame({key:np.sqrt(mean[:,j]) for j,key in enumerate(METRICS)})
                                df=df.assign(arm=arm,period_s=period*.02,fold=fold,seed=seed,cohort=cohort,log_id=log,stratum=stratum,local_age_s=np.arange(1,period+1)*.02,windows=int(choose.sum()));ages.append(df)
            # Fresh local measurement safety and suffix causality on actual query examples.
            b={k:v[query[:2]*10+3].clone() for k,v in flat[5].items()}
            with torch.no_grad():a=model(**b);b['future_reference'][:,3:]+=10;c=model(**b)
            assert all(torch.equal(a[k][:,:3],c[k][:,:3]) for k in a)
        print(f'Completed fold {fold}',flush=True)
    pd.concat(frames).to_csv(OUT/'prefix_per_window.csv',index=False);pd.concat(profiles).to_csv(OUT/'global_age_per_flight.csv',index=False);pd.concat(ages).to_csv(OUT/'local_age_per_flight.csv',index=False)
    base.write(OUT/'verification.json',dict(prefix_replays=checks,source_support=source_checks,all15_models_reference_prefix_causal=True,parent_windows=407,added_training_windows=0,sealed_test_accessed=False))
    base.write(OUT/'status.json',dict(status='complete',model_training=False,development_evaluated=False))

if __name__=='__main__':main()
