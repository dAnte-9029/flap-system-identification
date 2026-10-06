"""Frozen frequency-conditioned lag/phase audit on original training flights."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train
from reference_timing_v14 import curves,best_index,CORES,LAGS
from reference_axis_diagnostics_v10 import align_diagnostic
from run_reference_generalization_v7 import verify_manifest
OUT=base.ROOT/'docs/analysis/results/control_windows_v14'
V8=OUT.parent/'control_windows_v8';V10=OUT.parent/'control_windows_v10';V12=OUT.parent/'control_windows_v12';V13=OUT.parent/'control_windows_v13'
CONTRACT=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json'


def main():
    OUT.mkdir(exist_ok=False);x,y,w=load_train();contract=json.loads(CONTRACT.read_text())
    paths=[V8/'delivery_manifest.json',V10/'delivery_manifest.json',V12/'delivery_manifest.json',V13/'delivery_manifest.json',base.DATA/'manifest.json',CONTRACT,
        Path(__file__),base.ROOT/'scripts/reference_timing_v14.py',base.ROOT/'scripts/audit_reference_timing_v14.py',base.ROOT/'tests/test_reference_timing_v14.py',
        base.ROOT/'src/system_identification/data/reference_windows.py',base.ROOT/'scripts/audit_control_reference_semantics.py',base.ROOT/'scripts/prepare_reference_dynamics_data.py']
    for cache in w.cache.unique():
        p=base.ROOT/cache;assert base.sha(p)==contract['cache_sha256'][cache];paths.append(p)
    base.write(OUT/'protocol.json',dict(stage='v14 frequency-conditioned fixed-support lag and relative phase diagnosis',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],
        data='407originaltrain windows,5outerfolds,3v8holdseeds,fit+OOF; frozenv12 frequencymean strata; no newtraining, no timestamp modification, no development/sealed/reserved evaluation',
        original='reproduce v10 .5/1s maxlag5 gainclip0..3 onidentical fullprefixinterior, then stratify',
        blocks='fixed truthcore earlyindices5..19 (target.12-.40s) andlate30..44 (.62-.90s), equal15samples; same ±5lag candidates eachcore. Demean separately percore; unitgain and positivegain0..3 curves.',
        fixedlag='for eachfold/seed/axis and frequencystratum plusall, choose one unit-gain lag minimizing equalfitflight MSE of bothcores; evaluate unchanged on queryflight. Future prediction shifting remains offline, not deployable even when fitted onfitflights.',
        oracle='compare independently bestearly/late lags vs one perwindowsharedlag, sameunionof30truthsamples andunitgain; flexiblegain separatelydescriptive. A shortcore has aboutonecycle, lagambiguity/boundaryfractions must bereported.',
        phase='offline labels only: unwrapped cachedrelative phase fromorigin tofuture minus heldoriginfrequency phase, endpoints.25? use .24/.5/.74/1s; signed andabsoluteerrors. No assumed physicalzero. Reportassociation withlagdrift, not causal attribution.',
        timing='reconstruct publication/sample admissibility for selectedtrainwindows fromoriginalULog; verify cachedrates/phase/frequency/commands and ages; history state/ref origin-25..origin, actuator origin-26..origin-1, futureref origin..origin49, labels origin+1..origin50. Timestamps prove bookkeeping, not physicalsensor/servo latency.',
        conclusion='descriptive repeateddevelopment evidence. No shift/phase correction or controlpromotion based onoracle. Report allstrata, uncertainty and source-time limits.',sealed_test_accessed=False))
    for source in [V8,V10,V12,V13]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    strata=pd.read_csv(V12/'strata.csv');strata=strata[(strata.axis=='q')&(strata.condition=='frequency_mean')][['fold','cohort','array_index','stratum']]
    phase=[]
    for cache,g in w.groupby('cache'):
        with np.load(base.ROOT/cache) as z:a={k:z[k] for k in z.files}
        for r in g.itertuples():
            o=r.origin_index;phi=np.unwrap(a['phase_rad'][o:o+51].astype(float));freq=float(a['state_features'][o,11]);error=phi[1:]-phi[0]-2*np.pi*freq*np.arange(1,51)*.02
            d=dict(array_index=r.array_index,log_id=r.log_id)
            for k in [12,25,37,50]:d[f'phase_error_{k}']=float(error[k-1])
            d['late_minus_early_phase_error']=float(error[30:45].mean()-error[5:20].mean());phase.append(d)
    pd.DataFrame(phase).to_csv(OUT/'phase_per_window.csv',index=False)
    rows=[];original=[];choices=[];drifts=[];reproduction=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices']);dest=OUT/f'fold{fold}';dest.mkdir()
        for seed in [17,23,42]:
            for axis,j in [('p',0),('q',1)]:
                memo={}
                for cohort,ids in [('fit',fit),('oof',query)]:
                    with np.load(V8/f'fold{fold}/hold_{seed}/{cohort}_predictions.npz') as z:
                        np.testing.assert_array_equal(z['array_index'],ids);pred=z['rates_b'][:,:,j].astype(float)
                    truth=y['rates_b'][ids,:,j].numpy().astype(float);meta=w.iloc[ids][['array_index','log_id']].copy();meta['fold']=fold;meta['seed']=seed;meta['axis']=axis;meta['cohort']=cohort
                    meta=meta.merge(strata[(strata.fold==fold)&(strata.cohort==cohort)],on=['fold','cohort','array_index'],validate='one_to_one')
                    cs=[curves(pred,truth,core) for core in CORES];memo[cohort]=(meta,cs)
                    np.savez_compressed(dest/f'{cohort}_{axis}_{seed}_curves.npz',array_index=ids,early_unit=cs[0]['unit'],late_unit=cs[1]['unit'],early_gain=cs[0]['gain'],late_gain=cs[1]['gain'])
                    for k in [25,50]:
                        d=align_diagnostic(pred[:,:k],truth[:,:k]);df=meta.copy();df['horizon_s']=k*.02
                        for key,val in d.items():df[key]=val
                        original.append(df)
                    a,b=cs;ie=best_index(a['unit']);il=best_index(b['unit']);shared=best_index(a['unit']+b['unit']);ii=np.arange(len(ids));d=meta.copy()
                    d['early_lag_s']=LAGS[ie]*.02;d['late_lag_s']=LAGS[il]*.02;d['lag_drift_s']=d.late_lag_s-d.early_lag_s
                    d['same_lag']=ie==il;d['boundary']=(abs(LAGS[ie])==5)|(abs(LAGS[il])==5)
                    d['zero_mse']=(a['unit'][:,5]+b['unit'][:,5])/2;d['shared_oracle_mse']=(a['unit'][ii,shared]+b['unit'][ii,shared])/2;d['separate_oracle_mse']=(a['unit'][ii,ie]+b['unit'][ii,il])/2
                    d['ambiguous']=((np.sort(a['unit'],axis=1)[:,1]-a['unit'][ii,ie])<.01*np.maximum(a['unit'][:,5],1e-12))|((np.sort(b['unit'],axis=1)[:,1]-b['unit'][ii,il])<.01*np.maximum(b['unit'][:,5],1e-12))
                    drifts.append(d)
                fm,fc=memo['fit']
                for group in ['all','low','mid','high']:
                    mask=np.ones(len(fm),bool) if group=='all' else fm.stratum.eq(group).to_numpy();cost=(fc[0]['unit']+fc[1]['unit'])/2
                    mean=pd.DataFrame(cost[mask]).assign(log_id=fm.loc[mask,'log_id'].to_numpy()).groupby('log_id').mean().mean().to_numpy();idx=int(best_index(mean[None,:])[0])
                    choices.append(dict(fold=fold,seed=seed,axis=axis,stratum=group,lag_s=float(LAGS[idx]*.02),fit_windows=int(mask.sum())))
                    for cohort,(meta,cc) in memo.items():
                        chosen=np.ones(len(meta),bool) if group=='all' else meta.stratum.eq(group).to_numpy()
                        df=meta[chosen].copy();df['stratum']=group
                        df['zero_mse']=((cc[0]['unit'][:,5]+cc[1]['unit'][:,5])/2)[chosen]
                        df['fixed_mse']=((cc[0]['unit'][:,idx]+cc[1]['unit'][:,idx])/2)[chosen];rows.append(df)
        print(f'Completed fold {fold}',flush=True)
    orig=pd.concat(original);orig.to_csv(OUT/'fullprefix_oracles.csv',index=False)
    old=pd.read_csv(V10/'pq_per_window.csv');keys=['array_index','fold','seed','axis','cohort','horizon_s'];merged=orig.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one');assert len(merged)==len(old)
    err=max(float(abs(merged[k+'_new']-merged[k+'_old']).max()) for k in ['best_lag_s','best_lag_gain_mse','core_centered_mse']);assert err<1e-10
    pd.concat(rows).to_csv(OUT/'fixed_lag_per_window.csv',index=False);pd.DataFrame(choices).to_csv(OUT/'fit_lag_choices.csv',index=False);pd.concat(drifts).to_csv(OUT/'block_drift_per_window.csv',index=False)
    base.write(OUT/'verification.json',dict(v10_reproduction_max_abs=err,no_model_training=True,sealed_test_accessed=False))

if __name__=='__main__':main()
