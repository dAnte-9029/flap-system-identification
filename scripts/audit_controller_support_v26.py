"""Frozen v25 history-support audit; no training, prediction intervention or selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from controller_support_v26 import BLOCKS,summaries,support,quantile,eta_squared,correlation,q_errors
from reference_condition_diagnostics_v12 import weighted_terciles,bins

OUT=base.ROOT/'docs/analysis/results/control_windows_v26'
V12=OUT.parent/'control_windows_v12';V23=OUT.parent/'control_windows_v23'
V24=OUT.parent/'control_windows_v24';V25=OUT.parent/'control_windows_v25'
ARMS=['zero_history_control','observed_history_control']


def verify_parent(root):
    d=json.loads((root/'delivery_manifest.json').read_text())
    for path,digest in d['artifacts'].items():
        resolved=base.ROOT/path if path.startswith(str(root.relative_to(base.ROOT))+'/') else root/path
        assert base.sha(resolved)==digest,path
    for path,digest in d['code'].items():assert base.sha(base.ROOT/path)==digest,path


def main():
    for root in [V12,V23,V24,V25]:verify_parent(root)
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),base.ROOT/'scripts/controller_support_v26.py',base.ROOT/'tests/test_controller_support_v26.py']
    dataset=json.loads((V24/'dataset_manifest.json').read_text())
    protocol=dict(stage='v26 historical-controller support diagnostic',status='frozen_before_diagnostic',
        parents={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in [V12,V23,V24,V25]},
        code={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        scope='407original train parents, frozen30v25checkpoints and predictions; no training/modelmutation/windowremoval/developmentvalidation/sealed/reserved access.',
        marginal='Each channel: fit-flight-balanced1/99percentile plus fitmin/max on past26points, query point fractions outside. Report shift using fixedv24training normalization. No outcome based thresholds.',
        joint='Each block integral3/compression3/age2/all8 uses past mean,std,last perchannel. Fit-flight-balanced center/std, scale floor1e-6. RMS Euclidean standardized summary distance. Fit calibration nearest neighbor excludes entire same flight; query nearest among all4fitflights. Threshold fit-flight-balanced95percentile of cross-flight calibration distances; weak if distance>threshold. Marginal/joint flags are descriptive support proxies, not probability/causal validity or deletion rules.',
        structure='Report fit-only flight-balanced eta2 byflight for each mean/std/last summary, and fit correlations with pastfrequencymean and mode. Does not establish flight identity as learned cause.',
        strata='Reuse v12 q frequency_mean/frequency_std terciles and mode verbatim; verify thresholds using fit histories. Append4support strata andall; no outcome-dependent feature bins.',
        outcome='q full1s squared-error difference observed-minus-zero; exact bias/centered andfirst/last.5s decomposition. Fit andOOF; flight RMSE then equalflight/seed. Spearman supportdistance vs deltaMSE withinflight perseed; >=8cases and nonconstant, undefined retained. No pvalues, significance or rootcause claim.',
        limits='5repeated-development flights, correlated features and unequal mode coverage; summarydistance ignores full temporal ordering. No inference-time ablation/distribution modification. All407 retained; future logged rate references remain feedback-conditioned.',
        phase_contract=dataset['phase_contract'],frequency_contract=dataset['frequency_contract'],
        model_promoted=False,validation_accessed=False,sealed_test_accessed=False)
    base.write(OUT/'protocol.json',protocol)
    with np.load(V24/'history_controller.npz') as z:
        h=z['history_controller'].astype(float);times=z['query_us']
        np.testing.assert_array_equal(z['array_index'],np.arange(407))
        assert (z['publication_us']<=times[:,:,None]).all()
    with np.load(V23/'windows.npz') as z:
        truth=z['label_rates_b'][:,:,1].astype(float);state=z['input_history_state'].astype(float)
    w=pd.read_csv(V23/'windows.csv');assert w.partition.eq('train').all() and len(w)==407
    logs=w.log_id.to_numpy();features=dataset['features'];freq=state[:,:,11].mean(1)
    summary={block:summaries(h,ch) for block,ch in BLOCKS.items()}
    raw=pd.DataFrame(summary['all'],columns=[f'{f}_{stat}' for stat in ['mean','std','last'] for f in features])
    raw.insert(0,'array_index',np.arange(407));raw.insert(1,'log_id',logs);raw.to_csv(OUT/'history_summaries.csv',index=False)
    oldstrata=pd.read_csv(V12/'strata.csv')
    oldstrata=oldstrata[(oldstrata.axis=='q')&oldstrata.condition.isin(['frequency_mean','frequency_std','mode'])].drop(columns='axis')
    thresholds=pd.read_csv(V12/'thresholds.csv')
    distances=[];marginals=[];structure=[];threshold_rows=[];outcomes=[];calibration_checks=[]
    for fold in range(5):
        info=json.loads((V23/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
        assert set(logs[fit]).isdisjoint(logs[query])
        # Check existing strata independently using only original past frequency.
        for feature,values in [('frequency_mean',freq),('frequency_std',state[:,:,11].std(1))]:
            cut=weighted_terciles(values[fit],logs[fit]);old=thresholds[(thresholds.fold==fold)&(thresholds.axis=='q')&(thresholds.condition==feature)]
            np.testing.assert_allclose(cut,old[['low','high']].to_numpy()[0],atol=1e-12)
            for cohort,ids in [('fit',fit),('oof',query)]:
                recorded=oldstrata[(oldstrata.fold==fold)&(oldstrata.cohort==cohort)&(oldstrata.condition==feature)].set_index('array_index').loc[ids]
                np.testing.assert_array_equal(recorded.stratum,bins(values[ids],cut))
        fitted={}
        for block,values in summary.items():
            r=support(values[fit],logs[fit],values[query]);fitted[block]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items() if k in ['mean','scale','threshold']}
            threshold_rows.append(dict(fold=fold,block=block,threshold=r['threshold']))
            for cohort,ids,d in [('fit',fit,r['fit_distance']),('oof',query,r['query_distance'])]:
                df=w.iloc[ids][['array_index','log_id','mode']].copy().assign(fold=fold,cohort=cohort,block=block,distance=d,
                    threshold=r['threshold'],weak_support=d>r['threshold'])
                distances.append(df)
            # Independent brute-force cross-flight calibration verification.
            scaled=(values-r['mean'])/r['scale']
            check=np.array([np.sqrt(np.mean((scaled[i]-scaled[fit[logs[fit]!=logs[i]]])**2,axis=1)).min() for i in fit])
            np.testing.assert_allclose(check,r['fit_distance'],rtol=1e-12,atol=1e-12)
            calibration_checks.append(dict(fold=fold,block=block,fit_only=True,same_flight_excluded=True))
        dest=OUT/f'fold{fold}';dest.mkdir();base.write(dest/'support_parameters.json',dict(fit_indices=fit.tolist(),query_indices=query.tolist(),blocks=fitted))
        with np.load(V24/f'fold{fold}/controller_normalization.npz') as z:modelmean=z['mean'];modelscale=z['scale']
        for j,feature in enumerate(features):
            fitted_values=h[fit,:,j].ravel();fitlogs=np.repeat(logs[fit],26)
            lower,upper=[quantile(fitted_values,fitlogs,p) for p in [.01,.99]]
            minimum,maximum=fitted_values.min(),fitted_values.max()
            for log in np.unique(logs):
                ids=np.flatnonzero(logs==log);v=h[ids,:,j].ravel();z=(v-modelmean[j])/modelscale[j]
                marginals.append(dict(fold=fold,cohort='oof' if log==logs[query[0]] else 'fit',log_id=log,feature=feature,windows=len(ids),
                    mean=v.mean(),std=v.std(),p01=np.quantile(v,.01),median=np.median(v),p99=np.quantile(v,.99),
                    fit_lower=lower,fit_upper=upper,fit_min=minimum,fit_max=maximum,
                    outside_central_fraction=np.mean((v<lower)|(v>upper)),outside_range_fraction=np.mean((v<minimum)|(v>maximum)),
                    model_z_mean=z.mean(),model_z_std=z.std()))
            for stat,values in [('mean',h[:,:,j].mean(1)),('std',h[:,:,j].std(1)),('last',h[:,-1,j])]:
                structure.append(dict(fold=fold,feature=feature,stat=stat,fit_flight_eta2=eta_squared(values[fit],logs[fit]),
                    fit_frequency_spearman=correlation(pd.Series(values[fit]).rank(),pd.Series(freq[fit]).rank()),
                    fit_mode_correlation=correlation(values[fit],w.iloc[fit]['mode'].to_numpy())))
        for seed in [17,23,42]:
            for cohort,ids in [('fit',fit),('oof',query)]:
                df=w.iloc[ids][['array_index','log_id','mode']].copy().assign(fold=fold,seed=seed,cohort=cohort)
                for arm,prefix in zip(ARMS,['zero','observed']):
                    root=V25/f'fold{fold}/{arm}/seed{seed}'
                    with np.load(root/f'{cohort}_predictions.npz') as z:
                        np.testing.assert_array_equal(z['array_index'],ids);pred=z['rates_b'][:,:,1].astype(float)
                    errors=q_errors(pred,truth[ids])
                    np.testing.assert_allclose(errors['mse'],errors['bias_mse']+errors['centered_mse'])
                    for key,value in errors.items():df[prefix+'_'+key]=value
                for key in errors:df['delta_'+key]=df['observed_'+key]-df['zero_'+key]
                outcomes.append(df)
    d=pd.concat(distances,ignore_index=True);o=pd.concat(outcomes,ignore_index=True)
    d.to_csv(OUT/'support_per_window.csv',index=False);o.to_csv(OUT/'q_per_window.csv',index=False)
    pd.DataFrame(marginals).to_csv(OUT/'marginal_distribution.csv',index=False)
    pd.DataFrame(structure).to_csv(OUT/'fit_feature_structure.csv',index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT/'support_thresholds.csv',index=False)
    keys=['fold','cohort','array_index','log_id']
    assignments=[oldstrata[keys+['condition','stratum']]]
    assignments.append(d.assign(condition='support_'+d.block,stratum=np.where(d.weak_support,'weak','covered'))[keys+['condition','stratum']])
    assignments.append(d[d.block=='all'].assign(condition='all',stratum='all')[keys+['condition','stratum']])
    assignments=pd.concat(assignments,ignore_index=True)
    assert not assignments.duplicated(keys+['condition']).any()
    assignments.to_csv(OUT/'strata.csv',index=False)
    expanded=o.merge(assignments,on=keys,validate='many_to_many');assert len(expanded)==len(o)*8
    values=[c for c in o.columns if c.endswith('mse')]
    group=['cohort','seed','condition','stratum','log_id']
    pf=expanded.groupby(group)[values].mean()
    for prefix in ['zero','observed']:pf[prefix+'_rmse']=np.sqrt(pf[prefix+'_mse'])
    counts=expanded.groupby(group).agg(cases=('array_index','size'),unique_windows=('array_index','nunique'))
    pf=pf.join(counts).reset_index();pf['improvement_percent']=100*(1-pf.observed_rmse/pf.zero_rmse)
    pf.to_csv(OUT/'q_strata_per_flight.csv',index=False)
    avg=pf.groupby(['cohort','condition','stratum'])[['zero_rmse','observed_rmse']+values].mean().reset_index()
    avg['improvement_percent']=100*(1-avg.observed_rmse/avg.zero_rmse)
    avg.to_csv(OUT/'q_strata_summary.csv',index=False)
    coverage=assignments[assignments.cohort=='oof'].groupby(['condition','stratum']).agg(windows=('array_index','size'),flights=('log_id','nunique'))
    coverage.to_csv(OUT/'strata_coverage.csv')
    assoc=o.merge(d.drop(columns='mode'),on=keys,validate='many_to_many')
    correlations=[]
    for key,g in assoc.groupby(['cohort','seed','block','log_id']):
        correlations.append(dict(zip(['cohort','seed','block','log_id'],key),cases=len(g),
            spearman=correlation(g.distance.rank(),g.delta_mse.rank())))
    pd.DataFrame(correlations).to_csv(OUT/'support_error_association.csv',index=False)
    # Full parent q metrics independently reproduced, with no full metrics CSV load.
    old=pd.read_csv(V25/'comparison.csv');old=old[(old.metric=='q')&(old.kind=='wave')&(old.horizon==50)]
    for cohort in ['fit','oof']:
        for seed in [17,23,42]:
            got=pf[(pf.condition=='all')&(pf.cohort==cohort)&(pf.seed==seed)]
            want=old[(old.cohort==cohort)&(old.seed==seed)].iloc[0]
            np.testing.assert_allclose(got[['zero_rmse','observed_rmse']].mean(),want[ARMS].to_numpy(dtype=float),rtol=1e-12)
    for seed,g in o[o.cohort=='oof'].groupby('seed'):assert sorted(g.array_index)==list(range(407))
    base.write(OUT/'verification.json',dict(status='passed',parent_hashes=True,original_strata_recomputed=True,
        calibration_checks=calibration_checks,exact_mse_decomposition=True,v25_fit_oof_q_reproduced=True,
        each_seed_407_unique_oof=True,windows_retained=407,training_started=False,validation_accessed=False,sealed_test_accessed=False))
    print(avg[avg.cohort=='oof'][['condition','stratum','zero_rmse','observed_rmse','improvement_percent']].to_string(index=False))


if __name__=='__main__':main()
