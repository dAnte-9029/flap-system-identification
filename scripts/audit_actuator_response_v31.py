"""Frozen five-flight support and response localization; no training or selection."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from controller_support_v26 import support, quantile, correlation
from reference_condition_diagnostics_v12 import weighted_terciles, bins
from reference_error_structure_v9 import decompose
from actuator_diagnostics_v31 import modal_commands, features, integrate_rates, angle_squared, lag_errors

OUT = base.ROOT/'docs/analysis/results/control_windows_v31'
DATA = OUT.parent/'control_windows_v29'
V12 = OUT.parent/'control_windows_v12'
V23 = OUT.parent/'control_windows_v23'
V28 = OUT.parent/'control_windows_v28'
V30 = OUT.parent/'control_windows_v30'


def read(p): return json.loads(p.read_text())


def main():
    torch.set_num_threads(1)
    for root in [V12,V23,V28,DATA,V30]: verify_parent(root)
    pins = {str(p.relative_to(base.ROOT)):base.sha(p) for p in [Path(__file__).resolve(),
        base.ROOT/'scripts/actuator_diagnostics_v31.py',base.ROOT/'tests/test_actuator_diagnostics_v31.py',
        base.ROOT/'scripts/controller_support_v26.py',base.ROOT/'scripts/reference_condition_diagnostics_v12.py',
        base.ROOT/'scripts/reference_error_structure_v9.py',base.ROOT/'src/system_identification/models/trajectory_main_v1.py']}
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'protocol.json',dict(stage='v31 frozen actuator response/support localization',status='frozen_before_diagnostics',
        parents={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in [V12,V23,V28,DATA,V30]},code=pins,
        scope='All407 original train windows and15v30+15v23 frozen predictions, fit and OOF; no training/checkpoint changes/exclusions or validation/sealed/reserved access.',
        support='Physical command combinations motor,common=(servo0+servo1)/2,diff=(servo1-servo0)/2,yaw. Each history/future block mean,std,last,stepRMS; past-state same summaries of columns0..8,11..13. Four blocks: past command16, future command16, past state48, past command+state64. Fit-flight-balanced standardization floor1e-6; RMS nearest neighbor; fit calibration excludes entire same flight,95percentile fit-flight-balanced threshold. Query cannot affect statistics. Future-command support is OFFLINE only; no online trust claim.',
        marginal='Fit-flight-balanced1/99percentile andmin/max per command channel separately history/future. Do not drop tails or interpret coverage as causal validity.',
        strata='Reuse v12 q history frequency_mean/frequency_std terciles and mode verbatim, independently verify thresholds. Add input support covered/weak and v28 allocator-limited flag. No error-selected bins.',
        rate_errors='p/q/r waveform total=bias+centered=bias+amplitude+shape. Full1s,first/last.5s, signedbias, timewise residual/MSE. Paired candidate-minus-baseline, both fit/OOF, by flight andseed. No pvalues or causal root-cause claim.',
        lag='q only; lags -5..5samples of20ms on fixed truth indices5:45, centeredMSE. Positive means later prediction aligned with earlier truth. Best group-level lag after seeing truth is oracle diagnostic, NOT fitted input shift, physical latency or corrected deployable prediction.',
        attitude='Replay midpoint integration of saved predicted rates from original initial attitude/rates; exact match stored quaternions. Then replace one predicted axis or all3 with future true rates and reintegrate only kinematics, preserving other predicted rates. Offline oracle; no GRU reroll/state feedback. Axis improvements nonadditive and do not establish causal actuator effect. All-rate case checks label/integration discrepancy.',
        comparison='Preserve v30 metric definitions and equal-flight then equal-seed averaging. Reproduce frozen q andattitude metrics before interpreting; all407 retained.',
        baseline_retained='A1600',model_promoted=False,training_started=False,
        phase_contract=read(DATA/'dataset_manifest.json')['phase_contract'],frequency_contract=read(DATA/'dataset_manifest.json')['frequency_contract'],
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    w = pd.read_csv(DATA/'windows.csv'); logs = w.log_id.to_numpy()
    assert len(w)==407 and w.partition.eq('train').all()
    with np.load(DATA/'windows.npz') as z: arrays = {k:z[k] for k in z.files}
    state = arrays['input_history_state'].astype(float)
    commands = {name:modal_commands(arrays['input_'+key]) for name,key in
                [('past','history_command'),('future','future_command')]}
    fs = features(state[:,:,[0,1,2,3,4,5,6,7,8,11,12,13]])
    fc = {name:features(a) for name,a in commands.items()}
    blocks = dict(past_command=fc['past'],future_command=fc['future'],past_state=fs,
                  past_joint=np.concatenate([fc['past'],fs],1))
    oldstrata = pd.read_csv(V12/'strata.csv')
    strata = oldstrata[(oldstrata.axis=='q')&oldstrata.condition.isin(['frequency_mean','frequency_std','mode'])].drop(columns='axis').copy()
    thresholds = pd.read_csv(V12/'thresholds.csv')
    limited = pd.read_csv(V28/'allocation_limits_per_window.csv').set_index('array_index').loc[w.array_index]
    assert np.array_equal(limited.log_id,logs)
    rate_rows=[];time_rows=[];lag_rows=[];attitude_rows=[];distances=[];marginal=[];checks=[]
    for fold in range(5):
        info=read(DATA/f'fold{fold}/manifest.json'); fit=np.array(info['fit_indices']); query=np.array(info['query_indices'])
        assert set(logs[fit]).isdisjoint(logs[query])
        for name,val in [('frequency_mean',state[:,:,11].mean(1)),('frequency_std',state[:,:,11].std(1))]:
            cut=weighted_terciles(val[fit],logs[fit]); row=thresholds[(thresholds.fold==fold)&(thresholds.axis=='q')&(thresholds.condition==name)]
            np.testing.assert_allclose(cut,row[['low','high']].to_numpy()[0],rtol=1e-12)
            for cohort,ids in [('fit',fit),('oof',query)]:
                old=strata[(strata.fold==fold)&(strata.cohort==cohort)&(strata.condition==name)].set_index('array_index').loc[ids]
                np.testing.assert_array_equal(old.stratum,bins(val[ids],cut))
        pars={}
        for name,value in blocks.items():
            result=support(value[fit],logs[fit],value[query]); pars[name]={k:np.asarray(result[k]).tolist() for k in ['mean','scale','threshold']}
            # Independent nearest neighbor reference (same-flight exclusion).
            norm=(value-result['mean'])/result['scale']
            for cohort,ids,key in [('fit',fit,'fit_distance'),('oof',query,'query_distance')]:
                oracle=np.array([np.sqrt(np.mean((norm[i]-norm[fit[logs[fit]!=logs[i]]])**2,axis=1)).min() for i in ids])
                np.testing.assert_allclose(oracle,result[key],rtol=1e-12,atol=1e-12)
                d=w.iloc[ids][['array_index','log_id','mode']].copy().assign(fold=fold,cohort=cohort,block=name,
                    distance=result[key],threshold=result['threshold'],weak=result[key]>result['threshold'])
                distances.append(d)
                st=d[['array_index','log_id','fold','cohort']].copy(); st['condition']='support_'+name;st['stratum']=np.where(d.weak,'weak','covered')
                strata=pd.concat([strata,st],ignore_index=True)
        dest=OUT/f'fold{fold}';dest.mkdir();base.write(dest/'support_parameters.json',dict(fit_indices=fit.tolist(),query_indices=query.tolist(),blocks=pars))
        for name,a in commands.items():
            for j,channel in enumerate(['motor','common','diff','yaw']):
                fv=a[fit,:,j].ravel();fl=np.repeat(logs[fit],a.shape[1]);low=quantile(fv,fl,.01);high=quantile(fv,fl,.99)
                for cohort,ids in [('fit',fit),('oof',query)]:
                    for log in np.unique(logs[ids]):
                        ix=ids[logs[ids]==log];v=a[ix,:,j]
                        marginal.append(dict(fold=fold,cohort=cohort,log_id=log,block=name,channel=channel,
                            fit_low=low,fit_high=high,fit_min=fv.min(),fit_max=fv.max(),
                            outside_central_fraction=np.mean((v<low)|(v>high)),outside_range_fraction=np.mean((v<fv.min())|(v>fv.max())),
                            mean=v.mean(),std=v.std()))
        for cohort,ids in [('fit',fit),('oof',query)]:
            for condition,values in [('all',np.full(len(ids),'all')),
                 ('allocator_flag',np.where(limited.iloc[ids].limited_grid_points>0,'limited','unflagged'))]:
                strata=pd.concat([strata,pd.DataFrame(dict(array_index=ids,log_id=logs[ids],fold=fold,cohort=cohort,
                                                           condition=condition,stratum=values))],ignore_index=True)
        for seed in [17,23,42]:
            for cohort,ids in [('fit',fit),('oof',query)]:
                truths=arrays['label_rates_b'][ids]
                results={}
                for arm,root in [('baseline',V23),('actuator',V30)]:
                    with np.load(root/f'fold{fold}/seed{seed}/{cohort}_predictions.npz') as z:
                        np.testing.assert_array_equal(z['array_index'],ids); pred=z['rates_b']; quat=z['quaternion']
                    results[arm]=decompose(pred,truths)
                    residual=pred.astype(float)-truths.astype(float)
                    results[arm].update(early_mse=np.mean(residual[:,:25]**2,1),late_mse=np.mean(residual[:,25:]**2,1))
                    d=results[arm]
                    np.testing.assert_allclose(d['total_mse'],d['bias_mse']+d['centered_mse'],atol=1e-12)
                    np.testing.assert_allclose(d['total_mse'],d['bias_mse']+d['amplitude_mse']+d['shape_mse'],atol=1e-12)
                    np.testing.assert_allclose(d['total_mse'],.5*(d['early_mse']+d['late_mse']),atol=1e-12)
                    lags=lag_errors(pred[:,:,1],truths[:,:,1])
                    for log in np.unique(logs[ids]):
                        keep=logs[ids]==log
                        for lag,value in lags.items(): lag_rows.append(dict(fold=fold,seed=seed,cohort=cohort,log_id=log,arm=arm,lag_steps=lag,centered_mse=value[keep].mean()))
                        for axis,j in zip(['p','q','r'],range(3)):
                            for k in range(50):time_rows.append(dict(fold=fold,seed=seed,cohort=cohort,log_id=log,arm=arm,axis=axis,step=k+1,
                                mse=np.mean(residual[keep,k,j]**2),signed_bias=np.mean(residual[keep,k,j])))
                    replay=integrate_rates(arrays['input_initial_quaternion'][ids],arrays['input_initial_rates_b'][ids],pred,arrays['input_dt_s'][ids])
                    np.testing.assert_array_equal(replay,quat)
                    attitude=w.iloc[ids][['array_index','log_id']].copy().assign(fold=fold,seed=seed,cohort=cohort,arm=arm)
                    for replacement in ['none','p','q','r','all']:
                        if replacement=='none': estimated=quat
                        else:
                            changed=pred.copy()
                            if replacement=='all':changed=truths.copy()
                            else:
                                j=['p','q','r'].index(replacement);changed[:,:,j]=truths[:,:,j]
                            estimated=integrate_rates(arrays['input_initial_quaternion'][ids],arrays['input_initial_rates_b'][ids],changed,arrays['input_dt_s'][ids])
                        err=angle_squared(estimated,arrays['label_quaternion'][ids])
                        attitude[replacement+'_wave_mse']=err.mean(1)
                        attitude[replacement+'_half_mse']=err[:,24]
                        attitude[replacement+'_end_mse']=err[:,49]
                    attitude_rows.append(attitude)
                for axis,j in zip(['p','q','r'],range(3)):
                    frame=w.iloc[ids][['array_index','log_id','mode']].copy().assign(fold=fold,seed=seed,cohort=cohort,axis=axis)
                    for key in results['baseline']:
                        frame['baseline_'+key]=results['baseline'][key][:,j]
                        frame['actuator_'+key]=results['actuator'][key][:,j]
                        frame['delta_'+key]=frame['actuator_'+key]-frame['baseline_'+key]
                    rate_rows.append(frame)
            checks.append(dict(fold=fold,seed=seed,quaternion_reintegration_exact=True,rate_decomposition_exact=True))
        print(f'fold{fold}: support, waveform and kinematic diagnostics complete',flush=True)
    rate=pd.concat(rate_rows,ignore_index=True); att=pd.concat(attitude_rows,ignore_index=True); dist=pd.concat(distances,ignore_index=True)
    rate.to_csv(OUT/'rate_errors.csv',index=False); att.to_csv(OUT/'attitude_oracles.csv',index=False)
    dist.to_csv(OUT/'support_per_window.csv',index=False);pd.DataFrame(marginal).to_csv(OUT/'marginal_support.csv',index=False)
    pd.DataFrame(time_rows).to_csv(OUT/'error_timecourse.csv',index=False);pd.DataFrame(lag_rows).to_csv(OUT/'lag_curves.csv',index=False)
    assert not strata.duplicated(['fold','cohort','array_index','condition']).any()
    strata.to_csv(OUT/'strata.csv',index=False)
    q=rate[rate.axis=='q']; joined=q.merge(strata,on=['fold','cohort','array_index','log_id'],validate='many_to_many')
    columns=[k for k in rate if k.startswith(('baseline_','actuator_','delta_'))]
    agg=joined.groupby(['fold','seed','cohort','log_id','condition','stratum'])[columns].mean().reset_index()
    counts=joined.groupby(['fold','seed','cohort','log_id','condition','stratum']).size().reset_index(name='windows')
    agg=agg.merge(counts); agg['baseline_rmse']=np.sqrt(agg.baseline_total_mse);agg['actuator_rmse']=np.sqrt(agg.actuator_total_mse)
    # Each partition of each flight/seed must reproduce its unstratified MSE.
    for (fold,seed,cohort,log,condition),g in agg.groupby(['fold','seed','cohort','log_id','condition']):
        raw=q[(q.fold==fold)&(q.seed==seed)&(q.cohort==cohort)&(q.log_id==log)]
        assert g.windows.sum()==len(raw)
        np.testing.assert_allclose(np.average(g.delta_total_mse,weights=g.windows),raw.delta_total_mse.mean(),atol=1e-12)
    agg.to_csv(OUT/'q_strata.csv',index=False)
    correlations=[]
    for (fold,cohort,log,block),group in dist.groupby(['fold','cohort','log_id','block']):
        for seed in [17,23,42]:
            v=q[(q.fold==fold)&(q.seed==seed)&(q.cohort==cohort)&(q.log_id==log)].set_index('array_index').loc[group.array_index]
            correlations.append(dict(fold=fold,cohort=cohort,log_id=log,block=block,seed=seed,
                spearman=correlation(pd.Series(group.distance.to_numpy()).rank(),pd.Series(v.delta_total_mse.to_numpy()).rank()),windows=len(v)))
    pd.DataFrame(correlations).to_csv(OUT/'support_error_correlations.csv',index=False)
    # Independently preserve the published v30 all-flight/seed metric aggregation.
    for key in ['p','q','r']:
        v=rate[(rate.cohort=='oof')&(rate.axis==key)]
        for arm in ['baseline','actuator']:
            actual=v.groupby(['seed','log_id'])[arm+'_total_mse'].mean().pow(.5).groupby('seed').mean()
            published=pd.read_csv(V30/'comparison.csv');published=published[(published.cohort=='oof')&(published.kind=='wave')&(published.horizon==50)&(published.metric==key)].set_index('seed')
            np.testing.assert_allclose(actual,published.loc[actual.index,arm],atol=1e-12)
    for arm in ['baseline','actuator']:
        a=att[(att.cohort=='oof')&(att.arm==arm)].groupby(['seed','log_id']).none_wave_mse.mean().pow(.5).groupby('seed').mean()
        published=pd.read_csv(V30/'comparison.csv');p=published[(published.cohort=='oof')&(published.kind=='wave')&(published.horizon==50)&(published.metric=='attitude')].set_index('seed')
        np.testing.assert_allclose(a,p.loc[a.index,arm],atol=1e-12)
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_actuator_diagnostics_v31.py','tests/test_controller_support_v26.py'],text=True,capture_output=True,cwd=base.ROOT)
    base.write(OUT/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0,test.stdout+test.stderr
    base.write(OUT/'verification.json',dict(status='passed',checks=checks,tests_passed=6,
        independent_support_nearest_neighbors=True,original_frequency_strata_verified=True,
        strata_weighted_mse_recompose=True,v30_rate_and_attitude_metrics_reproduced=True,
        training_started=False,validation_accessed=False,sealed_test_accessed=False))
    for root in [V12,V23,V28,DATA,V30]:verify_parent(root)
    for name,digest in pins.items():assert base.sha(base.ROOT/name)==digest
    print('v31 diagnostics complete; no training or window deletion',flush=True)


if __name__=='__main__':main()
