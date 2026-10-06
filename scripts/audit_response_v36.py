"""Frozen v35/v30 observed-response audit; no training, fitting delays or exclusions."""
from pathlib import Path
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from train_joint_increment_v34 import DATA, CONTROL, OUT as V35, LOCAL, read
from actuator_diagnostics_v31 import modal_commands, features, integrate_rates, angle_squared
from controller_support_v26 import quantile
from reference_error_structure_v9 import decompose
from response_audit_v36 import anchors, response_trace, response_metrics, fixed_support_lags

OUT = V35.parent/'control_windows_v36'
SUPPORT = V35.parent/'control_windows_v31'
CHANNELS = ['motor', 'common', 'diff', 'yaw']


def main():
    torch.set_num_threads(1)
    parents = [DATA, CONTROL, SUPPORT, LOCAL, V35]
    for root in parents: verify_parent(root)
    paths = ['scripts/audit_response_v36.py', 'scripts/response_audit_v36.py',
             'tests/test_response_audit_v36.py', 'scripts/actuator_diagnostics_v31.py',
             'scripts/controller_support_v26.py', 'scripts/reference_error_structure_v9.py']
    protocol = dict(stage='v36 frozen observed maneuver response audit',
        parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in parents},
        code={p:base.sha(base.ROOT/p) for p in paths},
        scope='All407 original train parents, five OOF flights, seeds17/23/42. Same v30/v35 models. No training, checkpoint modification, window deletion or validation/sealed/reserved access.',
        strata='Reuse immutable v31 mode/frequency/allocator and support strata verbatim. Verify fit-only support scaling/distances. Future-command support is offline only.',
        events='Input-only retrospective anchors: per parent and modal command, max absolute mean(next5)-mean(previous5) in first25 command positions0..24. Origin command is future0; history origin omitted from previous5. Qualify if nonzero and >=75th percentile of corresponding FIT-parent maxima, fit-flight-balanced. One anchor per channel/parent; several channels may share support. No output-based thresholds or exclusions. All parents remain in full metrics.',
        response='Each prediction and true curve minus its own pre5 rate mean, post25samples (next20..500ms). Pre5 uses true pre-origin history and its own predicted/true subsequent history. Report raw response errors separately from full rollout errors. Late mean samples5:25 defines sign, numerical-zero tolerance1e-12 only; weak/opposite/neutral outcomes retained, not causal control directions.',
        timing='Oracle alignment +/-5samples on fixed truth5:20; predictions5+lag:20+lag, separately centered. Positive lag means later prediction aligns with earlier truth. Best per-flight/seed lag after seeing outcomes is not measured actuator delay or a deployed correction.',
        amplitude_coupling='All4modal channels x3rate axes, RMS amplitude and sustained difference, shape cosine. Cooccurrence if any other normalized contrast at anchor >=0.5 target contrast, normalizer=fit event75th-percentile floor1e-6. Modal command-difference correlations and covariance eigenvalues standardized on fit only. These are descriptive coupled closed-loop observations, not isolated gains.',
        persistence='Exact full1s bias/amplitude/shape decomposition, early/late MSE. v35 rolling derivative error mean and per-parent accumulated p/q/r retained and stratified. Attitude integration with single-axis truth substitutions is offline oracle only, no GRU reroll.',
        weighting='Within-flight MSE then root, equal flights within seed then equal seeds. Sign means likewise equal-flight/seed. Report event/parent/flight counts; related channel anchors and rolling observations are not independent experiments.',
        decisions='No new promotion threshold or response-selected subset. Preserve v34 absolute and relative gates; control readiness remains false without causal-response and closed-loop evidence. Decide next research step from complete audit, not best subset.',
        phase_contract=read(DATA/'dataset_manifest.json')['phase_contract'],
        frequency_contract=read(DATA/'dataset_manifest.json')['frequency_contract'],
        model_promoted=False, training_started=False, validation_accessed=False, sealed_test_accessed=False,reserved_accessed=False)
    if OUT.exists(): assert read(OUT/'protocol.json') == protocol, 'frozen audit protocol changed'
    else: OUT.mkdir(); base.write(OUT/'protocol.json', protocol)
    base.write(OUT/'dataset_manifest.json', dict(source=str(DATA.relative_to(base.ROOT)),
        source_manifest_sha256=base.sha(DATA/'dataset_manifest.json'), arrays_sha256=base.sha(DATA/'windows.npz'),
        support_manifest_sha256=base.sha(SUPPORT/'delivery_manifest.json'), partitions=['original_train_5flight_OOF'],
        windows=407, phase_contract=protocol['phase_contract'], frequency_contract=protocol['frequency_contract']))
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_response_audit_v36.py'],capture_output=True,text=True)
    base.write(OUT/'test_results.json', dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
    assert test.returncode == 0, test.stdout+test.stderr
    w = pd.read_csv(DATA/'windows.csv'); assert len(w)==407 and w.partition.eq('train').all()
    logs = w.log_id.to_numpy()
    with np.load(DATA/'windows.npz') as z: a = {k:z[k] for k in z.files}
    h, f = modal_commands(a['input_history_command']), modal_commands(a['input_future_command'])
    offsets, change, contrast = anchors(h, f)
    state = a['input_history_state'].astype(float)
    fs = features(state[:,:,[0,1,2,3,4,5,6,7,8,11,12,13]])
    fc = features(h)
    blocks = dict(past_command=fc, future_command=features(f), past_state=fs, past_joint=np.concatenate([fc,fs],1))
    strata = pd.read_csv(SUPPORT/'strata.csv'); strata = strata[strata.cohort=='oof'].drop(columns='cohort')
    distances = pd.read_csv(SUPPORT/'support_per_window.csv')
    events=[]; thresholds=[]; rates=[]; event_rows=[]; lag_rows=[]; input_stats=[]; att_rows=[]
    checked=[]
    for fold in range(5):
        info=read(DATA/f'fold{fold}/manifest.json'); fit=np.array(info['fit_indices']); ids=np.array(info['query_indices'])
        assert set(logs[fit]).isdisjoint(logs[ids])
        # Recompute query distance using only frozen FIT statistics and neighbors.
        pars=read(SUPPORT/f'fold{fold}/support_parameters.json')['blocks']
        for name,value in blocks.items():
            norm=(value-np.array(pars[name]['mean']))/np.array(pars[name]['scale'])
            d=np.array([np.sqrt(((norm[i]-norm[fit])**2).mean(1)).min() for i in ids])
            old=distances[(distances.fold==fold)&(distances.cohort=='oof')&(distances.block==name)].set_index('array_index').loc[ids]
            np.testing.assert_allclose(d,old.distance,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(d>pars[name]['threshold'],old.weak)
        cut=np.array([quantile(abs(change[fit,j]),logs[fit],.75) for j in range(4)])
        for j,channel in enumerate(CHANNELS): thresholds.append(dict(fold=fold,channel=channel,threshold=cut[j]))
        local_events=[]
        for i in ids:
            for j,channel in enumerate(CHANNELS):
                if abs(change[i,j])<cut[j] or abs(change[i,j])<=1e-12: continue
                k=int(offsets[i,j]); normcontrast=abs(contrast[i,k])/np.maximum(cut,1e-6)
                coupled=normcontrast>=.5*normcontrast[j];coupled[j]=False
                record=dict(fold=fold,array_index=int(i),log_id=logs[i],channel=channel,offset=k,
                    event_time_after_origin_s=.02*k,command_change=change[i,j],threshold=cut[j],
                    simultaneous_other_channels=int(coupled.sum()),positive_command=bool(change[i,j]>0))
                events.append(record);local_events.append(record)
        # Difference correlations are input support descriptors, not transfer gains.
        delta=np.diff(np.concatenate([h[:,-2:-1],f],1),axis=1)
        scale=np.maximum(delta[fit].reshape(-1,4).std(0),1e-6)
        for cohort,indices in [('fit',fit),('oof',ids)]:
            v=delta[indices].reshape(-1,4)/scale
            cov=np.cov(v,rowvar=False);den=np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
            corr=np.divide(cov,den,out=np.full_like(cov,np.nan),where=den>1e-12)
            eig=np.linalg.eigvalsh(cov)
            base.write(OUT/f'fold{fold}_{cohort}_input_covariance.json',dict(correlation=corr.tolist(),eigenvalues=eig.tolist(),
                scaled_on_fit=True,condition_number=float(eig[-1]/eig[0]) if eig[0]>1e-12 else None))
        for seed in [17,23,42]:
            for arm,root in [('baseline',CONTROL/f'fold{fold}/seed{seed}'),('joint',V35/f'fold{fold}/joint_increment/seed{seed}')]:
                with np.load(root/'oof_predictions.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ids); pred=z['rates_b']; quat=z['quaternion']
                truth=a['label_rates_b'][ids]; err=pred.astype(float)-truth.astype(float)
                dec=decompose(pred,truth); dec.update(early_mse=(err[:,:25]**2).mean(1),late_mse=(err[:,25:]**2).mean(1))
                np.testing.assert_allclose(dec['total_mse'],dec['bias_mse']+dec['amplitude_mse']+dec['shape_mse'],atol=1e-12)
                for j,axis in enumerate('pqr'):
                    df=w.iloc[ids][['array_index','log_id','mode']].copy().assign(fold=fold,seed=seed,arm=arm,axis=axis)
                    for key,val in dec.items():df[key]=val[:,j]
                    rates.append(df)
                # Exact replay; replacing one rate only evaluates kinematic error contribution.
                replay=integrate_rates(a['input_initial_quaternion'][ids],a['input_initial_rates_b'][ids],pred,a['input_dt_s'][ids])
                np.testing.assert_array_equal(replay,quat)
                for axis in ['none','p','q','r','all']:
                    mixed=pred.copy()
                    if axis=='all':mixed=truth.copy()
                    elif axis!='none':j='pqr'.index(axis);mixed[:,:,j]=truth[:,:,j]
                    est=quat if axis=='none' else integrate_rates(a['input_initial_quaternion'][ids],a['input_initial_rates_b'][ids],mixed,a['input_dt_s'][ids])
                    sq=angle_squared(est,a['label_quaternion'][ids])
                    for idx,val in zip(ids,sq[:,-1]):att_rows.append(dict(fold=fold,seed=seed,arm=arm,array_index=idx,log_id=logs[idx],replacement=axis,end_mse=val))
                for event in local_events:
                    i,k=event['array_index'],event['offset'];ii=np.flatnonzero(ids==i).item()
                    hist=a['input_history_state'][i,:,3:6]
                    np.testing.assert_array_equal(hist[-1],a['input_initial_rates_b'][i])
                    p=response_trace(hist,pred[ii],k);t=response_trace(hist,truth[ii],k)
                    met=response_metrics(p,t); lags=fixed_support_lags(p,t)
                    for j,axis in enumerate('pqr'):
                        row=dict(**event,seed=seed,arm=arm,axis=axis)
                        row.update({key:val[j] for key,val in met.items()})
                        event_rows.append(row)
                        for lag,val in lags.items():lag_rows.append(dict(**event,seed=seed,arm=arm,axis=axis,lag=lag,mse=val[j]))
                checked.append(dict(fold=fold,seed=seed,arm=arm,rate_decomposition_exact=True,quaternion_replay_exact=True))
        print(f'fold{fold}: {len(local_events)} input-defined channel anchors, all parents retained',flush=True)
    events=pd.DataFrame(events); events.to_csv(OUT/'events.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'event_thresholds.csv',index=False)
    rate=pd.concat(rates,ignore_index=True);rate.to_csv(OUT/'rate_errors.csv',index=False)
    for _,g in rate.groupby(['seed','arm','axis']):np.testing.assert_array_equal(np.sort(g.array_index),np.arange(407))
    cols=['total_mse','bias_mse','centered_mse','amplitude_mse','shape_mse','early_mse','late_mse','signed_bias','predicted_std','truth_std']
    joined=rate.merge(strata,on=['fold','array_index','log_id'],validate='many_to_many')
    group=['seed','arm','axis','fold','log_id','condition','stratum']
    ag=joined.groupby(group)[cols].mean().reset_index();counts=joined.groupby(group).size().reset_index(name='windows');ag=ag.merge(counts)
    for key,g in ag.groupby(['seed','arm','axis','fold','log_id','condition']):
        seed,arm,axis,fold,log,condition=key
        raw=rate[(rate.seed==seed)&(rate.arm==arm)&(rate.axis==axis)&(rate.fold==fold)]
        assert g.windows.sum()==len(raw)
        np.testing.assert_allclose(np.average(g[cols],weights=g.windows,axis=0),raw[cols].mean(),rtol=1e-11,atol=1e-11)
    ag.to_csv(OUT/'rate_by_strata.csv',index=False)
    ev=pd.DataFrame(event_rows);ev.to_csv(OUT/'event_response.csv',index=False)
    met=list(response_metrics(np.zeros((25,3)),np.zeros((25,3))))
    flight=ev.groupby(['seed','arm','channel','axis','log_id'])[met].mean().reset_index()
    rms=['response_mse','predicted_power','truth_power','sustained_error_squared']
    flight[rms]=np.sqrt(flight[rms]);flight.to_csv(OUT/'event_by_flight.csv',index=False)
    flight.groupby(['seed','arm','channel','axis'])[met].mean().groupby(['arm','channel','axis']).mean().to_csv(OUT/'event_summary.csv')
    event_strata=ev.merge(strata,on=['fold','array_index','log_id'])
    event_strata.groupby(['seed','arm','channel','axis','log_id','condition','stratum'])[met].mean().to_csv(OUT/'event_by_strata.csv')
    lag=pd.DataFrame(lag_rows).groupby(['seed','arm','channel','axis','log_id','lag']).mse.mean().reset_index()
    lag.to_csv(OUT/'event_lag_curves.csv',index=False)
    best=lag.loc[lag.groupby(['seed','arm','channel','axis','log_id']).mse.idxmin()]
    best.to_csv(OUT/'event_best_lag.csv',index=False)
    pd.DataFrame(att_rows).to_csv(OUT/'attitude_oracles.csv',index=False)
    roll=pd.read_csv(V35/'rolling_per_window.csv')
    roll.merge(strata,on=['fold','array_index','log_id']).groupby(['seed','arm','axis','log_id','condition','stratum'])[
        ['local_mse','accumulated_mse','accumulated_end_mse','derivative_bias']].mean().to_csv(OUT/'rolling_by_strata.csv')
    p=roll[roll.axis=='p'].pivot(index=['fold','seed','array_index','log_id'],columns='arm',values='accumulated_mse').reset_index()
    p['delta_mse']=p.joint-p.baseline;p.sort_values('delta_mse',ascending=False).to_csv(OUT/'p_accumulation_contributions.csv',index=False)
    # Reproduce all original v35 per-seed rate waveform RMSE values independently.
    published=pd.read_csv(V35/'comparison.csv')
    for (seed,arm,axis),g in rate.groupby(['seed','arm','axis']):
        value=g.groupby('log_id').total_mse.mean().pow(.5).mean()
        row=published[(published.seed==seed)&(published.cohort=='oof')&(published.metric==axis)&(published.horizon==50)&(published.kind=='wave')]
        np.testing.assert_allclose(value,row[arm].item(),rtol=1e-12,atol=1e-12)
    for root in parents: verify_parent(root)
    for path,digest in protocol['code'].items():assert base.sha(base.ROOT/path)==digest
    base.write(OUT/'verification.json',dict(status='passed',tests_passed=3,checks=checked,
        frozen_support_distances_reproduced=True,strata_recompose_all_original_parents=True,
        full_v35_metrics_reproduced=True,input_only_events=True,all407_retained=True,
        trained_models=0,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    print('v36 response audit complete',flush=True)


if __name__=='__main__':main()
