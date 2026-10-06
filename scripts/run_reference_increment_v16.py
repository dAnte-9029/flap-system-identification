"""Freeze and run original 20ms increment diagnostics, without model changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train, model_from, SEEDS
from reference_condition_diagnostics_v12 import weighted_terciles, bins
from reference_increment_v16 import history_spectrum, clipped_increment, metrics
from run_reference_generalization_v7 import verify_manifest

OUT=base.ROOT/'docs/analysis/results/control_windows_v16'
V8=OUT.parent/'control_windows_v8'
V12=OUT.parent/'control_windows_v12'
V15=OUT.parent/'control_windows_v15'


def capture(model, x, ids):
    predictions=[]; heads=[]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            local=[]
            hook=model.head.register_forward_hook(lambda module,args,out:local.append(out.detach().numpy().copy()))
            try:
                pred=model(**{k:v[ids[start:start+64]] for k,v in x.items()})
            finally:
                hook.remove()
            assert len(local)==50
            predictions.append(pred['rates_b'].numpy());heads.append(np.stack(local,axis=1))
    return np.concatenate(predictions),np.concatenate(heads)


def summarize(frame):
    records=[]
    for condition,column in [('all',None),('history_hf_rms','hf_stratum'),('history_frequency','frequency_stratum')]:
        keys=['fold','seed','cohort','log_id','axis']+([column] if column else [])
        for key,g in frame.groupby(keys):
            record=dict(zip(keys,key));record['condition']=condition
            record['stratum']=record.pop(column) if column else 'all'
            record.update(metrics(g));records.append(record)
    perflight=pd.DataFrame(records)
    keys=['cohort','axis','condition','stratum']
    # Fit observations recur under four separately fitted models: average those
    # models first within source flight, then give flights and seeds equal weight.
    exclude={'fold','seed','n','sign_eligible'}
    columns=[c for c in perflight.select_dtypes('number').columns if c not in exclude]
    source=perflight.groupby(keys+['log_id','seed'])[columns].mean().reset_index()
    summary=source.groupby(keys)[columns].mean().reset_index()
    counts=source.groupby(keys).agg(flights=('log_id','nunique'),flight_seed_groups=('seed','size')).reset_index()
    summary=summary.merge(counts,on=keys,validate='one_to_one')
    return perflight,source,summary


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    authority=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json'
    contract=json.loads(authority.read_text())
    paths=[Path(__file__),base.ROOT/'scripts/reference_increment_v16.py',
        base.ROOT/'tests/test_reference_increment_v16.py',
        base.ROOT/'scripts/reference_groupcv_v8.py',base.ROOT/'scripts/reference_condition_diagnostics_v12.py',
        base.ROOT/'src/system_identification/models/reference_gru_unit_gravity.py',
        base.ROOT/'src/system_identification/models/reference_gru_phase_schedule.py',
        base.DATA/'manifest.json',authority]+[p/'delivery_manifest.json' for p in [V8,V12,V15]]
    base.write(OUT/'protocol.json',dict(stage='v16 original-origin 20ms increments',
        source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        scope='407 original train windows, 5 frozen v8 folds x seeds17/23/42 hold models, fit and OOF; no new selection, training, development or sealed/reserved evaluation',
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],
        prediction='Use original observed initial_rates_b; first-step truth/pred increment equals next rate minus this common origin. Replay all50 steps and capture pre-clamp head. Saved model/normalization unchanged.',
        limits='Model raw head clamp +/-6 times scale20 rad/s2 times dt.02 = +/-2.4 rad/s perstep. Report firststep and full50step saturation separately; pointwise clipped-truth oracle lower bound on clamp-compatible increment RMSE, not an attainable predictor.',
        metrics='Per-flight increment RMSE,bias,RMS,std,pred/truth ratios,Pearson,descriptive pred-on-truth slope,5/50/95%quantiles,zero-increment hold gain. Sign denominator abs(truth)>.01rad/s; nearzero predicted increments abs(pred)<=.01 count incorrect and reported separately. All other metrics retain every window.',
        history_spectrum='Only original26 observed history p/q/r; remove sample mean then Hann, one-sided periodogram normalized by N*sum(Hann^2). >8Hz energy RMS/share; frequency spacing50/26=1.923Hz; Nyquist25Hz. Coarse cached-signal covariate, not noise attribution or original IMU spectrum. No future labels used for spectrum or strata.',
        strata='Highfrequency RMS terciles per axis fit on 4 fit flights with equalflight empirical inverseCDF; OOF uses frozen cuts. Existing v12 q/historyfrequency strata shared across axes. No threshold tuning/selection by target error.',
        aggregation='Within flight/model first; average repeated fit folds per sourceflight/seed, then equalflight/seed. Report individual flights and counts. Fit windows are in-sample and OOF is repeatedly explored train-flight holdout, not blind test.',
        decision='Diagnostic only; no control gate or model promotion, no clamp relaxation. Inspect saturation, amplitude and correlation, history highfrequency/error association without claiming causal noise.',
        sealed_test_accessed=False))
    for src in [V8,V12,V15]:verify_manifest(src/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();assert len(w)==407 and w.log_id.nunique()==5
    hf,share=history_spectrum(x['history_state'].numpy()[:,:,3:6])
    dt=x['dt_s'].numpy();np.testing.assert_allclose(dt,.02)
    freq=pd.read_csv(V12/'strata.csv')
    freq=freq[(freq.axis=='q')&(freq.condition=='frequency_mean')][['fold','cohort','array_index','stratum']]
    frames=[];cuts_list=[];checks=[];pins={}
    for fold in range(5):
        folder=V8/f'fold{fold}'
        manifest=json.loads((folder/'manifest.json').read_text())
        fit=np.array(manifest['fit_indices']);query=np.array(manifest['query_indices'])
        assert set(fit).isdisjoint(query) and sorted([*fit,*query])==list(range(407))
        assert set(w.iloc[fit].log_id).isdisjoint(w.iloc[query].log_id)
        cuts=np.stack([weighted_terciles(hf[fit,j],w.iloc[fit].log_id.to_numpy()) for j in range(3)])
        for j,axis in enumerate('pqr'):
            cuts_list.append(dict(fold=fold,axis=axis,low_cut=cuts[j,0],high_cut=cuts[j,1],fit_indices=fit.tolist(),query_indices=query.tolist()))
        dest=OUT/f'fold{fold}';dest.mkdir()
        for seed in SEEDS:
            checkpoint=folder/f'hold_{seed}/final.pt';pins[str(checkpoint.relative_to(base.ROOT))]=base.sha(checkpoint)
            model=model_from(checkpoint);scale=model.derivative_scale[3:].numpy()
            np.testing.assert_array_equal(scale,[20.,20.,20.])
            for cohort,ids in [('fit',fit),('oof',query)]:
                pred,heads=capture(model,x,ids)
                with np.load(folder/f'hold_{seed}/{cohort}_predictions.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ids)
                    replay=float(np.max(abs(pred-z['rates_b'])))
                    assert replay<1e-6
                origin=x['initial_rates_b'][ids].numpy()
                truth=y['rates_b'][ids,0].numpy()-origin
                delta=pred[:,0]-origin
                reconstructed=clipped_increment(heads[:,0,3:],dt[ids,0,None],scale)
                residual=float(np.max(abs(reconstructed-delta)));assert residual<1e-6
                # Check the hook's scaling and clamp over the entire recurrent trajectory.
                step_delta=np.diff(np.concatenate([origin[:,None],pred],axis=1),axis=1)
                full_residual=float(np.max(abs(clipped_increment(heads[:,:,3:],dt[ids,:,None],scale)-step_delta)))
                assert full_residual<1e-6
                np.testing.assert_allclose(delta-truth,pred[:,0]-y['rates_b'][ids,0].numpy(),atol=3e-7)
                np.savez_compressed(dest/f'{cohort}_seed{seed}.npz',array_index=ids,raw_head=heads,
                    truth_delta=truth,pred_delta=delta,initial_rates_b=origin)
                checks.append(dict(fold=fold,seed=seed,cohort=cohort,replay_max_abs=replay,
                    first_increment_reconstruction_max_abs=residual,full_increment_reconstruction_max_abs=full_residual))
                meta=w.iloc[ids][['array_index','log_id']].copy().assign(fold=fold,seed=seed,cohort=cohort)
                meta=meta.merge(freq,on=['fold','cohort','array_index'],validate='one_to_one').rename(columns={'stratum':'frequency_stratum'})
                np.testing.assert_array_equal(meta.array_index,ids)
                for j,axis in enumerate('pqr'):
                    frames.append(meta.assign(axis=axis,truth_delta=truth[:,j],pred_delta=delta[:,j],raw_head=heads[:,0,3+j],
                        limit=6*scale[j]*dt[ids,0],rollout_saturation=np.mean(abs(heads[:,:,3+j])>=6,axis=1),
                        hf_rms=hf[ids,j],hf_share=share[ids,j],hf_stratum=bins(hf[ids,j],cuts[j])))
            print(f'Completed fold{fold} seed{seed}',flush=True)
    frame=pd.concat(frames,ignore_index=True)
    oof=frame[frame.cohort=='oof']
    assert oof.groupby(['seed','axis']).array_index.nunique().eq(407).all()
    assert not oof.duplicated(['seed','axis','array_index']).any()
    frame.to_csv(OUT/'per_window.csv',index=False)
    perflight,source,summary=summarize(frame)
    perflight.to_csv(OUT/'per_flight.csv',index=False)
    source.to_csv(OUT/'source_flight_summary.csv',index=False)
    summary.to_csv(OUT/'summary.csv',index=False)
    base.write(OUT/'history_cuts.json',cuts_list)
    base.write(OUT/'model_pins.json',pins)
    base.write(OUT/'verification.json',dict(replays=checks,original_windows=407,folds=5,seeds=SEEDS,
        no_oof_duplicates=True,clamp_rad_s2=120.,increment_limit_rad_s=2.4,unchanged_model=True,
        development_evaluated=False,sealed_test_accessed=False))
    base.write(OUT/'status.json',dict(status='diagnostics_complete',model_promoted=False))


if __name__=='__main__':main()
