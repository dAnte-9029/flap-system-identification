"""OOF error structure and fold-isolated historical-information diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,SEEDS
from reference_error_structure_v9 import decompose,history_features,fit_predict,score_mean_readout
from run_reference_generalization_v7 import verify_manifest

OUT=base.ROOT/'docs/analysis/results/control_windows_v9'
V8=OUT.parent/'control_windows_v8'
ARMS=['zero','constant','residual_state','residual_full','mean_state','mean_full']


def freeze():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((base.DATA/'manifest.json').read_text())
    pins=[V8/'delivery_manifest.json',base.DATA/'manifest.json',base.ROOT/'dataset/control_windows_v2/windows.npz',
        Path(__file__),base.ROOT/'scripts/reference_error_structure_v9.py',base.ROOT/'tests/test_reference_error_structure_v9.py']
    base.write(OUT/'protocol.json',dict(task='OOF angular-rate mean versus waveform error and history-only predictability',
        source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in pins},data_authority=str((base.DATA/'manifest.json').relative_to(base.ROOT)),
        partitions='original407trainingwindows/5flights only; existing v8 OOF and same-model fit predictions; no new developmentvalidation evaluation; sealed/reserved not accessed',
        phase_contract=manifest['phase_contract'],frequency_contract=manifest['frequency_contract'],
        decomposition='perwindow float64 MSE=bias^2+centeredMSE; centeredMSE=(stdpred-stdtrue)^2+2*(stdpred*stdtrue-cov). Last term combines waveform/timing discrepancies, not pure phase error.',
        diagnostic_horizons_steps=[5,10,25,50],learning_horizons_steps=[25,50],seeds=SEEDS,
        decomposition_models='both v8 hold andphase, fit andOOF, no new neural training',
        predictors=ARMS,ridge_alpha=100,features='58state-only or94state+reference+actuator; last/mean/std/lineartrend of observed26samples, pluscurrentmode; no future reference, future measurement, flightID or errors as features',
        learning='for each outerfold andseed use its holdGRU only. Residual labels from that same model on its4fitflights; query5th excluded from GRU, regression, normalization and targets. Never combine otherfold OOF residuals as training labels.',
        residual_limitation='calibration residuals are in-sample for GRU, although regression evaluation is outerflight-honest; transfer may fail due optimism of these labels. This does not prove residuals unlearnable.',
        direct_mean='separate history-only predictor trained on4fitflights truth windowmean; no GRU-generated training targets, distinguishes predictability of mean from residual calibration transfer',
        interpretation='constant rate-readout shift is algebraic diagnosis only, changes trajectory mean; no reintegration or reference-response validation, not a dynamics/control model',
        primary='OOF flight-equal RMSE of normalized windowmean error, equal axespqr and horizons0.5/1s and seeds; normalization from respective4fitflight GRU loss scale',
        followup_gate='for each prespecified learnedarm: >=10percent primary improvement vszero, all3seedpositive, >=4of5flightpositive, noaxis normalizedmeanRMSE>5percent worse. Gate only supports further mean-error research, not control promotion.',
        association='within-flight Spearman, report every fixed historical covariate vs signedbias/centeredMSE; descriptive, no significance or causality claims',
        bootstrap='2000pairedwholeflight resamples seed909, fixedtrainedmodels,5flights; exploratory only',
        validation_status='training flights used in prior development; not blind test',sealed_test_accessed=False))


def read_predictions(path,ids):
    with np.load(path) as z:
        np.testing.assert_array_equal(z['array_index'],ids)
        return z['rates_b'].astype(float)


def main():
    freeze();verify_manifest(V8/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();truth=y['rates_b'].numpy().astype(float);logs=w.log_id.to_numpy();features={kind:history_features(x,kind) for kind in ['state','full']}
    decomposed=[];scores=[];predictions=[];provenance=[];max_identity=0.;reload_checks=[]
    base.write(OUT/'status.json',dict(status='running'))
    for fold in range(5):
        source=V8/f'fold{fold}';info=json.loads((source/'manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
        assert not set(logs[fit])&set(logs[query]);assert (w.iloc[np.r_[fit,query]].partition=='train').all()
        with np.load(source/'normalization.npz') as z:rate_scale=z['rate_scale'].astype(float)
        dest=OUT/f'fold{fold}';dest.mkdir()
        for seed in SEEDS:
            hold={cohort:read_predictions(source/f'hold_{seed}'/f'{cohort}_predictions.npz',ids) for cohort,ids in [('fit',fit),('oof',query)]}
            for arm in ['hold','phase']:
                for cohort,ids in [('fit',fit),('oof',query)]:
                    pred=hold[cohort] if arm=='hold' else read_predictions(source/f'phase_{seed}'/f'{cohort}_predictions.npz',ids)
                    for k in [5,10,25,50]:
                        d=decompose(pred[:,:k],truth[ids,:k])
                        maximum=max(float(np.max(abs(d['total_mse']-d['bias_mse']-d['centered_mse']))),float(np.max(abs(d['centered_mse']-d['amplitude_mse']-d['shape_mse']))))
                        max_identity=max(max_identity,maximum)
                        for ax,axis in enumerate(['p','q','r']):
                            df=w.iloc[ids][['array_index','window_id','log_id']].copy();df['fold']=fold;df['seed']=seed;df['arm']=arm;df['cohort']=cohort;df['horizon_s']=k*.02;df['axis']=axis
                            for name,value in d.items():df[name]=value[:,ax]
                            decomposed.append(df)
            for k in [25,50]:
                fit_truth_mean=truth[fit,:k].mean(1);fit_bias=(hold['fit'][:,:k]-truth[fit,:k]).mean(1)
                for arm in ARMS:
                    kind='state' if arm.endswith('state') else 'full'
                    target=fit_truth_mean if arm.startswith('mean_') else fit_bias
                    estimates={};params=None
                    for cohort,ids in [('fit',fit),('oof',query)]:
                        if arm=='zero':estimated=np.zeros((len(ids),3))
                        else:
                            value,params=fit_predict(features[kind][fit],target,features[kind][ids],logs[fit],'constant' if arm=='constant' else 'ridge')
                            estimated=hold[cohort][:,:k].mean(1)-value if arm.startswith('mean_') else value
                        estimates[cohort]=estimated
                        scored=score_mean_readout(hold[cohort][:,:k],truth[ids,:k],estimated)
                        for ax,axis in enumerate(['p','q','r']):
                            df=w.iloc[ids][['array_index','log_id']].copy();df['fold']=fold;df['seed']=seed;df['predictor']=arm;df['cohort']=cohort;df['horizon_s']=k*.02;df['axis']=axis
                            for metric,values in scored.items():df[metric]=values[:,ax]
                            df['normalized_mean_error_squared']=scored['mean_error_squared'][:,ax]/rate_scale[ax]**2
                            scores.append(df)
                    if params is not None:
                        path=dest/f'{arm}_{seed}_k{k}.npz';np.savez_compressed(path,fit_indices=fit,query_indices=query,feature_kind=kind,target_kind='truth_mean' if arm.startswith('mean_') else 'in_sample_GRU_residual',**params)
                        with np.load(path) as saved:
                            reload=np.tile(saved['target_mean'],(len(query),1)) if arm=='constant' else (features[kind][query]-saved['mean'])/saved['scale']@saved['coefficient']+saved['target_mean']
                        if arm.startswith('mean_'):reload=hold['oof'][:,:k].mean(1)-reload
                        error=float(np.max(abs(reload-estimates['oof'])));assert error<1e-10;reload_checks.append(error)
                    for row,idx in enumerate(query):
                        predictions.append(dict(fold=fold,seed=seed,predictor=arm,horizon_s=k*.02,array_index=int(idx),bias_p=float(estimates['oof'][row,0]),bias_q=float(estimates['oof'][row,1]),bias_r=float(estimates['oof'][row,2])))
            provenance.append(dict(fold=fold,seed=seed,base_model_fit_indices=fit.tolist(),calibrator_fit_indices=fit.tolist(),query_indices=query.tolist(),source_run_protocol_sha256=base.sha(source/f'hold_{seed}/run_protocol.json')))
        print(f'Completed fold{fold}',flush=True)
    assert max_identity<1e-10
    df=pd.concat(decomposed);df.to_csv(OUT/'decomposition_per_window.csv',index=False)
    metrics=['total_mse','signed_bias','bias_mse','centered_mse','amplitude_mse','shape_mse','predicted_std','truth_std']
    flight=df.groupby(['arm','cohort','seed','log_id','horizon_s','axis'])[metrics].mean().reset_index();flight.to_csv(OUT/'decomposition_per_flight.csv',index=False)
    summary=flight.groupby(['arm','cohort','horizon_s','axis'])[metrics].mean().reset_index()
    for component in ['bias','centered','amplitude','shape']:summary[component+'_share']=summary[component+'_mse']/summary.total_mse
    summary.to_csv(OUT/'decomposition_summary.csv',index=False)
    sc=pd.concat(scores);sc.to_csv(OUT/'predictor_per_window.csv',index=False)
    sf=sc.groupby(['predictor','cohort','seed','log_id','horizon_s','axis'])[['mean_error_squared','normalized_mean_error_squared','waveform_error_squared']].mean().pow(.5).reset_index()
    sf=sf.rename(columns={c:c.replace('squared','rmse') for c in ['mean_error_squared','normalized_mean_error_squared','waveform_error_squared']});sf.to_csv(OUT/'predictor_per_flight.csv',index=False)
    # Repeated fit flights belong to different frozen models, not new independent observations.
    sf.groupby(['predictor','cohort','seed','log_id','horizon_s','axis']).mean(numeric_only=True).groupby(['predictor','cohort','horizon_s','axis']).mean().to_csv(OUT/'predictor_summary.csv')
    pd.DataFrame(predictions).to_csv(OUT/'oof_bias_predictions.csv',index=False)
    association(x,w,df)
    base.write(OUT/'verification.json',dict(decomposition_max_identity_error=max_identity,predictor_reload_max_abs=max(reload_checks),predictor_artifacts=len(reload_checks),
        fold_provenance=provenance,feature_source='observedhistoryonly',no_other_fold_GRU_residuals_used=True,sealed_test_accessed=False))
    base.write(OUT/'status.json',dict(status='complete',decomposition_rows=len(df),predictor_rows=len(sc)))


def association(x,w,df):
    hs=x['history_state'].numpy();ref=x['history_reference'].numpy();act=x['history_actuator'].numpy()
    cov=w[['array_index','log_id']].copy();cov['frequency']=hs[:,-1,11];cov['speed']=np.linalg.norm(hs[:,-1,:3],axis=1)
    cov['history_rate_rms']=np.sqrt(np.mean(hs[:,:,3:6]**2,axis=(1,2)))
    for j,axis in enumerate(['p','q','r']):cov['past_mean_'+axis]=hs[:,:,3+j].mean(1)
    cov['roll_reference']=ref[:,-1,0];cov['pitch_reference']=ref[:,-1,1];cov['actuator_rms']=np.sqrt(np.mean(act**2,axis=(1,2)))
    cov.to_csv(OUT/'historical_covariates.csv',index=False)
    errors=df[(df.arm=='hold')&(df.cohort=='oof')].groupby(['array_index','log_id','axis','horizon_s'])[['signed_bias','centered_mse']].mean().reset_index().merge(cov,on=['array_index','log_id'],validate='many_to_one')
    rows=[]
    for (log,axis,h),g in errors.groupby(['log_id','axis','horizon_s']):
        for feature in cov.columns[2:]:
            for target in ['signed_bias','centered_mse']:
                corr=g[[feature,target]].corr(method='spearman').iloc[0,1]
                rows.append(dict(log_id=log,axis=axis,horizon_s=h,feature=feature,target=target,windows=len(g),spearman=corr))
    a=pd.DataFrame(rows);a.to_csv(OUT/'within_flight_associations.csv',index=False)
    a.groupby(['axis','horizon_s','feature','target']).spearman.agg(['median','min','max','count']).to_csv(OUT/'association_summary.csv')

if __name__=='__main__':main()
