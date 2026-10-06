"""Frozen channel counterfactual diagnostics, input support and q-bias conditions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,model_from,predict,SEEDS
from reference_channels_v20 import CHANNELS,MASKS,masked_reference,features,support,quantiles
from reference_increment_v16 import safe_corr
from run_reference_generalization_v7 import verify_manifest

OUT=base.ROOT/'docs/analysis/results/control_windows_v20'
V8=OUT.parent/'control_windows_v8';V18=OUT.parent/'control_windows_v18';V19=OUT.parent/'control_windows_v19'
BLOCKS={'early100ms':(0,5),'middle120_500ms':(5,25),'late520_1000ms':(25,50),'full1s':(0,50)}


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    authority=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json';contract=json.loads(authority.read_text())
    paths=[Path(__file__),base.ROOT/'scripts/reference_channels_v20.py',base.ROOT/'tests/test_reference_channels_v20.py',base.ROOT/'scripts/reference_groupcv_v8.py',
        base.ROOT/'scripts/reference_condition_diagnostics_v12.py',base.DATA/'manifest.json',authority]+[p/'delivery_manifest.json' for p in [V8,V18,V19]]
    base.write(OUT/'protocol.json',dict(stage='v20 channel masks and input support',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],
        scope='Same407 originaltrain parents,5folds x3seeds paired frozenv8/v18,fit+OOF; no training/selection, development or sealed/reserved.',
        masks=MASKS,channels=CHANNELS,blocks=BLOCKS,
        interventions='Factual/allheld,5 only-channel-changing,5 hold-one-channel. Always firstfuturevalue forheld; observedhistories unchanged. Deduplicate identical fullreference tapes. Synthetic reference combinations are diagnostic only, no real counterfactual truth.',
        metrics='qresponse RMS vs allheld and vs factual; factual-label meanerror and RMSE only descriptive under artificialinputs, not predictive utility certification. Sum(only_j-held) versus factual-held residual measures nonadditivity, no additive channel attribution assumed.',
        features='20 plannedreference summaries:initial,range,stepRMS,netchange each5channels;12 history features: v12 qcovariates8 plus qmean,pmean,pstd,rstd. No future measuredstates, commands or predictionerrors.',
        support='Fit-flight-equal mean/std floor1e-4; nearestfit RMS standardized featuredistance for reference20/history12/joint32. Threshold=flightweighted95percentile of fit distances to nearest OTHER-flight fitwindow. Scaling fittedonall4fitflights, therefore descriptive supportnot formal conformal or independent probability. OOF distances exclude entirequeryflight. Same originalfit calibration for artificialmask support. Featurewise fitmin/max andweighted1/99percent bounds also retained.',
        bias='OOF withinflight Spearman of all32 predeclaredinput features vs factual full1s signedqmeanerror and its square, perarm/seed. Compare v18-v8 bias_square and signedbias. Report allfeatures incl undefinedconstant correlations; exploratoryposthoc associations not causal and no outcome-based windowremoval.',
        strata='Reuse v19 plannedreference activitystrata exactly fororiginalwindows; masked cases keeporiginal labels, no outcome-based regrouping.',
        decision='Keep existingfailed model gates. No coefficientsearch, new controller or data exclusion.',sealed_test_accessed=False))
    for src in [V8,V18,V19]:verify_manifest(src/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();f=features(x);f.to_csv(OUT/'input_features.csv',index=False)
    allx={};allf={}
    for name,mask in MASKS.items():
        xx=dict(x);xx['future_reference']=masked_reference(x['future_reference'],mask);allx[name]=xx;allf[name]=features(xx)
    oldstrata=pd.read_csv(V19/'reference_per_window.csv');oldstrata=oldstrata[(oldstrata.arm=='baseline')&(oldstrata.seed==17)&(oldstrata.block=='full1s')][['fold','cohort','array_index','stratum']]
    rows=[];supportrows=[];bounds=[];correlations=[];checks=[];calibrations=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices']);logs=w.iloc[fit].log_id.to_numpy()
        assert set(logs).isdisjoint(w.iloc[query].log_id)
        dest=OUT/f'fold{fold}';dest.mkdir()
        for feature in f:
            low,high=quantiles(f.iloc[fit][feature].to_numpy(),logs,[.01,.99]);lo=f.iloc[fit][feature].min();hi=f.iloc[fit][feature].max()
            vals=f.iloc[query][feature].to_numpy();tol=1e-8*(1+max(abs(lo),abs(hi)))
            for idx,value in zip(query,vals):bounds.append(dict(fold=fold,array_index=idx,log_id=w.iloc[idx].log_id,feature=feature,value=value,fit_min=lo,fit_max=hi,fit_p01=low,fit_p99=high,outside_minmax=bool(value<lo-tol or value>hi+tol),outside_01_99=bool(value<low-tol or value>high+tol)))
        for group,columns in [('references',list(f.columns[:20])),('history',list(f.columns[20:])),('joint',list(f.columns))]:
            for name in MASKS:
                audit=support(f.iloc[fit][columns],allf[name].iloc[query][columns],logs)
                for pos,idx in enumerate(query):supportrows.append(dict(fold=fold,array_index=idx,log_id=w.iloc[idx].log_id,mask=name,group=group,distance=audit['distance'][pos],threshold=audit['threshold'],outside=bool(audit['distance'][pos]>audit['threshold']),nearest_fit_array_index=int(fit[audit['nearest_fit_position'][pos]])))
                if name=='factual':calibrations.append(dict(fold=fold,group=group,columns=columns,mean=audit['mean'].tolist(),scale=audit['scale'].tolist(),threshold=audit['threshold'],fit_indices=fit.tolist(),query_indices=query.tolist(),fit_otherflight_distances=audit['calibration'].tolist()))
        for seed in SEEDS:
            for arm,folder in [('baseline',V8/f'fold{fold}/hold_{seed}'),('aux',V18/f'fold{fold}/seed{seed}')]:
                model=model_from(folder/'final.pt')
                for cohort,ids in [('fit',fit),('oof',query)]:
                    with np.load(folder/f'{cohort}_predictions.npz') as z:np.testing.assert_array_equal(z['array_index'],ids);factual=z['rates_b'][:,:,1]
                    with np.load(V19/f'fold{fold}/{cohort}_{arm}_held_seed{seed}.npz') as z:held=z['rates_b'][:,:,1]
                    predictions={'factual':factual,'held':held};aliases={}
                    for name in MASKS:
                        if name in predictions:continue
                        matching=next((k for k in predictions if torch.equal(allx[name]['future_reference'][ids],allx[k]['future_reference'][ids])),None)
                        if matching is not None:predictions[name]=predictions[matching];aliases[name]=matching
                        else:predictions[name]=predict(model,allx[name],ids)['rates_b'][:,:,1]
                        np.testing.assert_allclose(predictions[name][:,0],factual[:,0],atol=1e-6,rtol=0)
                    np.savez_compressed(dest/f'{cohort}_{arm}_seed{seed}.npz',array_index=ids,**predictions)
                    checks.append(dict(fold=fold,seed=seed,arm=arm,cohort=cohort,aliases=aliases,all_first_steps_equal=True))
                    meta=w.iloc[ids][['array_index','window_id','log_id','origin_index','start_timestamp_us','end_timestamp_us','mode']].copy().assign(fold=fold,seed=seed,arm=arm,cohort=cohort)
                    meta=meta.merge(oldstrata,on=['fold','cohort','array_index'],validate='one_to_one');np.testing.assert_array_equal(meta.array_index,ids)
                    truth=y['rates_b'][ids,:,1].numpy().astype(float);fac=factual.astype(float);h=held.astype(float)
                    isolated=sum(predictions['only_'+c].astype(float)-h for c in CHANNELS)
                    for name,pred in predictions.items():
                        p=pred.astype(float)
                        for block,(a,b) in BLOCKS.items():
                            error=p[:,a:b]-truth[:,a:b]
                            rows.append(meta.assign(mask=name,block=block,signed_bias=error.mean(1),bias_mse=error.mean(1)**2,mse=np.mean(error**2,1),
                                versus_held_mse=np.mean((p[:,a:b]-h[:,a:b])**2,1),versus_factual_mse=np.mean((p[:,a:b]-fac[:,a:b])**2,1),
                                nonadditive_mse=np.mean(((fac-h-isolated)[:,a:b])**2,1)))
                    if cohort=='oof':
                        e=(fac-truth).mean(1)
                        for feature in f:
                            ranked=f.iloc[ids][feature].rank().to_numpy()
                            correlations.append(dict(fold=fold,seed=seed,arm=arm,log_id=w.iloc[ids[0]].log_id,feature=feature,n=len(ids),signed_bias_spearman=safe_corr(ranked,pd.Series(e).rank()),bias_square_spearman=safe_corr(ranked,pd.Series(e**2).rank())))
            print(f'Completed channel diagnostics fold{fold} seed{seed}',flush=True)
    pd.concat(rows,ignore_index=True).to_csv(OUT/'response_per_window.csv',index=False)
    pd.DataFrame(supportrows).to_csv(OUT/'support_per_window.csv',index=False)
    pd.DataFrame(bounds).to_csv(OUT/'feature_bounds.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'bias_conditions.csv',index=False)
    base.write(OUT/'support_calibration.json',calibrations)
    base.write(OUT/'verification.json',dict(mask_checks=checks,original_windows=407,no_training=True,no_window_removal=True,sealed_test_accessed=False))
    base.write(OUT/'status.json',dict(status='diagnostics_complete'))


if __name__=='__main__':main()
