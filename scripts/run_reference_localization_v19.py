"""Frozen paired v8/v18 timing, integration and planned-reference localization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,model_from,predict,SEEDS
from reference_error_structure_v9 import decompose
from reference_condition_diagnostics_v12 import weighted_terciles,bins
from reference_localization_v19 import BLOCKS,angle,integrate,reference_activity,rate_tape_attribution
from run_reference_generalization_v7 import verify_manifest

OUT=base.ROOT/'docs/analysis/results/control_windows_v19'
V8=OUT.parent/'control_windows_v8';V18=OUT.parent/'control_windows_v18'


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    authority=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json'
    contract=json.loads(authority.read_text())
    paths=[Path(__file__),base.ROOT/'scripts/reference_localization_v19.py',base.ROOT/'tests/test_reference_localization_v19.py',
        base.ROOT/'scripts/reference_error_structure_v9.py',base.ROOT/'scripts/reference_condition_diagnostics_v12.py',
        base.ROOT/'scripts/reference_groupcv_v8.py',V8/'delivery_manifest.json',V18/'delivery_manifest.json',base.DATA/'manifest.json',authority]
    base.write(OUT/'protocol.json',dict(stage='v19 frozen qgain and attitude/reference localization',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],
        scope='407 originaltrain windows,5folds x3seeds paired v8hold/v18aux,fit+OOF. No retraining, selection, newweights, development or sealed/reserved access.',
        blocks=BLOCKS,q_decomposition='Exact perwindow MSE=bias_squared+std_amplitude_difference_squared+shape_term; timeblocks fixed20-40,60-100,120-200,220-500,520-1000ms,full1s.2pointblock mathematicallyvalid but not frequency/time-delay identification.',
        attitude='Independent float64 numpy midpoint bodyrate right quaternion product using observedinitialquat/rate. Compare frozen quaternions atall50steps; maxangle discrepancy<2e-5rad. Integrate loggedfuture rates only as offline consistency diagnostic, not deployable prediction or errorfloor.',
        attribution='For each paired rate tape, all8 p/q/r replacements integrated without GRU feedback; exact3player Shapley of squared quaternion error over .5/1s prefix. Also4 early(first5endpoints20-100ms)/late(6-50) tape replacements, exact2player allocation. Contributions sum to integratedaux-baseline MSE; signedrad2. This is algebraic tape attribution, not a realizable hybrid or causal network mechanism.',
        reference='Fresh replay bothmodels withall5 plannedrefs held at firstfuturevalue, allfit+OOF. qfactual/held RMSE and outputdifference RMS per fixedblock/flight/seed. Verify v18 OOF held predictions and full1s reference metrics.',
        strata='Plannedreference activity=RMS(diff(reference)/foldfit_reference_scale) over49changes and5channels. Flightbalanced fit-terciles from4fitflights, queryusesfixedcuts. Uses known full plannedtape, no future measuredresponses; descriptive offline grouping, not causal online confidence. Keep all strata/counts incl smallcells.',
        weighting='Within model/sourceflight first, average repeatedfitmodels per sourceflight then equalseed/flight. Perwindow provenance and whole-flight records retained. No outcome-based dropping or posthoc causal/physicaldelay claim.',
        decision='Diagnostic only. Existingv18 gates remainfailed; no promotion, no tuning. Locate failures before choosing next experiment.',sealed_test_accessed=False))
    for src in [V8,V18]:verify_manifest(src/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();assert len(w)==407 and w.log_id.nunique()==5
    held_x={k:v.clone() for k,v in x.items()};held_x['future_reference']=x['future_reference'][:,:1].expand_as(x['future_reference']).clone()
    frames=[];attributions=[];references=[];checks=[];cutrows=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
        assert set(w.iloc[fit].log_id).isdisjoint(w.iloc[query].log_id)
        with np.load(V8/f'fold{fold}/normalization.npz') as z:activity=reference_activity(x['future_reference'].numpy(),z['reference_scale'])
        cuts=weighted_terciles(activity[fit],w.iloc[fit].log_id.to_numpy());strata=bins(activity,cuts)
        cutrows.append(dict(fold=fold,cuts=cuts.tolist(),fit_indices=fit.tolist(),query_indices=query.tolist()))
        dest=OUT/f'fold{fold}';dest.mkdir()
        for seed in SEEDS:
            folders={'baseline':V8/f'fold{fold}/hold_{seed}','aux':V18/f'fold{fold}/seed{seed}'}
            models={arm:model_from(folder/'final.pt') for arm,folder in folders.items()}
            for cohort,ids in [('fit',fit),('oof',query)]:
                meta=w.iloc[ids][['array_index','window_id','log_id','origin_index','start_timestamp_us','end_timestamp_us','mode']].copy()
                meta=meta.assign(fold=fold,seed=seed,cohort=cohort,stratum=strata[ids],reference_activity=activity[ids])
                initialq=x['initial_quaternion'][ids].numpy();initialw=x['initial_rates_b'][ids].numpy();dt=x['dt_s'][ids].numpy()
                tq=y['quaternion'][ids].numpy();tw=y['rates_b'][ids].numpy();preds={}
                truth_integrated=integrate(initialq,initialw,tw,dt);closure=angle(truth_integrated,tq)**2
                for arm,folder in folders.items():
                    with np.load(folder/f'{cohort}_predictions.npz') as z:
                        np.testing.assert_array_equal(z['array_index'],ids);preds[arm]={k:z[k] for k in ['rates_b','quaternion']}
                    pred=preds[arm];qi=integrate(initialq,initialw,pred['rates_b'],dt)
                    diff=float(angle(qi,pred['quaternion']).max());assert diff<2e-5
                    held=predict(models[arm],held_x,ids)['rates_b'];replay=None
                    if cohort=='oof' and arm=='aux':
                        with np.load(folder/'oof_held_reference_predictions.npz') as z:replay=float(np.max(abs(held-z['rates_b'])))
                        assert replay<1e-6
                    np.savez_compressed(dest/f'{cohort}_{arm}_held_seed{seed}.npz',array_index=ids,rates_b=held)
                    checks.append(dict(fold=fold,seed=seed,cohort=cohort,arm=arm,quaternion_max_angle=diff,v18_held_max_abs=replay))
                    p=pred['rates_b'][:,:,1].astype(float);t=tw[:,:,1].astype(float);h=held[:,:,1].astype(float)
                    for block,(start,end) in BLOCKS.items():
                        d=decompose(p[:,start:end,None],t[:,start:end,None])
                        np.testing.assert_allclose(d['total_mse'],d['bias_mse']+d['amplitude_mse']+d['shape_mse'],atol=1e-12)
                        frame=meta.assign(arm=arm,block=block,**{k:v[:,0] for k,v in d.items()});frames.append(frame)
                        references.append(meta.assign(arm=arm,block=block,
                            factual_mse=((p[:,start:end]-t[:,start:end])**2).mean(1),
                            held_mse=((h[:,start:end]-t[:,start:end])**2).mean(1),
                            effect_mse=((p[:,start:end]-h[:,start:end])**2).mean(1)))
                attribution={};values={}
                for kind,names in [('axis',['p','q','r']),('time',['early_20_100ms','late_120_1000ms'])]:
                    attr,v0,v1=rate_tape_attribution(initialq,initialw,preds['baseline']['rates_b'],preds['aux']['rates_b'],tq,dt,kind)
                    attribution[kind]=attr;values[kind]=(v0,v1)
                    for horizon in [25,50]:
                        attributions.append(meta.assign(kind=kind,horizon_s=horizon*.02,
                            baseline_mse=v0[:,:horizon].mean(1),aux_mse=v1[:,:horizon].mean(1),
                            recorded_rate_closure_mse=closure[:,:horizon].mean(1),
                            **{name:attr[:,:horizon,j].mean(1) for j,name in enumerate(names)}))
                np.testing.assert_allclose(values['axis'][0],values['time'][0],atol=1e-12)
                np.testing.assert_allclose(values['axis'][1],values['time'][1],atol=1e-12)
                np.savez_compressed(dest/f'{cohort}_attitude_seed{seed}.npz',array_index=ids,axis_contributions=attribution['axis'],time_contributions=attribution['time'],baseline_squared_angle=values['axis'][0],aux_squared_angle=values['axis'][1],recorded_rate_closure_squared_angle=closure)
            print(f'Completed localization fold{fold} seed{seed}',flush=True)
    pd.concat(frames,ignore_index=True).to_csv(OUT/'q_decomposition_per_window.csv',index=False)
    pd.concat(references,ignore_index=True).to_csv(OUT/'reference_per_window.csv',index=False)
    pd.concat(attributions,ignore_index=True).to_csv(OUT/'attitude_attribution_per_window.csv',index=False)
    base.write(OUT/'reference_cuts.json',cutrows)
    base.write(OUT/'verification.json',dict(integration_and_reference_checks=checks,original_windows=407,no_training=True,no_window_removal=True,sealed_test_accessed=False))
    base.write(OUT/'status.json',dict(status='diagnostics_complete'))


if __name__=='__main__':main()
