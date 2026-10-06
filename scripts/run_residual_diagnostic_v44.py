"""Frozen all-window fit/OOF and conditional/coupled residual diagnosis."""
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_attitude_v42 import OUT as V43,PREP as V42,DATA,read
from evaluate_attitude_v43 import V35,V40,SEEDS
from verify_attitude_v43 import independent_squares
from residual_diagnostic_v44 import rate_parts,kinematics
from coupled_replay_v40 import VARIANTS
from audit_controller_support_v26 import verify_parent

OUT=V43.parent/'control_windows_v44'
PARENTS=[DATA,V35,V40,V42,V43]
METRICS=['p','q','r','velocity','attitude','roll','pitch','tilt']


def iter_sources():
    for fold in range(5):
        info=read(DATA/f'fold{fold}/manifest.json')
        for seed in SEEDS:
            for cohort in ['fit','oof']:
                ids=info['fit_indices' if cohort=='fit' else 'query_indices']
                yield fold,seed,cohort,'conditional',ids,[V35/f'fold{fold}/joint_increment/seed{seed}/{cohort}_predictions.npz',V43/f'fold{fold}/seed{seed}/{cohort}_predictions.npz']
            for variant,_,_ in VARIANTS:
                yield fold,seed,'oof',variant,info['query_indices'],[V40/f'fold{fold}/{variant}_v35_seed{seed}.npz',V43/f'coupled/fold{fold}/{variant}_seed{seed}.npz']


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for p in PARENTS:verify_parent(p)
    OUT.mkdir(exist_ok=False)
    names=['scripts/residual_diagnostic_v44.py','scripts/run_residual_diagnostic_v44.py','tests/test_residual_diagnostic_v44.py']
    base.write(OUT/'protocol.json',dict(status='frozen_before_diagnostic',
        code={n:base.sha(base.ROOT/n) for n in names},parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in PARENTS},
        data='All407 originaltrain parents; pairedv35/v43 per5fold3seed. Fit predictions count the same source window under4different trainingfolds, not new independent data.',
        scope='No training, weight selection, outcome-selected window removal, controller changes or new validation/sealed/reserved access.',
        decomposition='For every axis and fixed full50/first25/last25 samples: mean(error^2)=mean(error)^2+mean((error-mean(error))^2). Window-specific mean, not global signed mean.',
        curves='All50 endpoint times for p/q/r/velocity/attitude/roll/pitch/tilt. Group each sourceflight within each modelfold/seed, then equal sourceflight, modelfold andseed. Mode-specific cells retain counts, no balanced-mode extrapolation.',
        kinematics='All8baseline/candidate p/q/r tape combinations on common initialq/r anddt. Endpoints0/7 must exactly reproduce native quaternion. Three-axis Shapley on full1s roll/pitch/tilt MSE sums to candidate-baseline; symmetric accounting of path interactions, not causal attribution or model/control rerollout.',
        oracles='Also remove each window future rate-error mean and integrate fulltrue rates; future measurement-dependent diagnostic only, not a correction or physical noise floor.',
        coupling='For same OOF window compare conditional MSE change to each coupled change; difference-in-differences describes the changed input/state feedback path, not isolated controller causality.',
        units='Rate components rad^2/s^2, anglecomponents rad^2. RMSE derived only after flight averaging, Shapley operates on MSE not RMSE.',
        promotion=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    base.write(OUT/'dataset_manifest.json',dict(authority=str((V43/'dataset_manifest.json').relative_to(base.ROOT)),authority_sha256=base.sha(V43/'dataset_manifest.json'),contract=read(V43/'dataset_manifest.json')))
    with np.load(DATA/'windows.npz') as z:raw={k:z[k] for k in z.files}
    windows=pd.read_csv(DATA/'windows.csv');window_fold={}
    for fold in range(5):
        for i in read(DATA/f'fold{fold}/manifest.json')['query_indices']:window_fold[i]=fold
    rows=[];curves=[];attribution=[];oracle=[];actionrows=[];count=0
    for fold,seed,cohort,variant,ids,paths in iter_sources():
        preds=[]
        for path in paths:
            with np.load(path) as z:
                np.testing.assert_array_equal(z['array_index'],ids);preds.append({k:z[k] for k in z.files if k!='array_index'})
        truth={k[6:]:v[ids].astype(float) for k,v in raw.items() if k.startswith('label_')}
        errors=np.stack([p['rates_b'].astype(float)-truth['rates_b'] for p in preds],1)
        squared=np.stack([np.stack([independent_squares(p,truth)[m] for m in METRICS],-1) for p in preds],1)
        kin=kinematics(raw['input_initial_quaternion'][ids],raw['input_initial_rates_b'][ids],raw['input_dt_s'][ids],*preds,
                       raw['label_quaternion'][ids],raw['label_rates_b'][ids])
        dest=OUT/f'fold{fold}';dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/f'{cohort}_{variant}_seed{seed}.npz',array_index=ids,rate_error=errors,state_squared=squared,**kin)
        common=dict(fold=fold,seed=seed,cohort=cohort,variant=variant)
        for arm,label in enumerate(['baseline','candidate']):
            for period,sl in [('full',slice(None)),('first',slice(0,25)),('last',slice(25,50))]:
                parts=rate_parts(errors[:,arm,sl])
                for axis,name in enumerate('pqr'):
                    for n,i in enumerate(ids):
                        rows.append(dict(**common,arm=label,array_index=i,source_fold=window_fold[i],mode=int(windows.iloc[i]['mode']),period=period,metric=name,
                                         **{k:v[n,axis] for k,v in parts.items()}))
            for source_fold in sorted(set(window_fold[i] for i in ids)):
                ix=np.asarray([window_fold[i]==source_fold for i in ids]);modes=windows.iloc[np.asarray(ids)[ix]]['mode'].unique()
                for mode in ['all',*map(str,sorted(modes))]:
                    take=ix if mode=='all' else ix&(windows.iloc[ids]['mode'].to_numpy()==int(mode))
                    for j,metric in enumerate(METRICS):
                        mse=squared[take,arm,:,j].mean(0)
                        for k,value in enumerate(mse):curves.append(dict(**common,arm=label,source_fold=source_fold,mode=mode,metric=metric,step=k+1,time_s=(k+1)*.02,windows=int(take.sum()),mse=value))
        for n,i in enumerate(ids):
            meta=dict(**common,array_index=i,source_fold=window_fold[i],mode=int(windows.iloc[i]['mode']))
            for j,metric in enumerate(['roll','pitch','tilt']):
                attribution.append(dict(**meta,metric=metric,baseline_mse=kin['factorial_mse'][n,0,j],candidate_mse=kin['factorial_mse'][n,7,j],
                    delta_mse=kin['factorial_mse'][n,7,j]-kin['factorial_mse'][n,0,j],p=kin['shapley_mse'][n,0,j],q=kin['shapley_mse'][n,1,j],r=kin['shapley_mse'][n,2,j]))
                for arm,label in enumerate(['baseline','candidate']):oracle.append(dict(**meta,arm=label,metric=metric,native_mse=kin['factorial_mse'][n,7*arm,j],debiased_mse=kin['debiased_mse'][n,arm,j],true_rate_mse=kin['true_rate_mse'][n,j]))
        if variant!='conditional':
            for arm,label in enumerate(['baseline','candidate']):
                error=preds[arm]['commands']-raw['input_future_command'][ids]
                parts=rate_parts(error)
                for j in range(4):
                    for n,i in enumerate(ids):actionrows.append(dict(**common,arm=label,array_index=i,source_fold=window_fold[i],mode=int(windows.iloc[i]['mode']),axis=j,**{k:v[n,j] for k,v in parts.items()}))
        count+=len(ids);base.write(OUT/'status.json',dict(status='running',paired_trajectories=count))
        if seed==42:print(f'fold{fold} {cohort} {variant} complete',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'rate_parts_per_window.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'state_curves_per_flight.csv',index=False)
    pd.DataFrame(attribution).to_csv(OUT/'kinematic_attribution_per_window.csv',index=False)
    pd.DataFrame(oracle).to_csv(OUT/'kinematic_oracles_per_window.csv',index=False)
    pd.DataFrame(actionrows).to_csv(OUT/'action_parts_per_window.csv',index=False)
    for parent in PARENTS:verify_parent(parent)
    for path,digest in read(OUT/'protocol.json')['code'].items():assert base.sha(base.ROOT/path)==digest
    assert count==407*3*(4+1+4)
    base.write(OUT/'verification.json',dict(status='passed',paired_trajectories=count,native_quaternions_exact=count*2,all8_rate_combinations=True,
        all_bias_variance_identities=True,all_shapley_sum_identities=True,parent_hashes_unchanged=True,training=False,model_promoted=False))
    base.write(OUT/'status.json',dict(status='diagnostic_complete_summary_pending'))


if __name__=='__main__':main()
