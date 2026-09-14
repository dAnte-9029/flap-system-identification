#!/usr/bin/env python3
"""Single-factor body-z rate supervision test, fully free-running evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,time,math,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_september_history_lengths import matched_inputs,history_prefix
from run_september_step2 import status,write_json,summarize_curves
from system_identification.data.september_trajectory import file_hash
from system_identification.training.trajectory_main_v1 import MainV1Config,MainV1Stats,assemble_history_trajectory_windows,fit_history_trajectory_model,predict_history_trajectory_model
from system_identification.evaluation.trajectory_rollout_diagnostics import prediction_errors,error_curve
SOURCE=ROOT/'artifacts/september_history_lengths_20260911'
OUTPUT=ROOT/'artifacts/september_body_z_weight_20260911'

def run():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.ones(1,device='cuda:1');status(OUTPUT,'verifying')
    manifest,entry,samples,windows,_=matched_inputs()
    for part,w in windows.items():
        if not w.equals(pd.read_parquet(SOURCE/f'{part}_windows.parquet')):raise ValueError('Window mismatch')
        w.to_parquet(OUTPUT/f'{part}_windows.parquet',index=False)
    prior=json.loads((SOURCE/'manifest.json').read_text())
    training_path='src/system_identification/training/trajectory_main_v1.py'
    for p,h in prior['source_hashes'].items():
        if p!=training_path and file_hash(ROOT/p)!=h:raise ValueError(f'Unexpected source change: {p}')
    stats=MainV1Stats(**dict(np.load(SOURCE/'normalization.npz')))
    original=torch.load(SOURCE/'history_26.pt',weights_only=True,map_location='cpu')
    common={k:v for k,v in original['config'].items() if k not in ['model_name','body_rate_axis_weights']}
    configs=[MainV1Config(model_name='baseline_z1',body_rate_axis_weights=(1.,1.,1.),**common),
             MainV1Config(model_name='candidate_z4',body_rate_axis_weights=(1.,1.,4.),**common)]
    files=[SOURCE/f for f in ['manifest.json','normalization.npz','history_26.pt','train_windows.parquet','validation_windows.parquet','train_history_26_errors.npz','validation_history_26_errors.npz']]
    hashes={str(p):file_hash(p) for p in files}
    code=[Path(__file__),ROOT/training_path,ROOT/'src/system_identification/models/trajectory_main_v1.py',ROOT/'src/system_identification/evaluation/trajectory_rollout_diagnostics.py']
    write_json(OUTPUT/'manifest.json',{'dataset':entry,'source_dataset_manifest':manifest,'configs':[asdict(c) for c in configs],
        'source_hashes':hashes,'source_code_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in code},
        'allowed_source_change':'body_rate_axis_weights default (1,1,1), default path unchanged and checkpoint replay gated',
        'phase_history_architecture_normalization':'unchanged history_26 baseline, shared saved train-only normalization',
        'hypothesis':'stronger body-z instantaneous rate supervision may reduce persistent rate drift; no promised benefit',
        'protocol':'docs/contracts/2026-09-11_september_body_z_weight.md','device':'cuda:1',
        'train_windows':len(windows['train']),'validation_windows':len(windows['validation']),
        'seed':17,'validation_fitting':False,'oracle_inputs':False,'sealed_test_opened':False})
    batch=history_prefix(assemble_history_trajectory_windows(samples['train'],windows['train'],history_steps=101),26)
    if not batch.history_mask.all():raise ValueError('Padded history')
    models={};budgets=[]
    for config in configs:
        status(OUTPUT,'training',model=config.model_name)
        torch.cuda.synchronize(1);start=time.monotonic()
        model,history=fit_history_trajectory_model(batch,stats,config,device='cuda:1')
        torch.cuda.synchronize(1)
        history.to_csv(OUTPUT/f'{config.model_name}_training.csv',index=False)
        torch.save({'state_dict':model.state_dict(),'config':asdict(config),'history_steps':26},OUTPUT/f'{config.model_name}.pt')
        if config.model_name=='baseline_z1':
            exact=all(torch.equal(v,original['state_dict'][k]) for k,v in model.state_dict().items())
            write_json(OUTPUT/'baseline_checkpoint_reproduction.json',{'bitwise_equal':exact})
            if not exact:raise ValueError('Baseline checkpoint did not reproduce; do not train candidate')
        budgets.append({'model':config.model_name,'wall_s':time.monotonic()-start,
                        'optimizer_updates':config.epochs*math.ceil(len(windows['train'])/config.batch_size)})
        pd.DataFrame(budgets).to_csv(OUTPUT/'training_budget.csv',index=False);models[config.model_name]=model
    del batch
    curves=[];endpoints=[];axis_rows=[]
    for part in ['train','validation']:
        batch=history_prefix(assemble_history_trajectory_windows(samples[part],windows[part],history_steps=101),26)
        if not batch.history_mask.all():raise ValueError('Padded evaluation history')
        truth=batch.trajectory.truth;dt=batch.trajectory.dt_s
        integrated=np.concatenate([truth.position_n[:,:1],truth.position_n[:,:1]+np.cumsum(.5*(truth.velocity_n[:,1:]+truth.velocity_n[:,:-1])*dt[:,:,None],axis=1)],axis=1)
        for name,model in models.items():
            status(OUTPUT,'evaluating',partition=part,model=name)
            pred=predict_history_trajectory_model(model,batch,use_history=True,batch_size=256,device='cuda:1')
            errors=prediction_errors(pred,truth)
            if name=='baseline_z1':
                saved=np.load(SOURCE/f'{part}_history_26_errors.npz')
                np.testing.assert_array_equal(saved['window_ids'].astype(str),batch.trajectory.window_ids.astype(str))
                for k,v in errors.items():np.testing.assert_allclose(v,saved[k],rtol=0,atol=0)
            aux=np.linalg.norm(pred.position_n-integrated,axis=-1)
            errors['integrated_position_error_m']=np.where(np.isfinite(aux),aux,np.inf)
            np.savez_compressed(OUTPUT/f'{part}_{name}_errors.npz',**errors,window_ids=batch.trajectory.window_ids.astype(str),log_ids=batch.trajectory.log_ids.astype(str),dt_s=dt)
            np.savez_compressed(OUTPUT/f'{part}_{name}_predictions.npz',**vars(pred),window_ids=batch.trajectory.window_ids.astype(str))
            curve=error_curve(errors,batch.trajectory.log_ids,dt,name,'continuous',0);curve['partition']=part;curves.append(curve)
            e=pred.angular_velocity_b-truth.angular_velocity_b
            ri=np.cumsum(.5*(e[:,1:]+e[:,:-1])*dt[:,:,None],axis=1)
            for step in ([50,100] if part=='train' else [50,100,150,250]):
                f=windows[part][['window_id','log_id','segment_id','start_sample_in_log']].copy()
                f['model']=name;f['partition']=part;f['horizon_s']=step/50
                for k,v in errors.items():f[k]=v[:,step]
                f['nonfinite']=~np.isfinite(f[list(errors)]).all(axis=1);endpoints.append(f)
                for log in np.unique(batch.trajectory.log_ids):
                    m=batch.trajectory.log_ids==log
                    for j,axis in enumerate(['body_x','body_y','body_z']):
                        z=e[m,step,j];i=ri[m,step-1,j]
                        axis_rows.append({'partition':part,'model':name,'log_id':log,'horizon_s':step/50,'axis':axis,'n_windows':int(m.sum()),
                            'rate_rmse_rad_s':float(np.sqrt(np.mean(z*z))),'signed_rate_bias_rad_s':float(z.mean()),
                            'integral_rmse_rad':float(np.sqrt(np.mean(i*i))),'signed_integral_bias_rad':float(i.mean())})
    curves=pd.concat(curves,ignore_index=True);curves.to_csv(OUTPUT/'per_log_error_curves.csv',index=False)
    macros=[]
    for part,g in curves.groupby('partition'):
        m=summarize_curves(g);m['partition']=part;macros.append(m)
    macro=pd.concat(macros,ignore_index=True);macro.to_csv(OUTPUT/'equal_log_error_curves.csv',index=False)
    macro[macro.global_horizon_s.isin([1,2,3,5])].to_csv(OUTPUT/'endpoint_metrics.csv',index=False)
    ep=pd.concat(endpoints,ignore_index=True);ep.to_parquet(OUTPUT/'window_endpoint_metrics.parquet',index=False)
    axes=pd.DataFrame(axis_rows);axes.to_csv(OUTPUT/'per_log_axis_endpoints.csv',index=False)
    cols=['rate_rmse_rad_s','signed_rate_bias_rad_s','integral_rmse_rad','signed_integral_bias_rad']
    axes.groupby(['partition','model','horizon_s','axis'])[cols].mean().reset_index().to_csv(OUTPUT/'equal_log_axis_endpoints.csv',index=False)
    for path,h in hashes.items():
        if file_hash(Path(path))!=h:raise ValueError('Source changed during run')
    write_json(OUTPUT/'summary.json',{'status':'completed','baseline_reproduced':True,'oracle_inputs':False,'sealed_test_opened':False,
        'endpoint_rows':len(ep),'nonfinite_rows':int(ep.nonfinite.sum()),'interpretation':'single-seed coefficient diagnostic; no automatic promotion'})
    status(OUTPUT,'completed')

if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=False)
    try:run()
    except BaseException as e:
        status(OUTPUT,'failed',error=repr(e));(OUTPUT/'traceback.txt').write_text(traceback.format_exc());raise
