#!/usr/bin/env python3
"""Matched 26/51/101-sample causal-history experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,traceback,time,math
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_september_step2 import prepare,status,write_json
from run_september_fit_diagnostics import metadata,grouped_metrics
from system_identification.data.september_trajectory import file_hash
from system_identification.training.trajectory_main_v1 import MainV1Config,assemble_history_trajectory_windows,fit_history_trajectory_model,predict_history_trajectory_model,fit_main_v1_stats
from system_identification.evaluation.trajectory_rollout_diagnostics import prediction_errors,ERROR_NAMES
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/september_history_lengths_20260911'
REGISTRY=ROOT/'configs/data/trajectory_dataset_registry.yaml'

def matched_inputs():
    manifest,entry,samples,windows,_=prepare(REGISTRY,train_horizon=2)
    coverage=[]
    for part,w in windows.items():
        keep=(w.available_history_s>=2)&(w.start_sample_in_segment>=100)
        windows[part]=w.loc[keep].reset_index(drop=True)
        if set(windows[part].log_id)!=set(w.log_id):raise ValueError('Dropped full log')
        for log,g in w.groupby('log_id'):
            coverage.append({'partition':part,'log_id':log,'original_windows':len(g),'selected_windows':int((windows[part].log_id==log).sum())})
    return manifest,entry,samples,windows,pd.DataFrame(coverage)

def history_prefix(batch,steps):
    if steps not in [26,51,101] or batch.history_state_features.shape[1]!=101:raise ValueError('Unexpected history length')
    return replace(batch,history_state_features=batch.history_state_features[:,-steps:],history_controls=batch.history_controls[:,-steps:],history_mask=batch.history_mask[:,-steps:])

def run():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.ones(1,device='cuda:1');status(OUTPUT,'verifying')
    manifest,entry,samples,windows,coverage=matched_inputs()
    coverage.to_csv(OUTPUT/'coverage.csv',index=False)
    for part,w in windows.items():w.to_parquet(OUTPUT/f'{part}_windows.parquet',index=False)
    train=assemble_history_trajectory_windows(samples['train'],windows['train'],history_steps=101)
    if not train.history_mask.all():raise ValueError('Padded train history')
    stats=fit_main_v1_stats(samples['train'],train)
    np.savez(OUTPUT/'normalization.npz',**vars(stats))
    meta=metadata(samples['train'],windows['train'])
    bins={c:np.quantile(meta[c],[1/3,2/3]).tolist() for c in ['speed_m_s','abs_body_yaw_rate_rad_s']}
    common=dict(use_history=True,use_controls=True,objective_steps=100,hidden_size=64,epochs=80,batch_size=256,learning_rate=3e-4,weight_decay=1e-5,gradient_clip_norm=5,seed=17)
    configs={n:MainV1Config(model_name=f'history_{n}',**common) for n in [26,51,101]}
    sources=[Path(__file__),ROOT/'scripts/run_september_step2.py',ROOT/'scripts/run_september_fit_diagnostics.py',ROOT/'src/system_identification/training/trajectory_main_v1.py',ROOT/'src/system_identification/models/trajectory_main_v1.py',ROOT/'src/system_identification/evaluation/trajectory_rollout_diagnostics.py']
    write_json(OUTPUT/'manifest.json',{'dataset':entry,'dataset_id':manifest['dataset_id'],'split_contract':manifest['split_contract'],'roles':manifest['roles'],'sampling':manifest['sampling'],'configs':{n:asdict(c) for n,c in configs.items()},'history_steps':[26,51,101],'nominal_history_s':[.5,1,2],'normalization':'common train-only 101-sample batch and training transitions','bins_train_only':bins,'train_windows':len(windows['train']),'validation_windows':len(windows['validation']),'source_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in sources},'device':'cuda:1','sealed_test_opened':False,'validation_fitting':False,'interpretation':'single-seed same-update diagnostic; encoder compute differs; historical results use different cohorts'})
    models={};budget=[]
    for n,config in configs.items():
        status(OUTPUT,'training',history_steps=n)
        torch.cuda.synchronize(1);start=time.monotonic()
        model,h=fit_history_trajectory_model(history_prefix(train,n),stats,config,device='cuda:1')
        torch.cuda.synchronize(1)
        budget.append({'history_steps':n,'wall_s':time.monotonic()-start,'optimizer_updates':80*math.ceil(len(windows['train'])/256)})
        pd.DataFrame(budget).to_csv(OUTPUT/'training_budget.csv',index=False)
        h.to_csv(OUTPUT/f'history_{n}_training.csv',index=False)
        torch.save({'state_dict':model.state_dict(),'config':asdict(config),'history_steps':n},OUTPUT/f'history_{n}.pt');models[n]=model
    records=[]
    for part in ['train','validation']:
        batch=train if part=='train' else assemble_history_trajectory_windows(samples[part],windows[part],history_steps=101)
        if not batch.history_mask.all():raise ValueError('Padded evaluation history')
        meta=metadata(samples[part],windows[part])
        for n,model in models.items():
            status(OUTPUT,'evaluating',partition=part,history_steps=n)
            prediction=predict_history_trajectory_model(model,history_prefix(batch,n),use_history=True,batch_size=256,device='cuda:1')
            errors=prediction_errors(prediction,batch.trajectory.truth)
            np.savez_compressed(OUTPUT/f'{part}_history_{n}_errors.npz',**errors,window_ids=batch.trajectory.window_ids.astype(str),log_ids=batch.trajectory.log_ids.astype(str),dt_s=batch.trajectory.dt_s)
            for step in ([50,100] if part=='train' else [50,100,150,250]):
                f=meta.copy();f['partition']=part;f['cohort']=part;f['model']=f'history_{n}';f['global_horizon_s']=step/50
                for name,error in errors.items():f[name]=error[:,step]
                records.append(f)
    frame=pd.concat(records,ignore_index=True)
    for c,edges in bins.items():frame[c+'_bin']=np.searchsorted(edges,frame[c],side='right')
    frame['nonfinite']=~np.isfinite(frame[list(ERROR_NAMES)]).all(axis=1)
    frame['large_error']=(frame.position_error_m>10)|(frame.attitude_error_deg>60)|frame.nonfinite
    frame.to_parquet(OUTPUT/'window_endpoint_metrics.parquet',index=False)
    for group in [None,'speed_m_s_bin','abs_body_yaw_rate_rad_s_bin']:
        per,macro=grouped_metrics(frame,group)
        per.to_csv(OUTPUT/f'per_log_{group or "overall"}.csv',index=False)
        macro.to_csv(OUTPUT/f'macro_{group or "overall"}.csv',index=False)
    write_json(OUTPUT/'summary.json',{'status':'completed','endpoint_rows':len(frame),'nonfinite_rows':int(frame.nonfinite.sum()),'sealed_test_opened':False})
    status(OUTPUT,'completed')

if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=False)
    try:run()
    except BaseException as error:
        status(OUTPUT,'failed',error=repr(error));(OUTPUT/'traceback.txt').write_text(traceback.format_exc());raise
