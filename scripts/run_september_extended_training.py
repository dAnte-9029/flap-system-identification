#!/usr/bin/env python3
"""120-epoch fixed full-2s experiment; evaluate frozen 40/80/120 checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,traceback
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_september_step2 import prepare,status,write_json
from run_september_fit_diagnostics import metadata,grouped_metrics
from system_identification.data.september_trajectory import file_hash
from system_identification.training.trajectory_main_v1 import MainV1Config,MainV1Stats,assemble_history_trajectory_windows,fit_history_trajectory_model,predict_history_trajectory_model
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.evaluation.trajectory_rollout_diagnostics import prediction_errors,ERROR_NAMES
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/september_multihorizon_20260911'
DIAG=ROOT/'artifacts/september_fit_diagnostics_20260911'
OUTPUT=ROOT/'artifacts/september_extended_training_20260911'

def run():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.ones(1,device='cuda:1')
    status(OUTPUT,'verifying')
    manifest,entry,samples,windows,_=prepare(ROOT/'configs/data/trajectory_dataset_registry.yaml',train_horizon=2)
    for part in ['train','validation']:
        if not windows[part].equals(pd.read_parquet(SOURCE/f'{part}_selected_windows.parquet')):raise ValueError('Window mismatch')
    prior=json.loads((SOURCE/'manifest.json').read_text())
    training_path='src/system_identification/training/trajectory_main_v1.py'
    for p,h in prior['source_hashes'].items():
        if p!=training_path and file_hash(ROOT/p)!=h:raise ValueError(f'Source changed: {p}')
    original=torch.load(SOURCE/'models/full_2s.pt',map_location='cpu',weights_only=True)
    config=replace(MainV1Config(**original['config']),epochs=120)
    stats=MainV1Stats(**dict(np.load(SOURCE/'normalization.npz')))
    diag=json.loads((DIAG/'manifest.json').read_text())
    for p,h in diag['source_artifact_hashes'].items():
        if file_hash(Path(p))!=h:raise ValueError(f'Prior artifact changed: {p}')
    long=pd.read_parquet(DIAG/'matched_train_5s_windows.parquet')
    origins=['log_id','segment_id','start_sample_in_log']
    if not pd.MultiIndex.from_frame(long[origins]).isin(pd.MultiIndex.from_frame(windows['train'][origins])).all():raise ValueError('Unmatched origins')
    write_json(OUTPUT/'manifest.json',{'dataset':entry,'dataset_id':manifest['dataset_id'],'split_contract':manifest['split_contract'],'roles':manifest['roles'],'sampling':manifest['sampling'],'config':asdict(config),'device':'cuda:1','sealed_test_opened':False,'validation_fitting':False,'restart':'deterministic seed-17 replay; original checkpoint lacks optimizer state','checkpoint_epochs':[40,80,120],'source_artifact_hashes':diag['source_artifact_hashes'],'training_source_hash':file_hash(ROOT/training_path),'runner_hash':file_hash(Path(__file__)),'train_windows':len(windows['train']),'train_5s_windows':len(long),'validation_windows':len(windows['validation']),'bins_train_only':diag['bins_train_only']})
    def checkpoint(epoch,model,optimizer,generator,history):
        if epoch not in [40,80,120]:return
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==40:
            exact=all(torch.equal(state[k],v) for k,v in original['state_dict'].items())
            write_json(OUTPUT/'epoch40_reproduction.json',{'bitwise_equal':exact})
            if not exact:raise ValueError('Epoch 40 does not reproduce original; stop unmatched continuation')
        torch.save({'state_dict':state,'config':asdict(config),'epoch':epoch,'optimizer':optimizer.state_dict(),'shuffle_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},OUTPUT/f'epoch_{epoch}.pt')
        history.to_csv(OUTPUT/'training_history.csv',index=False)
    batch=assemble_history_trajectory_windows(samples['train'],windows['train'],history_steps=26)
    if not batch.history_mask.all():raise ValueError('Padded history')
    status(OUTPUT,'training_120_epochs')
    fit_history_trajectory_model(batch,stats,config,device='cuda:1',epoch_callback=checkpoint)
    del batch
    records=[]
    for part,cohort,selected in [('train','all_train_2s',windows['train']),('train','matched_train_5s',long),('validation','validation_5s',windows['validation'])]:
        selected.to_parquet(OUTPUT/f'{cohort}_windows.parquet',index=False)
        batch=assemble_history_trajectory_windows(samples[part],selected,history_steps=26)
        if not batch.history_mask.all():raise ValueError('Padded history')
        meta=metadata(samples[part],selected)
        for epoch in [40,80,120]:
            status(OUTPUT,'evaluating',epoch=epoch,cohort=cohort)
            saved=torch.load(OUTPUT/f'epoch_{epoch}.pt',map_location='cpu',weights_only=True)
            model=CausalHistoryTrajectoryModel(hidden_size=config.hidden_size,use_controls=True,**vars(stats))
            model.load_state_dict(saved['state_dict'],strict=True)
            prediction=predict_history_trajectory_model(model,batch,use_history=True,batch_size=256,device='cuda:1')
            errors=prediction_errors(prediction,batch.trajectory.truth)
            np.savez_compressed(OUTPUT/f'{cohort}_epoch{epoch}_errors.npz',**errors,window_ids=batch.trajectory.window_ids.astype(str),dt_s=batch.trajectory.dt_s)
            for step in ([50,100] if cohort=='all_train_2s' else [50,100,150,250]):
                f=meta.copy();f['partition']=part;f['cohort']=cohort;f['model']=f'epoch_{epoch}';f['global_horizon_s']=step/50
                for name,error in errors.items():f[name]=error[:,step]
                records.append(f)
        del batch
    frame=pd.concat(records,ignore_index=True)
    for c,edges in diag['bins_train_only'].items():frame[c+'_bin']=np.searchsorted(edges,frame[c],side='right')
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
