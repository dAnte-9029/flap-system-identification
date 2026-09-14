#!/usr/bin/env python3
"""Frozen checkpoint train/validation rollout and operating-condition diagnostics."""
import json
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_september_step2 import prepare, status, write_json
from system_identification.data.september_trajectory import file_hash
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows, predict_history_trajectory_model
from system_identification.evaluation.trajectory_rollout_diagnostics import ERROR_NAMES, prediction_errors

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/september_multihorizon_20260911'
OUTPUT = ROOT / 'artifacts/september_fit_diagnostics_20260911'
REGISTRY = ROOT / 'configs/data/trajectory_dataset_registry.yaml'


def metadata(samples, windows):
    keys = pd.MultiIndex.from_frame(windows[['log_id', 'start_sample_in_log']].rename(columns={'start_sample_in_log':'sample_in_log'}))
    s = samples.set_index(['log_id', 'sample_in_log']).loc[keys]
    result = windows[['window_id', 'log_id', 'segment_id', 'start_sample_in_log']].reset_index(drop=True).copy()
    result['speed_m_s'] = np.linalg.norm(s[['velocity_ned_m_s_x','velocity_ned_m_s_y','velocity_ned_m_s_z']].to_numpy(), axis=1)
    result['abs_body_yaw_rate_rad_s'] = np.abs(s.angular_velocity_body_rad_s_z.to_numpy())
    return result


def grouped_metrics(frame, group):
    keys = ['partition', 'cohort', 'model', 'global_horizon_s', 'log_id'] + ([group] if group else [])
    values = list(ERROR_NAMES)
    per = frame.groupby(keys, observed=True)[values].agg(lambda x: float(np.sqrt(np.mean(x.to_numpy() ** 2)))).reset_index()
    counts = frame.groupby(keys, observed=True).size().rename('n_windows').reset_index()
    per = per.merge(counts, on=keys, validate='one_to_one')
    macro_keys = [k for k in keys if k != 'log_id']
    macro = per.groupby(macro_keys, observed=True)[values].mean().reset_index()
    counts = per.groupby(macro_keys, observed=True).agg(n_logs=('log_id','nunique'), n_windows=('n_windows','sum')).reset_index()
    return per, macro.merge(counts,on=macro_keys,validate='one_to_one')


def run():
    torch.set_num_threads(4)
    torch.ones(1,device='cuda:1')
    status(OUTPUT,'verifying')
    source_manifest = json.loads((SOURCE/'manifest.json').read_text())
    for path, expected in source_manifest['source_hashes'].items():
        if file_hash(ROOT/path) != expected: raise ValueError(f'Source changed: {path}')
    manifest,entry,samples,windows,_ = prepare(REGISTRY,train_horizon=2)
    saved = pd.read_parquet(SOURCE/'train_selected_windows.parquet')
    if not windows['train'].equals(saved): raise ValueError('Training windows differ')
    if not windows['validation'].equals(pd.read_parquet(SOURCE/'validation_selected_windows.parquet')):
        raise ValueError('Validation windows differ')
    _,_,_,long_windows,_ = prepare(REGISTRY,train_horizon=5)
    origin_keys = ['log_id','segment_id','start_sample_in_log']
    short_keys = pd.MultiIndex.from_frame(saved[origin_keys])
    long = long_windows['train']
    long = long.loc[pd.MultiIndex.from_frame(long[origin_keys]).isin(short_keys)].reset_index(drop=True)
    if long.empty: raise ValueError('No matched training origins with five-second coverage')
    train_meta = metadata(samples['train'],saved)
    bins = {c: np.quantile(train_meta[c],[1/3,2/3]).tolist() for c in ['speed_m_s','abs_body_yaw_rate_rad_s']}
    artifacts = [SOURCE/'manifest.json',SOURCE/'normalization.npz',SOURCE/'window_endpoint_metrics.parquet',SOURCE/'train_selected_windows.parquet',SOURCE/'validation_selected_windows.parquet'] + list((SOURCE/'models').glob('*.pt'))
    write_json(OUTPUT/'manifest.json',{'source_run':str(SOURCE),'dataset':entry,'dataset_id':manifest['dataset_id'],
        'split_contract':manifest['split_contract'],'sampling':manifest['sampling'],'roles':manifest['roles'],
        'source_artifact_hashes':{str(p):file_hash(p) for p in artifacts},'script_hash':file_hash(Path(__file__)),
        'bins_train_only':bins,'sealed_test_opened':False,'refit':False,'device':'cuda:1',
        'train_all_2s':len(saved),'train_matched_5s':len(long),'validation_5s':len(windows['validation']),
        'turn_proxy':'absolute body z angular rate at origin, not flight-path curvature',
        'interpretation':'Train 1/2s assesses fitting; 3/5s extrapolation only on matched eligible origins. Cross-day gap mixes distribution shift and generalization.'})
    records=[]
    stats=dict(np.load(SOURCE/'normalization.npz'))
    for cohort, selected in [('all_train_2s',saved),('matched_train_5s',long)]:
        selected.to_parquet(OUTPUT/f'{cohort}_windows.parquet',index=False)
        batch=assemble_history_trajectory_windows(samples['train'],selected,history_steps=26)
        if not batch.history_mask.all(): raise ValueError('Padded history')
        meta=metadata(samples['train'],selected)
        for name in ['matched_1s','full_2s','joint_1s_2s']:
            status(OUTPUT,'evaluating_train',model=name,cohort=cohort)
            checkpoint=torch.load(SOURCE/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
            config=checkpoint['config']
            model=CausalHistoryTrajectoryModel(hidden_size=config['hidden_size'],use_controls=config['use_controls'],**stats)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            prediction=predict_history_trajectory_model(model,batch,use_history=True,batch_size=256,device='cuda:1')
            errors=prediction_errors(prediction,batch.trajectory.truth)
            np.savez_compressed(OUTPUT/f'{cohort}_{name}_errors.npz',**errors,window_ids=batch.trajectory.window_ids.astype(str),log_ids=batch.trajectory.log_ids.astype(str),dt_s=batch.trajectory.dt_s)
            for step in ([50,100] if cohort=='all_train_2s' else [50,100,150,250]):
                f=meta.copy();f['partition']='train';f['cohort']=cohort;f['model']=name;f['global_horizon_s']=step/50
                for metric,error in errors.items(): f[metric]=error[:,step]
                records.append(f)
    val=pd.read_parquet(SOURCE/'window_endpoint_metrics.parquet')
    meta=metadata(samples['validation'],windows['validation'])
    val=val.merge(meta,on=['window_id','log_id'],validate='many_to_one')
    val['partition']='validation';val['cohort']='validation_5s';records.append(val)
    frame=pd.concat(records,ignore_index=True)
    for column,edges in bins.items():
        frame[column+'_bin']=np.searchsorted(edges,frame[column].to_numpy(),side='right')
    frame['nonfinite']=~np.isfinite(frame[list(ERROR_NAMES)]).all(axis=1)
    frame['large_error']=(frame.position_error_m>10)|(frame.attitude_error_deg>60)|frame.nonfinite
    frame.to_parquet(OUTPUT/'window_endpoint_metrics.parquet',index=False)
    for group in [None,'speed_m_s_bin','abs_body_yaw_rate_rad_s_bin']:
        per,macro=grouped_metrics(frame,group)
        per.to_csv(OUTPUT/f'per_log_{group or "overall"}.csv',index=False)
        macro.to_csv(OUTPUT/f'macro_{group or "overall"}.csv',index=False)
    write_json(OUTPUT/'summary.json',{'status':'completed','endpoint_rows':len(frame),'nonfinite_rows':int(frame.nonfinite.sum()),'sealed_test_opened':False,'refit':False})
    status(OUTPUT,'completed')


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=False)
    try: run()
    except BaseException as error:
        status(OUTPUT,'failed',error=repr(error))
        (OUTPUT/'traceback.txt').write_text(traceback.format_exc())
        raise
