#!/usr/bin/env python3
"""Frozen offline oracle diagnostics; outputs are NOT normal forecast scores."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_september_step2 import write_json,status
from run_september_history_lengths import matched_inputs,history_prefix
from system_identification.data.september_trajectory import file_hash
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory import TrajectoryPrediction
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.trajectory_oracle import oracle_rollout,MODES
from system_identification.evaluation.trajectory_rollout_diagnostics import prediction_errors,error_curve
SOURCE=ROOT/'artifacts/september_history_lengths_20260911'


def load_inputs():
    manifest,entry,samples,windows,_=matched_inputs()
    for part,w in windows.items():
        if not w.equals(pd.read_parquet(SOURCE/f'{part}_windows.parquet')):raise ValueError('Window mismatch')
    prior=json.loads((SOURCE/'manifest.json').read_text())
    for path,h in prior['source_hashes'].items():
        if file_hash(ROOT/path)!=h:raise ValueError(f'Source code mismatch: {path}')
    ck=torch.load(SOURCE/'history_26.pt',weights_only=True,map_location='cpu')
    model=CausalHistoryTrajectoryModel(hidden_size=ck['config']['hidden_size'],use_controls=True,**dict(np.load(SOURCE/'normalization.npz')))
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    return manifest,entry,samples,windows,model


def tensor_inputs(batch,start,end,device):
    t=batch.trajectory.truth
    def f(a):return torch.as_tensor(a[start:end],dtype=torch.float32,device=device)
    x=dict(history_state_features=f(batch.history_state_features),history_controls=f(batch.history_controls),
           history_mask=torch.as_tensor(batch.history_mask[start:end],dtype=torch.bool,device=device),
           future_controls=f(batch.trajectory.controls),dt_s=f(batch.trajectory.dt_s))
    for field in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz']:
        x[field]=f(getattr(t,field)[:,0])
    return x


def predict(model,batch,mode):
    device='cuda:1';model.to(device).eval();blocks={}
    with torch.no_grad():
        for start in range(0,len(batch.trajectory.window_ids),256):
            end=min(start+256,len(batch.trajectory.window_ids));x=tensor_inputs(batch,start,end,device)
            extra={}
            if mode in ['oracle_attitude_rotation','oracle_attitude_feedback']:
                extra['oracle_quaternion']=torch.as_tensor(batch.trajectory.truth.quaternion_nb[start:end],dtype=torch.float32,device=device)
            if mode=='oracle_rate':extra['oracle_rate']=torch.as_tensor(batch.trajectory.truth.angular_velocity_b[start:end],dtype=torch.float32,device=device)
            result=oracle_rollout(model,mode=mode,**extra,**x).to_numpy()
            for name,value in vars(result).items():blocks.setdefault(name,[]).append(value)
    model.cpu()
    return TrajectoryPrediction(**{k:np.concatenate(v) for k,v in blocks.items()})


def verify_reference(pred,batch,part,indices=None):
    saved=np.load(SOURCE/f'{part}_history_26_errors.npz')
    ix=np.arange(len(saved['window_ids'])) if indices is None else indices
    np.testing.assert_array_equal(saved['window_ids'][ix].astype(str),batch.trajectory.window_ids.astype(str))
    errors=prediction_errors(pred,batch.trajectory.truth)
    delta={}
    for name,value in errors.items():
        delta[name]=float(np.max(np.abs(value-saved[name][ix])))
        np.testing.assert_allclose(value,saved[name][ix],rtol=1e-6,atol=1e-4)
    return delta


def preflight():
    torch.set_num_threads(4);torch.ones(1,device='cuda:1')
    _,_,samples,windows,model=load_inputs()
    evidence={}
    for part in ['train','validation']:
        selected=windows[part].groupby('log_id',sort=False).head(2)
        indices=selected.index.to_numpy()
        batch=history_prefix(assemble_history_trajectory_windows(samples[part],selected,history_steps=101),26)
        if not batch.history_mask.all():raise ValueError('Padded history')
        pred=predict(model,batch,'free_run');evidence[part]=verify_reference(pred,batch,part,indices)
        model.to('cuda:1').eval();x=tensor_inputs(batch,0,len(selected),'cuda:1')
        with torch.no_grad():
            production=model(**x);diagnostic=oracle_rollout(model,**x)
            for a,b in zip(production,diagnostic):torch.testing.assert_close(a,b,rtol=0,atol=0)
        model.cpu()
    print(json.dumps({'preflight':'passed','baseline_max_error_differences':evidence},indent=2))


def run(output):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.ones(1,device='cuda:1');status(output,'verifying')
    manifest,entry,samples,windows,model=load_inputs()
    sources=[SOURCE/f for f in ['manifest.json','history_26.pt','normalization.npz','train_windows.parquet','validation_windows.parquet','train_history_26_errors.npz','validation_history_26_errors.npz']]
    hashes={str(p):file_hash(p) for p in sources}
    code=[Path(__file__),ROOT/'src/system_identification/evaluation/trajectory_oracle.py',ROOT/'src/system_identification/evaluation/trajectory_rollout_diagnostics.py',ROOT/'src/system_identification/models/trajectory_main_v1.py']
    write_json(output/'manifest.json',{'experiment':'offline_oracle_diagnostic','not_forecast_performance':True,
        'dataset':entry,'source_dataset_manifest':manifest,'modes':list(MODES),'checkpoint':'history_26.pt',
        'source_hashes':hashes,'source_code_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in code},
        'protocol':'docs/contracts/2026-09-11_september_oracle_diagnostics.md','device':'cuda:1',
        'sealed_test_opened':False,'trained':False,'same_windows':True,
        'forced_metrics':{'oracle_attitude_feedback':['attitude'],'oracle_rate':['body_rate']},
        'rate_oracle':'true omega[k] and omega[k+1], predicted angular acceleration discarded; integrate q',
        'rotation_oracle':'only direct substitution is R(q_true[k]); indirect velocity and hidden feedback remain',
        'feedback_oracle':'true q[k] for features and rotation, q[k+1] in recurrent next-state features and state',
        'auxiliary_position_target':'p_true[0] + cumulative trapezoid integral of true NED velocity, diagnostic only'})
    all_curves=[];all_endpoints=[];reproduction={}
    for part in ['train','validation']:
        w=windows[part];w.to_parquet(output/f'{part}_windows.parquet',index=False)
        batch=history_prefix(assemble_history_trajectory_windows(samples[part],w,history_steps=101),26)
        if not batch.history_mask.all():raise ValueError('Padded history')
        truth=batch.trajectory.truth;dt=batch.trajectory.dt_s
        integrated=np.concatenate([truth.position_n[:,:1],truth.position_n[:,:1]+np.cumsum(.5*(truth.velocity_n[:,1:]+truth.velocity_n[:,:-1])*dt[:,:,None],axis=1)],axis=1)
        for mode in MODES:
            status(output,'evaluating',partition=part,mode=mode)
            pred=predict(model,batch,mode)
            if mode=='free_run':
                reproduction[part]=verify_reference(pred,batch,part)
                write_json(output/'baseline_reproduction.json',reproduction)
            errors=prediction_errors(pred,truth)
            auxiliary=np.linalg.norm(pred.position_n-integrated,axis=-1)
            errors['integrated_position_error_m']=np.where(np.isfinite(auxiliary),auxiliary,np.inf)
            np.savez_compressed(output/f'{part}_{mode}_errors.npz',**errors,window_ids=batch.trajectory.window_ids.astype(str),log_ids=batch.trajectory.log_ids.astype(str),dt_s=dt)
            np.savez_compressed(output/f'{part}_{mode}_predictions.npz',**vars(pred),window_ids=batch.trajectory.window_ids.astype(str))
            curve=error_curve(errors,batch.trajectory.log_ids,dt,'history_26',mode,0)
            curve['partition']=part;curve['oracle']=mode!='free_run'
            curve['attitude_forced']=mode=='oracle_attitude_feedback';curve['body_rate_forced']=mode=='oracle_rate'
            # Cross-mode large-error statistic uses position alone, since q may be forced.
            curve=curve.drop(columns=['large_error_fraction'])
            position_fail=[]
            for row in curve.itertuples():
                k=int(round(row.local_horizon_s*50));mask=batch.trajectory.log_ids==row.log_id
                position_fail.append(float(np.mean(errors['position_error_m'][mask,k]>10)))
            curve['position_over_10m_fraction']=position_fail
            for forced,prefix in [(mode=='oracle_attitude_feedback','attitude'),(mode=='oracle_rate','body_rate')]:
                if forced:
                    for col in curve.columns:
                        if col.startswith(prefix) and col!=prefix+'_forced':curve[col]=np.nan
            all_curves.append(curve)
            for step in ([50,100] if part=='train' else [50,100,150,250]):
                f=w[['window_id','log_id','segment_id','start_sample_in_log']].copy()
                f['partition']=part;f['mode']=mode;f['horizon_s']=step/50
                f['actual_horizon_s']=dt[:,:step].sum(axis=1)
                for name,value in errors.items():f[name]=value[:,step]
                f['attitude_forced']=mode=='oracle_attitude_feedback';f['body_rate_forced']=mode=='oracle_rate'
                f['nonfinite']=~np.isfinite(f[list(errors)]).all(axis=1)
                all_endpoints.append(f)
    curves=pd.concat(all_curves,ignore_index=True);curves.to_csv(output/'per_log_error_curves.csv',index=False)
    values=[c for c in curves if 'rmse' in c or c.endswith('_p95') or c.endswith('_fraction')]
    macro=curves.groupby(['partition','mode','local_horizon_s'],sort=False)[values].mean().reset_index()
    macro['attitude_forced']=macro['mode']=='oracle_attitude_feedback';macro['body_rate_forced']=macro['mode']=='oracle_rate'
    macro.to_csv(output/'equal_log_error_curves.csv',index=False)
    macro[macro.local_horizon_s.isin([1,2,3,5])].to_csv(output/'endpoint_metrics.csv',index=False)
    endpoints=pd.concat(all_endpoints,ignore_index=True);endpoints.to_parquet(output/'window_endpoint_metrics.parquet',index=False)
    for path,h in hashes.items():
        if file_hash(Path(path))!=h:raise ValueError('Source changed during run')
    write_json(output/'summary.json',{'status':'completed','not_forecast_performance':True,'trained':False,'sealed_test_opened':False,
        'train_windows':len(windows['train']),'validation_windows':len(windows['validation']),
        'endpoint_rows':len(endpoints),'nonfinite_rows':int(endpoints.nonfinite.sum()),'baseline_reproduced':True})
    status(output,'completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/september_oracle_diagnostics_20260911')
    args=parser.parse_args()
    if args.preflight:preflight()
    else:
        output=args.output_root.resolve();output.mkdir(parents=True,exist_ok=False)
        try:run(output)
        except BaseException as e:
            status(output,'failed',error=repr(e));(output/'traceback.txt').write_text(traceback.format_exc());raise
