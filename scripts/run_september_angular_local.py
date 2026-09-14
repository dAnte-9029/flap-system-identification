#!/usr/bin/env python3
"""Local angular predictability versus persistence and uninterrupted rollout."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-angular-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/flap-angular-cache')
import sys,json,argparse,traceback,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_september_history_lengths import matched_inputs
from run_september_oracle_diagnostics import tensor_inputs
from run_september_step2 import status,write_json
from system_identification.data.september_trajectory import file_hash
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.trajectory_rollout_diagnostics import shifted_local_windows
from system_identification.evaluation.angular_local import capture_rollout,angular_metrics
SOURCE=ROOT/'artifacts/september_history_lengths_20260911'
FREE=ROOT/'artifacts/september_body_z_weight_20260911'


def load_model():
    ck=torch.load(SOURCE/'history_26.pt',map_location='cpu',weights_only=True)
    stats=dict(np.load(SOURCE/'normalization.npz'))
    model=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True,**stats)
    model.load_state_dict(ck['state_dict'],strict=True);return model.eval()


def evaluate(model,batch,device):
    model.to(device).eval();rates=[];heads=[]
    for start in range(0,len(batch.trajectory.window_ids),256):
        end=min(start+256,len(batch.trajectory.window_ids))
        result,raw=capture_rollout(model,tensor_inputs(batch,start,end,device))
        rates.append(result.angular_velocity_b.cpu().numpy().astype(float));heads.append(raw.cpu().numpy().astype(float))
    model.cpu();return np.concatenate(rates),np.concatenate(heads)


def run(output,device):
    torch.set_num_threads(4);status(output,'verifying')
    manifest,entry,samples,windows,_=matched_inputs();model=load_model()
    old=json.loads((SOURCE/'manifest.json').read_text());new=json.loads((FREE/'manifest.json').read_text())
    # Model/data sources must match; training code intentionally gained axis weights.
    for p,h in old['source_hashes'].items():
        if p!='src/system_identification/training/trajectory_main_v1.py' and file_hash(ROOT/p)!=h:raise ValueError(f'Source changed: {p}')
    for p,h in new['source_hashes'].items():
        if file_hash(Path(p))!=h:raise ValueError(f'Artifact changed: {p}')
    for part in ['train','validation']:
        if not windows[part].equals(pd.read_parquet(SOURCE/f'{part}_windows.parquet')):raise ValueError('Window mismatch')
    artifacts=[SOURCE/'history_26.pt',SOURCE/'normalization.npz']+[FREE/f'{part}_baseline_z1_predictions.npz' for part in ['train','validation']]
    hashes={str(p):file_hash(p) for p in artifacts}
    code=[Path(__file__),ROOT/'src/system_identification/evaluation/angular_local.py',ROOT/'src/system_identification/models/trajectory_main_v1.py']
    write_json(output/'manifest.json',{'dataset':entry,'source_dataset_manifest':manifest,'source_hashes':hashes,
        'code_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in code},'device':device,'trained':False,'sealed_test_opened':False,
        'offset_steps':{'train':[0,50],'validation':[0,50,100,150,200]},'local_endpoint_steps':[1,5,10,25],
        'local_history_samples':26,'future_inputs':'controls only after local origin; all local states free-running',
        'interpretation':'reinitialized local forecasts use real history at each selected origin; not one uninterrupted 5s prediction',
        'protocol':'docs/contracts/2026-09-12_september_angular_local.md'})
    bins=old['bins_train_only'];metric_rows=[];clip_rows=[];all_endpoints=[];repro={}
    for part,offsets in [('train',[0,50]),('validation',[0,50,100,150,200])]:
        saved=np.load(FREE/f'{part}_baseline_z1_predictions.npz')
        np.testing.assert_array_equal(saved['window_ids'],windows[part].window_id.to_numpy().astype(str))
        full_batch=assemble_history_trajectory_windows(samples[part],windows[part],history_steps=26)
        full_rates,full_raw=evaluate(model,full_batch,device)
        np.testing.assert_allclose(full_rates,saved['angular_velocity_b'],rtol=1e-6,atol=1e-5)
        np.savez_compressed(output/f'{part}_parent_derivatives.npz',unclipped_normalized_derivative=full_raw,
                            window_ids=windows[part].window_id.to_numpy().astype(str))
        for log in np.unique(full_batch.trajectory.log_ids):
            mask=full_batch.trajectory.log_ids==log
            for j,axis in enumerate('xyz'):
                v=full_raw[mask,:,j+3]
                clip_rows.append({'partition':part,'log_id':log,'offset_s':0,'axis':axis,
                    'state':'parent_free_run','n_values':int(v.size),'saturation_fraction':float(np.mean(abs(v)>=6)),
                    'first_step_saturation_fraction':float(np.mean(abs(v[:,0])>=6)),
                    'abs_raw_p99':float(np.quantile(abs(v),.99)),'abs_raw_max':float(np.max(abs(v)))})
        del full_batch,full_rates,full_raw
        for offset in offsets:
            status(output,'evaluating',partition=part,offset_s=offset/50)
            selected=shifted_local_windows(samples[part],windows[part],offset,length_steps=25)
            batch=assemble_history_trajectory_windows(samples[part],selected,history_steps=26)
            if not batch.history_mask.all():raise ValueError('Incomplete causal history')
            predicted,raw=evaluate(model,batch,device);true=batch.trajectory.truth.angular_velocity_b
            persistent=np.repeat(true[:,:1],26,axis=1)
            full=saved['angular_velocity_b'][:,offset:offset+26]
            if offset==0:
                difference=float(np.max(abs(predicted-full)));repro[part]=difference
                np.testing.assert_allclose(predicted,full,rtol=1e-6,atol=1e-5)
                write_json(output/'zero_offset_reproduction.json',repro)
            speed=np.linalg.norm(batch.trajectory.truth.velocity_n[:,0],axis=1)
            zrate=np.abs(true[:,0,2]);groups={'all':np.repeat('all',len(true)),
                'speed_bin':np.searchsorted(bins['speed_m_s'],speed).astype(str),
                'abs_body_z_rate_bin':np.searchsorted(bins['abs_body_yaw_rate_rad_s'],zrate).astype(str)}
            np.savez_compressed(output/f'{part}_offset{offset}_rates.npz',local_predicted_rate=predicted,true_rate=true,
                full_predicted_rate=full,unclipped_normalized_derivative=raw,window_ids=selected.window_id.to_numpy().astype(str),
                parent_window_ids=selected.parent_window_id.to_numpy().astype(str),log_ids=selected.log_id.to_numpy().astype(str),dt_s=batch.trajectory.dt_s)
            logs=batch.trajectory.log_ids
            for log in np.unique(logs):
                mask=logs==log
                for j,axis in enumerate('xyz'):
                    v=raw[mask,:,j+3]
                    clip_rows.append({'partition':part,'log_id':log,'offset_s':offset/50,'axis':axis,
                        'state':'local_predicted_rollout','n_values':int(v.size),'saturation_fraction':float(np.mean(abs(v)>=6)),
                        'first_step_saturation_fraction':float(np.mean(abs(v[:,0])>=6)),
                        'abs_raw_p99':float(np.quantile(abs(v),.99)),'abs_raw_max':float(np.max(abs(v)))})
            for k in [1,5,10,25]:
                for name,values in [('local_model',predicted),('hold_rate',persistent),('parent_free_run',full)]:
                    error=values[:,k]-true[:,k]
                    f=selected[['window_id','parent_window_id','log_id','segment_id','start_sample_in_log']].copy()
                    f['partition']=part;f['model']=name;f['offset_s']=offset/50;f['local_horizon_s']=k/50
                    f['actual_local_horizon_s']=batch.trajectory.dt_s[:,:k].sum(axis=1)
                    for j,axis in enumerate('xyz'):f[f'{axis}_signed_error_rad_s']=error[:,j]
                    all_endpoints.append(f)
                    for grouping,labels in groups.items():
                        for log in np.unique(logs):
                            for label in np.unique(labels[logs==log]):
                                mask=(logs==log)&(labels==label)
                                metric_rows.append({'partition':part,'model':name,'offset_s':offset/50,'local_horizon_s':k/50,
                                    'grouping':grouping,'group':label,'log_id':log,'n_windows':int(mask.sum()),**angular_metrics(error[mask])})
    per=pd.DataFrame(metric_rows);per.to_csv(output/'per_log_metrics.csv',index=False)
    keys=['partition','model','offset_s','local_horizon_s','grouping','group'];metrics=list(angular_metrics(np.zeros((1,3))))
    macro=per.groupby(keys)[metrics].agg(lambda x:float(np.mean(x.to_numpy()))).reset_index()
    counts=per.groupby(keys).agg(n_logs=('log_id','nunique'),n_windows=('n_windows','sum')).reset_index()
    macro.merge(counts,on=keys,validate='one_to_one').to_csv(output/'equal_log_metrics.csv',index=False)
    pd.DataFrame(clip_rows).to_csv(output/'saturation_per_log.csv',index=False)
    ep=pd.concat(all_endpoints,ignore_index=True);ep.to_parquet(output/'window_endpoint_errors.parquet',index=False)
    for p,h in hashes.items():
        if file_hash(Path(p))!=h:raise ValueError('Source changed during run')
    write_json(output/'summary.json',{'status':'completed','trained':False,'sealed_test_opened':False,
        'parent_train_windows':len(windows['train']),'parent_validation_windows':len(windows['validation']),
        'endpoint_rows':len(ep),'nonfinite_rows':int((~np.isfinite(ep[[f'{a}_signed_error_rad_s' for a in 'xyz']]).all(axis=1)).sum()),
        'note':'overlapping local forecasts are not independent samples; no automatic model promotion'})
    status(output,'completed')

def preflight(device):
    torch.set_num_threads(4);torch.ones(1,device=device)
    _,_,samples,windows,_=matched_inputs();model=load_model();evidence={}
    for part in ['train','validation']:
        parents=windows[part].groupby('log_id',sort=False).head(1)
        saved=np.load(FREE/f'{part}_baseline_z1_predictions.npz')
        local=shifted_local_windows(samples[part],parents,0,length_steps=25)
        batch=assemble_history_trajectory_windows(samples[part],local,history_steps=26)
        pred,raw=evaluate(model,batch,device)
        expected=saved['angular_velocity_b'][parents.index.to_numpy(),:26]
        np.testing.assert_allclose(pred,expected,rtol=1e-6,atol=1e-5)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite captured output')
        evidence[part]={'windows':len(parents),'maximum_rate_difference':float(np.max(abs(pred-expected)))}
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/september_angular_local_20260912')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--launch',action='store_true')
    a=parser.parse_args();out=a.output_root.resolve()
    if a.preflight:
        print(json.dumps(preflight(a.device),indent=2))
    elif a.launch:
        log=out.with_suffix('.run.log');launch=out.with_suffix('.launch.json')
        for p in [out,log,launch]:
            if p.exists():raise SystemExit(f'Refusing duplicate launch: {p}')
        evidence=preflight(a.device)
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--device',a.device,'--output-root',str(out)]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with log.open('x') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        info={'pid':process.pid,'command':cmd,'started_at':datetime.now(timezone.utc).isoformat(),'monitor':False,
              'preflight':evidence,'device':a.device,'log':str(log)}
        launch.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
    else:
        out.mkdir(parents=True,exist_ok=False)
        try:run(out,a.device)
        except BaseException as e:
            status(out,'failed',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
