#!/usr/bin/env python3
"""Frozen-model Step 3 pretraining diagnostics. Reads train/validation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-step3')
import sys,json,hashlib,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_main_v2_free_running import load_simulator,make_initial,rollout,sha
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,_model_call
from system_identification.training.main_v2_objectives import loss_components
from system_identification.evaluation.main_v2_free_running import select_windows,HORIZONS,PHYSICAL_FIELDS,magnitude,describe
from system_identification.evaluation.trajectory import VELOCITY_COLUMNS,BODY_RATE_COLUMNS


def spectrum(series,dt):
    # Offline diagnostic only: resample each separate native-time window onto a
    # 20ms grid before FFT. Never use this interpolation as simulator input.
    t=np.r_[0,np.cumsum(dt)]
    if len(series)==len(dt):t=t[:-1]
    grid=np.arange(t[0],t[-1],.02)
    y=np.stack([np.interp(grid,t,series[:,j]) for j in range(series.shape[1])],axis=-1)
    f,p=welch(y,fs=50.,nperseg=min(128,len(y)),axis=0,detrend='constant')
    return f,p.sum(axis=-1)


def diagnostics(pred,truth,dt,*,experiment,partition,ids,log_ids,out):
    rows=[];spec=[];phase=[]
    true={n:getattr(truth,n) for n in PHYSICAL_FIELDS}
    true['acceleration_n']=np.diff(truth.velocity_n,axis=1)/dt[:,:,None]
    true['angular_acceleration_b']=np.diff(truth.angular_velocity_b,axis=1)/dt[:,:,None]
    pred['acceleration_n']=np.diff(pred['velocity_n'],axis=1)/dt[:,:,None]
    pred['angular_acceleration_b']=np.diff(pred['angular_velocity_b'],axis=1)/dt[:,:,None]
    for horizon,k in HORIZONS.items():
        for key in ('velocity_n','angular_velocity_b','acceleration_n','angular_acceleration_b','gru_hidden'):
            for mode,data in [('pred',pred),('truth',true)]:
                if key not in data:continue
                a=data[key][:,:k] if 'acceleration' in key else data[key][:,:k+1]
                std=magnitude(np.std(a,axis=1));rms=np.sqrt(np.mean(np.sum(a*a,axis=-1),axis=1))
                mean=magnitude(a.mean(axis=1));diff=magnitude(np.diff(a,axis=1)).mean(axis=1)
                rows.append(dict(experiment=experiment,partition=partition,horizon_s=horizon,signal=key,source=mode,
                    temporal_std_median=float(np.median(std)),rms_median=float(np.median(rms)),mean_vector_norm_median=float(np.median(mean)),
                    temporal_delta_norm_mean=float(diff.mean()),norm_at_horizon_median=float(np.median(magnitude(a[:,-1])))))
        # Exact decomposition of phase error into predicted frequency integral
        # versus measured frequency integral and measured phase inconsistency.
        pf=pred['flap_frequency_hz'][:,:k+1];tf=truth.flap_frequency_hz[:,:k+1]
        integral=2*np.pi*np.sum(.5*((pf[:,:-1]-tf[:,:-1])+(pf[:,1:]-tf[:,1:]))*dt[:,:k],axis=1)
        measured_increment=np.sum(np.angle(np.exp(1j*np.diff(truth.relative_phase_rad[:,:k+1],axis=1))),axis=1)
        measured_integral=2*np.pi*np.sum(.5*(tf[:,:-1]+tf[:,1:])*dt[:,:k],axis=1)
        residual=measured_integral-measured_increment
        actual=np.angle(np.exp(1j*(pred['relative_phase_rad'][:,k]-truth.relative_phase_rad[:,k])))
        reconstructed=np.angle(np.exp(1j*(integral+residual)))
        omega_error=magnitude(pred['angular_velocity_b'][:,k]-truth.angular_velocity_b[:,k])
        phase.append(pd.DataFrame(dict(experiment=experiment,partition=partition,window_id=ids,log_id=log_ids,horizon_s=horizon,
            phase_error_rad=actual,frequency_integral_error_rad=integral,measured_phase_frequency_residual_rad=residual,
            reconstruction_error_rad=np.angle(np.exp(1j*(actual-reconstructed))),body_rate_error=omega_error,
            initial_frequency_error=pf[:,0]-tf[:,0],mean_frequency_bias=np.mean(pf-tf,axis=1),
            frequency_delta_rmse=np.sqrt(np.mean((np.diff(pf,axis=1)/dt[:,:k]-np.diff(tf,axis=1)/dt[:,:k])**2,axis=1)))))
    for key in ('velocity_n','angular_velocity_b','acceleration_n','angular_acceleration_b'):
        for mode,data in [('pred',pred),('truth',true)]:
            ps=[]
            for i,a in enumerate(data[key]):
                f,p=spectrum(a,dt[i]);ps.append(p)
            mean=np.mean(ps,axis=0)
            spec.extend(dict(experiment=experiment,partition=partition,signal=key,source=mode,frequency_hz=float(x),power=float(y)) for x,y in zip(f,mean))
    pd.DataFrame(rows).to_csv(out/'dynamics_statistics.csv',index=False)
    pd.DataFrame(spec).to_csv(out/'spectra.csv',index=False)
    pd.concat(phase).to_csv(out/'phase_analysis.csv',index=False)
    return rows,spec


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,default=ROOT/'docs/analysis/results/main_v2_training_objective_ablation/pretraining');pa.add_argument('--device',default='cuda:1');args=pa.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    baseline=ROOT/'docs/analysis/results/main_v2_free_running_5s';s=json.loads((baseline/'summary.json').read_text())
    for n,h in s['source_hashes'].items():
        if sha(ROOT/n)!=h:raise ValueError(f'frozen source changed {n}')
    sim=load_simulator(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',args.device)
    dataset=ROOT/'dataset/trajectory_v1_august_f5_c4';summary={};lossrows=[]
    for split in ('train','validation'):
        print('diagnosing',split,flush=True)
        samples=pd.read_parquet(dataset/f'samples_{split}.parquet')
        windows=pd.read_csv(baseline/'windows.csv') if split=='validation' else select_windows(samples)[0]
        batch=assemble_history_trajectory_windows(samples,windows,history_steps=26)
        predparts=[]
        with torch.inference_mode():
            for start in range(0,len(windows),128):
                idx=np.arange(start,min(start+128,len(windows)))
                c=torch.tensor(batch.trajectory.controls[idx],device=args.device,dtype=torch.float32)
                dt=torch.tensor(batch.trajectory.dt_s[idx],device=args.device,dtype=torch.float32)
                p,d=rollout(sim,make_initial(sim,batch,idx,'warm26',args.device),c,dt);predparts.append(p)
                if split=='train':
                    for k in (10,25,50,100):
                        pp,tt=_model_call(sim.model,batch,idx,use_history=True,rollout_steps=k,device=torch.device(args.device))
                        v=loss_components(pp,tt,k)
                        lossrows.append(dict(steps=k,count=len(idx),**{n:float(x) for n,x in v.items()}))
        pred={k:np.concatenate([p[k] for p in predparts]) for k in predparts[0]}
        part=out/split;part.mkdir()
        diagnostics(pred,batch.trajectory.truth,batch.trajectory.dt_s,experiment='frozen_MainV2',partition=split,
                    ids=windows.window_id,log_ids=windows.log_id,out=part)
        # Local predictions at every eligible timestamp: each call resets from
        # real past history, diagnostic-only; not a free-running trajectory.
        local=[]
        for (log_id,segment),g in samples[samples.valid_core].groupby(['log_id','segment_id']):
            for start in range(25,len(g)-1):local.append(dict(log_id=log_id,segment_id=segment,start_sample_in_segment=start,state_sample_count=2,window_id=f'{log_id}:{segment}:{start}'))
        local=pd.DataFrame(local);b=assemble_history_trajectory_windows(samples,local,history_steps=26)
        lparts=[]
        with torch.inference_mode():
            for start in range(0,len(local),2048):
                idx=np.arange(start,min(start+2048,len(local)))
                st=make_initial(sim,b,idx,'warm26',args.device)
                ns,d=sim.step(st,torch.tensor(b.trajectory.controls[idx,0],device=args.device,dtype=torch.float32),torch.tensor(b.trajectory.dt_s[idx,0],device=args.device,dtype=torch.float32))
                lparts.append({n:getattr(ns,n).cpu().numpy() for n in ('velocity_n','angular_velocity_b')})
        lr=[];acf=[];local_psd=[];lag_stats=[]
        for n in ('velocity_n','angular_velocity_b'):
            nextpred=np.concatenate([p[n] for p in lparts]);t=getattr(b.trajectory.truth,n)
            derivpred=(nextpred-t[:,0])/b.trajectory.dt_s[:,0,None];derivtruth=(t[:,1]-t[:,0])/b.trajectory.dt_s[:,0,None]
            for label,a in [('next_pred',nextpred),('next_truth',t[:,1]),('derivative_pred',derivpred),('derivative_truth',derivtruth)]:
                for j in range(3):lr.append(dict(signal=n,source=label,axis=j,mean=float(a[:,j].mean()),std=float(a[:,j].std()),rms=float(np.sqrt(np.mean(a[:,j]**2)))))
            lr.append(dict(signal=n,source='derivative_error',axis='vector',rms=float(np.sqrt(np.mean(np.sum((derivpred-derivtruth)**2,axis=-1))))))
            for (log,seg),g in local.groupby(['log_id','segment_id']):
                ix=g.index.to_numpy();dt=b.trajectory.dt_s[ix,0]
                for label,a in [('derivative_pred',derivpred[ix]),('derivative_truth',derivtruth[ix]),('state_truth',t[ix,0]),('state_local_pred',nextpred[ix])]:
                    if len(a)<128:continue
                    f,p=spectrum(a,dt);local_psd.extend(dict(log_id=log,segment_id=seg,signal=n,source=label,frequency_hz=x,power=y) for x,y in zip(f,p))
                    for lag in (1,2,3,5,10,25):
                        aa=a-a.mean(0);corr=np.sum(aa[:-lag]*aa[lag:])/np.sqrt(np.sum(aa[:-lag]**2)*np.sum(aa[lag:]**2)+1e-30)
                        acf.append(dict(log_id=log,segment_id=seg,signal=n,source=label,lag_steps=lag,correlation=corr))
                    if label=='state_truth' and n=='angular_velocity_b':
                        for lag in (1,2,3,5):
                            da=a[lag:]-a[:-lag]
                            lag_stats.append(dict(lag_steps=lag,delta_squared_mean=float(np.mean(np.sum(da*da,axis=-1))),count=len(da)))
        pd.DataFrame(lr).to_csv(part/'one_step_statistics.csv',index=False)
        pd.DataFrame(acf).to_csv(part/'autocorrelation.csv',index=False)
        pd.DataFrame(local_psd).to_csv(part/'local_spectra.csv',index=False)
        pd.DataFrame(lag_stats).to_csv(part/'delta_lag_statistics.csv',index=False)
        np.savez_compressed(part/'free_run.npz',**pred,window_ids=windows.window_id.to_numpy(str),dt_s=batch.trajectory.dt_s)
        summary[split]=dict(windows=len(windows),local_origins=len(local),one_step_dt_s=describe(b.trajectory.dt_s))
    pd.DataFrame(lossrows).to_csv(out/'train_loss_components.csv',index=False)
    summary.update(device=args.device,sealed_test_opened=False,training_performed=False,baseline_hashes=s['source_hashes'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary.keys(),flush=True)

if __name__=='__main__':main()
