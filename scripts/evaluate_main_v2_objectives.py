#!/usr/bin/env python3
"""Shared frozen 722-origin benchmark for Step 3 checkpoints; no training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-step3')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from run_main_v2_free_running import load_simulator,make_initial,rollout,sha,context
from diagnose_main_v2_objectives import diagnostics,spectrum
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import HORIZONS,PHYSICAL_FIELDS,magnitude,endpoint_errors,aggregate,stability
from system_identification.models.trajectory_main_v1 import _rotation_body_to_ned


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/main_v2_training_objective_ablation');pa.add_argument('--results',type=Path,default=ROOT/'docs/analysis/results/main_v2_training_objective_ablation');pa.add_argument('--device',default='cuda:1');pa.add_argument('--experiment');args=pa.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    oldroot=ROOT/'docs/analysis/results/main_v2_free_running_5s';old=json.loads((oldroot/'summary.json').read_text())
    windows=pd.read_csv(oldroot/'windows.csv');samples=pd.read_parquet(ROOT/'dataset/trajectory_v1_august_f5_c4/samples_validation.parquet')
    b=assemble_history_trajectory_windows(samples,windows,history_steps=26);dt=b.trajectory.dt_s;truth=b.trajectory.truth
    np.testing.assert_equal(len(windows),722)
    envelopes=json.loads((oldroot/'observed_envelope.json').read_text())['train']
    protocol=json.loads((args.results/'protocol.json').read_text())
    assert sha(oldroot/'windows.csv')==protocol['frozen_validation_windows_sha256']
    init=samples.set_index(['log_id','segment_id','sample_in_segment']).loc[[(r.log_id,r.segment_id,r.start_sample_in_segment) for r in windows.itertuples()]].reset_index()
    for k,v in context(init).items():windows[k+'_bin']=np.searchsorted(old['bin_edges_train_tertiles'][k],v,side='right')
    trspec=pd.read_csv(args.results/'pretraining/train/local_spectra.csv')
    trspec=trspec[(trspec.source=='state_truth')&(trspec.signal=='angular_velocity_b')].groupby('frequency_hz').power.mean()
    cutoff=float(np.interp(.95,np.cumsum(trspec)/trspec.sum(),trspec.index))
    names=[c['name'] for c in protocol['experiments'] if args.experiment is None or c['name']==args.experiment]
    for name in names:
        ckpt=args.artifacts/name/'model.pt';folder=args.results/name
        if folder.exists() and (folder/'summary.json').exists():continue
        if folder.exists() and any(folder.iterdir()):raise FileExistsError(folder)
        folder.mkdir(exist_ok=True)
        sim=load_simulator(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',args.device)
        state=torch.load(ckpt,map_location='cpu',weights_only=False)['state_dict'];sim.model.load_state_dict(state,strict=True)
        chunks=[];diags=[]
        print('EVALUATE',name,flush=True)
        with torch.inference_mode():
            for start in range(0,722,128):
                idx=np.arange(start,min(start+128,722))
                p,d=rollout(sim,make_initial(sim,b,idx,'warm26',args.device),
                    torch.tensor(b.trajectory.controls[idx],device=args.device,dtype=torch.float32),torch.tensor(dt[idx],device=args.device,dtype=torch.float32))
                chunks.append(p);diags.append(d)
        pred={k:np.concatenate([p[k] for p in chunks]) for k in chunks[0]};diag={k:np.concatenate([d[k] for d in diags]) for k in diags[0]}
        np.savez_compressed(args.artifacts/name/'validation_rollout.npz',**pred,**diag,dt_s=dt,window_ids=windows.window_id.to_numpy(str))
        _,flags=stability(pred,diag,dt,envelopes)
        rows=[];variation=[];dynamic=[]
        ta=np.diff(truth.angular_velocity_b,axis=1)/dt[:,:,None];pa=np.diff(pred['angular_velocity_b'],axis=1)/dt[:,:,None]
        tv=np.diff(truth.velocity_n,axis=1)/dt[:,:,None];pv=np.diff(pred['velocity_n'],axis=1)/dt[:,:,None]
        for h,k in HORIZONS.items():
            frame=windows.copy();frame['experiment']=name;frame['horizon_s']=h
            errs=endpoint_errors(pred,truth,k)
            for n,v in errs.items():frame[n]=v
            frame['failed']=flags['failed'][:,:k].any(1);frame['numerical_failed']=flags['numerical'][:,:k].any(1)
            frame['support_failed']=flags['envelope'][:,:k].any(1);frame['clipping_failed']=flags['clipping'][:,:k].any(1)
            for n in ('frequency_clipped','derivative_clipped','drive_clipped','residual_clipped'):frame[n+'_events']=diag[n][:,:k].sum(1)
            for n,key in [('omega','angular_velocity_b'),('velocity','velocity_n'),('frequency','flap_frequency_hz')]:
                a=pred[key][:,:k+1].std(1);z=getattr(truth,key)[:,:k+1].std(1)
                if a.ndim==2:a=magnitude(a);z=magnitude(z)
                frame[n+'_variation_ratio']=a/np.maximum(z,1e-8)
                variation.append(dict(experiment=name,horizon_s=h,signal=n,median=float(np.median(a/np.maximum(z,1e-8))),p10=float(np.quantile(a/np.maximum(z,1e-8),.1)),p90=float(np.quantile(a/np.maximum(z,1e-8),.9))))
            frame['angular_acceleration_rmse']=np.sqrt(np.mean(np.sum((pa[:,:k]-ta[:,:k])**2,axis=-1),axis=1))
            frame['linear_acceleration_rmse']=np.sqrt(np.mean(np.sum((pv[:,:k]-tv[:,:k])**2,axis=-1),axis=1))
            rows.append(frame)
        a=magnitude(pred['angular_velocity_b'][:,-51:].std(1));z=magnitude(truth.angular_velocity_b[:,-51:].std(1))
        variation.append(dict(experiment=name,horizon_s=5.,signal='omega_last1s',median=float(np.median(a/z)),p10=float(np.quantile(a/z,.1)),p90=float(np.quantile(a/z,.9))))
        for n,p,t in [('omega',pred['angular_velocity_b'],truth.angular_velocity_b),('angular_acceleration',pa,ta),('velocity',pred['velocity_n'],truth.velocity_n),('linear_acceleration',pv,tv)]:
            dynamic.append(dict(experiment=name,signal=n,one_step_rmse=float(np.sqrt(np.mean(np.sum((p[:,0 if 'acceleration' in n else 1]-t[:,0 if 'acceleration' in n else 1])**2,axis=-1)))),
                full_rmse=float(np.sqrt(np.mean(np.sum((p-t)**2,axis=-1)))),pred_std=float(magnitude(p.reshape(-1,3).std(0))),truth_std=float(magnitude(t.reshape(-1,3).std(0)))))
        rows=pd.concat(rows);metrics=list(errs)
        rows.to_csv(folder/'per_rollout.csv',index=False)
        aggregate(rows,['experiment','horizon_s'],metrics).to_csv(folder/'per_horizon.csv',index=False)
        aggregate(rows,['experiment','log_id','horizon_s'],metrics).to_csv(folder/'per_flight.csv',index=False)
        pd.DataFrame(variation).to_csv(folder/'variation.csv',index=False);pd.DataFrame(dynamic).to_csv(folder/'derivatives.csv',index=False)
        regimes=[]
        for key in old['bin_edges_train_tertiles']:
            r=aggregate(rows,['experiment',key+'_bin','horizon_s'],metrics).rename(columns={key+'_bin':'bin'});r['variable']=key;regimes.append(r)
        pd.concat(regimes).to_csv(folder/'regimes.csv',index=False)
        diagnostics(pred.copy(),truth,dt,experiment=name,partition='validation',ids=windows.window_id,log_ids=windows.log_id,out=folder)
        ps=pd.read_csv(folder/'spectra.csv');freqs=sorted(ps.frequency_hz.unique());sp=ps[ps.signal=='angular_velocity_b'].pivot(index='frequency_hz',columns='source',values='power')
        hfratio=float(sp.loc[sp.index>cutoff,'pred'].sum()/sp.loc[sp.index>cutoff,'truth'].sum())
        mismatch=float((sp.pred-sp.truth).abs().sum()/sp.truth.sum())
        # Exact algebraic rotation/input decomposition, not an oracle rollout.
        rp=_rotation_body_to_ned(torch.tensor(pred['quaternion_nb'][:,:-1])).numpy()
        rt=_rotation_body_to_ned(torch.tensor(truth.quaternion_nb[:,:-1],dtype=torch.float32)).numpy()
        rot=np.einsum('btij,btj->bti',rp-rt,diag['acceleration_b'])
        apbodytrue=np.einsum('btji,btj->bti',rt,tv)
        body=np.einsum('btij,btj->bti',rt,diag['acceleration_b']-apbodytrue)
        chain=[]
        for h,k in HORIZONS.items():chain.append(dict(experiment=name,horizon_s=h,rotation_component_rms=float(np.sqrt(np.mean(np.sum(rot[:,:k]**2,axis=-1)))),body_derivative_component_rms=float(np.sqrt(np.mean(np.sum(body[:,:k]**2,axis=-1)))),vector_sum_rms=float(np.sqrt(np.mean(np.sum((rot[:,:k]+body[:,:k])**2,axis=-1))))))
        pd.DataFrame(chain).to_csv(folder/'drift_chain.csv',index=False)
        final=rows[rows.horizon_s==5.]
        summary=dict(experiment=name,checkpoint_sha256=sha(ckpt),window_count=722,window_sha256=sha(oldroot/'windows.csv'),
            numeric_failures=int(final.numerical_failed.sum()),support_failures=int(final.support_failed.sum()),clipping_failures=int(final.clipping_failed.sum()),
            high_frequency_cutoff_hz_train95=cutoff,omega_high_frequency_energy_ratio=hfratio,omega_spectral_l1_relative=mismatch,sealed_test_opened=False)
        (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
    if all((args.results/c['name']/'summary.json').exists() for c in protocol['experiments']):
        for output,source in [('benchmark_per_horizon.csv','per_horizon.csv'),('variation_summary.csv','variation.csv'),('regime_summary.csv','regimes.csv'),('phase_analysis.csv','phase_analysis.csv')]:
            pd.concat([pd.read_csv(args.results/c['name']/source) for c in protocol['experiments']]).to_csv(args.results/output,index=False)

if __name__=='__main__':main()
