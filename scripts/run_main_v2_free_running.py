#!/usr/bin/env python3
"""Frozen August Main V2 validation benchmark; never trains or reads sealed test.

Exit 2 means the completed benchmark failed its declared warm-start gates.
Outputs are saved before that exit. Output directories must be new/empty.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-main-v2-free-running')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch

from system_identification.models.main_v2_simulator import MainV2Simulator, SimulatorState
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import (
    HORIZONS,PHYSICAL_FIELDS,select_windows,envelope_values,describe,
    endpoint_errors,aggregate,stability,magnitude,
)
from system_identification.evaluation.trajectory import (
    VELOCITY_COLUMNS,QUATERNION_COLUMNS,BODY_RATE_COLUMNS,
)
from system_identification.data.trajectory_dataset import CONTROL_COLUMNS

MODES=('warm26','warm13','warm5','cold_single','cold_zero','cold_repeated26')
EXPECTED_CHECKPOINT_SHA='0ca42b8a3cca07f507b4a52a910daf130bd98da31d590669bf6c47b6ac3031ff'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_simulator(path,device):
    if sha(path)!=EXPECTED_CHECKPOINT_SHA:
        raise ValueError('checkpoint differs from Step 1 frozen Main V2')
    c=torch.load(path,map_location='cpu',weights_only=False)
    s=c['state_dict']
    stats={k:s['base_model.'+k].numpy() for k in ('feature_mean','feature_std','control_mean','control_std','derivative_mean','derivative_std')}
    base=CausalHistoryTrajectoryModel(hidden_size=c['base_config']['hidden_size'],use_controls=False,**stats)
    model=ActuatorAwareTrajectoryModel(base_model=base,tail_mean=c['tail_mean'],tail_std=c['tail_std'],
        **{k:c['config'][k] for k in ('use_drive','use_tail','gated_tail','drive_tau_s','tail_tau_s','initial_tail_gate')})
    model.load_state_dict(s,strict=True)
    return MainV2Simulator(model.to(device).eval())


def make_initial(sim,batch,indices,mode,device):
    def t(v):
        return torch.as_tensor(v[indices],device=device,dtype=torch.float32)
    physical={name:t(getattr(batch.trajectory.truth,name)[:,0]) for name in PHYSICAL_FIELDS}
    if mode.startswith('warm'):
        count=int(mode[4:])
        return sim.reset(**physical,
            history_state_features=t(batch.history_state_features[:,-count:]),
            history_controls=t(batch.history_controls[:,-count:]),
            history_mask=torch.as_tensor(batch.history_mask[indices,-count:],device=device))
    return sim.cold_reset(strategy=mode[5:],command=t(batch.trajectory.controls[:,0]),**physical)


def rollout(sim,initial,commands,dt):
    """No truth argument or log access. Outputs include t0, diagnostics do not."""
    state=initial
    states=[state]
    diagnostics=[]
    for k in range(commands.shape[1]):
        state,d=sim.step(state,commands[:,k],dt[:,k])
        states.append(state)
        diagnostics.append(d)
    pred={f.name:torch.stack([getattr(s,f.name) for s in states],1).cpu().numpy() for f in fields(initial)}
    diag={f.name:torch.stack([getattr(d,f.name) for d in diagnostics],1).cpu().numpy() for f in fields(diagnostics[0])}
    return pred,diag


def euler(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1))


def context(samples):
    roll,pitch=euler(samples[list(QUATERNION_COLUMNS)].to_numpy())
    return dict(vertical_speed=samples[VELOCITY_COLUMNS[2]].to_numpy(),
                abs_body_yaw_rate=np.abs(samples[BODY_RATE_COLUMNS[2]].to_numpy()),
                abs_roll_deg=np.abs(np.rad2deg(roll)),pitch_deg=np.rad2deg(pitch),
                motor_command=samples[CONTROL_COLUMNS[0]].to_numpy(),
                tail_command_magnitude=magnitude(samples[list(CONTROL_COLUMNS[1:])].to_numpy()))


def plots(rows,per_horizon,output):
    warm=per_horizon[per_horizon.initialization=='warm26']
    for metric,short in [('position_m','position'),('velocity_m_s','velocity'),('attitude_deg','attitude'),
                         ('body_rate_rad_s','body_rate'),('frequency_hz','frequency')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for stat in ('rmse','median','p95','max'):
            ax.plot(warm.horizon_s,warm[f'{metric}_{stat}'],marker='o',label=stat)
        ax.set(xlabel='Nominal horizon (s); actual dt integrated',ylabel=metric,title='Warm26 validation; includes finite failed rollouts')
        ax.legend(); ax.grid(alpha=.3);fig.tight_layout()
        fig.savefig(output/f'error_vs_horizon_{short}.png',dpi=150);plt.close(fig)
    for log_id,g in rows[rows.initialization=='warm26'].groupby('log_id'):
        fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
        for ax,name in zip(axes,('velocity_m_s','attitude_deg','body_rate_rad_s','position_m')):
            norm=Normalize(float(g[name].min()),float(g[name].max()))
            for _,segment in g.groupby('segment_id'):
                pivot=segment.pivot(index='horizon_s',columns='start_time_s',values=name)
                times=pivot.columns.to_numpy()
                half_step=.5
                xedges=np.r_[times-half_step,times[-1]+half_step] if len(times)==1 else np.r_[times[0]-half_step,(times[1:]+times[:-1])/2,times[-1]+half_step]
                mesh=ax.pcolormesh(xedges,[.05,.35,.75,1.5,2.5,4.,6.],pivot.to_numpy(),norm=norm,shading='flat')
            ax.set(ylabel='Horizon (s)',title=name,yticks=list(HORIZONS),ylim=(.05,6.))
            fig.colorbar(mesh,ax=ax)
        axes[-1].set_xlabel('Log timestamp (s) at rollout start')
        fig.suptitle(log_id);fig.tight_layout()
        fig.savefig(output/f'heatmap_{Path(log_id).stem}.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'docs/analysis/results/main_v2_free_running_5s')
    parser.add_argument('--trajectory-root',type=Path,default=ROOT/'artifacts/main_v2_free_running_5s')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--stride-steps',type=int,default=50)
    parser.add_argument('--max-windows-per-flight',type=int,default=0,help='nonzero is smoke-only')
    args=parser.parse_args()
    if args.stride_steps<1 or args.batch_size<1 or args.max_windows_per_flight<0:
        parser.error('invalid batch/stride/window limit')
    for path in (args.output_root,args.trajectory_root):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'refusing nonempty output: {path}')
        path.mkdir(parents=True,exist_ok=True)
    out=args.output_root
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    started=time.time()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    logging.info('device=%s; frozen weights, validation rollout only',device)
    checkpoint=ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt'
    dataset=ROOT/'dataset/trajectory_v1_august_f5_c4'
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest['split_contract']['materialized_partitions']!=['train','validation'] or manifest['split_contract']['sealed_test_opened']:
        raise ValueError('unexpected partition contract')
    files=[checkpoint,dataset/'manifest.json',dataset/'samples_train.parquet',dataset/'samples_validation.parquet']
    source_files=[Path(__file__),ROOT/'src/system_identification/models/main_v2_simulator.py',
                  ROOT/'scripts/report_main_v2_free_running.py',
                  ROOT/'src/system_identification/models/trajectory_main_v2.py',
                  ROOT/'src/system_identification/models/trajectory_main_v1.py',
                  ROOT/'src/system_identification/evaluation/main_v2_free_running.py',
                  ROOT/'src/system_identification/evaluation/trajectory.py',
                  ROOT/'src/system_identification/training/trajectory_main_v1.py']
    frozen_hashes={str(p.relative_to(ROOT)):sha(p) for p in files+source_files}
    train=pd.read_parquet(dataset/'samples_train.parquet')
    validation=pd.read_parquet(dataset/'samples_validation.parquet')
    for name,frame in [('train',train),('validation',validation)]:
        if set(frame.log_id)!=set(manifest['split_contract']['assignments'][name]):
            raise ValueError('sample log IDs differ from frozen manifest')
    envelopes={name:{key:describe(v) for key,v in envelope_values(frame).items()} for name,frame in [('train',train),('validation',validation)]}
    (out/'observed_envelope.json').write_text(json.dumps(envelopes,indent=2))
    windows,coverage=select_windows(validation,stride_steps=args.stride_steps)
    if args.max_windows_per_flight:
        windows=windows.groupby('log_id',sort=False).head(args.max_windows_per_flight)
    windows=windows.reset_index(drop=True)
    windows.to_csv(out/'windows.csv',index=False);coverage.to_csv(out/'segment_coverage.csv',index=False)
    batch=assemble_history_trajectory_windows(validation,windows,history_steps=26)
    assert batch.history_mask.all()
    train_context=context(train.loc[train.valid_core])
    bin_edges={k:np.quantile(v,[1/3,2/3]).tolist() for k,v in train_context.items()}
    init_rows=validation.set_index(['log_id','segment_id','sample_in_segment']).loc[
        [(r.log_id,r.segment_id,r.start_sample_in_segment) for r in windows.itertuples()]].reset_index()
    start_context=context(init_rows)
    for k,v in start_context.items():
        windows[k]=v
        windows[k+'_bin']=np.searchsorted(bin_edges[k],v,side='right')
    sim=load_simulator(checkpoint,device)
    rows=[]; traces=[]; baseline_rows=[]; probe={}
    logging.info('%d eligible validation starts across %d flights; modes=%s',len(windows),windows.log_id.nunique(),MODES)
    with torch.inference_mode():
        # Frozen checkpoint parity and future-label NaN test on all five flights.
        import copy
        probe_i=np.array([g.index[0] for _,g in windows.groupby('log_id')])
        commands=torch.tensor(batch.trajectory.controls[probe_i],device=device,dtype=torch.float32)
        dt=torch.tensor(batch.trajectory.dt_s[probe_i],device=device,dtype=torch.float32)
        state=make_initial(sim,batch,probe_i,'warm26',device)
        pp,_=rollout(sim,state,commands,dt)
        poisoned=copy.deepcopy(batch)
        for name in PHYSICAL_FIELDS:
            getattr(poisoned.trajectory.truth,name)[:,1:]=np.nan
        npred,_=rollout(sim,make_initial(sim,poisoned,probe_i,'warm26',device),commands,dt)
        probe['future_labels_nan_bitwise_equal']=all(np.array_equal(pp[n],npred[n]) for n in pp)
        def tensor(v): return torch.tensor(v[probe_i],device=device,dtype=torch.float32)
        legacy=sim.model(history_state_features=tensor(batch.history_state_features),history_controls=tensor(batch.history_controls),
             history_mask=torch.tensor(batch.history_mask[probe_i],device=device),
             **{n:tensor(getattr(batch.trajectory.truth,n)[:,0]) for n in PHYSICAL_FIELDS},future_controls=commands,dt_s=dt)
        probe['legacy_250_step_max_abs_difference']={n:float(np.max(np.abs(pp[n]-getattr(legacy,n).cpu().numpy()))) for n in PHYSICAL_FIELDS}
        state=make_initial(sim,batch,probe_i,'warm26',device)
        for k in range(100): state,_=sim.step(state,commands[:,k],dt[:,k])
        snapshot_path=args.trajectory_root/'resume_state.pt'
        torch.save(state.snapshot(),snapshot_path)
        state=sim.reset(state=SimulatorState.from_snapshot(torch.load(snapshot_path,weights_only=True),device=device))
        tail,_=rollout(sim,state,commands[:,100:],dt[:,100:])
        probe['resume_all_state_bitwise_equal']=all(np.array_equal(pp[n][:,100:],tail[n]) for n in pp)
        assert probe['future_labels_nan_bitwise_equal'] and probe['resume_all_state_bitwise_equal']
        assert max(probe['legacy_250_step_max_abs_difference'].values())==0
        for mode in MODES:
            logging.info('starting %s',mode)
            for start in range(0,len(windows),args.batch_size):
                indices=np.arange(start,min(start+args.batch_size,len(windows)))
                commands=torch.tensor(batch.trajectory.controls[indices],device=device,dtype=torch.float32)
                dt=torch.tensor(batch.trajectory.dt_s[indices],device=device,dtype=torch.float32)
                pred,diag=rollout(sim,make_initial(sim,batch,indices,mode,device),commands,dt)
                dt_np=batch.trajectory.dt_s[indices]
                signals,flags=stability(pred,diag,dt_np,envelopes['train'])
                # Truth is consulted only after completion of each rollout batch.
                from system_identification.models.trajectory import TrajectoryPrediction
                truth=TrajectoryPrediction(**{n:getattr(batch.trajectory.truth,n)[indices] for n in PHYSICAL_FIELDS})
                if mode=='warm26':
                    from system_identification.models.trajectory import ConstantTwistPredictor, InitialTrajectoryState
                    reference=ConstantTwistPredictor().rollout(
                        InitialTrajectoryState(**{n:getattr(truth,n)[:,0] for n in PHYSICAL_FIELDS}),
                        batch.trajectory.controls[indices],dt_np)
                    for horizon,k in HORIZONS.items():
                        ref_frame=windows.iloc[indices].copy()
                        ref_frame['horizon_s']=horizon
                        ref_frame['initialization']='constant_twist'
                        ref_frame['failed']=False;ref_frame['numerical_failed']=False
                        for n,v in endpoint_errors(vars(reference),truth,k).items(): ref_frame[n]=v
                        baseline_rows.append(ref_frame)
                truth_pred={n:getattr(truth,n) for n in PHYSICAL_FIELDS}
                truth_diag=dict(acceleration_b=np.diff(truth.velocity_n,axis=1)/dt_np[:,:,None],
                                angular_acceleration_b=np.diff(truth.angular_velocity_b,axis=1)/dt_np[:,:,None],
                                **{k:np.zeros_like(dt_np) for k in ('frequency_clipped','derivative_clipped','drive_clipped','residual_clipped')})
                _,truth_flags=stability(truth_pred,truth_diag,dt_np,envelopes['train'])
                for horizon,k in HORIZONS.items():
                    frame=windows.iloc[indices].copy()
                    frame['initialization']=mode;frame['horizon_s']=horizon
                    frame['actual_horizon_s']=dt_np[:,:k].sum(axis=1)
                    errors=endpoint_errors(pred,truth,k)
                    for n,v in errors.items(): frame[n]=v
                    frame['failed']=flags['failed'][:,:k].any(axis=1)
                    frame['numerical_failed']=flags['numerical'][:,:k].any(axis=1)
                    frame['envelope_failed']=flags['envelope'][:,:k].any(axis=1)
                    frame['clipping_failed']=flags['clipping'][:,:k].any(axis=1)
                    frame['truth_envelope_failed']=truth_flags['envelope'][:,:k].any(axis=1)
                    for name,flag in flags.items():
                        if name not in {'failed','numerical','envelope','clipping'}:
                            frame[name+'_steps']=flag[:,:k].sum(axis=1)
                    for name in ('frequency_clipped','derivative_clipped','drive_clipped','residual_clipped'):
                        frame[name+'_events']=diag[name][:,:k].sum(axis=1)
                        frame[name+'_steps']=(diag[name][:,:k]>0).sum(axis=1)
                    for n,v in signals.items():
                        frame[n+'_max']=v[:,:k].max(axis=1)
                    first=np.argmax(flags['failed'][:,:k],axis=1)
                    frame['first_failure_time_s']=np.where(frame.failed,np.take_along_axis(np.cumsum(dt_np[:,:k],axis=1),first[:,None],axis=1)[:,0],np.nan)
                    rows.append(frame)
                # Paired late-rollout diagnostics are descriptive, not causal proof.
                tr=windows.iloc[indices].copy();tr['initialization']=mode
                for name,key in [('velocity','velocity_n'),('body_rate','angular_velocity_b'),('frequency','flap_frequency_hz')]:
                    a=pred[key][:,-51:];b=getattr(truth,key)[:,-51:]
                    av=np.std(a,axis=1);bv=np.std(b,axis=1)
                    if av.ndim==2: av=magnitude(av);bv=magnitude(bv)
                    tr[name+'_last1s_std_pred']=av;tr[name+'_last1s_std_truth']=bv
                    tr[name+'_variation_ratio']=av/np.maximum(bv,1e-8)
                    delta=pred[key][:,-1]-pred[key][:,0];td=getattr(truth,key)[:,-1]-getattr(truth,key)[:,0]
                    tr[name+'_drift_pred']=magnitude(delta) if delta.ndim==2 else delta
                    tr[name+'_drift_truth']=magnitude(td) if td.ndim==2 else td
                for name,key in [('speed','velocity_n'),('rate','angular_velocity_b')]:
                    ps=np.sum(pred[key]**2,axis=-1);ts=np.sum(getattr(truth,key)**2,axis=-1)
                    tr[name+'_squared_growth_pred']=ps[:,-1]-ps[:,0]
                    tr[name+'_squared_growth_truth']=ts[:,-1]-ts[:,0]
                    tr[name+'_positive_increment_fraction']=np.mean(np.diff(ps,axis=1)>0,axis=1)
                tr['command_total_variation']=magnitude(np.diff(batch.trajectory.controls[indices],axis=1)).sum(axis=1)
                from system_identification.models.trajectory import attitude_error_deg
                tr['attitude_drift_pred_deg']=attitude_error_deg(pred['quaternion_nb'][:,-1],pred['quaternion_nb'][:,0])
                tr['attitude_drift_truth_deg']=attitude_error_deg(truth.quaternion_nb[:,-1],truth.quaternion_nb[:,0])
                # Low terminal variation AND unsupported terminal state: equilibrium suspect.
                tr['unsupported_equilibrium_suspect']=(tr.velocity_variation_ratio<.25)&(tr.body_rate_variation_ratio<.25)&flags['envelope'][:,-1]
                tr['angular_damping_suspect']=(tr.body_rate_variation_ratio<.5)&(tr.body_rate_last1s_std_truth>1e-3)
                traces.append(tr)
                if mode=='warm26':
                    np.savez_compressed(args.trajectory_root/f'warm26_{start:05d}.npz',window_ids=windows.iloc[indices].window_id.to_numpy(dtype=str),
                                        dt_s=dt_np,**pred,**diag)
            logging.info('completed %s',mode)
    per_rollout=pd.concat(rows,ignore_index=True)
    trace=pd.concat(traces,ignore_index=True)
    # Descriptive suspects, not assertions of causal energy injection.
    low_tv=float(trace[trace.initialization=='warm26'].command_total_variation.quantile(1/3))
    trace['low_command_variation']=trace.command_total_variation<=low_tv
    trace['energy_growth_suspect']=trace.low_command_variation & (
        ((trace.speed_squared_growth_pred>0)&(trace.speed_squared_growth_pred>trace.speed_squared_growth_truth)&(trace.speed_positive_increment_fraction>.75)) |
        ((trace.rate_squared_growth_pred>0)&(trace.rate_squared_growth_pred>trace.rate_squared_growth_truth)&(trace.rate_positive_increment_fraction>.75)))
    metric_names=list(errors)
    baseline=aggregate(pd.concat(baseline_rows),['horizon_s'],metric_names)
    baseline.to_csv(out/'constant_twist_baseline.csv',index=False)
    per_horizon=aggregate(per_rollout,['initialization','horizon_s'],metric_names)
    per_flight=aggregate(per_rollout,['initialization','log_id','horizon_s'],metric_names)
    per_rollout.to_csv(out/'per_rollout.csv',index=False)
    per_horizon.to_csv(out/'per_horizon.csv',index=False)
    per_flight.to_csv(out/'per_flight.csv',index=False)
    per_rollout[per_rollout.failed].to_csv(out/'failure_cases.csv',index=False)
    per_horizon.to_csv(out/'initialization_ablation.csv',index=False)
    trace.to_csv(out/'trajectory_diagnostics.csv',index=False)
    regime=[]
    for key in bin_edges:
        r=aggregate(per_rollout[per_rollout.initialization=='warm26'],[key+'_bin','horizon_s'],metric_names)
        r=r.rename(columns={key+'_bin':'bin'});r['variable']=key;regime.append(r)
    pd.concat(regime,ignore_index=True).to_csv(out/'regime_bins.csv',index=False)
    # Command-transition bins are offline future-tape diagnostics only, not inputs.
    command_variation=trace[trace.initialization=='warm26'].set_index('window_id').command_total_variation
    transition_edges=np.quantile(command_variation,[1/3,2/3]).tolist()
    wr=per_rollout[per_rollout.initialization=='warm26'].copy()
    wr['transition_bin']=np.searchsorted(transition_edges,wr.window_id.map(command_variation),side='right')
    aggregate(wr,['transition_bin','horizon_s'],metric_names).to_csv(out/'command_transition_bins.csv',index=False)
    plots(per_rollout,per_horizon,out)
    warm=per_rollout[per_rollout.initialization=='warm26']
    final=warm[warm.horizon_s==5.0]
    gates={str(h):dict(n_rollouts=len(g),n_valid=int((~g.failed).sum()),n_failed=int(g.failed.sum()),
          failure_pct=float(100*g.failed.mean()),numerical_failure_pct=float(100*g.numerical_failed.mean()),
          envelope_failure_pct=float(100*g.envelope_failed.mean()),clipping_failure_pct=float(100*g.clipping_failed.mean()),
          truth_envelope_failure_pct=float(100*g.truth_envelope_failed.mean()),actual_horizon_s=describe(g.actual_horizon_s))
          for h,g in warm.groupby('horizon_s')}
    summary=dict(schema='main_v2_free_running_5s_v1',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=frozen_hashes,checkpoint_unchanged=sha(checkpoint)==EXPECTED_CHECKPOINT_SHA,
        partitions_read=['train_for_observed_envelope_only','validation'],sealed_test_opened=False,training_performed=False,
        dataset_contract=manifest,device=device,torch_version=torch.__version__,
        initialization_modes=MODES,window_count=len(windows),flight_count=windows.log_id.nunique(),
        stride_steps=args.stride_steps,smoke_only=bool(args.max_windows_per_flight),
        actual_dt_s=describe(batch.trajectory.dt_s),history_duration_s=describe(np.array([
            (init_rows.iloc[j].timestamp_us-validation[(validation.log_id==r.log_id)&(validation.segment_id==r.segment_id)&(validation.sample_in_segment==r.start_sample_in_segment-25)].iloc[0].timestamp_us)*1e-6 for j,r in enumerate(windows.itertuples())])),
        history_filter_dt_s=sim.model.history_dt_s,bin_edges_train_tertiles=bin_edges,
        transition_edges_validation_descriptive_only=transition_edges,
        probes=probe,gates=gates,
        gate_policy={'numeric':'all persistent state/derivatives finite; quaternion norm deviation <= 1e-4',
                     'observed_support':'all five magnitudes within train min/max; p1/p99 excursions separate',
                     'clipping':'any actual pre-clamp clipping is a failure signal; count component events and affected steps',
                     'failed':'union of numeric, observed_support and clipping; finite failed rollouts retained in metrics',
                     'not_certified_safety_envelope':True},
        warm5s_diagnostics={name:float(trace[trace.initialization=='warm26'][name].mean()) for name in ('unsupported_equilibrium_suspect','angular_damping_suspect','energy_growth_suspect')},
        benchmark_gate_pass=not bool(final.failed.any()),wall_time_s=time.time()-started)
    if not all(sha(ROOT/p)==digest for p,digest in frozen_hashes.items()):
        raise RuntimeError('input/source changed during benchmark')
    summary['output_sha256']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='run.log'}
    summary['trajectory_files']={str(p):sha(p) for p in sorted(args.trajectory_root.iterdir())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    logging.info('completed: warm5s failures=%d/%d; numeric=%d; gate=%s',final.failed.sum(),len(final),final.numerical_failed.sum(),summary['benchmark_gate_pass'])
    if not args.max_windows_per_flight and args.stride_steps == 50:
        subprocess.run([sys.executable,str(ROOT/'scripts/report_main_v2_free_running.py'),
                        '--results-root',str(out),'--report',str(out/'report.md')],check=True)
    return 0 if summary['benchmark_gate_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())
