#!/usr/bin/env python3
"""Audit frozen September state consistency without modifying data or training."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_september_step2 import write_json,status
from run_september_history_lengths import matched_inputs,history_prefix
from system_identification.data.september_trajectory import file_hash
from system_identification.evaluation.trajectory_consistency import integration_residual,metric_tables,shifted_velocity_residual
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
SOURCE=ROOT/'artifacts/september_history_lengths_20260911'


def optional_topic(ulog,name,instance=0):
    return next((x.data for x in ulog.data_list if x.name==name and x.multi_id==instance),None)


def previous_indices(t, query):
    return np.searchsorted(t,query,side='right')-1


def main_run(args,output):
    status(output,'verifying_registered_data')
    manifest,entry,samples,windows,coverage=matched_inputs()
    for part,w in windows.items():
        if not w.equals(pd.read_parquet(SOURCE/f'{part}_windows.parquet')):raise ValueError('Source window mismatch')
        w.to_parquet(output/f'{part}_windows.parquet',index=False)
    source_manifest=json.loads((SOURCE/'manifest.json').read_text())
    for p,h in source_manifest['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError(f'Source code mismatch: {p}')
    sources=[SOURCE/'manifest.json',SOURCE/'normalization.npz',SOURCE/'history_26.pt',SOURCE/'validation_history_26_errors.npz',SOURCE/'step2_error_source_diagnostics.npz']
    source_hashes={str(p):file_hash(p) for p in sources}
    paths=[Path(__file__),ROOT/'src/system_identification/evaluation/trajectory_consistency.py']
    run_manifest={'experiment':'september_state_consistency','dataset':entry,'dataset_id':manifest['dataset_id'],
        'source_dataset_manifest':manifest,'partitions':['train','validation'],'sealed_test_opened':False,
        'source_hashes':source_hashes,'implementation_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
        'mode':'replay_checkpoint' if args.replay else 'verified_cached_predictions',
        'aggregation':'vector Euclidean RMSE within each log, then equal-log mean; pooled reported separately',
        'lag_grid_s':[-.1,-.05,0,.05,.1],'lag_scope':'diagnostic-only; common interior with .11s margins',
        'future_truth':'offline diagnostics only; no fitting, label editing or normal-predictor future truth input',
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'protocol':'docs/contracts/2026-09-11_september_state_consistency.md'}
    write_json(output/'manifest.json',run_manifest)
    # Preserve the other agent's exact temporary implementations as provenance.
    archive=output/'legacy_source';archive.mkdir()
    for name in ['run_diag.py','analyze_diag.py','finalize_step2.py']:
        source=Path('/tmp')/name
        if source.exists():shutil.copyfile(source,archive/name)
    batch=assemble_history_trajectory_windows(samples['validation'],windows['validation'],history_steps=101)
    if args.replay:
        import torch
        from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
        from system_identification.training.trajectory_main_v1 import predict_history_trajectory_model
        torch.set_num_threads(4)
        ck=torch.load(SOURCE/'history_26.pt',weights_only=True,map_location='cpu')
        model=CausalHistoryTrajectoryModel(hidden_size=ck['config']['hidden_size'],use_controls=True,**dict(np.load(SOURCE/'normalization.npz')))
        model.load_state_dict(ck['state_dict'],strict=True)
        pred=predict_history_trajectory_model(model,history_prefix(batch,26),use_history=True,batch_size=256,device='cuda:1')
        pp,pv=pred.position_n,pred.velocity_n
    else:
        # Original local cache stores object-typed IDs. No external untrusted input accepted.
        cached=np.load(SOURCE/'step2_error_source_diagnostics.npz',allow_pickle=True)
        for name,expected in [('window_ids',batch.trajectory.window_ids),('log_ids',batch.trajectory.log_ids),('dt_s',batch.trajectory.dt_s),('true_position',batch.trajectory.truth.position_n),('true_velocity',batch.trajectory.truth.velocity_n)]:
            np.testing.assert_array_equal(cached[name],expected)
        pp,pv=cached['pred_position'],cached['pred_velocity']
    tp,tv,dt=batch.trajectory.truth.position_n,batch.trajectory.truth.velocity_n,batch.trajectory.dt_s
    original=np.load(SOURCE/'validation_history_26_errors.npz')
    np.testing.assert_array_equal(original['window_ids'].astype(str),batch.trajectory.window_ids.astype(str))
    for a,b,key in [(pp,tp,'position_error_m'),(pv,tv,'velocity_error_m_s')]:
        np.testing.assert_allclose(np.linalg.norm(a-b,axis=-1),original[key],rtol=0,atol=1e-4 if args.replay else 0)
    np.savez_compressed(output/'prediction_state_arrays.npz',pred_position=pp,pred_velocity=pv,true_position=tp,true_velocity=tv,
                        dt_s=dt,window_ids=batch.trajectory.window_ids.astype(str),log_ids=batch.trajectory.log_ids.astype(str))
    tr=integration_residual(tp,tv,dt);pr=integration_residual(pp,pv,dt)
    ev=pv-tv;ep=pp-tp
    integrated=np.cumsum(.5*(ev[:,1:]+ev[:,:-1])*dt[:,:,None],axis=1)
    vectors={'position_error_m':ep[:,1:],'velocity_error_m_s':ev[:,1:],
             'true_cumulative_residual_m':np.cumsum(tr,axis=1),'pred_cumulative_residual_m':np.cumsum(pr,axis=1),
             'integrated_velocity_error_m':integrated,
             'decomposition_closure_m':ep[:,1:]-ep[:,:1]-integrated+np.cumsum(tr,axis=1)-np.cumsum(pr,axis=1)}
    per,macro=metric_tables(vectors,batch.trajectory.log_ids,[50,100,150,250])
    per.to_csv(output/'prediction_per_log.csv',index=False);macro.to_csv(output/'prediction_aggregate.csv',index=False)
    del batch
    status(output,'auditing_raw_local_position')
    raw_rows=[];clock_tables=[];lag_tables=[];window_rows=[]
    for part in ['train','validation']:
        for log,w in windows[part].groupby('log_id',sort=False):
            raw_path=Path(manifest['source']['root'])/log
            expected=manifest['source']['ulog_sha256'][log]
            if file_hash(raw_path)!=expected:raise ValueError(f'Raw source changed: {log}')
            ulog=ULog(str(raw_path),message_name_filter_list=['vehicle_local_position','estimator_selector_status','estimator_status'])
            raw=optional_topic(ulog,'vehicle_local_position')
            if raw is None:raise ValueError('Missing local position')
            g=samples[part].loc[samples[part].log_id==log].sort_values('sample_in_log')
            index=g.sample_in_log.to_numpy(dtype=int)
            np.testing.assert_array_equal(index,np.arange(len(raw['timestamp'])))
            for field,col in [('timestamp','timestamp_us'),('timestamp_sample','state_sample_timestamp_us'),('x','position_ned_m_x'),('y','position_ned_m_y'),('z','position_ned_m_z'),('vx','velocity_ned_m_s_x'),('vy','velocity_ned_m_s_y'),('vz','velocity_ned_m_s_z')]:
                np.testing.assert_array_equal(raw[field],g[col].to_numpy())
            ts=raw['timestamp'].astype(np.float64)*1e-6;event=raw['timestamp_sample'].astype(np.float64)*1e-6
            p=np.column_stack([raw[k] for k in ['x','y','z']]).astype(float)
            v=np.column_stack([raw[k] for k in ['vx','vy','vz']]).astype(float)
            zvel=v.copy();zvel[:,2]=raw['z_deriv']
            starts=w.start_sample_in_log.to_numpy(dtype=int);count=int(w.state_sample_count.iloc[0])
            indices=starts[:,None]+np.arange(count)
            times=ts[indices];edts=np.diff(event[indices],axis=1)
            if (edts<=0).any():raise ValueError('Nonmonotone sample clock in selected windows')
            pub=integration_residual(p[indices],v[indices],np.diff(times,axis=1))
            sample=integration_residual(p[indices],v[indices],edts)
            vertical=integration_residual(p[indices],zvel[indices],np.diff(times,axis=1))
            steps=[50,100] if part=='train' else [50,100,150,250]
            rv={'raw_publication_clock_m':np.cumsum(pub,axis=1),'raw_sample_clock_m':np.cumsum(sample,axis=1),
                'clock_change_effect_m':np.cumsum(pub-sample,axis=1),'raw_z_deriv_variant_m':np.cumsum(vertical,axis=1)}
            per,_=metric_tables(rv,np.repeat(log,len(w)),steps);per['partition']=part;clock_tables.append(per)
            flags={}
            for field in ['xy_reset_counter','z_reset_counter','vxy_reset_counter','vz_reset_counter','heading_reset_counter']:
                flags[field]=np.any(np.diff(raw[field][indices].astype(int),axis=1)!=0,axis=1)
            selector=optional_topic(ulog,'estimator_selector_status')
            selection=np.full(len(ts),-1,dtype=int)
            if selector is not None:
                si=previous_indices(selector['timestamp'],raw['timestamp']);valid=si>=0
                selection[valid]=selector['primary_instance'][si[valid]]
            flags['estimator_switch']=np.any(np.diff(selection[indices],axis=1)!=0,axis=1)
            wf=w[['window_id','log_id']].copy();wf['partition']=part
            for key,value in flags.items():wf[key]=value
            wf['duration_s']=times[:,-1]-times[:,0];wf['true_cumulative_residual_m']=np.linalg.norm(pub.sum(axis=1),axis=1)
            window_rows.append(wf)
            # Count each raw transition once, despite overlapping evaluation windows.
            covered=np.zeros(len(ts)-1,dtype=bool)
            for start in starts:covered[start:start+count-1]=True
            residual=integration_residual(p,v,np.diff(ts));rate=residual/np.diff(ts)[:,None]
            tracking=np.full(len(ts),np.nan)
            for instance in np.unique(selection[selection>=0]):
                est=optional_topic(ulog,'estimator_status',int(instance))
                if est is None:continue
                ei=previous_indices(est['timestamp'],raw['timestamp']);safe=np.clip(ei,0,len(est['timestamp'])-1)
                good=(selection==instance)&(ei>=0)&((raw['timestamp'].astype(float)-est['timestamp'][safe])<=250000)
                tracking[good]=est['output_tracking_error[2]'][safe[good]]
            good=covered&np.isfinite(tracking[1:]);corr=None
            if good.sum()>2 and np.std(tracking[1:][good])>0:
                corr=float(np.corrcoef(np.linalg.norm(rate[good],axis=1),tracking[1:][good])[0,1])
            lag_ms=(ts-event)*1000
            row={'partition':part,'log_id':log,'raw_samples':len(ts),'selected_windows':len(w),'unique_selected_intervals':int(covered.sum()),
                 'raw_values_equal_dataset':True,'position_velocity_same_packet':True,
                 'publication_lag_p50_ms':float(np.median(lag_ms[indices])), 'publication_lag_p99_ms':float(np.quantile(lag_ms[indices],.99)),
                 'unique_step_vector_residual_rmse_m':float(np.sqrt(np.mean(np.sum(residual[covered]**2,axis=1)))),
                 'position_tracking_error_correlation':corr,'tracking_pairs':int(good.sum())}
            for j,axis in enumerate(['north','east','down']):
                row[f'residual_rate_{axis}_rmse_m_s']=float(np.sqrt(np.mean(rate[covered,j]**2)))
            for key,val in flags.items():row[key+'_windows']=int(val.sum())
            raw_rows.append(row)
            for lag in [-.1,-.05,0,.05,.1]:
                r,duration=shifted_velocity_residual(p[indices],v[indices],times,lag)
                lag_tables.append({'partition':part,'log_id':log,'lag_s':lag,'n_windows':len(w),'duration_min_s':float(duration.min()),'duration_max_s':float(duration.max()),'vector_rmse':float(np.sqrt(np.mean(np.sum(r*r,axis=1))))})
            if file_hash(raw_path)!=expected:raise ValueError('Raw changed during read')
    pd.DataFrame(raw_rows).to_csv(output/'raw_source_audit.csv',index=False)
    clocks=pd.concat(clock_tables,ignore_index=True);clocks.to_csv(output/'clock_per_log.csv',index=False)
    cols=['vector_rmse','north_rmse','east_rmse','down_rmse']
    clocks.groupby(['partition','metric','nominal_horizon_s'])[cols].mean().reset_index().to_csv(output/'clock_equal_log.csv',index=False)
    lags=pd.DataFrame(lag_tables);lags.to_csv(output/'lag_per_log.csv',index=False)
    lags.groupby(['partition','lag_s']).vector_rmse.mean().reset_index().to_csv(output/'lag_equal_log.csv',index=False)
    pd.concat(window_rows,ignore_index=True).to_parquet(output/'window_raw_audit.parquet',index=False)
    for path,h in source_hashes.items():
        if file_hash(Path(path))!=h:raise ValueError('Source artifact changed during audit')
    write_json(output/'summary.json',{'status':'completed','raw_logs_audited':len(raw_rows),'raw_values_equal_dataset':True,
        'prediction_errors_reproduce_original':True,'validation_windows':len(windows['validation']),'train_windows':len(windows['train']),
        'sealed_test_opened':False,'dataset_modified':False,'trained':False})
    status(output,'completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/september_state_consistency_20260911')
    parser.add_argument('--replay',action='store_true',help='Recompute predictions on GPU 1 rather than validate cached arrays')
    args=parser.parse_args();output=args.output_root.resolve();output.mkdir(parents=True,exist_ok=False)
    try:main_run(args,output)
    except BaseException as e:
        status(output,'failed',error=repr(e));(output/'traceback.txt').write_text(traceback.format_exc());raise
