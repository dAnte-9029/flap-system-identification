#!/usr/bin/env python3
"""Additional increment, axis, phase, derivative and noise diagnostics for S models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-increment')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from scipy.signal import welch
from scipy.spatial.transform import Rotation
from run_main_v2_free_running import load_simulator,make_initial,sha
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import HORIZONS,aggregate,magnitude,describe
from main_v2_step4_tools import harmonic_design


def rows_for_error(error,logs,**meta):
    rows=[]
    for log in pd.unique(logs):
        e=error[np.asarray(logs)==log]
        # Error arrays [origins, (time), axes]; do not assume origins independent.
        axis=np.sqrt(np.mean(e.reshape(-1,e.shape[-1])**2,axis=0))
        rows.append(dict(**meta,log_id=log,n_origins=len(e),rmse=float(np.linalg.norm(axis)),**{'axis_'+str(j)+'_rmse':float(v) for j,v in enumerate(axis)}))
    return rows


def axis_spectrum(t,y):
    grid=np.arange(t[0],t[-1],.02)
    a=np.column_stack([np.interp(grid,t,y[:,j]) for j in range(y.shape[1])])
    return welch(a,fs=50,nperseg=min(128,len(a)),axis=0,detrend='constant')


def phase_diagnostic(pred,true,phase,logs,*,experiment,mode):
    rows=[]
    for log in pd.unique(logs):
        sel=np.asarray(logs)==log;phi=phase[sel].reshape(-1);p=pred[sel].reshape(-1,6);t=true[sel].reshape(-1,6)
        X=harmonic_design(phi,3);cp=np.linalg.lstsq(X,p,rcond=None)[0];ct=np.linalg.lstsq(X,t,rcond=None)[0]
        grid=np.linspace(0,2*np.pi,96,endpoint=False);H=harmonic_design(grid,3);ap=H@cp;at=H@ct
        for j,label in enumerate(['ax_n','ay_n','az_n','p_dot','q_dot','r_dot']):
            rows.append(dict(experiment=experiment,mode=mode,log_id=log,signal=label,
                harmonic_rmse=float(np.sqrt(np.mean((ap[:,j]-at[:,j])**2))),pred_harmonic_std=float(ap[:,j].std()),truth_harmonic_std=float(at[:,j].std()),
                shape_correlation=float(np.corrcoef(ap[:,j],at[:,j])[0,1]),bias=float(cp[0,j]-ct[0,j]),
                phase_bin_errors=json.dumps([float(v) for v in (ap[:,j]-at[:,j])[::4]])))
    return rows


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--results',type=Path,required=True);pa.add_argument('--artifacts',type=Path,required=True);pa.add_argument('--device',default='cuda:1');args=pa.parse_args();r=args.results
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protocol=json.loads((r/'protocol.json').read_text());names=[c['name'] for c in protocol['experiments']]
    samples=pd.read_parquet(ROOT/'dataset/trajectory_v1_august_f5_c4/samples_validation.parquet')
    windows=pd.read_csv(ROOT/'docs/analysis/results/main_v2_free_running_5s/windows.csv');b=assemble_history_trajectory_windows(samples,windows,history_steps=26);dt=b.trajectory.dt_s;t=b.trajectory.truth
    # Reuse exactly the fixed Step 4 legal teacher-state origins, now predicting two steps.
    m=pd.read_csv(ROOT/'docs/analysis/results/main_v2_dynamics_observability_phase/validation_points.csv')
    local=m[['log_id','segment_id','sample_in_segment']].rename(columns={'sample_in_segment':'start_sample_in_segment'}).copy();local['state_sample_count']=3
    local['window_id']=[f'{x.log_id}:{x.segment_id}:{x.start_sample_in_segment}' for x in local.itertuples()]
    lb=assemble_history_trajectory_windows(samples,local,history_steps=26);ldt=lb.trajectory.dt_s;lt=lb.trajectory.truth
    offline=np.load(ROOT/'docs/analysis/results/main_v2_dynamics_observability_phase/validation_diagnostic_arrays.npz')
    np.testing.assert_allclose(np.column_stack([(lt.velocity_n[:,1]-lt.velocity_n[:,0])/ldt[:,0,None],(lt.angular_velocity_b[:,1]-lt.angular_velocity_b[:,0])/ldt[:,0,None]]),offline['target_D0_raw'],atol=1e-8)
    tv=(t.velocity_n[:,1:]-t.velocity_n[:,:-1])/dt[:,:,None];ta=(t.angular_velocity_b[:,1:]-t.angular_velocity_b[:,:-1])/dt[:,:,None]
    diagnostics=pd.read_csv(ROOT/'docs/analysis/results/main_v2_free_running_5s/trajectory_diagnostics.csv');diagnostics=diagnostics[diagnostics.initialization=='warm26'].set_index('window_id')
    variation_edges=np.quantile(diagnostics.command_total_variation,[1/3,2/3])
    teacher=[];freerows=[];axisrows=[];variation=[];phase=[];spectra=[];acf=[];extra_regimes=[];derivative=[];derivative_stats=[]
    for name in names:
        print('INCREMENT DIAGNOSTICS',name,flush=True)
        sim=load_simulator(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',args.device)
        sim.model.load_state_dict(torch.load(args.artifacts/name/'model.pt',map_location='cpu',weights_only=False)['state_dict'])
        predparts=[]
        with torch.inference_mode():
            for start in range(0,len(local),2048):
                ix=np.arange(start,min(start+2048,len(local)));state=make_initial(sim,lb,ix,'warm26',args.device)
                p={}
                for k in range(2):
                    state,d=sim.step(state,torch.tensor(lb.trajectory.controls[ix,k],device=args.device,dtype=torch.float32),torch.tensor(ldt[ix,k],device=args.device,dtype=torch.float32))
                    p['v'+str(k+1)]=state.velocity_n.cpu().numpy();p['w'+str(k+1)]=state.angular_velocity_b.cpu().numpy()
                predparts.append(p)
        lp={key:np.concatenate([p[key] for p in predparts]) for key in predparts[0]}
        np.savez_compressed(args.artifacts/name/'teacher_two_step.npz',**lp,dt_s=ldt,point_ids=local.window_id.to_numpy(str))
        for signal,key,truth in [('velocity','v',lt.velocity_n),('omega','w',lt.angular_velocity_b)]:
            error=(lp[key+'2']-truth[:,0])-(truth[:,2]-truth[:,0])
            teacher.extend(rows_for_error(error,m.log_id,experiment=name,signal=signal,mode='teacher_state',lag_steps=2))
        ld=np.column_stack([(lp['v1']-lt.velocity_n[:,0])/ldt[:,0,None],(lp['w1']-lt.angular_velocity_b[:,0])/ldt[:,0,None]])
        for target in ['D0_raw','D1_lp6','D1_lp8','D1_lp12']:
            for signal,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:derivative.extend(rows_for_error(ld[:,sl]-offline['target_'+target][:,sl],m.log_id,experiment=name,mode='teacher_state',target=target,signal=signal))
        for source,data in [('prediction',ld)]+[(target,offline['target_'+target]) for target in ['D0_raw','D1_lp6','D1_lp8','D1_lp12']]:
            for log in m.log_id.unique():
                values=data[m.log_id.to_numpy()==log]
                for j,label in enumerate(['ax_n','ay_n','az_n','p_dot','q_dot','r_dot']):
                    derivative_stats.append(dict(experiment=name,mode='teacher_state',source=source,log_id=log,axis=label,
                        mean=float(values[:,j].mean()),std=float(values[:,j].std()),rms=float(np.sqrt(np.mean(values[:,j]**2)))))
        phase.extend(phase_diagnostic(ld,offline['target_D0_raw'],m.phase.to_numpy(),m.log_id.to_numpy(),experiment=name,mode='teacher_state'))
        trace=np.load(args.artifacts/name/'validation_rollout.npz');v=trace['velocity_n'];w=trace['angular_velocity_b'];pa=np.diff(w,axis=1)/dt[:,:,None];pv=np.diff(v,axis=1)/dt[:,:,None]
        phase.extend(phase_diagnostic(np.concatenate([pv,pa],axis=-1),np.concatenate([tv,ta],axis=-1),t.relative_phase_rad[:,:-1],windows.log_id.to_numpy(),experiment=name,mode='free_running'))
        for h,k in HORIZONS.items():
            for signal,pred,true in [('velocity',v,t.velocity_n),('omega',w,t.angular_velocity_b)]:
                err=(pred[:,2:k+1]-pred[:,:k-1])-(true[:,2:k+1]-true[:,:k-1])
                freerows.extend(rows_for_error(err,windows.log_id,experiment=name,horizon_s=h,signal=signal,lag_steps=2))
            # Axis-specific endpoints and SO(3) log pitch-axis drift, no quaternion subtraction.
            pq=trace['quaternion_nb'][:,k];tq=t.quaternion_nb[:,k]
            rot=(Rotation.from_quat(tq[:,[1,2,3,0]]).inv()*Rotation.from_quat(pq[:,[1,2,3,0]])).as_rotvec()*180/np.pi
            axisrows.extend(rows_for_error(w[:,k]-t.angular_velocity_b[:,k],windows.log_id,experiment=name,horizon_s=h,signal='body_rate_endpoint'))
            axisrows.extend(rows_for_error(rot,windows.log_id,experiment=name,horizon_s=h,signal='SO3_log_error_deg'))
            axisrows.extend(rows_for_error(pa[:,:k]-ta[:,:k],windows.log_id,experiment=name,horizon_s=h,signal='angular_acceleration'))
            ratio=magnitude(pa[:,:k].std(1))/np.maximum(magnitude(ta[:,:k].std(1)),1e-12)
            variation.append(dict(experiment=name,horizon_s=h,signal='angular_acceleration',median=float(np.median(ratio)),p10=float(np.quantile(ratio,.1)),p90=float(np.quantile(ratio,.9))))
        ratio=magnitude(pa[:,-50:].std(1))/np.maximum(magnitude(ta[:,-50:].std(1)),1e-12)
        variation.append(dict(experiment=name,horizon_s=5.,signal='angular_acceleration_last1s',median=float(np.median(ratio)),p10=float(np.quantile(ratio,.1)),p90=float(np.quantile(ratio,.9))))
        for label,pred,true in [('omega',w,t.angular_velocity_b),('angular_acceleration',pa,ta)]:
            for lag in [1,2,5,10,25]:
                for source,data in [('pred',pred),('truth',true)]:
                    x=data-data.mean(1,keepdims=True);num=np.sum(x[:,:-lag]*x[:,lag:],axis=(1,2));den=np.sqrt(np.sum(x[:,:-lag]**2,axis=(1,2))*np.sum(x[:,lag:]**2,axis=(1,2)))
                    acf.append(dict(experiment=name,signal=label,source=source,lag_steps=lag,median_correlation=float(np.median(num/np.maximum(den,1e-20)))))
            ps=[];ts=[]
            for j in range(722):
                time=np.r_[0,np.cumsum(dt[j])][:len(pred[j])];freq,pp=axis_spectrum(time,pred[j]);_,tt=axis_spectrum(time,true[j]);ps.append(pp);ts.append(tt)
            pp=np.mean(ps,axis=0);tt=np.mean(ts,axis=0)
            for axis in range(3):
                for freqval,pval,tval in zip(freq,pp[:,axis],tt[:,axis]):spectra.append(dict(experiment=name,signal=label,axis=['p','q','r'][axis],frequency_hz=freqval,pred_power=pval,truth_power=tval))
        frame=pd.read_csv(r/name/'per_rollout.csv');frame['transition_bin']=np.searchsorted(variation_edges,frame.window_id.map(diagnostics.command_total_variation),side='right')
        metrics=['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad']
        extra_regimes.append(aggregate(frame,['experiment','transition_bin','horizon_s'],metrics).rename(columns={'transition_bin':'bin'}).assign(variable='command_variation'))
    pd.DataFrame(teacher).to_csv(r/'teacher_increment_metrics.csv',index=False);pd.DataFrame(freerows).to_csv(r/'free_running_increment_metrics.csv',index=False);pd.DataFrame(axisrows).to_csv(r/'axis_summary.csv',index=False)
    pd.DataFrame(phase).to_csv(r/'phase_conditioned_metrics.csv',index=False);pd.DataFrame(spectra).to_csv(r/'axis_spectra.csv',index=False);pd.DataFrame(acf).to_csv(r/'autocorrelation.csv',index=False);pd.DataFrame(derivative).to_csv(r/'derivative_diagnostics.csv',index=False)
    pd.DataFrame(derivative_stats).to_csv(r/'teacher_derivative_statistics.csv',index=False)
    pd.concat([pd.read_csv(r/n/'variation.csv') for n in names]+[pd.DataFrame(variation)]).to_csv(r/'variation_summary.csv',index=False)
    pd.concat([pd.read_csv(r/n/'regimes.csv') for n in names]+extra_regimes).to_csv(r/'regime_summary.csv',index=False)
    pd.read_csv(r/'benchmark_per_horizon.csv').to_csv(r/'free_running_per_horizon.csv',index=False)
    sp=pd.DataFrame(spectra);summary=[]
    for keys,g in sp.groupby(['experiment','signal','axis']):
        summary.append(dict(experiment=keys[0],signal=keys[1],axis=keys[2],relative_spectral_l1=float((g.pred_power-g.truth_power).abs().sum()/g.truth_power.sum()),high_frequency_ratio=float(g.loc[g.frequency_hz>13.122682010077684,'pred_power'].sum()/g.loc[g.frequency_hz>13.122682010077684,'truth_power'].sum())))
    pd.DataFrame(summary).to_csv(r/'spectral_summary.csv',index=False)
    (r/'increment_evaluation_manifest.json').write_text(json.dumps(dict(teacher_origins=len(local),teacher_ids_sha256=sha(ROOT/'docs/analysis/results/main_v2_dynamics_observability_phase/validation_points.csv'),free_origins=722,
        teacher_increment_duration_s=describe(ldt.sum(1)),free_increment_duration_s=describe((dt[:,:-1]+dt[:,1:]).reshape(-1)),
        transition_bin_edges=variation_edges.tolist(),phase_offsets_fitted=False,
        filters=dict(source='frozen Step4 targets',usage='offline target diagnosis only; never simulator inputs',
            method='native states interpolated to 50 Hz; fourth-order Butterworth sosfiltfilt; interpolate to native timestamps then finite difference',cutoffs_hz=[6,8,12]),
        finished=True,sealed_test_opened=False),indent=2))
    print('INCREMENT EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
