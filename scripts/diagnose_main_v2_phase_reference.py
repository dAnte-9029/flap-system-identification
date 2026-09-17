#!/usr/bin/env python3
"""Offline lag/bias attribution; never feeds corrected truth back to a rollout."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter1d
from system_identification.models.phase_reference import circular,harmonic_design,rotate_coefficients,interval_design
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import HORIZONS
from prepare_main_v2_phase_reference import OLD,DATA


def alignment(phi,yp,yt):
    X=np.column_stack([np.ones(len(phi)),harmonic_design(phi)]);cond=np.linalg.cond(X)
    cycles=(np.unwrap(phi)[-1]-phi[0])/(2*np.pi)
    base=dict(observed_cycles=cycles,design_condition=cond,identifiable=bool(cycles>=1 and cond<1e3 and len(phi)>=7))
    if not base['identifiable']:return base,[]
    cp=np.linalg.lstsq(X,yp,rcond=None)[0];ct=np.linalg.lstsq(X,yt,rcond=None)[0]
    ap=np.hypot(cp[1::2],cp[2::2]);at=np.hypot(ct[1::2],ct[2::2]);theta=circular(np.arctan2(cp[2::2],cp[1::2])-np.arctan2(ct[2::2],ct[1::2]))
    amp=.5*(ap-at)**2;phase=ap*at*(1-np.cos(theta));dc=(cp[0]-ct[0])**2
    grid=np.linspace(-np.pi,np.pi,720,endpoint=False);rot=rotate_coefficients(cp[1:,None],grid)[:,:,0]
    cost=np.sum((rot-ct[None,1:])**2,axis=1);lag=grid[cost.argmin()]
    uniform=harmonic_design(np.arange(360)*2*np.pi/360)
    p=uniform@cp[1:];t=uniform@ct[1:]
    base.update(best_circular_lag_rad=float(lag),amplitude_mse=float(amp.sum()),phase_mse=float(phase.sum()),dc_mse=float(dc),
        harmonic_coefficient_error=float(np.linalg.norm(cp[1:]-ct[1:])),shape_correlation=float(np.corrcoef(p,t)[0,1]),
        pred_harmonic_rms=float(np.sqrt(.5*np.sum(ap*ap))),truth_harmonic_rms=float(np.sqrt(.5*np.sum(at*at))),
        native_residual_mse=float(np.mean(((yp-X@cp)-(yt-X@ct))**2)),
        amplitude_strength_ratio=float(np.linalg.norm(ap)/max(np.linalg.norm(at),1e-12)),
        lag_supported=bool(np.linalg.norm(ap)>=.1*np.linalg.norm(at)))
    rows=[dict(harmonic=k+1,pred_amplitude=ap[k],truth_amplitude=at[k],harmonic_phase_error_rad=theta[k],amplitude_mse=amp[k],phase_mse=phase[k]) for k in range(3)]
    return base,rows


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);a=pa.parse_args();out=a.output
    windows=pd.read_csv(ROOT/'docs/analysis/results/main_v2_free_running_5s/windows.csv');samples=pd.read_parquet(DATA/'samples_validation.parquet')
    b=assemble_history_trajectory_windows(samples,windows,history_steps=26);truth=b.trajectory.truth;dt=b.trajectory.dt_s
    tp=np.unwrap(truth.relative_phase_rad,axis=1);midphase=.5*(tp[:,:-1]+tp[:,1:]);time=np.cumsum(dt,axis=1)
    tq=np.diff(truth.angular_velocity_b[:,:,1],axis=1)/dt;tz=np.diff(truth.velocity_n[:,:,2],axis=1)/dt
    models={'S0':ROOT/'artifacts/main_v2_increment_supervision/S0','S1':ROOT/'artifacts/main_v2_increment_supervision/S1'}
    if (out/'full_models.json').exists():models.update({k:Path(v) for k,v in json.loads((out/'full_models.json').read_text()).items()})
    lagrows=[];harmonics=[];freqrows=[];freqroll=[]
    for name,path in models.items():
        trace=np.load(path/'validation_rollout.npz');np.testing.assert_array_equal(trace['window_ids'],windows.window_id.to_numpy(str))
        pq=np.diff(trace['angular_velocity_b'][:,:,1],axis=1)/dt;pz=np.diff(trace['velocity_n'][:,:,2],axis=1)/dt
        for i,row in enumerate(windows.itertuples()):
            for h,k in HORIZONS.items():
                for signal,p,t in [('q_dot',pq,tq),('az_n',pz,tz)]:
                    result,parts=alignment(midphase[i,:k],p[i,:k],t[i,:k]);meta=dict(model=name,window_id=row.window_id,log_id=row.log_id,horizon_s=h,signal=signal)
                    lagrows.append(dict(**meta,**result));harmonics.extend([dict(**meta,**z) for z in parts])
        f=trace['flap_frequency_hz'];tf=truth.flap_frequency_hz;e=.5*((f[:,:-1]-tf[:,:-1])+(f[:,1:]-tf[:,1:]));pf=np.unwrap(trace['relative_phase_rad'],axis=1)
        phaseerror=(pf[:,1:]-pf[:,:1])-(tp[:,1:]-tp[:,:1]);integ=2*np.pi*np.cumsum(e*dt,axis=1)
        closure=2*np.pi*np.cumsum(.5*(tf[:,:-1]+tf[:,1:])*dt,axis=1)-(tp[:,1:]-tp[:,:1])
        for log in windows.log_id.unique():
            sel=windows.log_id.to_numpy()==log;d=dt[sel];err=e[sel];const=float(np.sum(err*d)/np.sum(d))
            # Offline decomposition: centered ~1s local average, NEVER predictor input.
            slow=uniform_filter1d(err-const,size=51,axis=1,mode='nearest');noise=err-const-slow
            mean=np.sum(noise*d,axis=1)/np.sum(d,axis=1);slow+=mean[:,None];noise-=mean[:,None]
            ci=2*np.pi*const*np.cumsum(d,axis=1);si=2*np.pi*np.cumsum(slow*d,axis=1);ni=2*np.pi*np.cumsum(noise*d,axis=1)
            for h,k in HORIZONS.items():
                ph=phaseerror[sel,k-1];ie=integ[sel,k-1]
                freqrows.append(dict(model=name,log_id=log,horizon_s=h,mean_frequency_error_hz=const,slow_rms_hz=float(np.sqrt(np.mean(slow[:,:k]**2))),zero_mean_residual_rms_hz=float(np.sqrt(np.mean(noise[:,:k]**2))),
                    frequency_integral_rmse_rad=float(np.sqrt(np.mean(ie**2))),phase_drift_unwrapped_rmse_rad=float(np.sqrt(np.mean(ph**2))),phase_drift_circular_rmse_rad=float(np.sqrt(np.mean(circular(ph)**2))),
                    integral_vs_phase_correlation=float(np.corrcoef(ie,ph)[0,1]),encoder_frequency_closure_rmse_rad=float(np.sqrt(np.mean(closure[sel,k-1]**2))),
                    decomposition_identity_max_rad=float(np.max(np.abs(phaseerror[sel,:k]-integ[sel,:k]-closure[sel,:k])))))
                for j,window in enumerate(windows[sel].window_id):freqroll.append(dict(model=name,window_id=window,log_id=log,horizon_s=h,
                    constant_bias_integral_rad=ci[j,k-1],slow_bias_integral_rad=si[j,k-1],zero_mean_residual_integral_rad=ni[j,k-1],
                    total_frequency_error_integral_rad=ie[j],phase_error_rad=ph[j],encoder_closure_rad=closure[sel,k-1][j]))
        print('LAG/BIAS',name,'complete',flush=True)
    pd.DataFrame(lagrows).to_csv(out/'phase_lag_vs_horizon.csv',index=False);pd.DataFrame(harmonics).to_csv(out/'harmonic_alignment.csv',index=False)
    pd.DataFrame(freqrows).to_csv(out/'frequency_bias_analysis.csv',index=False);pd.DataFrame(freqroll).to_csv(out/'frequency_integral_per_rollout.csv',index=False)
    estimates=pd.read_csv(out/'phase_offset_estimation.csv');stability=[]
    for keys,g in estimates.groupby(['split','method','requested_history_steps','log_id','segment_id']):
        g=g.sort_values('timestamp_s');jumps=np.abs(circular(np.diff(g.offset_rad)));elapsed=np.diff(g.timestamp_s)
        if not len(jumps):continue
        stability.append(dict(split=keys[0],method=keys[1],history_steps=keys[2],log_id=keys[3],segment_id=keys[4],n=len(g),
            median_circular_jump_rad=float(np.median(jumps)),p95_circular_jump_rad=float(np.quantile(jumps,.95)),jump_over_45deg_fraction=float(np.mean(jumps>np.pi/4)),
            median_offset_rate_rad_s=float(np.median(jumps/elapsed))))
    pd.DataFrame(stability).to_csv(out/'offset_stability.csv',index=False)
    # Preserve frozen common-suite baselines; no phase intervention in these traces.
    prior=ROOT/'docs/analysis/results/main_v2_increment_supervision'
    for file in ['free_running_per_horizon.csv','variation_summary.csv','regime_summary.csv']:
        pd.read_csv(prior/file).to_csv(out/file,index=False)
    (out/'diagnostic_contract.json').write_text(json.dumps(dict(
        phase_fit='3 harmonics+DC; >=1 observed cycle, condition number<1000; otherwise lag not identifiable and left NaN; lag_supported also requires predicted harmonic RMS >=10percent truth, a descriptive weak-signal flag',
        lag_sign='positive lag means prediction must be evaluated at phi+lag to match truth; circular/periodic aliases remain possible',
        decomposition='exact harmonic-domain MSE = amplitude + phase + DC on uniform full phase grid; native residual is separate, not claimed additive under nonuniform partial coverage',
        frequency_slow='offline centered51-native-interval boxcar (~1s); residual demeaned per rollout; NOT an input/correction or proof of white measurement noise',
        autonomous_contract='no encoder update after t0; online observed-phase predictor and RL internal phase are different use cases',sealed_test_opened=False),indent=2))

if __name__=='__main__':main()
