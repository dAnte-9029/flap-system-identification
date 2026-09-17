#!/usr/bin/env python3
"""Step 4 train/validation-only phase, target, observability and sensitivity audit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-step4')
import sys,json,argparse,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from scipy.signal import welch
from main_v2_step4_tools import harmonic_design,offline_derivatives,integrate_intervals,vector_r2,nearest_indices
from run_main_v2_free_running import load_simulator,make_initial,sha
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.trajectory import VELOCITY_COLUMNS,BODY_RATE_COLUMNS,QUATERNION_COLUMNS,CONTROL_COLUMNS
from system_identification.models.trajectory_main_v1 import _rotation_body_to_ned
from pyulog import ULog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['p','q','r','ax_n','ay_n','az_n','p_dot','q_dot','r_dot']


def describe(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),p1=float(np.quantile(a,.01)),median=float(np.median(a)),p99=float(np.quantile(a,.99)),max=float(a.max())) if len(a) else dict(count=0)


def csv(out,name,rows):pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)


def load_segments(split):
    frame=pd.read_parquet(ROOT/f'dataset/trajectory_v1_august_f5_c4/samples_{split}.parquet')
    groups=[]
    for (log,seg),g in frame[frame.valid_core].groupby(['log_id','segment_id'],sort=True):
        g=g.sort_values('sample_in_segment').reset_index(drop=True)
        if len(g)<100:continue
        t=g.timestamp_us.to_numpy()*1e-6;y=g[list(VELOCITY_COLUMNS)+list(BODY_RATE_COLUMNS)].to_numpy()
        groups.append(dict(split=split,log=log,seg=seg,frame=g,t=t,y=y,derivatives=offline_derivatives(t,y)))
    return frame,groups


def phase_target_audit(out,parts):
    curves=[];harmonics=[];integration=[];dstat=[];durations=[];phase_windows=[];offset_data={}
    for split,(_,groups) in parts.items():
        for g in groups:
            t,y,frame=g['t'],g['y'],g['frame'];dt=np.diff(t);der=g['derivatives']
            for lag in [1,2,5,10,25]:durations.append(dict(split=split,log_id=g['log'],segment_id=g['seg'],steps=lag,**describe(t[lag:]-t[:-lag])))
            for name,a in der.items():
                for axis in range(6):dstat.append(dict(split=split,log_id=g['log'],segment_id=g['seg'],target=name,axis=axis,**describe(a[25:-25,axis])))
                for lag in [2,5,10,25]:
                    pred=integrate_intervals(a,dt,lag)[25:-25];true=(y[lag:]-y[:-lag])[25:-25]
                    for label,sl in [('velocity',slice(0,3)),('omega',slice(3,6))]:
                        err=pred[:,sl]-true[:,sl]
                        integration.append(dict(split=split,log_id=g['log'],segment_id=g['seg'],target=name,steps=lag,signal=label,count=len(err),rmse=float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),r2=vector_r2(pred[:,sl],true[:,sl]),truth_increment_rms=float(np.sqrt(np.mean(np.sum(true[:,sl]**2,axis=1))))))
        for log in sorted({g['log'] for g in groups}):
            lg=[g for g in groups if g['log']==log];cycles=[];cycle_ids=[];cycle_times=[]
            # Full cycles sampled at fixed phase; no interpolation across segments.
            for g in lg:
                phi=g['frame'].relative_flap_phase_unwrapped_rad.to_numpy();d=g['derivatives']['D0_raw']
                vals=np.column_stack([g['y'][:-1,3:],d[:,:3],d[:,3:]])
                for cycle in range(int(np.ceil(phi[25]/(2*np.pi))),int(np.floor(phi[-26]/(2*np.pi)))):
                    grid=(cycle+(np.arange(24)+.5)/24)*2*np.pi
                    ix=np.searchsorted(phi,grid)
                    if (ix<1).any() or (ix>=len(vals)).any() or np.max(np.diff(phi[max(0,ix[0]-1):ix[-1]+1]))>np.pi:continue
                    cycles.append(np.column_stack([np.interp(grid,phi[:-1],v) for v in vals.T]));cycle_ids.append((g['seg'],cycle));cycle_times.append(float(np.interp(grid[0],phi,g['t'])))
            a=np.asarray(cycles)
            if len(a)<20:continue
            # Ten-cycle blocks for uncertainty (not independent sample CI).
            blocks=np.array([a[i:i+10].mean(0) for i in range(0,len(a)-9,10)])
            rng=np.random.default_rng(415);boot=np.array([blocks[rng.integers(len(blocks),size=len(blocks))].mean(0) for _ in range(300)])
            low,high=np.quantile(boot,[.025,.975],axis=0)
            for b in range(24):
                for j,n in enumerate(NAMES):curves.append(dict(split=split,log_id=log,phase_bin=b,phase_rad=(b+.5)*2*np.pi/24,signal=n,mean=float(a[:,b,j].mean()),std=float(a[:,b,j].std()),ci_low=low[b,j],ci_high=high[b,j],n_cycles=len(a),n_blocks=len(blocks)))
            ph=np.tile((np.arange(24)+.5)*2*np.pi/24,len(a));yy=a.reshape(-1,9);train=np.repeat(np.arange(len(a))<len(a)//2,24)
            # Fit on earlier cycles, assess later cycles independently for each log.
            for order in [1,2,3]:
                X=harmonic_design(ph,order);coef=np.linalg.lstsq(X[train],yy[train],rcond=None)[0];pred=X@coef
                for j,n in enumerate(NAMES):harmonics.append(dict(split=split,log_id=log,representation='within_log_cycle',order=order,signal=n,fit_r2=vector_r2(pred[train,j:j+1],yy[train,j:j+1]),heldout_r2=vector_r2(pred[~train,j:j+1],yy[~train,j:j+1]),harmonic_variance_fraction=float(np.var(pred[~train,j])/max(np.var(yy[~train,j]),1e-20)),n_cycles=len(a)))
            if split=='train':offset_data[log]=(a[:len(a)//2].mean(0),a[len(a)//2:].mean(0))
            # Current representation A versus stored B on actual original windows.
            win=pd.read_parquet(ROOT/f'dataset/trajectory_v1_august_f5_c4/windows_{split}.parquet');win=win[win.log_id==log]
            phia=[];phib=[];targets=[];times=[]
            lookup={g['seg']:g for g in lg}
            for row in win.itertuples():
                if row.segment_id not in lookup:continue
                g=lookup[row.segment_id];s=int(row.start_sample_in_segment);e=s+50
                if s<25 or e>=len(g['t'])-25:continue
                phi=g['frame'].relative_flap_phase_unwrapped_rad.to_numpy();d=g['derivatives']['D0_raw'];y=g['y']
                phia.extend(phi[s:e]-phi[s]);phib.extend(phi[s:e]);targets.extend(np.column_stack([y[s:e,3:],d[s:e,:3],d[s:e,3:]]));times.extend(g['t'][s:e])
            if not targets:continue
            yy=np.asarray(targets);times=np.asarray(times);cut=np.median(times);fit=times<cut-1;test=times>cut+1
            for representation,ph in [('t0_relative',phia),('within_log_native',phib)]:
                X=harmonic_design(ph,3);coef=np.linalg.lstsq(X[fit],yy[fit],rcond=None)[0];pred=X@coef
                for j,n in enumerate(NAMES):harmonics.append(dict(split=split,log_id=log,representation=representation,order=3,signal=n,fit_r2=vector_r2(pred[fit,j:j+1],yy[fit,j:j+1]),heldout_r2=vector_r2(pred[test,j:j+1],yy[test,j:j+1]),harmonic_variance_fraction=float(np.var(pred[test,j])/max(np.var(yy[test,j]),1e-20)),n_cycles=len(a)))
            phase_windows.append(dict(split=split,log_id=log,n_windows=len(win),t0_encoder_phase_circular_resultant=float(abs(np.mean(np.exp(1j*np.asarray(phib)[::50])))),t0_network_sin=0.,t0_network_cos=1.))
    csv(out,'phase_curves',curves);csv(out,'harmonic_regression',harmonics);csv(out,'integration_consistency',integration);csv(out,'derivative_statistics',dstat);csv(out,'actual_durations',durations);csv(out,'window_phase_origins',phase_windows)
    # One constant offset per TRAIN log, estimated only from its earlier cycles.
    logs=list(offset_data);ref=offset_data[logs[0]][0][:,6:];ref=ref-ref.mean(0);scale=np.maximum(ref.std(0),1e-8);offsets={logs[0]:0};offrows=[]
    for log in logs:
        early,late=offset_data[log];e=early[:,6:]-early[:,6:].mean(0)
        errors=[np.mean(((np.roll(e,k,axis=0)-ref)/scale)**2) for k in range(24)];k=int(np.argmin(errors));offsets[log]=k
        offrows.append(dict(log_id=log,phase_offset_rad=k*2*np.pi/24,fit_scope='train earlier half cycles only',early_mse_before=errors[0],early_mse_after=errors[k]))
    # Compare across-log consistency on later halves; labels never fit validation offsets.
    late=np.array([offset_data[l][1][:,6:] for l in logs]);aligned=np.array([np.roll(offset_data[l][1][:,6:],offsets[l],axis=0) for l in logs])
    for row in offrows:
        row['heldout_cross_log_dispersion_before']=float(np.mean(np.var(late,axis=0)));row['heldout_cross_log_dispersion_after']=float(np.mean(np.var(aligned,axis=0)))
    csv(out,'train_phase_offsets',offrows)
    df=pd.DataFrame(curves)
    for split in parts:
        fig,axes=plt.subplots(3,3,figsize=(14,10))
        for ax,n in zip(axes.flat,NAMES):
            for log,g in df[(df.split==split)&(df.signal==n)].groupby('log_id'):
                ax.plot(g.phase_rad,g['mean'],label=Path(log).stem);ax.fill_between(g.phase_rad,g.ci_low,g.ci_high,alpha=.1)
            ax.set(title=n,xlabel='Log-local encoder phase rad')
        axes[0,0].legend(fontsize=5);fig.suptitle(split+'; each log has an unknown mechanical zero');fig.tight_layout();fig.savefig(out/f'phase_domain_{split}.png',dpi=160);plt.close(fig)
    return dict(train_offset_dispersion_before=float(np.mean(np.var(late,axis=0))),train_offset_dispersion_after=float(np.mean(np.var(aligned,axis=0))))


def telemetry_audit(out,parts,manifest):
    rows=[];availability=[];aligned={};rawhash={}
    topics=['airspeed','airspeed_validated','airspeed_quality_input','airspeed_selector_quality_status','wind','estimator_wind','airspeed_wind','sensor_gps','vehicle_gps_position','vehicle_local_position','actuator_servos','actuator_outputs','actuator_motors','actuator_servos_trim','servo_feedback','esc_status','esc_report','rpm','encoder_count','wing_phase','battery_status','vehicle_angular_velocity','vehicle_acceleration']
    for split,(frame,groups) in parts.items():
        for log in manifest['split_contract']['assignments'][split]:
            path=Path(manifest['source']['qgclogs_root'])/log;rawhash[str(path)]=sha(path);u=ULog(str(path),message_name_filter_list=topics)
            g=frame[(frame.log_id==log)&frame.valid_core];target=g.timestamp_us.to_numpy();features={}
            present={d.name for d in u.data_list}
            for topic in topics:availability.append(dict(split=split,log_id=log,topic=topic,present=topic in present))
            for d in u.data_list:
                t=np.asarray(d.data['timestamp']);ix=np.searchsorted(t,target,side='right')-1;safe=np.clip(ix,0,len(t)-1);age=(target-t[safe])*1e-6
                for field,v in d.data.items():
                    if field.startswith('timestamp'):continue
                    a=np.asarray(v,dtype=float);sel=(ix>=0)&(age>=0)&(age<.5);z=a[safe].copy();z[~sel]=np.nan
                    rows.append(dict(split=split,log_id=log,topic=d.name,instance=d.multi_id,field=field,rate_hz=float(1e6/np.median(np.diff(t))) if len(t)>1 else 0,max_gap_s=float(np.max(np.diff(t))*1e-6) if len(t)>1 else 0,core_fresh_finite_fraction=float(np.isfinite(z).mean()),**describe(z)))
                    if d.multi_id==0 and (d.name,field) in [('battery_status','current_a'),('battery_status','voltage_v'),('rpm','rpm_raw'),('rpm','rpm_estimate'),('airspeed_validated','airspeed_source')]:features[d.name+'__'+field]=z
            aligned[log]=pd.DataFrame(dict(sample_in_log=g.sample_in_log.to_numpy(),**features))
            print('telemetry',split,log,flush=True)
    csv(out,'telemetry_fields',rows);csv(out,'telemetry_availability',availability)
    return aligned,rawhash


def point_data(out,parts,device):
    sim=load_simulator(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',device)
    # A0 was independently retrained and bitwise matches original weights.
    ck=ROOT/'artifacts/main_v2_training_objective_ablation/A0_baseline_retrain/model.pt'
    sim.model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict'])
    data={};teacher=[];sensitivity=[]
    for split,(samples,groups) in parts.items():
        windows=[];meta=[];targets={};histtimes=[]
        for g in groups:
            for i in range(25,len(g['frame'])-26):
                f=g['frame'].iloc[i];windows.append(dict(log_id=g['log'],segment_id=g['seg'],start_sample_in_segment=i,state_sample_count=2,window_id=f"{g['log']}:{g['seg']}:{i}"))
                meta.append(dict(split=split,log_id=g['log'],segment_id=g['seg'],sample_in_segment=i,sample_in_log=int(f.sample_in_log),timestamp_s=g['t'][i],phase=f.relative_flap_phase_rad,frequency=f.flap_frequency_hz,airspeed=f.true_airspeed_m_s,wind_n=f.wind_ned_m_s_n,wind_e=f.wind_ned_m_s_e,airdata_valid=bool(f.valid_airdata)))
                histtimes.append([g['t'][i]-g['t'][i-k+1] for k in [5,13,26]])
                for name,a in g['derivatives'].items():targets.setdefault(name,[]).append(a[i])
                for k in [2,5,10,25]:targets.setdefault('increment'+str(k),[]).append((g['y'][i+k]-g['y'][i])/(g['t'][i+k]-g['t'][i]))
        w=pd.DataFrame(windows);m=pd.DataFrame(meta);b=assemble_history_trajectory_windows(samples,w,history_steps=26)
        features={};pred=[];h=[];proxy=[]
        with torch.inference_mode():
            for start in range(0,len(w),2048):
                ix=np.arange(start,min(start+2048,len(w)));st=make_initial(sim,b,ix,'warm26',device)
                ns,d=sim.step(st,torch.tensor(b.trajectory.controls[ix,0],device=device,dtype=torch.float32),torch.tensor(b.trajectory.dt_s[ix,0],device=device,dtype=torch.float32))
                R=_rotation_body_to_ned(st.quaternion_nb);an=torch.einsum('bij,bj->bi',R,d.acceleration_b)
                pred.append(torch.cat([an,d.angular_acceleration_b],1).cpu().numpy());h.append(st.gru_hidden.cpu().numpy());proxy.append(torch.cat([st.drive_state[:,None],st.tail_state],1).cpu().numpy())
        prediction=np.concatenate(pred);hidden=np.concatenate(h);proxies=np.concatenate(proxy)
        physical=b.history_state_features[:,-1];commands=b.history_controls[:,-1];features['instant']=np.column_stack([physical,commands]);features['actuator']=np.column_stack([physical,commands,proxies])
        for k in [5,13,26]:features['history'+str(k)]=np.column_stack([b.history_state_features[:,-k:].reshape(len(w),-1),b.history_controls[:,-k:].reshape(len(w),-1),proxies])
        features['gru26']=np.column_stack([features['actuator'],hidden])
        phase=np.column_stack([np.sin(m.phase),np.cos(m.phase)])
        features['gru26_phase']=np.column_stack([features['gru26'],phase])
        features['gru26_airdata']=np.column_stack([features['gru26'],m[['airspeed','wind_n','wind_e']].to_numpy()])
        targets={k:np.asarray(v) for k,v in targets.items()}
        for name,target in targets.items():
            for log,g in m.groupby('log_id'):
                idx=g.index.to_numpy()
                for sig,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:
                    p=prediction[idx,sl];z=target[idx,sl]
                    teacher.append(dict(split=split,log_id=log,target=name,signal=sig,n_points=len(idx),rmse=float(np.sqrt(np.mean(np.sum((p-z)**2,axis=1)))),r2=vector_r2(p,z),pred_vector_std=float(np.linalg.norm(p.std(0))),truth_vector_std=float(np.linalg.norm(z.std(0)))))
        # Deterministic equal-log operating points, no outcome-based selection.
        operating=np.concatenate([g.index.to_numpy()[np.linspace(0,len(g)-1,min(100,len(g)),dtype=int)] for _,g in m.groupby('log_id')])
        csv(out,split+'_operating_points',m.loc[operating].assign(point_index=operating))
        for start in range(0,len(operating),128):
            ix=operating[start:start+128];st=make_initial(sim,b,ix,'warm26',device);u=torch.tensor(commands[ix],device=device,dtype=torch.float32);dt=torch.tensor(b.trajectory.dt_s[ix,0],device=device,dtype=torch.float32)
            for channel,du in [('motor',[1,0,0,0]),('tail_sym',[0,1,1,0]),('tail_diff',[0,1,-1,0]),('rudder',[0,0,0,1])]:
                direction=torch.tensor(du,device=device,dtype=torch.float32)[None];eps=.001
                values=[]
                for sign in [-1,1]:
                    state=st;control=u+sign*eps*direction
                    vals=[]
                    for step in range(6):
                        nxt,d=sim.step(state,control,dt)
                        vals.append(torch.cat([d.acceleration_b,d.angular_acceleration_b,((nxt.flap_frequency_hz-state.flap_frequency_hz)/dt)[:,None]],1))
                        state=nxt
                    values.append(torch.stack(vals,1).detach().cpu().numpy())
                jac=(values[1]-values[0])/(2*eps)
                for j,idx in enumerate(ix):
                    for step in [0,1,5]:
                        for axis,val in enumerate(jac[j,step]):sensitivity.append(dict(split=split,log_id=m.loc[idx,'log_id'],point_index=idx,channel=channel,step=step,elapsed_s=float(dt[j])*step,output=['ax_b','ay_b','az_b','p_dot','q_dot','r_dot','frequency_dot'][axis],sensitivity=float(val),motor=float(u[j,0]),tail_magnitude=float(torch.linalg.vector_norm(u[j,1:])),perturbation_outside_unit=bool(((u[j]+eps*direction[0]).abs()>1).any() or ((u[j]-eps*direction[0]).abs()>1).any())))
        m.to_csv(out/(split+'_points.csv'),index=False)
        np.savez_compressed(out/(split+'_diagnostic_arrays.npz'),prediction=prediction,hidden=hidden,proxies=proxies,**{'target_'+k:v for k,v in targets.items()},**{'feature_'+k:v for k,v in features.items()})
        data[split]=dict(meta=m,features=features,targets=targets,prediction=prediction,history_times=np.asarray(histtimes))
        print('teacher points',split,len(w),flush=True)
    csv(out,'teacher_state_metrics',teacher);csv(out,'command_sensitivity',sensitivity)
    return data


def ambiguity(out,data,telemetry,device):
    rows=[];neighbor_rows=[]
    # Fix pools by sample position, not labels. Queries balanced across logs.
    ref=data['train'];ri=np.arange(0,len(ref['meta']),5)
    for split,d in data.items():
        mi=d['meta'];queries=np.concatenate([g.index.to_numpy()[np.linspace(0,len(g)-1,min(200,len(g)),dtype=int)] for _,g in mi.groupby('log_id')])
        for scope in (['same_log','other_log'] if split=='train' else ['train_only']):
            rm=ref['meta'].iloc[ri];qm=mi.iloc[queries]
            same=qm.log_id.to_numpy()[:,None]==rm.log_id.to_numpy()[None,:]
            close=np.abs(qm.timestamp_s.to_numpy()[:,None]-rm.timestamp_s.to_numpy()[None,:])<2.
            forbidden=(~same if scope=='same_log' else same if scope=='other_log' else np.zeros_like(same))|(same&close)
            for rep in d['features']:
                x=ref['features'][rep][ri];q=d['features'][rep][queries]
                # Missing airdata imputed from training only; validity reported separately.
                mean=np.nanmean(x,axis=0);std=np.nanstd(x,axis=0);std[std<1e-5]=1
                x=np.nan_to_num((x-mean)/std);q=np.nan_to_num((q-mean)/std)
                ix,dist=nearest_indices(q,x,forbidden,k=20,device=device)
                # Persist neighbor identity for exact reproducibility.
                for j,query in enumerate(queries):
                    for k in [5,20]:
                        neighbor_rows.append(dict(split=split,scope=scope,representation=rep,query_index=int(query),query_id=f"{mi.iloc[query].log_id}:{mi.iloc[query].segment_id}:{mi.iloc[query].sample_in_segment}",k=k,neighbor_indices=' '.join(map(str,ri[ix[j,:k]])),distance_rms=float(dist[j,:k].mean())))
                for name in ['D0_raw','D1_lp6','increment2','increment5','increment10']:
                    ty=ref['targets'][name][ri];qy=d['targets'][name][queries]
                    for k in [5,20]:
                        neighbors=ty[ix[:,:k]];pred=neighbors.mean(1);conditional=neighbors.var(1,ddof=1)
                        for log,g in qm.reset_index(drop=True).groupby('log_id'):
                            j=g.index.to_numpy()
                            for signal,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:
                                rows.append(dict(split=split,scope=scope,log_id=log,representation=rep,target=name,k=k,signal=signal,n_queries=len(j),distance_median=float(np.median(dist[j,:k])),conditional_variance_ratio=float(np.mean(conditional[j,sl].sum(1))/np.var(ty[:,sl],axis=0).sum()),knn_rmse=float(np.sqrt(np.mean(np.sum((pred[j,sl]-qy[j,sl])**2,axis=1)))),knn_r2=vector_r2(pred[j,sl],qy[j,sl])))
                print('neighbors',split,scope,rep,flush=True)
    csv(out,'conditional_variance',rows);csv(out,'neighbor_ids',neighbor_rows)


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,default=ROOT/'docs/analysis/results/main_v2_dynamics_observability_phase');pa.add_argument('--device',default='cuda:1');args=pa.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    old=ROOT/'docs/analysis/results/main_v2_training_objective_ablation';oldhash={str(p.relative_to(ROOT)):sha(p) for p in old.rglob('*') if p.is_file()}
    protocol=json.loads((old/'protocol.json').read_text())
    for path,h in protocol['baseline_hashes'].items():assert sha(ROOT/path)==h,path
    manifest=json.loads((ROOT/'dataset/trajectory_v1_august_f5_c4/manifest.json').read_text())
    run=dict(stage='analysis',device=args.device,seed=415,sealed_test_opened=False,partitions=['train','validation'],dataset='trajectory_v1_august_f5_c4',baseline_hashes=protocol['baseline_hashes'],step3_hashes=oldhash,filters=dict(usage='offline target diagnosis only; never simulator inputs',uniform_grid_s=.02,butterworth_order=4,zero_phase=True,cutoffs_hz=[4,6,8,12],primary_cutoff_hz=6,edge_exclusion_steps=25),knn=dict(k=[5,20],reference_stride=5,queries_per_log=200,time_exclusion_s=2,normalization='train-only per coordinate, RMS distance',limitations='finite-sample conditional ambiguity, not a Markov impossibility proof'),small_models_not_yet_trained=True)
    (out/'manifest.json').write_text(json.dumps(run,indent=2));parts={s:load_segments(s) for s in ['train','validation']}
    phase=phase_target_audit(out,parts);telemetry,rawhash=telemetry_audit(out,parts,manifest);data=point_data(out,parts,args.device)
    ambiguity(out,data,telemetry,args.device)
    for path,h in oldhash.items():assert sha(ROOT/path)==h,path
    run.update(stage='analysis_completed',phase_offsets=phase,raw_ulog_hashes=rawhash);(out/'manifest.json').write_text(json.dumps(run,indent=2))
    print('ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':main()
