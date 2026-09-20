"""Stage 2: causal log replay diagnostics and frozen train-only input-event screen.

This is a diagnostic approximation, not an equivalent PX4 controller or a causal
plant identification experiment. Never reads sealed/test or September 19 logs.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[1]
PX4 = Path('/home/zn/PX4-Autopilot')
LOGS = Path('/home/zn/QgcLogs')
SHA = 'e624a99f2955addbf76681e636c44162a0c03055'
CHANNELS = ['symmetric_tail', 'differential_tail', 'motor', 'rudder']


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def hold(data, fields, ref, clock='timestamp'):
    """Backward as-of only. Return NaN before first sample and source age."""
    t = data[clock].astype(np.int64)
    assert np.all(np.diff(t) >= 0)
    ix = np.searchsorted(t, ref, side='right') - 1
    values = np.column_stack([data[f][np.maximum(ix, 0)] for f in fields]).astype(float)
    values[ix < 0] = np.nan
    age = (ref - t[np.maximum(ix, 0)]) / 1e6
    age[ix < 0] = np.inf
    return values, age


def cols(d, stem, n=3):
    return np.column_stack([d[f'{stem}[{i}]'] for i in range(n)]).astype(float)


def metric(rows, log, stage, predicted, actual, mask=None, unit='normalized'):
    pred = np.asarray(predicted); obs = np.asarray(actual)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None: valid &= mask
    err = (pred - obs)[valid]
    rows.append(dict(log=log, diagnostic=stage, n=int(valid.sum()), n_total=int(valid.size),
                     rmse=float(np.sqrt(np.mean(err**2))) if len(err) else None,
                     p95_abs=float(np.quantile(abs(err), .95)) if len(err) else None,
                     max_abs=float(abs(err).max()) if len(err) else None, unit=unit))


def source_snapshot(out):
    paths = ['src/lib/rate_control/rate_control.cpp', 'src/lib/rate_control/gain_compression.cpp',
             'src/lib/rate_control/gain_compression.hpp', 'src/lib/mathlib/math/filter/AlphaFilter.hpp',
             'src/modules/fw_rate_control/FixedwingRateControl.cpp', 'src/modules/fw_rate_control/FixedwingRateControl.hpp',
             'src/lib/control_allocation/control_allocation/ControlAllocationPseudoInverse.cpp',
             'src/lib/control_allocation/actuator_effectiveness/ActuatorEffectiveness.hpp',
             'src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessFixedWing.cpp',
             'src/lib/mixer_module/mixer_module.cpp', 'src/lib/mixer_module/functions/FunctionMotors.hpp']
    result = {}
    for path in paths:
        data = subprocess.check_output(['git', '-C', str(PX4), 'show', f'{SHA}:{path}'])
        p = out/'firmware_sources'/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        result[path] = sha(p)
    return result


def replay(out):
    metrics=[]; timing=[]; coverage=[]; inputs=[]
    names = ['rate_ctrl_terms', 'rate_ctrl_status', 'vehicle_rates_setpoint', 'vehicle_angular_velocity',
             'vehicle_torque_setpoint', 'actuator_servos', 'actuator_outputs', 'actuator_motors',
             'airspeed_validated', 'gain_compression', 'vehicle_control_mode', 'vehicle_land_detected',
             'vehicle_status', 'control_allocator_status']
    for p in sorted((LOGS/'9.18数据').glob('*.ulg')):
        if p.name.startswith('._'): continue
        inputs.append(dict(path=str(p), sha256=sha(p)))
        u=ULog(str(p),message_name_filter_list=names)
        assert u.msg_info_dict['ver_sw']==SHA and not u.changed_parameters
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}; par=u.initial_parameters
        for name,d in ds.items():
            dt=np.diff(d['timestamp'].astype(float))/1e6
            timing.append(dict(log=p.name, topic=name, samples=len(d['timestamp']), median_dt_s=float(np.median(dt)), max_dt_s=float(dt.max())))
        term=ds['rate_ctrl_terms']; ref=term['timestamp'].astype(np.int64); sample=term['timestamp_sample'].astype(np.int64)
        omega,age=hold(ds['vehicle_angular_velocity'],[f'xyz[{i}]' for i in range(3)],sample,clock='timestamp_sample')
        rate,rate_age=hold(ds['vehicle_rates_setpoint'],['roll','pitch','yaw'],ref)
        P=np.array([par[f'FW_{a}R_P'] for a in ['R','P','Y']]); FF=np.array([par[f'FW_{a}R_FF'] for a in ['R','P','Y']])
        for axis in range(3):
            metric(metrics,p.name,f'term_sum_axis{axis}',sum(term[f'{x}_term[{axis}]'] for x in ['p','i','d','ff']),term[f'output[{axis}]'])
            metric(metrics,p.name,f'P_from_recorded_rate_axis{axis}',P[axis]*(rate[:,axis]-omega[:,axis]),term[f'p_term[{axis}]'],(age<=.011)&(rate_age<=.05))
        coverage.append(dict(log=p.name, rate_terms=len(ref), exact_logged_gyro_sample_fraction=float((age==0).mean()), rate_setpoint_age_p95_ms=float(np.quantile(rate_age[np.isfinite(rate_age)],.95)*1000)))
        # Run the pitch rate loop on recorded gyro publication updates. No subsequent
        # real I/output/gain feedback; warm-start I ONCE at first available term.
        gyro=ds['vehicle_angular_velocity']; tt=gyro['timestamp'].astype(np.int64)
        sp,spa=hold(ds['vehicle_rates_setpoint'],['pitch'],tt)
        air,aira=hold(ds['airspeed_validated'],['calibrated_airspeed_m_s'],tt)
        mode,ma=hold(ds['vehicle_control_mode'],['flag_control_rates_enabled'],tt)
        land,la=hold(ds['vehicle_land_detected'],['landed'],tt)
        reset,_=hold(ds['vehicle_rates_setpoint'],['reset_integral'],tt)
        sat,sata=hold(ds['control_allocator_status'],['unallocated_torque[1]'],tt)
        assert par['FW_PR_D']==0 and par['TRIM_PITCH']==0
        assert par['FW_DTRIM_P_VMIN']==0 and par['FW_DTRIM_P_VMAX']==0
        n=len(tt); pred=np.full(n,np.nan); ints=np.full(n,np.nan); gains=np.full(n,np.nan); scales=np.full(n,np.nan)
        integral=float(term['i_term[1]'][0]); gain=1.; hpf=0.; lpf=0.; prev=0.; af=par['FW_AIRSPD_TRIM']
        previous_sample=None; start_time=None
        for i in range(n):
            if tt[i]<ref[0] or not np.isfinite(sp[i,0]): continue
            if start_time is None: start_time=tt[i]
            dt=.01 if previous_sample is None else np.clip((float(gyro['timestamp_sample'][i])-previous_sample)/1e6,.002,.04)
            previous_sample=float(gyro['timestamp_sample'][i])
            if mode[i,0]!=1:
                integral=0.; gain=1.; continue
            airborne=land[i,0]==0
            if not airborne or reset[i,0]==1: integral=0.; gain=1.
            if par['FW_USE_AIRSPD'] and aira[i]<1 and np.isfinite(air[i,0]):
                af+=dt/(1+dt)*(max(.5,air[i,0])-af); speed=af
            else: speed=par['FW_AIRSPD_TRIM']
            scale=par['FW_AIRSPD_TRIM']/max(speed,par['FW_AIRSPD_STALL']) if par['FW_ARSP_SCALE_EN'] else 1.
            error=sp[i,0]-gyro['xyz[1]'][i]
            output=P[1]*error+integral+FF[1]/scale*sp[i,0]
            effort=gain*output*scale**2
            pred[i]=np.clip(effort,-1,1); ints[i]=integral; gains[i]=gain; scales[i]=scale
            if airborne:
                ei=error
                if sat[i,0]>np.finfo(np.float32).eps: ei=min(ei,0.)
                if sat[i,0]<-np.finfo(np.float32).eps: ei=max(ei,0.)
                integral=np.clip(integral+max(0.,1-(ei/np.deg2rad(400))**2)*par['FW_PR_I']*ei*dt,-par['FW_PR_IMAX'],par['FW_PR_IMAX'])
            if par['FW_GC_EN']:
                alpha=1/(1+2*np.pi*10*dt)
                hpf=alpha*hpf+alpha*(effort-prev); prev=effort
                lpf+=dt/(1/(2*np.pi*5)+dt)*(hpf*hpf-lpf)
                gain=np.clip(gain+(-200*max(gain-par['FW_GC_GAIN_MIN'],0)*lpf+.1*(1-gain))*dt,par['FW_GC_GAIN_MIN'],1)
            else: gain=1.
        sim={'timestamp':tt,'pitch':pred,'integral':ints,'gain':gains,'scale':scales}
        # Recorded target output compared at target publication timestamps (backward only).
        tor=ds['vehicle_torque_setpoint']; pr,pr_age=hold(sim,['pitch'],tor['timestamp'])
        md,_=hold(ds['vehicle_control_mode'],['flag_control_rates_enabled'],tor['timestamp'])
        mask=(md[:,0]==1)&(pr_age<=.02)&(tor['timestamp']>=ref[0]+5_000_000)
        metric(metrics,p.name,'pitch_stateful_recorded_grid',pr[:,0],tor['xyz[1]'],mask)
        pi,_=hold(sim,['integral','gain','scale'],ref)
        metric(metrics,p.name,'pitch_integral_stateful',pi[:,0],term['i_term[1]'],ref>=ref[0]+5_000_000)
        gc,gca=hold(ds['gain_compression'],['compression_gains[1]'],ref)
        metric(metrics,p.name,'pitch_gain_stateful',pi[:,1],gc[:,0],(ref>=ref[0]+5_000_000)&(gca<=.2))
        # Conditional diagnostic uses measured internal I and gain, NOT autonomous replay.
        cond=P[1]*(rate[:,1]-omega[:,1])+term['i_term[1]']+FF[1]/pi[:,2]*rate[:,1]
        conditional=np.clip(gc[:,0]*cond*pi[:,2]**2,-1,1)
        # Same controller gyro sample ID locates torque produced in that update.
        ts=tor['timestamp_sample'].astype(np.int64); ix=np.searchsorted(ts,sample); ix=np.minimum(ix,len(ts)-1)
        exact=(ts[ix]==sample)&(ref>=ref[0]+5_000_000)&(gca<=.2)&(age<=.011)&(rate_age<=.05)
        metric(metrics,p.name,'pitch_conditional_logged_I_gain',conditional,tor['xyz[1]'][ix],exact)
        # Fixed-wing pseudo-inverse, non-normalized R/P, no flaps or trim.
        sv=ds['actuator_servos']; torque,ta=hold(tor,[f'xyz[{i}]' for i in range(3)],sv['timestamp'])
        rcoef=par['CA_SV_CS1_TRQ_R']; pcoef=par['CA_SV_CS0_TRQ_P']
        allocated=np.clip(np.column_stack([torque[:,1]/(2*pcoef)-torque[:,0]/(2*rcoef),torque[:,1]/(2*pcoef)+torque[:,0]/(2*rcoef),torque[:,2]]),-1,1)
        for axis in range(3): metric(metrics,p.name,f'allocation_servo{axis}',allocated[:,axis],sv[f'control[{axis}]'],ta<=.03)
        # PWM midpoint mapping uses commanded actuator value, not physical angle.
        pwm=ds['actuator_outputs']; sr,sra=hold(sv,[f'control[{i}]' for i in range(3)],pwm['timestamp'])
        for axis,channel in enumerate([1,2,5]):
            sign=-1 if par['PWM_MAIN_REV']&(1<<(channel-1)) else 1
            low=par[f'PWM_MAIN_MIN{channel}']; high=par[f'PWM_MAIN_MAX{channel}']
            expected=(low+high)/2+sign*sr[:,axis]*(high-low)/2
            metric(metrics,p.name,f'PWM_MAIN{channel}_from_servo',expected,pwm[f'output[{channel-1}]'],sra<=.03,unit='microseconds')
        motor,moage=hold(ds['actuator_motors'],['control[0]'],pwm['timestamp'])
        assert par['THR_MDL_FAC']==0
        motor_pwm=par['PWM_MAIN_MIN3']+motor[:,0]*(par['PWM_MAIN_MAX3']-par['PWM_MAIN_MIN3'])
        metric(metrics,p.name,'PWM_MAIN3_from_motor',motor_pwm,pwm['output[2]'],(moage<=.03)&(motor[:,0]>=0),unit='microseconds')
        traces=pd.DataFrame({'timestamp_us':tor['timestamp'],'pitch_torque_recorded':tor['xyz[1]'],'pitch_torque_replay':pr[:,0],'scored':mask})
        traces.to_csv(out/f'{p.stem}_pitch_replay.csv',index=False)
        fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
        t=(tor['timestamp'].astype(float)-float(tor['timestamp'][0]))/1e6
        ax[0].plot(t,tor['xyz[1]'],lw=.5,label='recorded'); ax[0].plot(t,pr[:,0],lw=.5,label='stateful pitch replay'); ax[0].legend(); ax[0].set_ylabel('pitch torque command')
        ax[1].plot(t,(pr[:,0]-tor['xyz[1]']),lw=.5); ax[1].set_ylabel('replay - recorded'); ax[1].set_xlabel('seconds from log start'); fig.tight_layout(); fig.savefig(out/f'{p.stem}_pitch_replay.png',dpi=150); plt.close(fig)
        print('replay',p.name,flush=True)
    pd.DataFrame(metrics).to_csv(out/'controller_replay_metrics.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'controller_topic_timing.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'controller_alignment_coverage.csv',index=False)
    return inputs


def verify_dataset(out):
    reg=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text())
    entry=reg['datasets'][reg['default_dataset_id']]; mp=ROOT/entry['manifest_path']
    assert sha(mp)==entry['manifest_sha256']; m=json.loads(mp.read_text())
    verified={}
    for name,h in m['artifact_sha256'].items():
        assert 'test' not in name; actual=sha(mp.parent/name); assert actual==h; verified[name]=actual
    dump(out/'dataset_provenance.json',dict(dataset_id=reg['default_dataset_id'],manifest=str(mp),manifest_sha256=sha(mp),artifacts=verified,partitions=['train','validation'],frames=m['frames'],frequency_contract=m['frequency_contract'],phase_contract='logged_flap_phase_rad with valid_logged_phase; physical zero unconfirmed'))
    return mp.parent


def candidates(frame, partition, manifest, raw_records):
    """Every 0.2 s, inspect [-0.5,+1] without crossing log boundaries.
    Selection only uses flags, input commands, and current operating state.
    """
    rows=[]; curves={}; trajectory_id=0
    for log,g in frame.groupby('log_id',sort=False):
        g=g.sort_values('timestamp_us').reset_index(drop=True); t=g.timestamp_us.to_numpy()/1e6
        assert log in manifest['split_contract']['assignments'][partition]
        assert partition in ['train', 'validation']
        raw=LOGS/log; actual_hash=sha(raw)
        assert actual_hash==manifest['source']['ulog_sha256'][log]
        raw_records.append(dict(partition=partition,path=str(raw),sha256=actual_hash))
        needed=['vehicle_local_position','vehicle_attitude','vehicle_angular_velocity','actuator_servos','actuator_motors','flap_frequency','wing_phase']
        ul=ULog(str(raw),message_name_filter_list=needed)
        topic_times={d.name:d.data['timestamp'].astype(float)/1e6 for d in ul.data_list if d.multi_id==0}
        assert set(needed)<=set(topic_times)
        source_stale=np.zeros(len(t),dtype=bool)
        for topic,ts in topic_times.items():
            ix=np.searchsorted(ts,t,side='right')-1
            source_stale|=(ix<0)|(t-ts[np.maximum(ix,0)]>.05)
        raw_gaps={name:ts[1:][np.diff(ts)>.05] for name,ts in topic_times.items()}
        u=np.column_stack([(g.control_left_elevon_normalized+g.control_right_elevon_normalized)/2,
                           (g.control_right_elevon_normalized-g.control_left_elevon_normalized)/2,
                           g.control_flap_motor_normalized,g.control_rudder_normalized])
        rates=g[[f'angular_velocity_body_rad_s_{a}' for a in 'xyz']].to_numpy()
        freq=g.flap_frequency_hz.to_numpy(); speed=np.linalg.norm(g[[f'velocity_ned_m_s_{a}' for a in 'xyz']].to_numpy(),axis=1)
        q=g[['attitude_q_w','attitude_q_x','attitude_q_y','attitude_q_z']].to_numpy(); pitch=np.rad2deg(np.arcsin(np.clip(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1)))
        required=(g.valid_core & g.valid_logged_phase & g.valid_state & g.valid_control & g.valid_airborne_safe).to_numpy()
        boundary=(g.reset_boundary|g.mode_boundary|g.gap_boundary).to_numpy(); nav=g.nav_state.to_numpy()
        next_center=t[0]+.5
        for i in range(len(g)):
            if t[i]<next_center: continue
            next_center=t[i]+.2
            a=np.searchsorted(t,t[i]-.5); b=np.searchsorted(t,t[i]+1,side='right')
            reason=[]
            if a==0 or b==len(g) or t[b-1]-t[i]<.97: reason.append('incomplete_window')
            if not required[a:b].all(): reason.append('invalid_state_control_phase_or_airborne')
            if boundary[a:b].any() or not np.all(nav[a:b]==nav[i]): reason.append('mode_reset_or_gap_boundary')
            if b-a<3 or np.any(np.diff(t[a:b])>.05): reason.append('state_gap')
            if source_stale[a:b].any(): reason.append('raw_source_age_over_50ms')
            if any(np.any((ends>=t[a])&(ends<=t[b-1])) for ends in raw_gaps.values()): reason.append('raw_source_gap_over_50ms')
            # Phase publication freshness: dataset allows up to 0.1; tighten here.
            if np.any((g.timestamp_us.to_numpy()[a:b]-g.phase_source_timestamp_us.to_numpy()[a:b])>50_000): reason.append('stale_phase')
            pre=(t>=t[i]-.2)&(t<t[i]); post=(t>=t[i])&(t<t[i]+.2)
            tail=(t>=t[i]+.2)&(t<t[i]+.5); window=(t>=t[i]-.2)&(t<t[i]+1)
            if not pre.any() or not post.any() or not tail.any(): continue
            with np.errstate(invalid='ignore'):
                base=np.mean(u[pre],axis=0); delta=np.mean(u[post],axis=0)-base
                excursion=np.max(abs(u[window]-base),axis=0)
                pre_range=np.ptp(u[pre],axis=0); hold_delta=np.mean(u[tail],axis=0)-base
            common=dict(id=trajectory_id,partition=partition,log_id=log,timestamp_us=int(g.timestamp_us.iloc[i]),speed=float(speed[i]),pitch_deg=float(pitch[i]),frequency=float(freq[i]),base_reasons='|'.join(reason),base_valid=not reason)
            for j,c in enumerate(CHANNELS):
                common['delta_'+c]=float(delta[j]); common['excursion_'+c]=float(excursion[j]); common['pre_range_'+c]=float(pre_range[j]); common['hold_'+c]=float(hold_delta[j])
            rows.append(common)
            if not reason:
                timeline=np.arange(-.2,1.001,.02)
                curves[trajectory_id]=np.column_stack([np.interp(t[i]+timeline,t[a:b],rates[a:b,j]) for j in range(3)]+[np.interp(t[i]+timeline,t[a:b],freq[a:b])])
            trajectory_id+=1
    return pd.DataFrame(rows),curves


def screen(out, dataset):
    # Train is read first; freeze all thresholds before validation samples are read.
    train=pd.read_parquet(dataset/'samples_train.parquet')
    manifest=json.loads((dataset/'manifest.json').read_text()); raw_records=[]
    tr,tc=candidates(train,'train',manifest,raw_records); healthy=tr.loc[tr.base_valid]
    assert len(healthy)>0
    protocol={'version':1,'center_stride_s':.2,'window_s':[-.5,1.],'command_difference_windows_s':[[-.2,0],[0,.2]],
              'threshold_source':'train base-valid windows only; no response/model predictions used',
              'operating_envelope':{k:[float(healthy[k].quantile(.05)),float(healthy[k].quantile(.95))] for k in ['speed','pitch_deg','frequency']},
              'excitation':{c:float(max(.01,healthy['delta_'+c].abs().quantile(.75))) for c in CHANNELS},
              'other_channel_excursion_max':{c:float(max(.01,healthy['excursion_'+c].quantile(.25))) for c in CHANNELS},
              'pre_range_fraction_max':.5,'hold_signed_delta_fraction_min':.5,'nonoverlap_s':1.5,
              'response_use':'descriptive closed-loop association only; no causal gain/delay identification',
              'not_performance_acceptance_thresholds':True}
    # Preserve thresholds already fixed in the initial screen; this revision only
    # repairs source-freshness verification and the motor response observable.
    initial=ROOT/'docs/analysis/results/control_readiness_stage2_initial_diagnostics/event_protocol_frozen_before_validation.json'
    if initial.exists():
        protocol=json.loads(initial.read_text())
        protocol['threshold_origin_sha256']=sha(initial)
        protocol['integrity_revision']='Add raw topic age/gap <=50ms; motor response is flap frequency. No threshold relaxation; initial validation already inspected.'
    dump(out/'event_protocol_frozen_before_validation.json',protocol)
    val=pd.read_parquet(dataset/'samples_validation.parquet'); va,vc=candidates(val,'validation',manifest,raw_records)
    dump(out/'response_raw_log_provenance.json',raw_records)
    results=[]; summaries=[]; responses=[]; plotcurves={}
    for rows,curves in [(tr,tc),(va,vc)]:
        for channel in CHANNELS:
            last={}; accepted=[]
            for r in rows.to_dict('records'):
                why=[] if r['base_valid'] else [r['base_reasons']]
                if any(not lo<=r[k]<=hi for k,(lo,hi) in protocol['operating_envelope'].items()): why.append('outside_train_envelope')
                delta=r['delta_'+channel]
                if not np.isfinite(delta) or abs(delta)<protocol['excitation'][channel]: why.append('small_excitation')
                for other in CHANNELS:
                    if other!=channel and (not np.isfinite(r['excursion_'+other]) or r['excursion_'+other]>protocol['other_channel_excursion_max'][other]): why.append('coupled_'+other)
                if not np.isfinite(r['pre_range_'+channel]) or r['pre_range_'+channel]>.5*abs(delta): why.append('unstable_pre_command')
                if not np.isfinite(r['hold_'+channel]) or r['hold_'+channel]*np.sign(delta)<.5*abs(delta): why.append('not_held')
                if not why and (r['timestamp_us']-last.get(r['log_id'],-1e12))<1_500_000: why.append('overlap')
                keep=not why
                if keep: last[r['log_id']]=r['timestamp_us']; accepted.append(r)
                results.append(dict(**r,channel=channel,accepted=keep,reasons='|'.join(why)))
            partition=rows.partition.iloc[0]
            summaries.append(dict(partition=partition,channel=channel,centers=len(rows),base_valid=int(rows.base_valid.sum()),accepted=len(accepted),logs=len({r['log_id'] for r in accepted})))
            axis={'symmetric_tail':1,'differential_tail':0,'motor':3,'rudder':2}[channel]
            for r in accepted:
                curve=curves[r['id']][:,axis]; base=np.mean(curve[:10]); aligned=(curve-base)*np.sign(r['delta_'+channel])
                plotcurves.setdefault((partition,channel),[]).append(aligned)
                for horizon in [.1,.2,.5,1.]:
                    j=int(round((horizon+.2)/.02)); response=float(curve[j]-base)
                    responses.append(dict(partition=partition,channel=channel,log_id=r['log_id'],timestamp_us=r['timestamp_us'],horizon_s=horizon,command_delta=r['delta_'+channel],rate_axis=('x','y','z','flap_frequency_hz')[axis],rate_change=response,apparent_gain=response/r['delta_'+channel]))
    pd.DataFrame(results).to_csv(out/'event_candidates_all.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'event_counts.csv',index=False)
    pd.DataFrame(responses,columns=['partition','channel','log_id','timestamp_us','horizon_s','command_delta','rate_axis','rate_change','apparent_gain']).to_csv(out/'event_responses.csv',index=False)
    rejection=pd.DataFrame(results).assign(reason_list=lambda d:d.reasons.str.split('|')).explode('reason_list')
    rejection.groupby(['partition','channel','reason_list'],dropna=False).size().rename('count').reset_index().to_csv(out/'event_rejections.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for c,ax in zip(CHANNELS,axes.ravel()):
        found=False
        for part,color in [('train','tab:blue'),('validation','tab:orange')]:
            data=plotcurves.get((part,c),[])
            if not data: continue
            found=True; arr=np.array(data); t=np.arange(-.2,1.001,.02)
            for line in arr: ax.plot(t,line,color=color,alpha=.12,lw=.6)
            ax.plot(t,np.median(arr,axis=0),color=color,label=f'{part} n={len(arr)}')
        ax.axvline(0,color='gray',lw=.5); ax.set_title(c); ax.set_ylabel('signed frequency change (Hz)' if c=='motor' else 'signed rate change (rad/s)')
        if found: ax.legend()
        else: ax.text(.5,.5,'No qualifying events',ha='center',transform=ax.transAxes)
        ax.set_xlabel('seconds relative to command window start')
    fig.suptitle('All qualifying closed-loop events; descriptive, not causal step responses'); fig.tight_layout(); fig.savefig(out/'event_responses.png',dpi=150); plt.close(fig)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    return summaries


def self_check():
    d={'timestamp':np.array([100,200,400]),'x':np.array([1.,2.,4.])}
    v,a=hold(d,['x'],np.array([50,100,199,200,399]))
    assert np.isnan(v[0,0]) and np.array_equal(v[1:,0],[1,1,2,2])
    assert np.isinf(a[0]) and np.all(a[1:]>=0)
    # Independent expected inverse for pure roll and pitch; reversal affects PWM only.
    E=np.array([[-.55,.55,0],[1,1,0],[0,0,1.]])
    for target in [[.11,0,0],[0,.2,0],[0,0,.1]]:
        r,p,y=target; act=np.array([p/2-r/1.1,p/2+r/1.1,y]); assert np.allclose(E@act,target)
    print('causal join and actuator mapping checks passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=ROOT/'docs/analysis/results/control_readiness_stage2'); parser.add_argument('--self-check',action='store_true'); args=parser.parse_args()
    if args.self_check: self_check(); return
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    dump(out/'scope.json',dict(script_sha256=sha(Path(__file__)),python=sys.executable,target_commit=SHA,raw_log_scope='9.18 controller replay plus registered September train/validation source timing',dataset_scope='registered September train/validation only',sealed_test_opened=False,september19_opened=False,neural_inference=False))
    source=source_snapshot(out); dataset=verify_dataset(out)
    summaries=screen(out,dataset)
    inputs=replay(out)
    self_check()
    dump(out/'run_manifest.json',dict(script_sha256=sha(Path(__file__)),source_hashes=source,raw_inputs=inputs,events=summaries,completed=True,sealed_test_opened=False,september19_opened=False))

if __name__=='__main__': main()
