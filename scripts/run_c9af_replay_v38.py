"""Controller-only teacher-state replay; no learned-model predictions or tuning."""
from pathlib import Path
import concurrent.futures
import subprocess
import sys
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from train_joint_increment_v34 import read
from audit_controller_support_v26 import verify_parent
from replay_preflight_v37 import available_index
from system_identification.control.px4_c9af import C9Controller,FIRMWARE

OUT=base.ROOT/'docs/analysis/results/control_windows_v38'
PREV=OUT.parent/'control_windows_v37';DATA=OUT.parent/'control_windows_v29'
TOPICS=['vehicle_attitude','vehicle_attitude_setpoint','vehicle_angular_velocity','vehicle_rates_setpoint',
        'airspeed_validated','manual_control_setpoint','vehicle_control_mode','vehicle_land_detected','vehicle_status',
        'rate_ctrl_status','gain_compression','vehicle_torque_setpoint','actuator_motors','actuator_servos','autotune_attitude_control_status']
VARIANTS=[('cold',0,0,2500,'source_no_interleave'),('partial_plus',0,1,2500,'source_no_interleave'),
          ('partial_minus',0,-1,2500,'source_no_interleave'),('warm2_plus',2,1,2500,'source_no_interleave'),
          ('warm2_minus',2,-1,2500,'source_no_interleave'),('warm2_20ms',2,1,20000,'source_no_interleave'),
          ('warm2_delayed_feedback',2,1,2500,'delayed_allocation')]


def euler(d,prefix):
    q=np.stack([d[f'{prefix}[{j}]'] for j in range(4)],1).astype(float)
    return np.stack([np.arctan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2)),
                     np.arcsin(np.clip(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))],1)


def indices(d,times):
    available=np.maximum(d['timestamp'],d.get('timestamp_sample',d['timestamp'])).astype(np.int64)
    assert (np.diff(available)>=0).all()
    ix=np.searchsorted(available,times,side='right')-1
    if (ix<0).any():raise ValueError('No causal prehistory')
    return ix


def initialize(c,ds,start,sign):
    if sign==0:return
    s=c.state
    d=ds['rate_ctrl_status'];ix=available_index(d,start)
    s.integral=[float(d[axis+'speed_integ'][ix]) for axis in ['roll','pitch','yaw']]
    d=ds['gain_compression'];ix=available_index(d,start)
    s.gain=[float(d[f'compression_gains[{j}]'][ix]) for j in range(3)];s.cached_gain=s.gain.copy()
    s.hpf=[sign*np.sqrt(max(0,float(d[f'spectral_damper_hpf[{j}]'][ix]))) for j in range(3)]
    s.energy=[max(0,float(d[f'spectral_damper_out[{j}]'][ix])) for j in range(3)]
    d=ds['airspeed_validated'];ix=available_index(d,start);speed=float(d['calibrated_airspeed_m_s'][ix])
    s.airspeed_filter=max(.5,speed) if np.isfinite(speed) else c.p['FW_AIRSPD_TRIM']
    # Previous effort/yaw memory unavailable: explicit zero assumptions.


def flight(entry):
    src=entry['source'];assert src['partition']=='train' and base.sha(src['path'])==src['sha256']
    assert base.sha(base.ROOT/entry['cache'])==entry['cache_sha256']
    u=ULog(src['path'],message_name_filter_list=TOPICS);assert u.msg_info_dict['ver_sw']==FIRMWARE and not u.changed_parameters
    ds={d.name:d.data for d in u.data_list if d.multi_id==0}
    p={k:v for k,v in u.initial_parameters.items() if k.startswith(('FW_','CA_','VT_','TRIM_','PWM_MAIN_','FLAP_'))}
    w=pd.read_csv(DATA/'windows.csv');w=w[w.log_id==entry['log_id']]
    with np.load(base.ROOT/entry['cache']) as z:cached_times=z['timestamp_us']
    att=euler(ds['vehicle_attitude'],'q');ref=euler(ds['vehicle_attitude_setpoint'],'q_d')
    rate=np.stack([ds['vehicle_angular_velocity'][f'xyz[{j}]'] for j in range(3)],1).astype(float)
    accel=np.stack([ds['vehicle_angular_velocity'][f'xyz_derivative[{j}]'] for j in range(3)],1).astype(float)
    rate_sp=np.stack([ds['vehicle_rates_setpoint'][axis] for axis in ['roll','pitch','yaw']],1).astype(float)
    ts=np.stack([ds['vehicle_torque_setpoint'][f'xyz[{j}]'] for j in range(3)],1).astype(float)
    serv=np.stack([ds['actuator_servos'][f'control[{j}]'] for j in range(3)],1).astype(float)
    rows=[];arrays=[];snapshots=[];metadata=[]
    for row in w.itertuples():
        origin=int(cached_times[row.origin_index]);query=origin+np.arange(50)*20000
        target_indices={name:indices(ds[name],query) for name in ['vehicle_rates_setpoint','vehicle_torque_setpoint','actuator_motors','actuator_servos']}
        truth=dict(rates=rate_sp[target_indices['vehicle_rates_setpoint']],torque=ts[target_indices['vehicle_torque_setpoint']],
            commands=np.column_stack([ds['actuator_motors']['control[0]'][target_indices['actuator_motors']],serv[target_indices['actuator_servos']]]))
        for name,warm,sign,tick,feedback in VARIANTS:
            start=origin-int(warm*1e6);times=np.arange(start,query[-1]+1,tick,dtype=np.int64)
            ix={topic:indices(ds[topic],times) for topic in TOPICS if topic in ds}
            if 'autotune_attitude_control_status' in ds:
                states=ds['autotune_attitude_control_status']['state'][ix['autotune_attitude_control_status']]
                # Fail rather than silently ignore active autotune.
                assert not np.isin(states,[1,2,3,4,5,6,7]).any(), 'Inspect active autotune before replay'
            c=[C9Controller(p),C9Controller(p)]
            for controller in c:initialize(controller,ds,start,sign)
            for branch,controller in zip(['rate_input','attitude_chain'],c):
                snapshots.append(dict(array_index=row.array_index,variant=name,branch=branch,start_us=int(start),initial=controller.snapshot()))
            prev_att=-1;last_ref=None;observations=[[],[]];invalid_warm=0;invalid_future=0
            for n,t in enumerate(times):
                mode=ix['vehicle_control_mode'][n];md=ds['vehicle_control_mode']
                status=ix['vehicle_status'][n];vd=ds['vehicle_status']
                valid=bool(md['flag_control_rates_enabled'][mode] and md['flag_control_attitude_enabled'][mode] and not vd['is_vtol'][status])
                if not valid:
                    if t<origin:invalid_warm+=1
                    else:invalid_future+=1
                    for controller in c:controller.state.integral=[0.]*3;controller.state.gain=[1.]*3
                    continue
                gi=ix['vehicle_angular_velocity'][n];ai=ix['vehicle_attitude'][n];ri=ix['vehicle_attitude_setpoint'][n];spi=ix['vehicle_rates_setpoint'][n]
                avi=ix['airspeed_validated'][n];air=float(ds['airspeed_validated']['calibrated_airspeed_m_s'][avi]);age=(int(t)-int(ds['airspeed_validated']['timestamp'][avi]))*1e-6
                manual=bool(md['flag_control_manual_enabled'][mode]);mi=ix['manual_control_setpoint'][n];yaw=float(ds['manual_control_setpoint']['yaw'][mi])
                landed=bool(ds['vehicle_land_detected']['landed'][ix['vehicle_land_detected'][n]])
                if ai!=prev_att:
                    last_ref=c[1].attitude(*att[ai],*ref[ri],air,age,manual=manual,yaw_stick=yaw);prev_att=ai
                for b,controller in enumerate(c):
                    sp=rate_sp[spi] if b==0 else last_ref
                    motor=float(ds['vehicle_rates_setpoint' if b==0 else 'vehicle_attitude_setpoint']['thrust_body[0]'][spi if b==0 else ri])
                    reset=bool(ds['vehicle_rates_setpoint']['reset_integral'][spi]) if b==0 else landed
                    result=controller.rate(rate[gi],sp,accel[gi],air,age,motor,tick*1e-6,landed=landed,reset_integral=reset,feedback=feedback)
                    if t>=origin:observations[b].append((int(t),list(sp),result['torque'],result['commands']))
            if invalid_future:raise RuntimeError(f'Unsupported scored mode at parent{row.array_index}; no silent exclusion')
            for b,branch in enumerate(['rate_input','attitude_chain']):
                values=observations[b];time=np.array([v[0] for v in values]);select=np.searchsorted(time,query,side='right')-1;assert (select>=0).all()
                pred={k:np.array([v[j+1] for v in values])[select] for j,k in enumerate(['rates','torque','commands'])}
                arrays.append(dict(array_index=row.array_index,variant=name,branch=branch,**pred))
                for kind,actual in truth.items():
                    error=pred[kind]-actual
                    for j in range(error.shape[1]):rows.append(dict(log_id=entry['log_id'],array_index=row.array_index,variant=name,branch=branch,kind=kind,axis=j,
                        mse=float((error[:,j]**2).mean()),bias=float(error[:,j].mean()),max_abs=float(abs(error[:,j]).max()),true_power=float((actual[:,j]**2).mean())))
                metadata.append(dict(array_index=row.array_index,variant=name,branch=branch,unsupported_warmup_cycles=invalid_warm,unsupported_future_cycles=invalid_future,
                    start_us=int(start),origin_us=origin,tick_us=tick,initialization_sign=sign,warmup_s=warm))
    dest=OUT/Path(src['path']).stem;dest.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False);pd.DataFrame(metadata).to_csv(dest/'coverage.csv',index=False)
    base.write(dest/'initial_states.json',snapshots)
    np.savez_compressed(dest/'predictions.npz',array_index=[r['array_index'] for r in arrays],variant=[r['variant'] for r in arrays],branch=[r['branch'] for r in arrays],
        **{k:np.stack([r[k] for r in arrays]) for k in ['rates','torque','commands']})
    return dict(log_id=entry['log_id'],parents=len(w),status='complete',runs=len(arrays),destination=str(dest.relative_to(OUT)))


def main():
    verify_parent(PREV);verify_parent(DATA)
    assert read(OUT/'source_check/verification.json')['status']=='passed'
    paths=['src/system_identification/control/px4_c9af.py','tests/test_px4_c9af.py','scripts/check_c9af_source_v38.py','scripts/run_c9af_replay_v38.py']
    protocol=dict(stage='v38 c9af controller-only teacher-state replay',status='frozen_before_replay',firmware=FIRMWARE,
        parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in [PREV,DATA]},code={p:base.sha(base.ROOT/p) for p in paths},
        variants=[dict(name=n,warmup_s=w,hpf_sign=s,tick_us=t,feedback=f) for n,w,s,t,f in VARIANTS],
        branches=['rate_input:logged rate reference+thrust to rate controller','attitude_chain:logged attitude reference/thrust/manual-yaw to attitude then rate controller'],
        initialization='cold: allsource initial states0/gain1; partial: causal postI/gain/energy, +/-sqrtHPF, rawCAS filter proxy, previous effort/yaw memory0. Only at each variant start, no logged hidden state read afterward. +/-all axes are two probes, not exhaustive8sign confidence bounds.',
        timing='Fixed2.5ms or20ms rate ticks anchored to origin; attitude updated when causal logged attitude index changes, before rate. Publication AND sample causal ZOH. Synthetic schedules, not recovered native timing. Targets same causal origin+0..49 command grid; no fitted shifts.',
        feedback='Primary no-interleaved-publication reading of c9af double update of same instance0 for nonVTOL: no fresh surface feedback consumed. Additional delayed_allocation variant uses preceding synthetic allocator residual, an alternative scheduling assumption. Do not infer actual thread interleaving from sparse logs.',
        scope='All407 originaltrain parents; actual future measured states allowed only for this controller component check; no learned-model rollout, future hidden state injection, tuning, exclusions or flight.',
        nonfinite='Fail on nonfinite physical state/command; invalid airspeed uses source fallback; unsupported scored mode fails, unsupported warmup modes resetI/gain and counted.',
        thresholds='Source-body arithmetic rtol1e-5 atol2e-6, no posthoc output gate. Teacher-replay errors all reported; no variant selected/promoted.',
        model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False,
        phase_contract=read(DATA/'dataset_manifest.json')['phase_contract'],frequency_contract=read(DATA/'dataset_manifest.json')['frequency_contract'])
    assert not (OUT/'protocol.json').exists();base.write(OUT/'protocol.json',protocol)
    base.write(OUT/'test_results.json',dict(passed=6,command='pytest -q tests/test_px4_c9af.py',reused_unchanged_source_result=True))
    entries=read(PREV/'dataset_manifest.json')['sources'];completed=[]
    base.write(OUT/'status.json',dict(status='running',completed=[]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
        for value in pool.map(flight,entries):
            completed.append(value);base.write(OUT/'status.json',dict(status='running',completed=completed));print(value,flush=True)
    base.write(OUT/'status.json',dict(status='replay_complete_evaluation_pending',completed=completed))


if __name__=='__main__':main()
