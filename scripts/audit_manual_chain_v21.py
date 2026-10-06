"""Frozen, read-only ULog/source audit for original train window 391."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from reference_timing_v14 import effective_indices

OUT = base.ROOT / 'docs/analysis/results/control_windows_v21'
PX4 = Path('/home/zn/PX4-Autopilot')
COMMIT = 'c9af571575af33d5403fbb479256848797c801e2'
CONTRACT = base.ROOT / 'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json'
SOURCES = [
    'src/modules/fw_att_control/FixedwingAttitudeControl.cpp',
    'src/modules/fw_att_control/fw_yaw_controller.cpp',
    'src/modules/fw_rate_control/FixedwingRateControl.cpp',
    'src/modules/rc_update/rc_update.cpp',
    'src/modules/manual_control/ManualControl.cpp',
    'src/modules/manual_control/ManualControlSelector.cpp',
    'src/modules/logger/logged_topics.cpp',
    'src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessControlSurfaces.cpp',
    'src/lib/mixer_module/functions/FunctionServos.hpp',
    'src/lib/mixer_module/output_functions.yaml',
    'src/lib/mixer_module/mixer_module.cpp',
    'src/drivers/px4io/px4io.cpp',
]
TOPICS = ['manual_control_setpoint', 'manual_control_input', 'input_rc', 'rc_channels',
          'vehicle_status', 'vehicle_control_mode', 'vehicle_attitude_setpoint',
          'vehicle_rates_setpoint', 'vehicle_torque_setpoint', 'rate_ctrl_status',
          'rate_ctrl_terms', 'control_allocator_status', 'actuator_servos',
          'actuator_outputs', 'vehicle_angular_velocity', 'vehicle_attitude',
          'airspeed_validated', 'autotune_attitude_control_status']


def rc_normalize(value, minimum, trim, maximum, reverse):
    if not minimum < trim < maximum:
        raise ValueError('invalid RC calibration')
    return np.interp(value, [minimum, trim, maximum], [-1., 0., 1.]) * (-1 if reverse < 0 else 1)


def servo_pwm(value, minimum, maximum, reverse=False):
    """Linear map when center is absent; outputs rounded as positive C lroundf."""
    v = np.asarray(value, float) * (-1 if reverse else 1)
    return np.floor(minimum + (v + 1) * .5 * (maximum-minimum) + .5)


def stats(v):
    a = np.asarray(v, float)
    if a.size == 0:
        return dict(n=0)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite values in reported field')
    return dict(n=int(a.size), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(a.max()),
                mean=float(a.mean()), rms=float(np.sqrt(np.mean(a*a))))


def hold(d, q):
    return effective_indices(d['timestamp'], d.get('timestamp_sample', d['timestamp']), q)


def main():
    target_path = base.ROOT/'docs/analysis/results/control_windows_v20/window391.json'
    target = json.loads(target_path.read_text())
    contract = json.loads(CONTRACT.read_text())
    source_path = base.ROOT/'dataset/reference_dynamics_data_v1/manifest.json'
    ready_path = base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_ready/manifest.json'
    source = json.loads(source_path.read_text())
    assert base.sha(ready_path) == contract['parent_sha256']
    assert base.sha(source_path) == json.loads(ready_path.read_text())['parent_sha256']
    assert source['parent_manifest_sha256'] == contract['source_dataset_manifest_sha256']
    src = source['sources'][target['log_id']]
    assert src['partition'] == target['partition'] == 'train'
    assert base.sha(src['path']) == src['sha256']
    cache_path = base.ROOT/target['cache']
    assert base.sha(cache_path) == contract['cache_sha256'][target['cache']]
    with np.load(cache_path) as z:
        cache = {k: z[k] for k in z.files}
    origin = target['origin_timestamp_us']
    lo, hi = origin-3000000, origin+3000000
    protocol = dict(stage='v21', target=target, interval_us=[lo, hi],
                    methods=['all native rows plus preceding row', 'causal zero order hold using max(publication,sample)',
                             'exact sample timestamp joins where available', 'fixed source semantics; no fitted lag'],
                    exclusions='none; target selected for v20 diagnosis, not representative of all windows',
                    raw_source=src, firmware_commit=COMMIT,
                    pins={str(p.relative_to(base.ROOT)): base.sha(p) for p in [target_path, CONTRACT, source_path, ready_path, cache_path]},
                    phase_contract=contract['phase_contract'], frequency_contract=contract['frequency_contract'],
                    models_changed=False, training_started=False, validation_accessed=False, sealed_test_accessed=False)
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        base.write(OUT/'protocol.json', protocol)
    (OUT/'source').mkdir(exist_ok=True)
    blobs = {}
    for path in SOURCES:
        content = subprocess.check_output(['git', '-C', str(PX4), 'show', f'{COMMIT}:{path}'])
        dst = OUT/'source'/Path(path).name
        dst.write_bytes(content)
        blobs[path] = dict(blob=subprocess.check_output(['git', '-C', str(PX4), 'rev-parse', f'{COMMIT}:{path}'], text=True).strip(),
                           snapshot=str(dst.relative_to(OUT)), sha256=base.sha(dst))
    base.write(OUT/'source_provenance.json', dict(commit=COMMIT, files=blobs,
               limitation='Logged version identifies source revision; does not independently attest binary or uncommitted build patches.'))
    u = ULog(src['path'], message_name_filter_list=TOPICS)
    assert u.msg_info_dict['ver_sw'] == COMMIT
    ds = {d.name: d.data for d in u.data_list if d.multi_id == 0}
    required = set(TOPICS)-{'manual_control_input', 'rc_channels', 'autotune_attitude_control_status'}
    assert required <= ds.keys(), required-ds.keys()
    p = u.initial_parameters
    parameters = {k: v for k,v in p.items() if k.startswith(('RC4_', 'PWM_MAIN_', 'CA_SV', 'FW_Y', 'FW_MAN_Y'))
                  or k in ['RC_MAP_YAW','CA_AIRFRAME','CA_METHOD','FW_RLL_TO_YAW_FF','TRIM_YAW','SDLOG_PROFILE']}
    base.write(OUT/'parameters.json', parameters)
    ranges = dict(context=(lo,hi), window=(target['start_timestamp_us'],target['end_timestamp_us']),
                  forecast=(origin,origin+980000))
    (OUT/'raw').mkdir(exist_ok=True)
    timings = []
    for topic, d in ds.items():
        t = d['timestamp'].astype(np.int64)
        ids = np.flatnonzero((t >= lo) & (t <= hi))
        before = np.searchsorted(t,lo,side='left')-1
        if before >= 0:
            ids = np.unique(np.r_[before,ids])
        frame = pd.DataFrame({k:v[ids] for k,v in d.items()})
        frame.insert(1,'relative_time_s',(t[ids]-origin)*1e-6)
        frame.to_csv(OUT/'raw'/f'{topic}.csv',index=False)
        for name,(start,end) in ranges.items():
            ix = np.flatnonzero((t>=start)&(t<=end))
            interval = np.diff(t[ix])*.001
            timings.append(dict(topic=topic, interval=name, rows=len(ix),
                                has_sample_timestamp='timestamp_sample' in d,
                                period_median_ms=float(np.median(interval)) if len(interval) else None,
                                period_max_ms=float(interval.max()) if len(interval) else None))
    pd.DataFrame(timings).to_csv(OUT/'topic_timing.csv',index=False)
    idx = np.arange(target['origin_index']-26,target['origin_index']+51)
    q = cache['timestamp_us'][idx].astype(np.int64)
    assert (q[0],q[-1],q[26]) == (target['start_timestamp_us'],target['end_timestamp_us'],origin)
    man = ds['manual_control_setpoint']; mi = hold(man,q)
    yaw = np.where(cache['nav_state'][idx]==15,man['yaw'][mi],0)
    error = float(np.max(abs(yaw-cache['references'][idx,4])))
    assert error < 1e-7
    # Eligibility needs both timestamps in the past; cache age is the OLDER
    # timestamp's age (max of publication/sample ages), as in the data contract.
    age = (q-np.minimum(man['timestamp'][mi],man['timestamp_sample'][mi]))*1e-6
    np.testing.assert_allclose(age,cache['manual_reference_age_s'][idx],atol=1e-7)
    grid = pd.DataFrame(dict(timestamp_us=q, relative_time_s=(q-origin)*1e-6,
                             manual_yaw=yaw, manual_pub_us=man['timestamp'][mi],
                             manual_sample_us=man['timestamp_sample'][mi], manual_max_age_ms=age*1000,
                             yaw_additive_rate_rad_s=np.clip(yaw*np.deg2rad(p['FW_MAN_YR_MAX']),
                                                            -np.deg2rad(p['FW_Y_RMAX']),np.deg2rad(p['FW_Y_RMAX']))))
    fields = {'vehicle_rates_setpoint':['roll','pitch','yaw'], 'vehicle_torque_setpoint':[f'xyz[{i}]' for i in range(3)],
              'actuator_servos':[f'control[{i}]' for i in range(3)],'actuator_outputs':['output[0]','output[1]','output[4]'],
              'vehicle_angular_velocity':[f'xyz[{i}]' for i in range(3)],
              'vehicle_attitude_setpoint':['yaw_sp_move_rate'],
              'control_allocator_status':['torque_setpoint_achieved','unallocated_torque[2]'],
              'vehicle_status':['nav_state','failsafe'], 'vehicle_control_mode':['flag_control_manual_enabled','flag_control_rates_enabled','flag_control_attitude_enabled']}
    for topic, keys in fields.items():
        d=ds[topic];ix=hold(d,q)
        for k in keys:
            grid[f'{topic}.{k}']=d[k][ix]
    grid.to_csv(OUT/'window_grid.csv',index=False)
    # Preserve raw RC observations without treating the slower RC stream as a replacement.
    rc=ds['input_rc']; ri=np.flatnonzero((rc['timestamp']>=lo)&(rc['timestamp']<=hi))
    assert p['RC_MAP_YAW']==4
    rcframe=pd.DataFrame(dict(timestamp_us=rc['timestamp'][ri],relative_time_s=(rc['timestamp'][ri].astype(np.int64)-origin)*1e-6,
                             rc4=rc['values[3]'][ri], rc_lost=rc['rc_lost'][ri],rc_failsafe=rc['rc_failsafe'][ri],
                             normalized_yaw=rc_normalize(rc['values[3]'][ri],p['RC4_MIN'],p['RC4_TRIM'],p['RC4_MAX'],p['RC4_REV'])))
    rmi=hold(man,rc['timestamp'][ri])
    rcframe['last_signal_us']=rc['timestamp_last_signal'][ri]
    rcframe['preceding_manual_yaw']=man['yaw'][rmi]
    rcframe['preceding_manual_sample_us']=man['timestamp_sample'][rmi]
    rcframe['sample_separation_ms']=(rc['timestamp_last_signal'][ri]-man['timestamp_sample'][rmi])*.001
    rcframe['difference_from_sparse_hold']=rcframe.normalized_yaw-rcframe.preceding_manual_yaw
    rcframe.to_csv(OUT/'rc_calibrated.csv',index=False)
    # Exact acquisition-time match avoids interpreting asynchronous holds as allocation error.
    torque=ds['vehicle_torque_setpoint'];servo=ds['actuator_servos']
    ti=np.flatnonzero((torque['timestamp']>=lo)&(torque['timestamp']<=hi))
    si=np.flatnonzero((servo['timestamp']>=lo)&(servo['timestamp']<=hi))
    left=pd.DataFrame(dict(sample_us=torque['timestamp_sample'][ti],torque_z=torque['xyz[2]'][ti]))
    right=pd.DataFrame(dict(sample_us=servo['timestamp_sample'][si],servo2=servo['control[2]'][si]))
    assert left.sample_us.is_unique and right.sample_us.is_unique
    matched=left.merge(right,on='sample_us',validate='one_to_one')
    matched['difference']=matched.servo2-matched.torque_z
    matched['relative_time_s']=(matched.sample_us.astype(np.int64)-origin)*1e-6
    matched.to_csv(OUT/'allocation_exact_sample_pairs.csv',index=False)
    # Fixed linear output mapping. The causal servo record can itself be stale.
    output=ds['actuator_outputs'];oi=np.flatnonzero((output['timestamp']>=lo)&(output['timestamp']<=hi))
    sx=hold(servo,output['timestamp'][oi]);sv=servo['control[2]'][sx]
    assert p['PWM_MAIN_FUNC5']==203 and p['PWM_MAIN_REV']&(1<<4)
    pwm=servo_pwm(sv,p['PWM_MAIN_MIN5'],p['PWM_MAIN_MAX5'],True)
    pf=pd.DataFrame(dict(timestamp_us=output['timestamp'][oi],relative_time_s=(output['timestamp'][oi].astype(np.int64)-origin)*1e-6,
                         servo2=sv,servo_record_age_ms=(output['timestamp'][oi]-servo['timestamp'][sx])*.001,
                         predicted_pwm5=pwm,logged_pwm5=output['output[4]'][oi]))
    pf['residual_us']=pf.logged_pwm5-pf.predicted_pwm5
    pf.to_csv(OUT/'pwm_causal_comparison.csv',index=False)
    summaries={}
    for name,(start,end) in ranges.items():
        item={}
        for topic, keys in fields.items():
            d=ds[topic];sel=(d['timestamp']>=start)&(d['timestamp']<=end)
            item[topic]={k:stats(d[k][sel]) for k in keys}
        for topic,keys in [('manual_control_setpoint',['yaw','valid','data_source']),
                           ('input_rc',['rc_lost','rc_failsafe']),('control_allocator_status',[f'actuator_saturation[{i}]' for i in range(4)]),
                           ('vehicle_attitude',['quat_reset_counter']),
                           ('rate_ctrl_terms',['p_term[2]','i_term[2]','d_term[2]','ff_term[2]','output[2]'])]:
            d=ds[topic];sel=(d['timestamp']>=start)&(d['timestamp']<=end)
            item.setdefault(topic,{}).update({k:stats(d[k][sel]) for k in keys})
        summaries[name]=item
    base.write(OUT/'summary.json',dict(target=target, firmware=COMMIT, missing_optional_topics=sorted(set(TOPICS)-ds.keys()),
               native=summaries, cache_max_abs_error=error, cache_manual_age_ms=stats(age*1000),
               forecast_manual_age_ms=stats(age[26:76]*1000),
               cache_repeated_manual_fraction=float(np.mean(np.diff(mi)==0)),
               parameter_change_count=len(u.changed_parameters),
               raw_dropouts=[dict(timestamp=int(d.timestamp),duration_ms=int(d.duration)) for d in u.dropouts],
               exact_allocation_match=dict(pairs=len(matched),error=stats(matched.difference)),
               pwm_causal_residual_us=stats(pf.residual_us),
               pwm_same_mapping_with_stale_servo_fraction=float(np.mean(abs(pf.residual_us)<=1)),
               model_ready_for_control=False,sealed_test_accessed=False))
    plot(ds,grid,origin,lo,hi)
    print(json.dumps(dict(output=str(OUT),cache_max_abs_error=error,allocation_pairs=len(matched)),ensure_ascii=False))


def plot(ds,grid,origin,lo,hi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(5,1,figsize=(12,12),sharex=True)
    def line(ax,topic,field,label):
        d=ds[topic];m=(d['timestamp']>=lo)&(d['timestamp']<=hi)
        ax.plot((d['timestamp'][m].astype(np.int64)-origin)*1e-6,d[field][m],'.-',ms=2,lw=.8,label=label)
    line(axes[0],'manual_control_setpoint','yaw','logged manual yaw (~5 Hz)')
    axes[0].step(grid.relative_time_s,grid.manual_yaw,where='post',label='model cache hold (50 Hz grid)')
    for c in ['roll','pitch','yaw']:line(axes[1],'vehicle_rates_setpoint',c,c+' rate reference')
    line(axes[2],'vehicle_torque_setpoint','xyz[2]','yaw torque demand')
    line(axes[2],'actuator_servos','control[2]','servo 3 command')
    line(axes[3],'actuator_outputs','output[4]','MAIN5 output (PWM us)')
    for j,c in enumerate(['p','q','r']):line(axes[4],'vehicle_angular_velocity',f'xyz[{j}]',c+' measured rad/s')
    for ax in axes:
        ax.axvspan(-.52,1.,alpha=.08,color='green');ax.axvline(0,color='k',ls='--',lw=.8)
        ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('PX4 time relative to origin 547.917212 s; green = original window')
    fig.suptitle('Window 391: native logged command chain; lines do not recover unlogged samples')
    fig.tight_layout();fig.savefig(OUT/'manual_chain.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
