"""Mine reference-only natural pulse candidates, preserving registered flight splits.

Observational segments are not randomized experiments or validated causal responses.
Only manifest train/validation ULogs are parsed; other files are inventoried by name.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import yaml
from pyulog import ULog

from audit_control_reference_semantics import past_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/analysis/results/existing_reference_ramps_v1'
LOGROOT = Path('/home/zn/QgcLogs')
DT = .02
TOPICS = ['vehicle_status', 'vehicle_control_mode', 'vehicle_land_detected',
          'vehicle_attitude_setpoint', 'vehicle_attitude', 'vehicle_angular_velocity',
          'actuator_servos', 'actuator_motors', 'manual_control_setpoint',
          'vehicle_local_position', 'vehicle_rates_setpoint', 'wing_phase', 'airspeed_validated',
          'sysid_excitation']


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def euler(q):
    norm = np.linalg.norm(q, axis=1)
    q = q / np.maximum(norm[:, None], 1e-12)
    w, x, y, z = q.T
    a = np.column_stack([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                         np.arcsin(np.clip(2*(w*y-z*x), -1, 1))])
    a[~np.isfinite(norm) | (norm < .5)] = np.nan
    return np.rad2deg(a)


def pulse_candidates(ref, valid, threshold):
    """Reference-only fixed pulse screen; no selection on measured motion.

1s baseline, 0.4s sustained excursion, return within 3s for 0.5s;
extract [-1,+6)s, greedily nonoverlapping per axis/threshold.
"""
    result = []
    for axis in range(2):
        free = 0
        for start, end in runs(valid):
            for i in range(start+50, end-300, 5):
                if i-50 < free:
                    continue
                before = ref[i-50:i, axis]
                base = np.median(before)
                if np.ptp(before) > threshold/2:
                    continue
                # Permit a 0.3s approach instead of demanding an instantaneous jump.
                delta = ref[i+15:i+35, axis]-base
                amplitude = float(np.median(delta))
                if not threshold <= abs(amplitude) <= 8:
                    continue
                sign = np.sign(amplitude)
                if np.min(sign*delta) < threshold*.7 or np.ptp(delta) > max(.5, abs(amplitude)*.5):
                    continue
                returned = None
                for j in range(i+35, i+166, 5):
                    if np.max(np.abs(ref[j:j+25, axis]-base)) <= max(.25, abs(amplitude)*.3):
                        returned = j
                        break
                if returned is None:
                    continue
                other = np.ptp(ref[i-50:returned+25, 1-axis])
                result.append(dict(index=i, axis=axis, threshold_deg=threshold, amplitude_deg=amplitude,
                                   baseline_deg=float(base), return_s=(returned-i)*DT,
                                   other_reference_range_deg=float(other),
                                   single_axis=bool(other <= max(1., abs(amplitude)*.5))))
                free = i+300
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    registry_path = ROOT/'configs/data/trajectory_dataset_registry.yaml'
    registry = yaml.safe_load(registry_path.read_text())
    did = registry['default_dataset_id']; entry = registry['datasets'][did]
    mp = ROOT/entry['manifest_path']
    assert digest(mp) == entry['manifest_sha256']
    manifest = json.loads(mp.read_text()); assignments = manifest['split_contract']['assignments']
    allowed = {p: part for part in ['train', 'validation'] for p in assignments[part]}
    excluded_names = {Path(p).name for k, paths in assignments.items() if k not in ['train', 'validation'] for p in paths}
    assert not any(Path(p).name in excluded_names for p in allowed)
    inventory = []
    for p in sorted(LOGROOT.rglob('*.ulg')):
        rel = str(p.relative_to(LOGROOT))
        inventory.append(dict(path=rel, bytes=p.stat().st_size,
            access=allowed.get(rel, 'protected_name_not_opened' if p.name in excluded_names else 'outside_current_whitelist_not_opened')))
    pd.DataFrame(inventory).to_csv(OUT/'inventory.csv', index=False)
    write(OUT/'protocol.json', dict(dataset_id=did, manifest_path=str(mp), manifest_sha256=digest(mp),
        registry_sha256=digest(registry_path), script_sha256=digest(__file__),
        helper_sha256=digest(ROOT/'scripts/audit_control_reference_semantics.py'),
        partitions=['train', 'validation'], sealed_test_accessed=False, allowed=allowed,
        phase_contract=manifest['phase_contract'], frequency_contract=manifest['frequency_contract'],
        selection='All raw registered train/validation; no existing origin subsampling; no training or split changes',
        mode='nav15, armed, manual+attitude+rates enabled, climb-rate disabled, not landed, no failsafe; stale mode >1.2s rejected',
        alignment='20ms query grid; last message with publication AND sample <=query; reference/state/command max age60ms',
        pulses='reference only; thresholds0.5/1/2deg, baseline1s range<=threshold/2, 0.3s permitted approach then sustained0.4s, excursion<=8deg, return<=3.3s within max0.25deg/30% amplitude for0.5s; context1s before/6s after; greedy nonoverlap per axis/threshold',
        followup='Strict rectangular screen preserved in existing_reference_excitation_v1; this separate screen permits finite approach time and exports all20s blocks',
        thresholds='descriptive mining choices, not safety limits or causal admission gates; fixed before results',
        long_windows='nonoverlapping20s blocks of mode/reference-valid runs, for later spectral analysis only',
        limitations='No inferred pre-generated excitation provenance; no outcome-based selection; pilot feedback/exogeneity unresolved; no actuator angle inferred from command'))
    logrows, segments, candidates, cadences, sources, blocks = [], [], [], [], {}, []
    (OUT/'segments').mkdir()
    for log, partition in sorted(allowed.items()):
        path = LOGROOT/log
        expected = manifest['source']['ulog_sha256'][log]
        assert digest(path) == expected, path
        sources[log] = dict(path=str(path), sha256=expected, partition=partition)
        ul = ULog(str(path), message_name_filter_list=TOPICS)
        topics = {d.name:d.data for d in ul.data_list if d.multi_id == 0}
        for name, d in topics.items():
            intervals = np.diff(d['timestamp'].astype(float))*1e-6
            cadences.append(dict(log_id=log, partition=partition, topic=name, n=len(d['timestamp']),
                median_dt_s=float(np.median(intervals)) if len(intervals) else None,
                p95_dt_s=float(np.quantile(intervals,.95)) if len(intervals) else None))
        refdata = topics['vehicle_attitude_setpoint']
        times = np.arange(int(refdata['timestamp'][0]), int(refdata['timestamp'][-1]), 20000, dtype=np.int64)
        def get(name, keys, age=.06):
            if name not in topics or not all(k in topics[name] for k in keys):
                return np.full((len(times), len(keys)), np.nan), np.zeros(len(times),bool)
            a, pubage, sampleage = past_values(topics[name], keys, times)
            good = np.isfinite(a).all(1)&(pubage<=age)&(sampleage<=age)
            return a, good
        q, rg = get('vehicle_attitude_setpoint',[f'q_d[{i}]' for i in range(4)])
        ref = euler(q)
        status, sg = get('vehicle_status',['nav_state','failsafe'],1.2)
        mode, mg = get('vehicle_control_mode',['flag_armed','flag_control_manual_enabled','flag_control_attitude_enabled','flag_control_rates_enabled','flag_control_climb_rate_enabled'],1.2)
        landed, lg = get('vehicle_land_detected',['landed'],1.2)
        valid = rg & sg & mg & lg & np.isfinite(ref).all(1) & (status[:,0]==15)&(status[:,1]==0)&(mode[:,:4]==1).all(1)&(mode[:,4]==0)&(landed[:,0]==0)
        rates, rateok = get('vehicle_angular_velocity',[f'xyz[{i}]' for i in range(3)])
        actualq, attok = get('vehicle_attitude',[f'q[{i}]' for i in range(4)])
        actual = euler(actualq)
        servos, cmdok = get('actuator_servos',[f'control[{i}]' for i in range(3)])
        motors, motorok = get('actuator_motors',['control[0]'])
        manual, manok = get('manual_control_setpoint',['roll','pitch','throttle','valid'],.3)
        velocity, vok = get('vehicle_local_position',['vx','vy','vz'])
        phase, phaseok = get('wing_phase',['phase_rad','flap_frequency_hz','phase_valid'])
        chainok = rateok & attok & cmdok & motorok
        params = {k:float(v) for k,v in ul.initial_parameters.items() if k.startswith(('FW_R','FW_P','CA_SV_CS','FLAP_'))}
        parameter_hash = hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
        parameter_updates = len(ul.changed_parameters)
        for a,b in runs(valid):
            if b-a >= 500:
                segments.append(dict(log_id=log, partition=partition, start_us=int(times[a]), end_us=int(times[b-1]),
                    duration_s=(b-a)*DT, nonoverlap_20s_blocks=(b-a)//1000,
                    chain_fresh_fraction=float(chainok[a:b].mean()), roll_reference_range_deg=float(np.ptp(ref[a:b,0])),
                    pitch_reference_range_deg=float(np.ptp(ref[a:b,1]))))
            for left in range(a,b-999,1000):
                right=left+1000
                ident=hashlib.sha256(f'{log}:block:{left}'.encode()).hexdigest()[:16]
                block=dict(block_id=ident,log_id=log,partition=partition,start_us=int(times[left]),end_us=int(times[right-1]),
                    seconds_from_log_start=(int(times[left])-ul.start_timestamp)*1e-6,
                    chain_fresh_fraction=float(chainok[left:right].mean()),
                    roll_reference_std_deg=float(np.std(ref[left:right,0])),pitch_reference_std_deg=float(np.std(ref[left:right,1])),
                    roll_reference_range_deg=float(np.ptp(ref[left:right,0])),pitch_reference_range_deg=float(np.ptp(ref[left:right,1])),
                    vertical_velocity_abs_p95=float(np.nanquantile(np.abs(velocity[left:right,2]),.95)),
                    groundspeed_median=float(np.nanmedian(np.linalg.norm(velocity[left:right,:2],axis=1))),
                    flap_frequency_median=float(np.nanmedian(phase[left:right,1])),
                    parameter_hash=parameter_hash,firmware=str(ul.msg_info_dict.get('ver_sw','unknown')),
                    parameter_update_count=parameter_updates,
                    reference_roll_pitch_correlation=float(np.corrcoef(ref[left:right].T)[0,1]) if np.min(np.std(ref[left:right],axis=0))>1e-6 else None,
                    servo_left_right_correlation=float(np.corrcoef(servos[left:right,:2].T)[0,1]) if np.isfinite(servos[left:right,:2]).all() and np.min(np.std(servos[left:right,:2],axis=0))>1e-6 else None)
                blocks.append(block)
                np.savez_compressed(OUT/'segments'/f'block_{ident}.npz',timestamp_us=times[left:right],
                    reference_deg=ref[left:right],attitude_deg=actual[left:right],rates_rad_s=rates[left:right],
                    servo_commands=servos[left:right],motor_command=motors[left:right],manual=manual[left:right],
                    velocity_n=velocity[left:right],wing_phase_frequency_valid=phase[left:right],chain_fresh=chainok[left:right])
        perlog=[]
        for threshold in [.5,1.,2.]:
            for item in pulse_candidates(ref,valid,threshold):
                i=item.pop('index');axis=item['axis'];a=i-50;b=i+300
                ident=hashlib.sha256(f'{log}:{i}:{axis}:{threshold}'.encode()).hexdigest()[:16]
                t0=int(times[i]); command=(servos[:,0]+servos[:,1])/2 if axis==1 else (servos[:,0]-servos[:,1])/2
                base=float(np.median(command[i-50:i]));window=command[i:i+150]-base
                record=dict(candidate_id=ident,log_id=log,partition=partition,axis_name=['roll','pitch'][axis],
                    onset_timestamp_us=t0, seconds_from_log_start=(t0-ul.start_timestamp)*1e-6,
                    start_us=int(times[a]),end_us=int(times[b-1]),**item,
                    chain_fresh_fraction=float(chainok[a:b].mean()), manual_fresh_fraction=float((manok[a:b]&(manual[a:b,3]==1)).mean()),
                    servo_abs_ge_098_fraction=float((np.abs(servos[a:b])>=.98).any(1).mean()),
                    command_peak_change=float(np.nanmax(np.abs(window))),
                    actual_axis_range_deg=float(np.ptp(actual[a:b,axis])),
                    rate_axis_std_rad_s=float(np.std(rates[a:b,axis])),
                    initial_groundspeed_m_s=float(np.linalg.norm(velocity[i,:2])) if vok[i] else None,
                    initial_vertical_velocity_m_s=float(velocity[i,2]) if vok[i] else None,
                    initial_flap_frequency_hz=float(phase[i,1]) if phaseok[i] else None,
                    firmware=str(ul.msg_info_dict.get('ver_sw','unknown')),parameter_hash=parameter_hash,
                    parameter_update_count=parameter_updates,
                    provenance='observed_reference_only; planned/pilot origin not certified')
                candidates.append(record);perlog.append(record)
                # Export primary1deg candidates with all signals, not selected for good response.
                if threshold==1.:
                    np.savez_compressed(OUT/'segments'/f'{ident}.npz',timestamp_us=times[a:b],relative_time_s=(times[a:b]-t0)*1e-6,
                        reference_deg=ref[a:b],attitude_deg=actual[a:b],rates_rad_s=rates[a:b],servo_commands=servos[a:b],
                        motor_command=motors[a:b],manual=manual[a:b],velocity_n=velocity[a:b],wing_phase_frequency_valid=phase[a:b],chain_fresh=chainok[a:b])
        logrows.append(dict(log_id=log,partition=partition,stabilized_reference_fresh_s=float(valid.sum()*DT),
            long_runs=sum(b-a>=500 for a,b in runs(valid)),pulse_candidates=len(perlog),
            sysid_excitation_topic_present='sysid_excitation' in topics,firmware=str(ul.msg_info_dict.get('ver_sw','unknown')),
            parameter_hash=parameter_hash,parameter_update_count=parameter_updates))
        print(len(logrows),len(allowed),log,'candidates',len(perlog),flush=True)
    pd.DataFrame(logrows).to_csv(OUT/'logs.csv',index=False)
    pd.DataFrame(segments).to_csv(OUT/'continuous_segments.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'candidates.csv',index=False)
    pd.DataFrame(cadences).to_csv(OUT/'topic_cadence.csv',index=False)
    pd.DataFrame(blocks).to_csv(OUT/'blocks.csv',index=False)
    write(OUT/'sources.json',sources)
    write(OUT/'completion.json',dict(status='complete',logs_parsed=len(logrows),sealed_test_accessed=False,
        outputs={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        causal_identification_validated=False,flight_replacement_established=False))


if __name__=='__main__':
    run()
