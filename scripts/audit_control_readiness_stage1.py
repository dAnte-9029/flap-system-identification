"""Read-only firmware/log provenance audit; no model evaluation or test access."""
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[1]
LOGS = Path('/home/zn/QgcLogs')
PX4 = Path('/home/zn/PX4-Autopilot')
OUT = ROOT / 'docs/analysis/results/control_readiness_stage1'
SHA = 'e624a99f2955addbf76681e636c44162a0c03055'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def save(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x), allow_nan=False)

def table(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def git(*args):
    return subprocess.check_output(['git', '-C', str(PX4), *args], text=True)

def relevant(p):
    return {k:v for k,v in p.items() if k.startswith(('FW_', 'FLAP_', 'CA_', 'PWM_', 'RPM_', 'IMU_', 'MC_', 'ASPD_', 'SDLOG_', 'SYS_AUTOSTART'))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save('scope.json', {'development_full_topics_day':18, 'header_only_days':list(range(11,18))+[19], 'reserved_from_this_audit_day':19, 'reservation_not_claim_of_prior_unseen':True, 'no_test_or_earlier_log_open':True, 'script_sha256':digest(Path(__file__)), 'python':sys.executable, 'target_commit':SHA, 'source_branch':git('branch','--show-current').strip()})
    registry=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text())
    entry=registry['datasets'][registry['default_dataset_id']]
    mp=ROOT/entry['manifest_path']
    assert digest(mp)==entry['manifest_sha256']
    manifest=json.loads(mp.read_text())
    checks=[]
    for name, expected in manifest['artifact_sha256'].items():
        assert 'test' not in name
        p=mp.parent/name; actual=digest(p); assert actual==expected, name
        checks.append(dict(path=str(p), sha256=actual))
    save('dataset_provenance.json', dict(dataset_id=registry['default_dataset_id'], registry_entry=entry, manifest_absolute_path=str(mp), manifest_sha256=digest(mp), verified_artifacts=checks, frames=manifest['frames'], frequency_contract=manifest['frequency_contract'], phase_zero=yaml.safe_load((ROOT/entry['config_path']).read_text())['phase_zero'], partitions=['train','validation']))
    source_paths=['src/modules/fw_att_control/FixedwingAttitudeControl.cpp','src/modules/fw_rate_control/FixedwingRateControl.cpp','src/modules/fw_rate_control/BumplessRollITransfer.hpp','src/modules/flap_aug_shadow/module.yaml','src/modules/logger/logged_topics.cpp','docs/flapping-logging.md','msg/versioned/VehicleStatus.msg','msg/FlapAugShadow.msg','msg/FlapB2bAdaptive.msg']
    for path in source_paths:
        p=OUT/'firmware_sources'/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(git('show',f'{SHA}:{path}'))
    (OUT/'firmware_changes.txt').write_text(git('diff','--stat','c9af571575af33d5403fbb479256848797c801e2',SHA,'--','src/modules/fw_att_control','src/modules/fw_rate_control','src/modules/flap_aug_shadow','src/modules/control_allocator','src/modules/wing_phase','src/drivers/rpm_capture'))
    topics='vehicle_status vehicle_land_detected mission_result position_setpoint_triplet vehicle_attitude vehicle_attitude_setpoint vehicle_rates_setpoint vehicle_angular_velocity actuator_outputs actuator_servos vehicle_torque_setpoint vehicle_thrust_setpoint rate_ctrl_status rate_ctrl_terms flap_aug_shadow flap_aug_v4_shadow flap_b2b_adaptive flap_frequency wing_phase hall_event vehicle_local_position airspeed_validated control_allocator_status'.split()
    rows=[]; groups=[]; rates=[]; missions=[]; modes=[]; changes=[]; planrows=[]; seen={}; config=yaml.safe_load((ROOT/entry['config_path']).read_text()); mismatch=[]
    plans={}
    for p in sorted(LOGS.rglob('*.plan')):
        if p.name.startswith('._') or not re.match(r'9\.(1[1-9])(?:数据|-)',p.relative_to(LOGS).parts[0]): continue
        obj=json.loads(p.read_text()); items=obj.get('mission',{}).get('items',[])
        coords=[i['params'][4:7] for i in items if i.get('command')==16 and len(i.get('params',[]))>=7]
        plans[str(p)]=coords
        planrows.append(dict(path=str(p),sha256=digest(p),items=len(items),waypoints=json.dumps(coords),speed_commands=json.dumps([i.get('params') for i in items if i.get('command')==178])))
    for p in sorted(LOGS.rglob('*.ulg')):
        rel=p.relative_to(LOGS); match=re.match(r'9\.(1[1-9])(?:数据|-)',rel.parts[0])
        if p.name.startswith('._') or not match: continue
        day=int(match[1]); assert 11<=day<=19
        h=digest(p); duplicate=seen.get(h,''); seen.setdefault(h,str(rel)); full=day==18 and not duplicate
        u=ULog(str(p),message_name_filter_list=topics if full else None,parse_header_only=not full)
        info=u.msg_info_dict; params=u.initial_parameters; subset=relevant(params)
        group=hashlib.sha256(json.dumps(subset,sort_keys=True).encode()).hexdigest()[:16]
        save('parameters/'+h+'.json',dict(initial=params,info=info,scope='selected_topics_full' if full else 'header_only',changed_parameters=u.changed_parameters if full else 'not_inspected')) if not duplicate else None
        row=dict(path=str(rel),sha256=h,bytes=p.stat().st_size,day=day,duplicate_of=duplicate,scope='selected_topics_full' if full else 'header_only',role='reserved_from_now' if day==19 else 'development_audit' if day==18 else 'provenance_only',firmware=info.get('ver_sw',''),branch=info.get('ver_sw_branch',''),target_matches=info.get('ver_sw')==SHA,config_group=group)
        rows.append(row)
        groups.append(dict(path=str(rel),group=group,FLAP_SLOW_EN=params.get('FLAP_SLOW_EN'),FLAP_V4_ENABLE=params.get('FLAP_V4_ENABLE'),FLAP_B2B_ADAPT=params.get('FLAP_B2B_ADAPT'),FLAP_B2B_CAP=params.get('FLAP_B2B_CAP'),FLAP_RATIO=params.get('FLAP_RATIO'),parameter_scope='initial_only' if not full else 'initial_and_changes'))
        for k,v in {**config['structural_parameters'],**config['output_mapping']}.items():
            actual=params.get(k)
            if actual is None or not np.isclose(actual,v,rtol=1e-6,atol=1e-6): mismatch.append(dict(path=str(rel),parameter=k,registered=v,actual=actual))
        if not full: continue
        for t,k,v in u.changed_parameters: changes.append(dict(path=str(rel),timestamp_us=t,parameter=k,value=v))
        for d in u.data_list:
            ts=d.data['timestamp'].astype(np.float64)/1e6; dt=np.diff(ts); positive=dt[dt>0]
            rates.append(dict(path=str(rel),topic=d.name,instance=d.multi_id,n=len(ts),span_s=float(ts[-1]-ts[0]) if len(ts)>1 else 0,median_hz=float(1/np.median(positive)) if len(positive) else '',max_gap_s=float(positive.max()) if len(positive) else '',nonpositive_intervals=int((dt<=0).sum()),fields=','.join(d.data)))
            if d.name=='vehicle_status':
                for field in ['nav_state','arming_state','failsafe']:
                    v=d.data[field]; starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]; ends=np.r_[starts[1:],len(v)-1]
                    for a,b in zip(starts,ends): modes.append(dict(path=str(rel),field=field,value=int(v[a]),start_s=float(ts[a]),end_s=float(ts[b]),duration_s=float(ts[b]-ts[a])))
            if d.name=='position_setpoint_triplet':
                data=d.data; mask=data['current.valid'].astype(bool)&np.isfinite(data['current.lat'])&np.isfinite(data['current.lon'])
                obs=np.unique(np.column_stack([data['current.lat'][mask],data['current.lon'][mask]]),axis=0)
                for plan,coords in plans.items():
                    if not coords or len(obs)==0: continue
                    c=np.array(coords,dtype=float)[:,:2]
                    distances=np.sqrt(((obs[:,None,0]-c[None,:,0])*111320)**2+((obs[:,None,1]-c[None,:,1])*111320*np.cos(np.deg2rad(obs[:,None,0])))**2)
                    count=int((distances.min(axis=1)<5).sum())
                    if count: missions.append(dict(log=str(rel),plan=plan,observed_unique_positions=len(obs),matched_positions_5m=count,max_nearest_distance_m=float(distances.min(axis=1).max()),evidence='coordinate_overlap_only_not_exact_mission_proof'))
        save('log_details/'+h+'.json',{'path':str(rel),'dropout_durations_ms':[x.duration for x in u.dropouts],'missing_requested_topics':sorted(set(topics)-{d.name for d in u.data_list}),'mission_result':[{k:v.tolist() for k,v in d.data.items()} for d in u.data_list if d.name=='mission_result']})
        print('full audit',rel,flush=True)
    for name,data in [('log_inventory.csv',rows),('firmware_log_map.csv',[{k:r[k] for k in ['path','firmware','branch','target_matches']} for r in rows]),('controller_config_groups.csv',groups),('topic_inventory.csv',rates),('mission_log_map.csv',missions),('mode_segments.csv',modes),('parameter_changes.csv',changes),('plan_inventory.csv',planrows),('structural_parameter_differences.csv',mismatch)]: table(name,data)
    save('verification.json',dict(log_count=len(rows),unique_logs=len(seen),full_topic_logs=sum(r['scope']=='selected_topics_full' for r in rows),reserved_logs=sum(r['day']==19 for r in rows),verified_dataset_artifacts=len(checks),target_matches=sum(r['target_matches'] for r in rows),structural_difference_rows=len(mismatch),parameter_changes=len(changes),test_opened=False))
    print(json.dumps(json.loads((OUT/'verification.json').read_text()),indent=2))

if __name__=='__main__': main()
