"""Exact-revision, five-log replay entry audit and causal origin-state inventory."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from audit_manual_chain_v21 import PX4,COMMIT
from audit_controller_support_v26 import verify_parent
from train_joint_increment_v34 import DATA,read
from replay_preflight_v37 import available_index,infer_cycle_dt

OUT=DATA.parent/'control_windows_v37'
PARENTS=[DATA.parent/f'control_windows_v{n}' for n in [23,24,28,29,35,36]]
TOPICS=['rate_ctrl_status','rate_ctrl_terms','gain_compression','vehicle_angular_velocity','vehicle_attitude',
        'vehicle_attitude_setpoint','vehicle_rates_setpoint','airspeed_validated','vehicle_control_mode',
        'vehicle_status','vehicle_land_detected','control_allocator_status','actuator_motors','actuator_servos',
        'vehicle_torque_setpoint','vehicle_thrust_setpoint','manual_control_setpoint','battery_status']
SOURCES=['src/lib/rate_control/gain_compression.hpp','src/lib/rate_control/gain_compression.cpp',
         'src/lib/rate_control/rate_control.cpp','src/lib/rate_control/rate_control.hpp',
         'src/modules/fw_rate_control/FixedwingRateControl.cpp','src/modules/fw_rate_control/FixedwingRateControl.hpp',
         'src/modules/fw_att_control/FixedwingAttitudeControl.cpp','src/modules/fw_att_control/FixedwingAttitudeControl.hpp',
         'src/modules/fw_att_control/fw_roll_controller.cpp','src/modules/fw_att_control/fw_roll_controller.h',
         'src/modules/fw_att_control/fw_pitch_controller.cpp','src/modules/fw_att_control/fw_pitch_controller.h',
         'src/modules/fw_att_control/fw_yaw_controller.cpp','src/modules/fw_att_control/fw_yaw_controller.h']


def main():
    for root in PARENTS:verify_parent(root)
    source={p:subprocess.check_output(['git','-C',str(PX4),'show',f'{COMMIT}:{p}']) for p in SOURCES}
    paths=['scripts/audit_replay_preflight_v37.py','scripts/replay_preflight_v37.py','tests/test_replay_preflight_v37.py',
        'src/system_identification/control/px4_pitch.py','src/system_identification/control/px4_flight_chain.py']
    protocol=dict(stage='v37 same-controller replay entry audit',firmware=COMMIT,
        parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in PARENTS},
        code={p:base.sha(base.ROOT/p) for p in paths},
        scope='Only5originaltrainULogs and407origins from v23 manifest; no model run/training, no firmware edits or deployment. Read full topic streams for source integrity; diagnostics restricted to original parent intervals, causal origin snapshots use only available past messages.',
        initialization='Latest publication with BOTH publication and sample<=origin. No future controller state, no interpolation; save source times and age separately. Snapshots are asynchronous partial observations, NOT restorable exact controller state.',
        clock='Source dt bounds2..40ms. Terms paired with first post status within1ms and before next terms. Infer per-axis dt from post-minus-pre integral divided by Ki*error*(1-(error/400deg)^2), error=Pterm/Kp; discard denominator<=.005 or near integral clamp and estimates outside source bounds. Conditional diagnostic assumes no reset/antiwindup, never substitutes for exact native scheduler.',
        boundary='Candidate intended attitude-reference+thrust conditioned inner-loop replay, not Mission navigation/TECS or exogenous manual experiment. Recorded future references are offline conditioning; future logged actuator commands/controller states forbidden after origin. Model dt contract20ms; native faster substeps require separately declared approximation and teacher replay evidence.',
        native_telemetry='Report gaps only within original windows. Logged topic cadence cannot certify complete producer updates.',
        state_gate='Require signed HPF, previous effort, compression gain/cached gain, integral phase, airspeed filter, attitude yaw state, controller clocks and prior allocator feedback for exact replay. Missing states must be represented as uncertainty, not silently zeroed.',
        phase_contract=read(DATA/'dataset_manifest.json')['phase_contract'],frequency_contract=read(DATA/'dataset_manifest.json')['frequency_contract'],
        model_promoted=False,training_started=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    OUT.mkdir(exist_ok=False);base.write(OUT/'protocol.json',protocol);(OUT/'source').mkdir()
    provenance={}
    for name,value in source.items():
        path=OUT/'source'/Path(name).name;path.write_bytes(value)
        provenance[name]=dict(snapshot=str(path.relative_to(OUT)),sha256=base.sha(path),
            blob=subprocess.check_output(['git','-C',str(PX4),'rev-parse',f'{COMMIT}:{name}'],text=True).strip())
    base.write(OUT/'source_provenance.json',dict(commit=COMMIT,files=provenance,limit='Logged revision is source provenance, not binary attestation.'))
    parent=read(DATA.parent/'control_windows_v23/dataset_manifest.json');w=pd.read_csv(DATA/'windows.csv')
    snapshots=[];ages=[];cadence=[];schemas=[];clock=[];parameters=[];lineage=[];compat=[]
    for entry in parent['sources']:
        src=entry['source'];assert src['partition']=='train' and base.sha(src['path'])==src['sha256']
        assert base.sha(base.ROOT/entry['cache'])==entry['cache_sha256']
        log=entry['log_id'];g=w[w.log_id==log]
        with np.load(base.ROOT/entry['cache']) as z:times=z['timestamp_us']
        u=ULog(src['path'],message_name_filter_list=TOPICS);assert u.msg_info_dict['ver_sw']==COMMIT and not u.changed_parameters
        ds={d.name:d.data for d in u.data_list if d.multi_id==0};p=u.initial_parameters
        parameters.append(dict(log_id=log,values={k:v for k,v in p.items() if k.startswith(('FW_','CA_','VT_','TRIM_','PWM_MAIN_','FLAP_'))},changes=0))
        # Existing adapter requires a parameter absent in this exact older revision.
        from system_identification.control.px4_flight_chain import FlightInnerLoop
        try:FlightInnerLoop(p);result='constructed (not proof of source parity)'
        except (KeyError,ValueError) as exc:result=type(exc).__name__+': '+str(exc)
        compat.append(dict(log_id=log,existing_adapter_firmware='e624a99f2955addbf76681e636c44162a0c03055',logged_firmware=COMMIT,
            FLAP_SLOW_EN_present='FLAP_SLOW_EN' in p,constructor_result=result))
        for topic in TOPICS:
            if topic not in ds:schemas.append(dict(log_id=log,topic=topic,present=False));continue
            d=ds[topic];schemas.append(dict(log_id=log,topic=topic,present=True,fields=list(d)))
            for r in g.itertuples():
                tt=d['timestamp'];ix=np.flatnonzero((tt>=r.start_timestamp_us)&(tt<=r.end_timestamp_us))
                delta=np.diff(tt[ix].astype(float))/1000
                if len(delta):cadence.append(dict(log_id=log,array_index=r.array_index,topic=topic,rows=len(ix),median_ms=np.median(delta),max_gap_ms=delta.max()))
        for r in g.itertuples():
            origin=int(times[r.origin_index]);record=dict(array_index=r.array_index,log_id=log,mode=int(r.mode),origin_us=origin,topics={},exact_restore_ready=False)
            for topic,d in ds.items():
                idx=available_index(d,origin)
                if idx is None:record['topics'][topic]=dict(available=False);continue
                pub=int(d['timestamp'][idx]);sample=int(d.get('timestamp_sample',d['timestamp'])[idx])
                assert pub<=origin and sample<=origin
                record['topics'][topic]=dict(available=True,source_index=idx,publication_us=pub,sample_us=sample,
                    has_sample='timestamp_sample' in d,publication_age_ms=(origin-pub)/1000,
                    values={k:(np.asarray(v[idx]).item() if np.isfinite(v[idx]) else None) for k,v in d.items() if k not in ['timestamp','timestamp_sample']},
                    nonfinite_fields=[k for k,v in d.items() if not np.isfinite(v[idx])])
                ages.append(dict(array_index=r.array_index,log_id=log,topic=topic,publication_age_ms=(origin-pub)/1000,source_index=idx))
            snapshots.append(record)
        # Diagnostics only on native terms inside the original parent intervals.
        term=ds['rate_ctrl_terms'];status=ds['rate_ctrl_status'];mask=np.zeros(len(term['timestamp']),bool)
        for r in g.itertuples():mask|=(term['timestamp']>=r.start_timestamp_us)&(term['timestamp']<=r.end_timestamp_us)
        tx=np.flatnonzero(mask);tt=term['timestamp'][tx].astype(np.int64)
        si=np.searchsorted(status['timestamp'],tt);good=si<len(status['timestamp']);si=np.minimum(si,len(status['timestamp'])-1)
        gap=status['timestamp'][si].astype(np.int64)-tt
        nextt=term['timestamp'][np.minimum(tx+1,len(term['timestamp'])-1)]
        good&=(gap>=0)&(gap<=1000)&(status['timestamp'][si]<nextt)
        assert len(np.unique(si[good]))==good.sum()
        for j,axis in enumerate(['roll','pitch','yaw']):
            key=['R','P','Y'][j];tr=tx[good];sr=si[good]
            error=term[f'p_term[{j}]'][tr].astype(float)/p[f'FW_{key}R_P']
            pre=term[f'i_term[{j}]'][tr];post=status[axis+'speed_integ'][sr]
            est=infer_cycle_dt(pre,post,error,p[f'FW_{key}R_I'],p[f'FW_{key}R_IMAX'])
            for i in np.flatnonzero(np.isfinite(est)):
                clock.append(dict(log_id=log,axis=axis,terms_publication_us=int(term['timestamp'][tr[i]]),
                    sample_us=int(term['timestamp_sample'][tr[i]]),status_publication_us=int(status['timestamp'][sr[i]]),
                    pre_integral=float(pre[i]),post_integral=float(post[i]),error=float(error[i]),dt_s=float(est[i])))
        lineage.append(entry)
        print(f'{log}: causal snapshots for {len(g)} origins; source revision verified',flush=True)
    assert sorted(r['array_index'] for r in snapshots)==list(range(407))
    base.write(OUT/'origin_observations.json',sorted(snapshots,key=lambda r:r['array_index']))
    base.write(OUT/'topic_schemas.json',schemas);base.write(OUT/'parameters_by_flight.json',parameters)
    base.write(OUT/'adapter_compatibility.json',compat)
    pd.DataFrame(ages).to_csv(OUT/'origin_observation_age.csv',index=False)
    pd.DataFrame(cadence).to_csv(OUT/'topic_cadence.csv',index=False)
    c=pd.DataFrame(clock);c.to_csv(OUT/'conditional_clock_estimates.csv',index=False)
    c.groupby(['log_id','axis']).dt_s.agg(['count','median','min','max']).to_csv(OUT/'clock_summary.csv')
    pd.DataFrame(cadence).groupby('topic').agg(median_of_window_medians_ms=('median_ms','median'),max_gap_ms=('max_gap_ms','max')).to_csv(OUT/'cadence_summary.csv')
    base.write(OUT/'dataset_manifest.json',dict(sources=lineage,windows=407,partitions=['original_train'],
        parent_manifest_sha256=base.sha(DATA/'dataset_manifest.json'),phase_contract=protocol['phase_contract'],frequency_contract=protocol['frequency_contract']))
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_replay_preflight_v37.py'],capture_output=True,text=True)
    base.write(OUT/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0
    for root in PARENTS:verify_parent(root)
    base.write(OUT/'verification.json',dict(status='passed',causal_partial_snapshots=407,exact_restore_claimed=False,
        raw_hash_and_firmware_verified=True,tests_passed=3,model_or_controller_replay_started=False,
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))


if __name__=='__main__':main()
