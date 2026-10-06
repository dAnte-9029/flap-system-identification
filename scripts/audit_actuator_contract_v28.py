"""Five train-log actuator chain audit and immutable conditional-input materialization."""
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from audit_manual_chain_v21 import PX4,COMMIT,hold
from audit_controller_support_v26 import verify_parent
from actuator_contract_v28 import CHANNELS,PORTS,held_commands,output_pwm,allocation,stats

OUT=base.ROOT/'docs/analysis/results/control_windows_v28';V23=OUT.parent/'control_windows_v23';V27=OUT.parent/'control_windows_v27'
TOPICS=['actuator_motors','actuator_servos','actuator_outputs','vehicle_torque_setpoint','vehicle_thrust_setpoint','control_allocator_status','actuator_armed']
SOURCES=['src/modules/control_allocator/ControlAllocator.cpp',
 'src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessFixedWing.cpp',
 'src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessFixedWing.hpp',
 'src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessRotors.cpp',
 'src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessControlSurfaces.cpp',
 'src/lib/control_allocation/actuator_effectiveness/ActuatorEffectiveness.hpp',
 'src/lib/control_allocation/control_allocation/ControlAllocation.cpp',
 'src/lib/control_allocation/control_allocation/ControlAllocationPseudoInverse.cpp',
 'src/lib/mixer_module/functions/FunctionMotors.hpp','src/lib/mixer_module/functions/FunctionServos.hpp',
 'src/lib/mixer_module/mixer_module.cpp','src/lib/mixer_module/output_functions.yaml','src/drivers/px4io/px4io.cpp',
 'msg/versioned/ActuatorMotors.msg','msg/versioned/ActuatorServos.msg','msg/ActuatorOutputs.msg']


def selected(d,intervals):
    mask=np.zeros(len(d['timestamp']),bool)
    for lo,hi in intervals:mask|=(d['timestamp']>=lo)&(d['timestamp']<=hi)
    return np.flatnonzero(mask)


def main():
    for root in [V23,V27]:verify_parent(root)
    parent=json.loads((V23/'dataset_manifest.json').read_text());w=pd.read_csv(V23/'windows.csv')
    assert len(w)==407 and w.partition.eq('train').all()
    # Resolve all exact-revision source paths before creating the frozen run.
    source={p:subprocess.check_output(['git','-C',str(PX4),'show',f'{COMMIT}:{p}']) for p in SOURCES}
    OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
    paths=[Path(__file__),base.ROOT/'scripts/actuator_contract_v28.py',base.ROOT/'tests/test_actuator_contract_v28.py']
    base.write(OUT/'protocol.json',dict(stage='v28 actuator input contract audit',parents={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in [V23,V27]},
        code={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},source_firmware=COMMIT,
        scope='Original407trainparents only;5authorized source ULogs; no training, selection, future state features, developmentvalidation/sealed/reserved.',
        window_support='77grid instants origin-26..origin+50 inclusive. Preserve existing history command origin-26..origin-1 and candidate future commands origin..origin+49. Labels remain origin+1..origin+50.',
        commands='[motor0,servo0,servo1,servo2] normalized post-allocation, independently latest messages with max(publication,sample)<=query. Store both times; sample timestamp is originating control/gyro sample, not physical actuator application.',
        allocation='Configured fixedwing pseudoinverse from source/parameters; join unique exact timestamp_sample among motor,servo,torque,thrust; report duplicate/unmatched counts. No fitted control mapping.',
        pwm='Normal active output mapping using fixedfunction,reverse,min/max,optionalcenter,THR_MDL_FAC; compare causal lastlogged commands with native actuator_outputs instance0, no fitted time shift. Asynchronous logs need not coincide with commands applied by driver.',
        coverage='Native update counts and gaps within each original parent; grid publication/sample ages; future range/change counts, saturation/status tags retained. No output response sign or model error selection.',
        independence='Within-window first differences motor/common/differential/yaw; perflight raw SVD and correlation, descriptive only, no causal identifiability claim.',
        phase_contract=parent['phase_contract'],frequency_contract=parent['frequency_contract'],training_started=False,sealed_test_accessed=False))
    blobs={}
    for p,data in source.items():
        path=OUT/'source'/Path(p).name;path.write_bytes(data)
        blobs[p]=dict(snapshot=str(path.relative_to(OUT)),sha256=base.sha(path),blob=subprocess.check_output(['git','-C',str(PX4),'rev-parse',f'{COMMIT}:{p}'],text=True).strip())
    base.write(OUT/'source_provenance.json',dict(commit=COMMIT,files=blobs,limit='Logged revision identifies source, not independent attestation of binary/uncommitted patches or hardware pose.'))
    command=np.zeros((407,77,4),np.float32);pub=np.zeros((407,77,2),np.int64);sample=pub.copy();query=np.zeros((407,77),np.int64);pwm=np.zeros((407,77,4),np.float32)
    grids=[];windows=[];timings=[];pairs=[];output_rows=[];parameters=[];presence=[];native_counts=[];checks=[];independence=[]
    for entry in parent['sources']:
        log=entry['log_id'];src=entry['source'];assert src['partition']=='train' and base.sha(src['path'])==src['sha256']
        assert base.sha(base.ROOT/entry['cache'])==entry['cache_sha256']
        with np.load(base.ROOT/entry['cache']) as z:cache={k:z[k] for k in ['timestamp_us','actuator_commands']}
        g=w[w.log_id==log];intervals=g[['start_timestamp_us','end_timestamp_us']].to_numpy();u=ULog(src['path'],message_name_filter_list=TOPICS)
        assert u.msg_info_dict['ver_sw']==COMMIT and not u.changed_parameters
        ds={d.name:d.data for d in u.data_list if d.multi_id==0};assert set(TOPICS)<=ds.keys()
        presence.extend(dict(log_id=log,topic=d.name,instance=d.multi_id,rows=len(d.data['timestamp'])) for d in u.data_list)
        p=u.initial_parameters;chosen={k:v for k,v in p.items() if k.startswith(('CA_','PWM_MAIN_','THR_MDL_')) or k=='COM_SPOOLUP_TIME'}
        parameters.append(dict(log_id=log,values=chosen))
        assert p['CA_AIRFRAME']==1 and p['CA_METHOD']==2 and p['CA_ROTOR_COUNT']==1 and p['CA_SV_CS_COUNT']==3
        assert [p[f'PWM_MAIN_FUNC{i}'] for i in PORTS]==[101,201,202,203]
        assert p['CA_ROTOR0_PY']==p['CA_ROTOR0_PZ']==0 and p['CA_ROTOR0_CT']>0
        assert all(p[f'CA_SV_CS{j}_{field}']==0 for j in range(3) for field in ['TRIM','FLAP','SPOIL'])
        assert all(v==0 for k,v in p.items() if re.fullmatch(r'CA_(R|SV)\d+_SLEW',k))
        motor,servo,output=[ds[t] for t in TOPICS[:3]]
        idx=np.unique(np.concatenate([np.arange(i-26,i+51) for i in g.origin_index]));t=cache['timestamp_us'][idx].astype(np.int64)
        a,pt,st,_=held_commands(motor,servo,t);np.testing.assert_array_equal(a,cache['actuator_commands'][idx])
        oi=hold(output,t);observed=np.column_stack([output[f'output[{port-1}]'][oi] for port in PORTS]);assert (output['noutputs'][oi]>=5).all()
        mi=hold(motor,t);assert (motor['reversible_flags'][mi]==0).all()
        ai=hold(ds['actuator_armed'],t);armed=ds['actuator_armed'];assert (armed['armed'][ai]==1).all()
        for flag in ['lockdown','kill','termination','in_esc_calibration_mode']:assert not armed[flag][ai].any()
        ca=ds['control_allocator_status'];ci=hold(ca,t)
        frame=pd.DataFrame(dict(log_id=log,cache_index=idx,query_us=t,motor_publication_us=pt[:,0],servo_publication_us=pt[:,1],motor_sample_us=st[:,0],servo_sample_us=st[:,1],output_publication_us=output['timestamp'][oi],
            allocator_status_age_ms=(t-ca['timestamp'][ci])*.001,torque_achieved=ca['torque_setpoint_achieved'][ci],thrust_achieved=ca['thrust_setpoint_achieved'][ci],
            failure_mask=ca['handled_motor_failure_mask'][ci],stop_mask=ca['motor_stop_mask'][ci]))
        for j,ch in enumerate(CHANNELS):
            frame[ch]=a[:,j];frame['pwm_'+ch]=observed[:,j];frame['saturation_'+ch]=ca[f'actuator_saturation[{j}]'][ci]
        grids.append(frame)
        for r in g.itertuples():
            loc=np.searchsorted(idx,np.arange(r.origin_index-26,r.origin_index+51));q=t[loc]
            assert q[0]==r.start_timestamp_us and q[-1]==r.end_timestamp_us
            command[r.array_index]=a[loc];pub[r.array_index]=pt[loc];sample[r.array_index]=st[loc];query[r.array_index]=q;pwm[r.array_index]=observed[loc]
            for j,ch in enumerate(CHANNELS):
                future=a[loc[26:76],j];topic='actuator_motors' if j==0 else 'actuator_servos';d=ds[topic]
                ix=hold(d,q);native=selected(d,[(q[0],q[-1])]);h=0 if j==0 else 1
                windows.append(dict(array_index=r.array_index,log_id=log,mode=r.mode,channel=ch,native_rows=len(native),
                    future_distinct_messages=len(np.unique(ix[26:76])),future_changes=int(np.count_nonzero(abs(np.diff(future))>1e-6)),
                    future_min=float(future.min()),future_max=float(future.max()),future_range=float(np.ptp(future)),future_step_rms=float(np.sqrt(np.mean(np.diff(future)**2))),
                    publication_age_max_ms=float(((q-pt[loc,h])*.001).max()),sample_age_max_ms=float(((q-st[loc,h])*.001).max()),
                    near_bound_fraction=float(np.mean(abs(future)>=.99)) if j else float(np.mean((future<=.01)|(future>=.99)))))
            for topic in TOPICS:
                d=ds[topic];native=selected(d,[(q[0],q[-1])]);gaps=np.diff(d['timestamp'][native].astype(np.int64))*.001
                if len(gaps):timings.append(dict(array_index=r.array_index,log_id=log,topic=topic,n=len(native),period_median_ms=float(np.median(gaps)),period_max_ms=float(gaps.max())))
        # Exact sample joins, exclude ambiguous repeated sample identifiers explicitly.
        tables=[]
        for topic,cols in [('actuator_motors',['control[0]']),('actuator_servos',[f'control[{j}]' for j in range(3)]),
                           ('vehicle_torque_setpoint',[f'xyz[{j}]' for j in range(3)]),('vehicle_thrust_setpoint',['xyz[0]'])]:
            d=ds[topic];ix=selected(d,intervals);table=pd.DataFrame(dict(sample_us=d['timestamp_sample'][ix],**{topic+'_'+k:d[k][ix] for k in cols}))
            duplicate=table.sample_us.duplicated(keep=False);native_counts.append(dict(log_id=log,topic=topic,native_rows=len(ix),duplicate_sample_rows=int(duplicate.sum())))
            tables.append(table[~duplicate])
        exact=tables[0]
        for table in tables[1:]:exact=exact.merge(table,on='sample_us',validate='one_to_one')
        expected=allocation(exact[[f'vehicle_torque_setpoint_xyz[{j}]' for j in range(3)]].to_numpy(),exact['vehicle_thrust_setpoint_xyz[0]'].to_numpy(),p)
        values=exact[['actuator_motors_control[0]']+[f'actuator_servos_control[{j}]' for j in range(3)]].to_numpy()
        exact.insert(0,'log_id',log)
        for j,ch in enumerate(CHANNELS):exact[ch+'_allocation_residual']=values[:,j]-expected[:,j]
        pairs.append(exact)
        ni=selected(output,intervals);qt=output['timestamp'][ni].astype(np.int64);causal,cp,cs,_=held_commands(motor,servo,qt)
        predicted=output_pwm(causal,p);actual=np.column_stack([output[f'output[{port-1}]'][ni] for port in PORTS])
        for j,ch in enumerate(CHANNELS):
            output_rows.append(pd.DataFrame(dict(log_id=log,channel=ch,timestamp_us=qt,command=causal[:,j],command_publication_age_ms=(qt-cp[:,0 if j==0 else 1])*.001,
                predicted_pwm=predicted[:,j],logged_pwm=actual[:,j],residual_us=actual[:,j]-predicted[:,j])))
        fut=command[g.array_index,26:76].astype(float);transformed=fut.copy();transformed[:,:,1]=(fut[:,:,1]+fut[:,:,2])*.5;transformed[:,:,2]=(fut[:,:,2]-fut[:,:,1])*.5
        delta=np.diff(transformed,axis=1).reshape(-1,4);singular=np.linalg.svd(delta-delta.mean(0),compute_uv=False)
        independence.append(dict(log_id=log,channels=['motor','common','differential','yaw'],delta_std=delta.std(0).tolist(),singular_values=singular.tolist(),
            singular_min_max_ratio=float(singular[-1]/singular[0]) if singular[0]>0 else None,correlation=np.corrcoef(delta,rowvar=False).tolist()))
        checks.append(dict(log_id=log,windows=len(g),grid_points=len(t),cache_commands_exact=True,raw_firmware_and_hashes=True,exact_allocation_pairs=len(exact),
            allocation_max_abs=float(np.max(abs(values-expected))),source=entry))
        print('Audited',log,len(g),'windows',flush=True)
    reference=parameters[0]['values'];assert all(item['values']==reference for item in parameters)
    with np.load(V23/'windows.npz') as z:np.testing.assert_array_equal(command[:,:26],z['input_history_actuator'])
    assert (pub<=query[:,:,None]).all() and (sample<=query[:,:,None]).all()
    np.savez_compressed(OUT/'actuator_grid.npz',array_index=np.arange(407),command_grid=command,query_us=query,publication_us=pub,sample_us=sample,pwm_grid=pwm)
    pd.concat(grids,ignore_index=True).to_csv(OUT/'grid_points.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'coverage_per_window.csv',index=False);pd.DataFrame(timings).to_csv(OUT/'native_timing.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(OUT/'allocation_exact_pairs.csv',index=False);pd.concat(output_rows,ignore_index=True).to_csv(OUT/'pwm_comparison.csv',index=False)
    pd.DataFrame(native_counts).to_csv(OUT/'native_sample_counts.csv',index=False);pd.DataFrame(presence).to_csv(OUT/'topic_instances.csv',index=False)
    base.write(OUT/'parameters_by_flight.json',parameters);base.write(OUT/'input_independence.json',independence)
    base.write(OUT/'verification.json',dict(status='passed',checks=checks,all_five_configs_equal=True,past_command_tensor_matches_v23=True,
        causal_timestamps=True,windows_retained=407,training_started=False,validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
