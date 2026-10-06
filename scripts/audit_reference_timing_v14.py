"""Read-only source-log audit of causal holds and model index contracts."""
import json
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train
from reference_timing_v14 import effective_indices
from run_reference_timing_v14 import OUT,CONTRACT
from system_identification.data.reference_windows import extract_window
TOPICS={'vehicle_angular_velocity':[f'xyz[{j}]' for j in range(3)],'wing_phase':['phase_rad','flap_frequency_hz'],
    'actuator_servos':[f'control[{j}]' for j in range(3)],'actuator_motors':['control[0]'],
    'vehicle_attitude':[f'q[{j}]' for j in range(4)],'vehicle_local_position':['vx','vy','vz'],
    'vehicle_attitude_setpoint':[f'q_d[{j}]' for j in range(4)]+['yaw_sp_move_rate','thrust_body[0]'],
    'manual_control_setpoint':['yaw','valid']}


def main():
    x,y,w=load_train();manifest_path=base.ROOT/'dataset/reference_dynamics_data_v1/manifest.json';m=json.loads(manifest_path.read_text());contract=json.loads(CONTRACT.read_text())
    assert base.sha(manifest_path)==contract['source_dataset_manifest_sha256']
    records=[];perwindow=[];pins={};checks=[]
    for cache,g in w.groupby('cache'):
        log=g.log_id.iloc[0];src=m['sources'][log];assert src['partition']=='train' and base.sha(src['path'])==src['sha256'];pins[log]=src
        with np.load(base.ROOT/cache) as z:a={k:z[k] for k in z.files}
        indices=np.unique(np.concatenate([np.arange(r.origin_index-26,r.origin_index+51) for r in g.itertuples()]));times=a['timestamp_us'][indices]
        ds={d.name:d.data for d in ULog(src['path'],message_name_filter_list=list(TOPICS)).data_list if d.multi_id==0}
        maps={};values_error={}
        for topic,fields in TOPICS.items():
            d=ds[topic];t=np.asarray(d['timestamp'],dtype=np.int64);sample=np.asarray(d.get('timestamp_sample',t),dtype=np.int64)
            ix=effective_indices(t,sample,times);v=np.column_stack([d[k][ix] for k in fields]).astype(float)
            pa=(times-t[ix])*1e-6;sa=(times-sample[ix])*1e-6;age=np.maximum(pa,sa)
            if topic=='vehicle_angular_velocity':expected=a['rates_b'][indices]
            elif topic=='wing_phase':expected=np.column_stack([a['phase_rad'][indices],a['state_features'][indices,11]])
            elif topic=='actuator_servos':expected=a['actuator_commands'][indices,1:]
            elif topic=='actuator_motors':expected=a['actuator_commands'][indices,:1]
            elif topic=='vehicle_attitude':expected=a['quaternion'][indices]
            elif topic=='vehicle_local_position':expected=a['velocity_n'][indices]
            elif topic=='vehicle_attitude_setpoint':
                q=v[:,:4]/np.linalg.norm(v[:,:4],axis=1,keepdims=True);qw,qx,qy,qz=q.T
                v=np.column_stack([np.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)),np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1)),v[:,4:]])
                expected=a['references'][indices,:4]
            else:
                v=np.where(a['nav_state'][indices,None]==15,v[:,:1],0);expected=a['references'][indices,4:5]
            error=float(np.max(abs(v-expected)));assert error<1e-5,(log,topic,error);values_error[topic]=error
            cached_age=a['manual_reference_age_s'][indices] if topic=='manual_control_setpoint' else a['age_s'][indices,list(a['age_topic_names']).index(topic)]
            np.testing.assert_allclose(age,cached_age,atol=1e-7,rtol=1e-6)
            hold=np.diff(times)==20000;period=(t[ix[ix>0]]-t[ix[ix>0]-1])*1e-6
            records.append(dict(log_id=log,topic=topic,has_timestamp_sample='timestamp_sample' in d,n_grid=len(times),
                pub_age_median_ms=float(np.median(pa)*1000),sample_age_median_ms=float(np.median(sa)*1000),age_p95_ms=float(np.quantile(age,.95)*1000),
                publication_minus_sample_median_ms=float(np.median(t[ix]-sample[ix])*.001),publication_period_median_ms=float(np.median(period)*1000),
                repeated_hold_fraction=float(np.mean(np.diff(ix)[hold]==0)),value_max_abs=error))
            maps[topic]=(times-t[ix],times-sample[ix])
            for r in g.itertuples():
                mask=(indices>=r.origin_index-25)&(indices<=r.origin_index+50);at=np.flatnonzero(indices==r.origin_index)[0]
                history_index=np.flatnonzero(indices==r.origin_index-1)[0] if topic.startswith('actuator_') else at
                perwindow.append(dict(array_index=r.array_index,log_id=log,topic=topic,mean_age_ms=float(age[mask].mean()*1000),origin_age_ms=float(age[at]*1000),
                    last_model_history_age_at_origin_ms=float(age[history_index]*1000+(20 if topic.startswith('actuator_') else 0))))
        for r in g.itertuples():
            inp,lab=extract_window(a,r.origin_index)
            for k,v in inp.items():np.testing.assert_allclose(x[k][r.array_index].numpy(),v,atol=1e-6,rtol=1e-6)
            for k,v in lab.items():np.testing.assert_allclose(y[k][r.array_index].numpy(),v,atol=1e-6,rtol=1e-6)
        checks.append(dict(log_id=log,windows=len(g),all_source_rows_causal=True,all_window_indices_match=True,cache_values_max_abs=max(values_error.values())))
        print(f'Audited {log}',flush=True)
    pd.DataFrame(records).to_csv(OUT/'source_timing_per_flight.csv',index=False);pd.DataFrame(perwindow).to_csv(OUT/'source_timing_per_window.csv',index=False)
    base.write(OUT/'source_timing_audit.json',dict(source_manifest_sha256=base.sha(manifest_path),source_logs=pins,checks=checks,
        limits='publication/sample bookkeeping only; sample_timestamp absent means physicalsampleage unknown. No independent sensor/servo latency measurement; intentionalone-step historyactuator offset retained.',sealed_test_accessed=False))

if __name__=='__main__':main()
