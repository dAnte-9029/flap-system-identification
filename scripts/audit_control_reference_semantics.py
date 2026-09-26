"""Train-only causal timestamp/reference-source and conditional relevance audit."""
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog

import run_paper_rollout_horizon_ablation as source
from identify_control_dynamics_local import design,fit,predict

m=source.m
BASE=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=BASE/'reference_semantics'


def past_values(data,keys,times):
    """Use only messages published and sampled at or before the query time."""
    t=np.asarray(data['timestamp'],dtype=np.int64)
    sample=np.asarray(data.get('timestamp_sample',t),dtype=np.int64)
    if np.any(np.diff(t)<0):raise ValueError('unsorted topic timestamps')
    effective=np.maximum(t,sample)
    # Monotonic in publication order is required to avoid silent time reordering.
    if np.any(np.diff(effective)<0):raise ValueError('nonmonotonic effective timestamp')
    ix=np.searchsorted(effective,times,side='right')-1
    ok=ix>=0;safe=ix.clip(0)
    result=np.column_stack([np.asarray(data[k])[safe] for k in keys]).astype(float)
    result[~ok]=np.nan
    age=np.where(ok,(times-t[safe])*1e-6,np.inf)
    sample_age=np.where(ok,(times-sample[safe])*1e-6,np.inf)
    return result,age,sample_age


def pitch_from_quaternion(q):
    q=np.asarray(q,dtype=float);norm=np.linalg.norm(q,axis=-1,keepdims=True)
    valid=np.isfinite(q).all(-1)&(norm[:,0]>.5)
    q=q/np.maximum(norm,1e-12)
    w,x,y,z=q.T
    return np.where(valid,np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.nan)


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    previous=m.read_json(BASE/'reference_availability/completion.json');m.verify_pins(previous['outputs'])
    ep=m.read_json(BASE/'ensemble_candidate/protocol.json');m.verify_pins(ep['pins'])
    sp,batches,_=source.inputs();b=batches['train'];origins=pd.read_csv(m.BASE/'train_origins.csv')
    np.testing.assert_array_equal(origins.window_id,b.trajectory.window_ids)
    mp=m.ROOT/sp['manifest_path'];manifest=m.read_json(mp)
    samples=pd.read_parquet(mp.parent/'samples_train.parquet',columns=['log_id','segment_id','sample_in_segment','timestamp_us','nav_state','valid_core'])
    samples=samples[samples.valid_core]
    lookup=samples.set_index(['log_id','segment_id','sample_in_segment'])
    assert lookup.index.is_unique
    idx=pd.MultiIndex.from_frame(origins[['log_id','segment_id','start_sample_in_segment']].rename(columns={'start_sample_in_segment':'sample_in_segment'}))
    current=lookup.reindex(idx)
    np.testing.assert_array_equal(current.timestamp_us,origins.origin_timestamp_us)
    source_dir=m.ROOT/'docs/analysis/results/flight_chain_alignment/sources/src/modules/fw_att_control'
    pins=dict(ep['pins'])
    for path in [Path(__file__),m.ROOT/'scripts/identify_control_dynamics_local.py',BASE/'local_identification/protocol.json',
        BASE/'reference_availability/raw_sources.json',source_dir/'FixedwingAttitudeControl.cpp',source_dir/'fw_pitch_controller.cpp']:
        pins[m.rel(path)]=m.file_hash(path)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        heldout_accessed=False,partitions=['train'],model_training=False,
        selection='original train origins only; currentnav15 AND fresh manual/attitude-enabled,climb-rate-disabled control-mode flags; no selection by future outcome',
        alignment='last message whose publication AND sample timestamps<=query; reference queried at t0,t0-100ms,t0-200ms',
        freshness='manual<=300ms, attitude/rate setpoint<=50ms, mode<=300ms, manual valid; raw times/ages kept for all origins',
        source_audit='compare held manual-derived pitch=-manual.pitch*FW_MAN_P_MAX+FW_PSP_OFF clipped to bounds versus quaternion attitude target; e624 source reference only, not all41 firmware identity assumption',
        conditional_stage='future100/200ms common mean target; state+past-command base from local identification versus adding three past/current reference values',
        models=['state_past_controls','plus_manual_pitch','plus_attitude_pitch','plus_rate_pitch'],
        estimator='fixed ridge0.01, five whole-train-flight folds, flight equal; both arms evaluated on SAME fresh-reference origins; diagnostic only',
        relevance='report OOF predictive gain and per-flight directions; never treat input relevance as instrument validity',
        exogeneity='not assumed for pilot reference; rate setpoint explicitly state dependent; no future reference input to predictor or model',
        stage='no second-stage plant fitting until provenance/timing/relevance reviewed'))
    raw=m.read_json(BASE/'reference_availability/raw_sources.json')
    records=[];params=[];reference=np.full((len(origins),3,3),np.nan)
    admitted=np.zeros(len(origins),bool)
    for log,g in origins.groupby('log_id',sort=True):
        src=raw[log];assert m.file_hash(Path(src['path']))==src['sha256']
        ul=ULog(src['path'],message_name_filter_list=['manual_control_setpoint','vehicle_attitude_setpoint','vehicle_rates_setpoint','vehicle_control_mode'])
        topics={d.name:d.data for d in ul.data_list if d.multi_id==0}
        needed=['manual_control_setpoint','vehicle_attitude_setpoint','vehicle_rates_setpoint','vehicle_control_mode']
        if not all(t in topics for t in needed):raise ValueError(f'missing reference topic for {log}')
        values=ul.initial_parameters
        if 'FW_MAN_P_MAX' not in values or 'FW_PSP_OFF' not in values:raise ValueError(f'missing pitch mapping for {log}')
        max_pitch=float(values['FW_MAN_P_MAX'])*np.pi/180;off=float(values['FW_PSP_OFF'])*np.pi/180
        params.append(dict(log_id=log,firmware_info=str(ul.msg_info_dict.get('ver_sw','unknown')),max_pitch_rad=max_pitch,pitch_offset_rad=off))
        indices=g.index.to_numpy();times=g.origin_timestamp_us.to_numpy(dtype=np.int64)
        histories=[];ages=[]
        for lag in [0,100000,200000]:
            query=times-lag
            manual,ma,msa=past_values(topics['manual_control_setpoint'],['pitch','valid'],query)
            quat,aa,_=past_values(topics['vehicle_attitude_setpoint'],[f'q_d[{i}]' for i in range(4)],query)
            rate,ra,_=past_values(topics['vehicle_rates_setpoint'],['pitch'],query)
            att=pitch_from_quaternion(quat)
            histories.append(np.column_stack([manual[:,0],att,rate[:,0]]))
            ages.append(np.column_stack([ma,aa,ra]))
            if lag==0:manual_now=manual;sample_age_now=msa
        h=np.stack(histories,1);a=np.stack(ages,1);reference[indices]=h
        flags,mode_age,_=past_values(topics['vehicle_control_mode'],['flag_control_manual_enabled','flag_control_attitude_enabled','flag_control_climb_rate_enabled'],times)
        nav=current.nav_state.to_numpy()[indices]
        mode_ok=(nav==15)&(flags[:,0]==1)&(flags[:,1]==1)&(flags[:,2]==0)&(mode_age<=.3)
        fresh=np.isfinite(h).all((1,2))&(a[:,:,0]<=.3).all(1)&(a[:,:,1:]<=.05).all((1,2))&(manual_now[:,1]==1)
        admit=mode_ok&fresh;admitted[indices]=admit
        mapped=np.clip(-manual_now[:,0]*max_pitch+off,-max_pitch,max_pitch)
        for j,i in enumerate(indices):
            records.append(dict(index=int(i),window_id=origins.window_id.iloc[i],log_id=log,nav_state=int(nav[j]),
                current_mode_stabilized=bool(mode_ok[j]),admitted=bool(admit[j]),manual_age_s=float(a[j,0,0]),
                manual_sample_age_s=float(sample_age_now[j]),attitude_age_s=float(a[j,0,1]),rate_age_s=float(a[j,0,2]),mode_age_s=float(mode_age[j]),
                manual_pitch=float(h[j,0,0]),attitude_pitch=float(h[j,0,1]),rate_pitch=float(h[j,0,2]),
                manual_mapped_attitude=float(mapped[j]),mapping_residual=float(h[j,0,1]-mapped[j])))
        print('references aligned',log,int(admit.sum()),flush=True)
    rec=pd.DataFrame(records).sort_values('index');rec.to_csv(OUT/'per_origin.csv',index=False)
    pd.DataFrame(params).to_csv(OUT/'mapping_parameters.csv',index=False)
    np.savez_compressed(OUT/'references.npz',past_current_reference=reference,admitted=admitted,window_ids=b.trajectory.window_ids.astype(str))
    folds=m.read_json(BASE/'local_identification/protocol.json')['folds']
    ids=b.trajectory.log_ids[admitted];fold=np.array([folds[x] for x in ids])
    assert len(set(ids))>=10 and len(ids)>=500,'insufficient reference coverage for grouped diagnostic'
    x0=design(b,5)[0][admitted]
    rows=[];predictions=[]
    for horizon in [5,10]:
        u=b.trajectory.controls[admitted,:horizon];dt=b.trajectory.dt_s[admitted,:horizon]
        y=np.sum((u[:,:,1]+u[:,:,2])/2*dt,axis=1)/dt.sum(1);y=y[:,None]
        variants=[('state_past_controls',x0)]+[(name,np.concatenate([x0,reference[admitted,:,j]],1)) for j,name in enumerate(['plus_manual_pitch','plus_attitude_pitch','plus_rate_pitch'])]
        for name,x in variants:
            result=np.full_like(y,np.nan)
            for j in range(5):
                tr=fold!=j;ho=~tr
                assert not set(ids[tr])&set(ids[ho])
                result[ho]=predict(fit(x[tr],y[tr],ids[tr]),x[ho])
            assert np.isfinite(result).all()
            for log in sorted(set(ids)):
                take=ids==log
                rows.append(dict(steps=horizon,variant=name,log_id=log,n=int(take.sum()),
                    rmse=float(np.sqrt(np.mean((y[take]-result[take])**2))),target_std=float(y[take].std())))
            for wid,truth,pred in zip(b.trajectory.window_ids[admitted],y[:,0],result[:,0]):
                predictions.append(dict(window_id=wid,steps=horizon,variant=name,target=float(truth),prediction=float(pred)))
    f=pd.DataFrame(rows);f.to_csv(OUT/'conditional_relevance_per_flight.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'conditional_relevance_predictions.csv',index=False)
    f.groupby(['steps','variant']).rmse.mean().reset_index().to_csv(OUT/'conditional_relevance_summary.csv',index=False)
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,admitted_origins=int(admitted.sum()),admitted_flights=len(set(ids)),
        instrument_validated=False,second_stage_fitted=False,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
