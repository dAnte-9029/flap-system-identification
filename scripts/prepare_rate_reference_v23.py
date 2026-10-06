"""Materialize train-only rate reference windows and verify original source semantics."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from audit_manual_chain_v21 import CONTRACT,PX4,hold
from rate_reference_v23 import OUT,V8,FIELDS,reference_values,reference_stats
from system_identification.data.reference_windows import extract_window


def main():
    contract=json.loads(CONTRACT.read_text())
    ready=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_ready/manifest.json'
    source=base.ROOT/'dataset/reference_dynamics_data_v1/manifest.json';sources=json.loads(source.read_text())
    assert base.sha(ready)==contract['parent_sha256'] and base.sha(source)==json.loads(ready.read_text())['parent_sha256']
    assert sources['parent_manifest_sha256']==contract['source_dataset_manifest_sha256']
    old=json.loads((base.DATA/'manifest.json').read_text())
    assert base.sha(base.DATA/'windows_v2.csv')==old['tables']['windows_v2.csv']
    w=pd.read_csv(base.DATA/'windows_v2.csv');w=w[w.partition=='train'].sort_values('array_index')
    assert np.array_equal(w.array_index,np.arange(407))
    v22=OUT.parent/'control_windows_v22';delivery=json.loads((v22/'delivery_manifest.json').read_text())
    for f,h in delivery['artifacts'].items():assert base.sha(v22/f)==h
    paths=[Path(__file__),base.ROOT/'scripts/rate_reference_v23.py',base.ROOT/'scripts/train_rate_reference_v23.py',
           base.ROOT/'tests/test_rate_reference_v23.py',CONTRACT,ready,source,base.DATA/'manifest.json',base.DATA/'windows_v2.csv',
           v22/'delivery_manifest.json',V8/'delivery_manifest.json',base.ROOT/'scripts/train_control_windows_v2.py',
           base.ROOT/'src/system_identification/data/reference_windows.py',
           base.ROOT/'src/system_identification/models/reference_gru.py',
           base.ROOT/'src/system_identification/models/reference_gru_unit_gravity.py',
           base.ROOT/'src/system_identification/models/reference_gru_phase_schedule.py']
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'protocol.json',dict(stage='v23 rate-reference conditional prediction',
        source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        data='407 original train parents, same H26 K50 50Hz, no new split or exclusion; development/sealed/reserved unused',
        reference_fields=FIELDS,units=['rad/s','rad/s','rad/s','normalized thrust demand, not newtons','boolean reset command'],
        subsystem='fixed PX4 body rate loop, allocation, actuators and aircraft; excludes attitude loop; same non-VTOL mode3/15 cruise context',
        timing='causal latest eligible rate message; history origin-25..origin, future origin..origin+49, targets origin+1..origin+50; actuator history origin-26..origin-1',
        reset='Retain reset_integral as fifth reference channel, binary mean0 scale1. Cruise materialization requires zero reset throughout original support; fail rather than remove windows.',
        semantics='Future logged rate/thrust references are feedback-generated conditional inputs, NOT exogenous attitude plans or evidence of new control performance.',
        architecture='unchanged ReferenceGRUPhaseSchedule with unit gravity, no phase schedule; 40070 parameters; five slots now rate3/thrust/reset instead of attitude2/yaw_move/thrust/manual_yaw',
        optimizer='AdamW lr3e-4 wd1e-4 clip1; original full1s loss; no q auxiliary loss',steps=1600,batch=32,seeds=[17,23,42],folds=5,
        pairing='same v8 fold train/query, all initial trainable parameters, exact stored 1600x32 sample draws; copy non-reference normalization, refit rate normalization on unique fit rows only',
        comparison='Reuse verified v8 hold baseline; same targets but different information boundary. No full-data refit or controller promotion in this stage.',
        evaluation='fit and OOF p/q/r full1s waveform, velocity/attitude; 20ms,.1s,.5s,1s endpoints and original absolute0.5s bounds. Reference-hold removes future changes of rate3/thrust, preserves reset, diagnostic only.',
        descriptive_gate='q full1s mean flight/seed RMSE improves >=5%, all seeds positive and >=4/5 flight positive; p/r full1s and velocity/attitude .5/1s <=5% regression for each seed; report original0.5s p/q<=.15rad/s attitude<=3deg velocity<=.5m/s. No causal/control promotion regardless of outcome.',
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],sealed_test_accessed=False))
    inputs={};labels={};references={};evidence=[];pins=[];pointrows=[]
    topics=['vehicle_rates_setpoint','vehicle_attitude_setpoint','vehicle_control_mode','vehicle_status','vehicle_land_detected']
    baseline_source=json.loads((OUT.parent/'control_windows_v21/source_provenance.json').read_text())['files']
    for cp,g in w.groupby('cache'):
        log=g.log_id.iloc[0];src=sources['sources'][log]
        assert src['partition']=='train' and base.sha(src['path'])==src['sha256']
        assert base.sha(base.ROOT/cp)==contract['cache_sha256'][cp]
        with np.load(base.ROOT/cp) as z:a={k:z[k] for k in z.files}
        u=ULog(src['path'],message_name_filter_list=topics);d={v.name:v.data for v in u.data_list if v.multi_id==0}
        assert not u.changed_parameters
        firmware=u.msg_info_dict['ver_sw'];blobs={}
        for name in ['src/modules/fw_att_control/FixedwingAttitudeControl.cpp','src/modules/fw_rate_control/FixedwingRateControl.cpp']:
            blob=subprocess.check_output(['git','-C',str(PX4),'rev-parse',f'{firmware}:{name}'],text=True).strip()
            assert blob==baseline_source[name]['blob'];blobs[name]=blob
        rr=d['vehicle_rates_setpoint']
        reference=np.full((len(a['timestamp_us']),5),np.nan,dtype=np.float32)
        ids=np.unique(np.concatenate([np.arange(r.origin_index-26,r.origin_index+51) for r in g.itertuples()]))
        q=a['timestamp_us'][ids];v,ix=reference_values(rr,q);reference[ids]=v
        assert not v[:,4].any(),'reset command present; retain data but revise contract before training'
        assert np.all(rr['thrust_body[1]'][ix]==0) and np.all(rr['thrust_body[2]'][ix]==0)
        assert (v[:,3]>=0).all() and (v[:,3]<=1).all()
        status=d['vehicle_status'];si=hold(status,q)
        assert not status['is_vtol'][si].any() and not status['failsafe'][si].any()
        cm=d['vehicle_control_mode'];ci=hold(cm,q)
        assert cm['flag_control_rates_enabled'][ci].all() and cm['flag_control_attitude_enabled'][ci].all()
        ld=d['vehicle_land_detected'];li=hold(ld,q);assert not ld['landed'][li].any()
        att=d['vehicle_attitude_setpoint'];ai=hold(att,q)
        age=(q-rr['timestamp'][ix])*.001
        for j,k in enumerate(ids):
            pointrows.append(dict(log_id=log,cache_index=int(k),timestamp_us=int(q[j]),publication_us=int(rr['timestamp'][ix[j]]),
                reference_age_ms=float(age[j]),p_sp=float(v[j,0]),q_sp=float(v[j,1]),r_sp=float(v[j,2]),thrust=float(v[j,3]),
                attitude_thrust=float(att['thrust_body[0]'][ai[j]]),reset_integral=int(v[j,4])))
        references[log]=reference
        for r in g.itertuples():
            original,y=extract_window(a,r.origin_index)
            b=dict(a);b['references']=reference;x,y2=extract_window(b,r.origin_index)
            for k in y:np.testing.assert_array_equal(y[k],y2[k])
            for k in x:
                if k not in ['history_reference','future_reference']:np.testing.assert_array_equal(x[k],original[k])
            inputs[r.array_index]=x;labels[r.array_index]=y
        evidence.append(dict(log_id=log,windows=len(g),points=len(q),firmware=firmware,source_blobs=blobs,
            battery_scaling=u.initial_parameters['FW_BAT_SCALE_EN'],reset_count=int(v[:,4].sum()),
            age_p50_ms=float(np.median(age)),age_p95_ms=float(np.quantile(age,.95)),age_max_ms=float(age.max()),
            thrust_vs_attitude_max_abs=float(np.max(abs(v[:,3]-att['thrust_body[0]'][ai]))),
            controller_parameter_snapshot={k:val for k,val in u.initial_parameters.items() if k.startswith(('FW_','CA_','PWM_MAIN_','TRIM_'))}))
        pins.append(dict(log_id=log,source=src,cache=cp,cache_sha256=base.sha(base.ROOT/cp)))
    x={k:np.stack([inputs[i][k] for i in range(407)]) for k in inputs[0]}
    y={k:np.stack([labels[i][k] for i in range(407)]) for k in labels[0]}
    np.savez_compressed(OUT/'windows.npz',**{'input_'+k:v for k,v in x.items()},**{'label_'+k:v for k,v in y.items()})
    w.to_csv(OUT/'windows.csv',index=False);pd.DataFrame(pointrows).to_csv(OUT/'source_points.csv',index=False)
    base.write(OUT/'source_audit.json',evidence)
    for fold in range(5):
        oldfold=json.loads((V8/f'fold{fold}/manifest.json').read_text());fit=oldfold['fit_indices'];query=oldfold['query_indices']
        assert set(w.iloc[fit].log_id).isdisjoint(w.iloc[query].log_id)
        vals=[]
        for log,g in w.iloc[fit].groupby('log_id'):
            ids=np.unique(np.concatenate([np.arange(i-25,i+50) for i in g.origin_index]))
            vals.append(references[log][ids])
        with np.load(V8/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
        assert base.sha(V8/f'fold{fold}/normalization.npz')==oldfold['normalization_sha256']
        stats['reference_mean'],stats['reference_scale']=reference_stats(np.concatenate(vals))
        dest=OUT/f'fold{fold}';dest.mkdir();np.savez(dest/'normalization.npz',**stats)
        base.write(dest/'manifest.json',dict(fold=fold,fit_indices=fit,query_indices=query,
                   source_manifest_sha256=base.sha(V8/f'fold{fold}/manifest.json'),normalization_sha256=base.sha(dest/'normalization.npz')))
    base.write(OUT/'dataset_manifest.json',dict(dataset_id='rate_reference_train_v23',partition='original_train_only',windows=407,
        authority=str((OUT/'protocol.json').relative_to(base.ROOT)),protocol_sha256=base.sha(OUT/'protocol.json'),
        arrays_sha256=base.sha(OUT/'windows.npz'),windows_sha256=base.sha(OUT/'windows.csv'),sources=pins,
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],sealed_test_accessed=False))
    print('Prepared 407 windows and 5 fit-only normalizations',flush=True)


if __name__=='__main__':main()
