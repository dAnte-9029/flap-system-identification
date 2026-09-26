"""Read-only, allowlisted ULog timestamp/filter audit; never changes frozen labels."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import yaml
from pyulog import ULog

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/analysis/results/angular_rate_measurement_audit_v1'
DATA=ROOT/'dataset/trajectory_v3_september_expanded'
TOPICS=['vehicle_local_position','vehicle_attitude','vehicle_angular_velocity','sensor_gyro',
        'sensor_gyro_fifo','vehicle_imu','sensor_selection','sensor_gyro_status']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stats(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {'count':0}
    return dict(zip(['count','mean','median','p05','p95','min','max'],
                    [len(x),float(x.mean()),*map(float,np.quantile(x,[.5,.05,.95,0,1]))]))

def past_index(times,reference):
    t=np.asarray(times,dtype=np.int64)
    if not len(t) or np.any(np.diff(t)<0): raise ValueError('source timestamps must be nondecreasing')
    return np.searchsorted(t,np.asarray(reference,dtype=np.int64),side='right')-1

def sample_aligned_index(publication,sample,reference_publication,reference_sample):
    # Both sample and publication clocks must precede their respective reference.
    return np.minimum(past_index(publication,reference_publication),past_index(sample,reference_sample))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists(): raise RuntimeError('refuse overwrite completed audit')
    manifest=json.loads((DATA/'manifest.json').read_text())
    allow=manifest['split_contract']['assignments']['validation']
    admission=yaml.safe_load((ROOT/'configs/control/identification_log_admission.yaml').read_text())
    roots=list(map(Path,manifest['source']['roots'])); entries=[]
    for identity in allow:
        hits=[root/identity for root in roots if (root/identity).is_file()]
        if not hits: raise ValueError((identity,hits))
        if len(hits)>1:
            assert all(sha(p)==manifest['source']['ulog_sha256'][identity] for p in hits), 'ambiguous raw copies'
        entries.append(dict(identity=identity,path=str(hits[0]),split='validation',sha256=manifest['source']['ulog_sha256'][identity]))
    for log in admission['logs']:
        p=Path(log['path']);identity=next((i for i in manifest['split_contract']['assignments']['train'] if p.as_posix().endswith(i)),None)
        assert identity is not None and log['role']=='development_audit'
        assert log['sha256']==manifest['source']['ulog_sha256'][identity]
        entries.append(dict(identity=identity,path=str(p),split='train',sha256=log['sha256']))
    files={str(DATA/'manifest.json'):sha(DATA/'manifest.json')}
    for split in ['train','validation']:
        p=DATA/f'samples_{split}.parquet';files[str(p)]=sha(p)
        assert files[str(p)]==manifest['artifact_sha256'][p.name]
    write('protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs=files,logs=entries,heldout_accessed_this_run=False,training=False,
        prior_access='First validation ULog inspected in pilot before this registration; no claim of blind preregistration.',
        row_selection='All cached valid_core rows in allowlisted logs, not a new benchmark origin set.',
        comparisons=['frozen publication-time past-only ZOH','sample-time past-only ZOH, additionally publication-available'],
        rule='Compare only common valid rows, sample-age <=50ms; retain coverage/exclusions. No lag optimization or model scoring.',
        limitations='Measurement timestamps are estimator output epoch tags, not proof of exact physical group delay. No true angular-rate reference.'))
    cached={s:pd.read_parquet(DATA/f'samples_{s}.parquet') for s in ['train','validation']}
    timing=[];topics=[];params=[];checks=[];firmwares=set()
    for e in entries:
        assert sha(e['path'])==e['sha256']
        with (OUT/'access_log.jsonl').open('a') as f:f.write(json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),**e},ensure_ascii=False)+'\n')
        u=ULog(e['path'],message_name_filter_list=TOPICS)
        fw=str(u.msg_info_dict.get('ver_sw','unknown'));firmwares.add(fw)
        base={'flight':e['identity'],'split':e['split'],'firmware':fw}
        for d in u.data_list:
            t=np.asarray(d.data['timestamp'],dtype=np.int64);ts=np.asarray(d.data.get('timestamp_sample',t),dtype=np.int64)
            dif=np.diff(t)*1e-3
            topics.append({**base,'topic':d.name,'instance':d.multi_id,'rows':len(t),'timestamp_sample_present':'timestamp_sample' in d.data,
              'mean_rate_hz':(len(t)-1)/((t[-1]-t[0])*1e-6) if len(t)>1 else np.nan,
              'duplicate_or_backward':int((np.diff(t)<=0).sum()),
              **{'interval_ms_'+k:v for k,v in stats(dif).items()},
              **{'publish_lag_ms_'+k:v for k,v in stats((t-ts)*1e-3).items()}})
        for k,v in u.initial_parameters.items():
            if k.startswith(('IMU_GYRO','IMU_DGYRO','SENS_BOARD','CAL_GYRO')):
                params.append({**base,'parameter':k,'value':v,'kind':'initial'})
        for t,k,v in u.changed_parameters:
            if k.startswith(('IMU_GYRO','IMU_DGYRO','SENS_BOARD','CAL_GYRO')):
                params.append({**base,'parameter':k,'value':v,'kind':'changed','timestamp':t})
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        a=ds['vehicle_angular_velocity'];at=ds['vehicle_attitude'];l=ds['vehicle_local_position']
        frame=cached[e['split']];frame=frame[frame.log_id==e['identity']]
        ref=frame.timestamp_us.to_numpy(np.int64);local_idx=past_index(l['timestamp'],ref)
        np.testing.assert_array_equal(np.asarray(l['timestamp'])[local_idx],ref)
        lp=np.asarray(l['timestamp_sample'],dtype=np.int64)[local_idx]
        stored=frame.state_sample_timestamp_us.to_numpy(float)
        stored_finite=np.isfinite(stored)
        np.testing.assert_array_equal(lp[stored_finite],stored[stored_finite])
        ap=np.asarray(a['timestamp'],np.int64);ass=np.asarray(a['timestamp_sample'],np.int64)
        ai=past_index(ap,ref);ci=sample_aligned_index(ap,ass,ref,lp);ati=past_index(at['timestamp'],ref)
        xyz=np.stack([a[f'xyz[{i}]'] for i in range(3)],axis=1)
        old=xyz[np.maximum(ai,0)].astype(float)
        old[ai<0]=np.nan
        labels=frame[[f'angular_velocity_body_rad_s_{i}' for i in 'xyz']].to_numpy()
        np.testing.assert_allclose(old,labels,rtol=0,atol=0,equal_nan=True)
        core=frame.valid_core.to_numpy(bool);valid=core&(ci>=0)&(ai>=0)&(ati>=0)
        safe=np.maximum(ci,0)
        valid&=(lp-ass[safe]>=0)&(lp-ass[safe]<=50000)
        assert np.all(ap[safe[valid]]<=ref[valid])
        row={**base,'cached_local_sample_timestamp_present_rows':int(stored_finite.sum()),'cached_rows':len(frame),'valid_core_rows':int(core.sum()),'comparison_rows':int(valid.sum()),'excluded_from_sensitivity':int((core&~valid).sum()),'label_max_abs_difference':0.0,
             'selected_sample_changed_fraction':float(np.mean(ai[valid]!=ci[valid]))}
        arrays={'gyro_publication_age_ms':(ref-ap[np.maximum(ai,0)])*.001,
          'gyro_sample_age_to_grid_ms':(ref-ass[np.maximum(ai,0)])*.001,
          'local_publication_lag_ms':(ref-lp)*.001,
          'gyro_minus_local_sample_ms':(ass[np.maximum(ai,0)]-lp)*.001,
          'gyro_minus_attitude_sample_ms':(ass[np.maximum(ai,0)]-np.asarray(at['timestamp_sample'],np.int64)[np.maximum(ati,0)])*.001,
          'causal_aligned_sample_age_ms':(lp-ass[safe])*.001}
        for name,x in arrays.items():row.update({name+'_'+k:v for k,v in stats(x[valid]).items()})
        delta=old[valid]-xyz[safe[valid]]
        for j,axis in enumerate('pqr'):row['timing_sensitivity_rmse_'+axis]=float(np.sqrt(np.mean(delta[:,j]**2)))
        timing.append(row);checks.append({'flight':e['identity'],'exact_cached_label_parity':True,'raw_sha256_valid':True})
        print(e['identity'],row['comparison_rows'],flush=True)
    for name,rows in [('topic_timing.csv',topics),('parameters.csv',params),('alignment_per_flight.csv',timing)]:pd.DataFrame(rows).to_csv(OUT/name,index=False)
    evidence=[]
    for fw in sorted(firmwares):
        proc=subprocess.run(['git','-C','/home/zn/PX4-Autopilot','show',f'{fw}:src/modules/sensors/vehicle_angular_velocity/VehicleAngularVelocity.cpp'],text=True,capture_output=True)
        evidence.append({'firmware':fw,'source_available':proc.returncode==0,'source_sha256':hashlib.sha256(proc.stdout.encode()).hexdigest() if proc.returncode==0 else None})
        if proc.returncode==0:
            lines=proc.stdout.splitlines();selected=[]
            for i,line in enumerate(lines):
                if any(w in line for w in ['_lp_filter_velocity[axis].apply','_calibration.Correct','angular_velocity.timestamp_sample =','angular_velocity.timestamp =','IMU_GYRO_CUTOFF','_lp_filter_acceleration[axis].update']):selected.append(f'{i+1}: {line.strip()}')
            (OUT/f'firmware_{fw[:12]}.txt').write_text('\n'.join(selected)+'\n')
    write('sanity_checks.json',{'flights':checks,'source_evidence':evidence,'heldout_accessed_this_run':False,'frozen_inputs_unchanged':all(sha(p)==h for p,h in files.items())})
    write('completion.json',{'completed':True,'logs':len(entries),'heldout_accessed_this_run':False,'new_training':False,'model_or_dataset_modified':False})

if __name__=='__main__':run()
