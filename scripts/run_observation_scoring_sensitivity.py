"""Frozen-prediction measurement-clock and fixed pre-sampling lowpass diagnostic."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
from pyulog import ULog
from scipy.spatial.transform import Rotation,Slerp
import run_model_flight_response_diagnostic as r
from system_identification.models.trajectory import attitude_error_deg
m=r.m;ROOT=r.ROOT
OUT=ROOT/'docs/analysis/results/observation_scoring_sensitivity_v1'
ART=ROOT/'artifacts/observation_scoring_sensitivity_v1'
TAU=1/(2*np.pi*10)

def interp(t,x,q):
    t=np.asarray(t,float);q=np.asarray(q,float)
    if np.any(np.diff(t)<=0) or q.min()<t[0] or q.max()>t[-1]:raise ValueError('nonmonotone clock or extrapolation')
    return np.stack([np.interp(q,t,x[:,j]) for j in range(x.shape[1])],axis=-1)

def filter_foh(t,x,tau=TAU):
    """Exact two cascaded first-order lowpasses for piecewise-linear input."""
    t=np.asarray(t,float);x=np.asarray(x,float)
    if tau<=0 or len(t)!=len(x) or np.any(np.diff(t)<=0):raise ValueError('invalid filter')
    y1=x[0].copy();y2=x[0].copy();out=[y2.copy()]
    for k,dt in enumerate(np.diff(t)):
        b=(x[k+1]-x[k])/dt;e=np.exp(-dt/tau)
        n2=x[k+1]-2*b*tau+(y2-x[k]+2*b*tau+(y1-x[k]+b*tau)*dt/tau)*e
        y1=x[k+1]-b*tau+(y1-x[k]+b*tau)*e;y2=n2;out.append(y2.copy())
    return np.array(out)

def sampled_filter(t,x,q):
    # Insert score instants before filtering, preserving raw knots; no resample-to-50 first.
    inside=(t>q[0])&(t<q[-1]);grid=np.unique(np.r_[q,t[inside]])
    raw=interp(t,x,grid);y=filter_foh(grid,raw)
    return y[np.searchsorted(grid,q)]

def quaternion_at(t,q,query):
    if query.min()<t[0] or query.max()>t[-1]:raise ValueError('quaternion extrapolation')
    rot=Rotation.from_quat(q[:,[1,2,3,0]])
    return Slerp(t,rot)(query).as_quat()[...,[3,0,1,2]]

def angle(p,t):
    return attitude_error_deg(p.reshape(-1,4),t.reshape(-1,4)).reshape(p.shape[:-1])

def run():
    if OUT.exists():raise RuntimeError('refuse overwrite existing experiment')
    OUT.mkdir(parents=True);ART.mkdir(parents=True,exist_ok=True)
    prior=m.read_json(ROOT/'docs/analysis/results/angular_rate_measurement_audit_v1/protocol.json')
    logs=[e for e in prior['logs'] if e['split']=='validation'];assert len(logs)==17
    sp,batches,stats=r.source.inputs();t=batches['validation'].trajectory
    select=pd.read_csv(r.OUT/'origin_selection.csv');np.testing.assert_array_equal(select.window_id,t.window_ids)
    sourcepins=m.read_json(r.OUT/'protocol.json')['frozen_input_sha256']
    pins={}
    for seed in [17,23,42]:
        path=r.source.predpath(seed);h=m.file_hash(path);assert h==sourcepins[m.rel(path)];pins[m.rel(path)]=h
    for path in [r.OUT/'origin_selection.csv',m.ART/'prepared.pt',Path(__file__),ROOT/'tests/test_observation_scoring_sensitivity.py']:
        pins[m.rel(path)]=m.file_hash(path)
    protocol=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model='frozen Standard GRU64 H26 K50',seeds=[17,23,42],origins=2582,logs=logs,pins=pins,
        heldout_data_accessed_this_run=False,new_inference=False,training=False,
        clock='Original publication-grid time t0+cumsum(native dt), unchanged for predictions; interpolate raw xyz and quaternion by timestamp_sample onto this common clock. Offline/acausal reference ONLY, no extrapolation, gap>50ms fails entire analysis.',
        conditions=['original','synchronized','synchronized_lowpass'],
        synchronization='linear body-rate interpolation, quaternion SLERP; no model/state reinitialization. Quaternion not filtered. Common-time means timestamp-tag synchronized, not latent physical group-delay corrected.',
        filter=dict(formula='H(s)=1/(1+tau*s)^2; tau=1/(2*pi*10) seconds; exact FOH solution',tau_s=TAU,pole_hz=10,effective_3db_hz=10*np.sqrt(np.sqrt(2)-1),attenuation_at_25hz_db=20*np.log10(1+(25/10)**2),
            application='Measured rate: original ~100Hz sample-time knots before evaluation-grid sampling; predicted rate: native forecast knots. Both piecewise-linear; two states initialize to their own value at t0; no cross-label initialization.',
            limitation='Fixed bandwidth restriction reduces alias risk; cannot remove already-aliased 100Hz logging content or repair forecast aliasing. Initial transient retained; report 0-500 and 500-1000ms separately. Not an alternative physical truth.'),
        metrics='Per-axis p/q/r and vector rate RMSE; attitude geodesic RMS for original/synchronized only. Endpoint5/10/25/50 and native-dt weighted interval1..25,26..50. Flight RMSE then equal-flight mean then seed mean/sample SD.',
        grouping='Reuse ALL/low/middle/high500ms frozen origin groups, no rethreshold.',
        failure='No failed origins silently dropped; save identity and stop. No lag/cutoff optimization.',
        interpretation='Scoring sensitivity, not improved model, noise attribution, physical delay identification or causal response validation.')
    m.write_json(OUT/'protocol.json',protocol)
    n=len(select);query_us=select.origin_timestamp_us.to_numpy()[:,None]+np.r_[np.zeros((1,n)),np.cumsum(t.dt_s,axis=1).T].T*1e6
    sync=np.empty((n,51,3));filt=np.empty_like(sync);qs=np.empty((n,51,4));coverage=[]
    maxgap=0.;maxparity=0.
    try:
        for entry in logs:
            assert m.file_hash(Path(entry['path']))==entry['sha256']
            with (OUT/'access_log.jsonl').open('a') as f:f.write(json.dumps(dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(),flight=entry['identity'],sha256=entry['sha256']))+'\n')
            u=ULog(entry['path'],message_name_filter_list=['vehicle_angular_velocity','vehicle_attitude'])
            ds={d.name:d.data for d in u.data_list if d.multi_id==0};a=ds['vehicle_angular_velocity'];at=ds['vehicle_attitude']
            start=min(float(a['timestamp_sample'][0]),float(at['timestamp_sample'][0]))
            rt=(np.asarray(a['timestamp_sample'],float)-start)*1e-6
            qt=(np.asarray(at['timestamp_sample'],float)-start)*1e-6
            xyz=np.stack([a[f'xyz[{j}]'] for j in range(3)],-1).astype(float)
            quat=np.stack([at[f'q[{j}]'] for j in range(4)],-1).astype(float)
            ids=np.flatnonzero(t.log_ids==entry['identity'])
            for i in ids:
                q=(query_us[i]-start)*1e-6
                for clock in [rt,qt]:
                    lo=np.searchsorted(clock,q[0],side='right')-1;hi=np.searchsorted(clock,q[-1],side='left')
                    if lo<0 or hi>=len(clock):raise ValueError('outside raw support')
                    gap=np.diff(clock[lo:hi+1]).max();maxgap=max(maxgap,float(gap))
                    if gap>.050001:raise ValueError('raw bracket gap exceeds frozen50ms rule')
                sync[i]=interp(rt,xyz,q);filt[i]=sampled_filter(rt,xyz,q);qs[i]=quaternion_at(qt,quat,q)
                pi=np.searchsorted(a['timestamp'],query_us[i]+.001,side='right')-1
                diff=np.max(np.abs(xyz[pi]-t.truth.angular_velocity_b[i]));maxparity=max(maxparity,float(diff))
                assert diff<1e-7
            coverage.append(dict(flight=entry['identity'],n_origins=len(ids),cohort=r.cohort(entry['identity'])))
            print(entry['identity'],len(ids),flush=True)
    except Exception as exc:
        m.write_json(OUT/'failure.json',dict(error=str(exc),flight=entry['identity'],origin_index=int(i),window_id=str(t.window_ids[i])))
        raise
    assert np.isfinite(sync).all() and np.isfinite(filt).all() and np.isfinite(qs).all()
    np.savez_compressed(ART/'scoring_references.npz',window_ids=t.window_ids,synchronized_rate=sync,filtered_rate=filt,synchronized_quaternion=qs,query_us=query_us)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    rows=[];initial=[]
    def collect(err,seed,condition,metric):
        for label,kind,k0,k1 in [('100ms','endpoint',5,5),('200ms','endpoint',10,10),('500ms','endpoint',25,25),('1s','endpoint',50,50),('0_500ms','interval',1,25),('500_1000ms','interval',26,50)]:
            mse=err[:,k1] if kind=='endpoint' else (err[:,k0:k1+1]*t.dt_s[:,k0-1:k1]).sum(1)/t.dt_s[:,k0-1:k1].sum(1)
            for group in ['ALL','low','middle','high']:
                mask=np.ones(n,bool) if group=='ALL' else (select.activity_group==group).to_numpy()
                for flight in sorted(set(t.log_ids[mask])):
                    take=mask&(t.log_ids==flight)
                    rows.append(dict(seed=seed,condition=condition,metric=metric,horizon=label,kind=kind,group=group,flight=flight,cohort=r.cohort(flight),n_origins=int(take.sum()),value=float(np.sqrt(mse[take].mean()))))
    for seed in [17,23,42]:
        with np.load(r.source.predpath(seed),allow_pickle=False) as f:
            np.testing.assert_array_equal(f['window_ids'],t.window_ids);pred=f['angular_velocity_b'].astype(float);pq=f['quaternion_nb'].astype(float)
        filtered=np.stack([filter_foh((query_us[i]-query_us[i,0])*1e-6,pred[i]) for i in range(n)])
        assert np.isfinite(pred).all() and np.isfinite(filtered).all()
        for name,pr,tr in [('original',pred,t.truth.angular_velocity_b),('synchronized',pred,sync),('synchronized_lowpass',filtered,filt)]:
            error=(pr-tr)**2
            for j,axis in enumerate('pqr'):collect(error[...,j],seed,name,axis+'_rad_s')
            collect(error.sum(-1),seed,name,'rate_rad_s')
            initial.append(dict(seed=seed,condition=name,initial_rate_rms=float(np.sqrt(error[:,0].sum(-1).mean()))))
        collect(angle(pq,t.truth.quaternion_nb)**2,seed,'original','attitude_deg')
        collect(angle(pq,qs)**2,seed,'synchronized','attitude_deg')
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_flight.csv',index=False);pd.DataFrame(initial).to_csv(OUT/'initial_mismatch.csv',index=False)
    keys=['condition','metric','horizon','kind','group','cohort']
    ex=pd.concat([frame,frame.assign(cohort='ALL')])
    per=ex.groupby(keys+['seed']).value.mean().reset_index();per.to_csv(OUT/'per_seed.csv',index=False)
    agg=per.groupby(keys).value.agg(['mean','std']).reset_index();agg.to_csv(OUT/'summary.csv',index=False)
    # Scoring deltas, not gains in model accuracy.
    piv=frame.pivot(index=['metric','horizon','kind','group','flight','cohort','seed'],columns='condition',values='value').reset_index()
    for c in ['synchronized','synchronized_lowpass']:piv[c+'_minus_original']=piv[c]-piv.original
    piv.to_csv(OUT/'paired_scoring_deltas.csv',index=False)
    m.verify_pins(pins)
    m.write_json(OUT/'sanity_checks.json',dict(status='passed',all_origins_preserved=n==2582,original_rate_label_parity_max_abs=maxparity,max_raw_bracket_gap_s=maxgap,
        synchronized_quaternion_max_norm_error=float(np.max(np.abs(np.linalg.norm(qs,axis=-1)-1))),heldout_data_accessed_this_run=False,pins_unchanged=True))
    m.write_json(OUT/'completion.json',dict(status='numerics_complete',training=0,new_predictions=0,heldout_data_accessed_this_run=False))

if __name__=='__main__':run()
