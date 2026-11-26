"""Replicate the previous QC and identical paired offset queries."""
import json
import numpy as np
import pandas as pd
from network import ROOT

def observations(d,flight,b):
    src=ROOT/'dataset/velocity_rate_multistep_v1'
    raw=pd.read_parquet(src/'native_vehicle_angular_velocity_validation.parquet')
    raw=raw.loc[raw.in_reviewed_flight].groupby('flight_id')
    cols=['flight_id','state_measurement_us','segment_id','valid_core','firmware_verified','review','provenance_review','excluded']
    state=pd.read_parquet(src/'samples_validation.parquet',columns=cols).groupby('flight_id')
    win=pd.read_parquet(ROOT/'outputs/history_slow_fast_20261119_cache/validation_windows.parquet').set_index('window_id')
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:
        win=win.loc[z['window']].reset_index()
    np.testing.assert_array_equal(win.flight_id.to_numpy(),flight)
    offsets=np.array([-125000,-20000,0])
    steps=[1,3,7]
    t=np.zeros((len(flight),3,3),np.float32)
    y=np.zeros((*t.shape,3),np.float32)
    mask=np.zeros(t.shape,bool)
    nominal=d['time'][:,steps,1].numpy()
    registered_target=d['y'][:,steps,1].numpy()+b['origin'][:,None]
    max_target_diff=0.
    for fid,group in win.groupby('flight_id'):
        ix=group.index.to_numpy()
        times=np.rint(nominal[ix].astype('float64')*1e6).astype('int64')+group.decision_us.to_numpy()[:,None]
        targets=times[:,:,None]+offsets
        gyro=raw.get_group(fid).sort_values('timestamp_sample_us').drop_duplicates('timestamp_sample_us')
        clock=gyro.timestamp_sample_us.to_numpy()
        sample=clock.searchsorted(targets).clip(1,len(clock)-1)
        sample-=np.abs(targets-clock[sample-1])<np.abs(targets-clock[sample])
        actual=clock[sample]
        value=gyro[['xyz[0]','xyz[1]','xyz[2]']].to_numpy()[sample]
        st=state.get_group(fid)
        tc=st.state_measurement_us.to_numpy()
        seg=st.segment_id.to_numpy()
        good=(st.valid_core&st.firmware_verified&~st.review&~st.provenance_review&~st.excluded&(seg>=0)).to_numpy()
        hi=tc.searchsorted(actual).clip(0,len(tc)-1)
        lo=np.where(tc[hi]==actual,hi,(hi-1).clip(0))
        valid=(np.abs(actual-targets)<=10000)&good[lo]&good[hi]&(seg[lo]==seg[hi])&(seg[hi]==seg[hi][:,:,-1:])
        valid &= np.isfinite(value).all(3) & (tc[lo]<=actual) & (tc[hi]>=actual)
        registered=d['mask'][ix][:,steps,1].numpy()
        assert np.all(actual[:,:,-1][registered]==times[registered])
        max_target_diff=max(max_target_diff,float(abs(value[:,:,-1][registered]-registered_target[ix][registered]).max()))
        mask[ix]=valid & registered[...,None] & b['ok'][ix,None,None]
        t[ix]=(actual-group.decision_us.to_numpy()[:,None,None])*1e-6
        y[ix]=value
    assert max_target_diff<1e-6
    # Exclude any straddling reset, as in the original registered windows.
    meta={m['flight_id']:m for m in json.loads((src/'flight_provenance.json').read_text()) if m['split']=='validation'}
    for fid,g in win.groupby('flight_id'):
        stops=np.sort(meta[fid]['reset_times_us'])
        now=g.decision_us.to_numpy()[:,None,None]
        mask[g.index] &= (stops.searchsorted(now,side='right')==stops.searchsorted(now+t[g.index]*1e6,side='right'))
    return t,y,mask
