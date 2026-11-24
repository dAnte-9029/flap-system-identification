"""Prepare decision-causal histories and separate native targets for the paired GPU experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
TOPICS = dict(vehicle_local_position=['x','y','z','vx','vy','vz'],
              vehicle_attitude=[f'q[{i}]' for i in range(4)],
              vehicle_angular_velocity=[f'xyz[{i}]' for i in range(3)],
              actuator_motors=['control[0]'], actuator_servos=[f'control[{i}]' for i in range(3)],
              encoder_count=['total_count'], flap_frequency=['frequency_hz'])
HGRID = np.arange(-500000, 1, 20000)
HORIZONS = np.array([.2, .5, 1., 2.])
LABELS = [f'target_{key}_{axis}' for key,axes in [('position_ned_m','xyz'),('velocity_ned_m_s','xyz'),
          ('attitude_q','wxyz'),('angular_velocity_body_rad_s','xyz')] for axis in axes]


def rotation(q):
    return Rotation.from_quat(np.asarray(q).reshape(-1,4)[:,[1,2,3,0]])


def rotate(q, vectors, inverse=False):
    return rotation(q).apply(vectors.reshape(-1,3), inverse=inverse).reshape(vectors.shape)


def slerp(a, b, frac):
    dot = (a*b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(abs(dot), 0., 1.))
    def factor(u):
        return u*np.sinc(u*angle/np.pi)/np.sinc(angle/np.pi)
    q = factor(1-frac)*a + factor(frac)*b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def knots(topic, grid, decision, lower, max_gap):
    t, pub, v, valid = topic
    # All state clocks are monotone within the reviewed flight.
    last = np.searchsorted(pub, decision, 'right')[:,None]-1
    first = np.searchsorted(t, lower)[:,None]
    left = np.minimum(np.searchsorted(t, grid, 'right')-1, last)
    lo = left.clip(0, len(t)-1)
    hi = np.minimum(lo+1, last).clip(0, len(t)-1)
    valid_lo = (left >= first) & (left >= 0) & valid[lo]
    between = valid_lo & valid[hi] & (hi>lo) & (t[hi]-t[lo]<=max_gap)
    age = grid-t[lo]
    span = np.maximum(1, t[hi]-t[lo])
    frac = np.where(between, age/span, 0.).clip(0,1)[...,None]
    before = (lo-1).clip(0)
    after = (hi+1).clip(max=len(t)-1)
    before = np.where((before>=first)&valid[before]&(t[lo]-t[before]<=max_gap),before,lo)
    after = np.where((after<=last)&valid[after]&(t[after]-t[hi]<=max_gap),after,hi)
    return lo, hi, before, after, frac, between, valid_lo & (age<=max_gap), age


def cubic(topic, knots_):
    t, _, v, _ = topic
    lo,hi,pre,post,u = knots_[:5]
    span = (t[hi]-t[lo])[...,None]
    ml = (v[hi]-v[pre])*span/np.maximum(1,t[hi]-t[pre])[...,None]
    mr = (v[post]-v[lo])*span/np.maximum(1,t[post]-t[lo])[...,None]
    return v[lo]+u*(ml+u*(3*(v[hi]-v[lo])-2*ml-mr+u*(2*(v[lo]-v[hi])+ml+mr)))


def attitude(topic, ks, gyro, grid, decision, lower):
    t,_,q,_ = topic
    lo,hi,pre,post,u,between,valid,age = ks
    # Spherical cubic Bezier, with local angular tangents at the two ends.
    left,right = rotation(q[lo]),rotation(q[hi])
    span = (t[hi]-t[lo]).ravel()
    def relative(a,b):
        return (a.inv()*b).as_rotvec()
    # Secants on either side expressed in each knot's tangent space.
    arc = relative(left,right)
    v0 = (arc-relative(left,rotation(q[pre])))*(span/np.maximum(1,(t[hi]-t[pre]).ravel()))[:,None]
    v1 = (relative(right,rotation(q[post]))-relative(right,left))*(span/np.maximum(1,(t[post]-t[lo]).ravel()))[:,None]
    def packed(rot):
        return rot.as_quat()[:,[3,0,1,2]].reshape(q[lo].shape)
    a = packed(left*Rotation.from_rotvec(v0/3))
    b = packed(right*Rotation.from_rotvec(-v1/3))
    l,m,r = slerp(q[lo],a,u),slerp(a,b,u),slerp(b,q[hi],u)
    result = slerp(slerp(l,m,u),slerp(m,r,u),u)
    tail = (~between) & valid & (age>0)
    if tail.any():
        gt,gpub,gvalue,good = gyro
        start,end = t[lo[tail]],grid[tail]
        i = gt.searchsorted(start,'right')-1
        last = np.minimum(gt.searchsorted(end,'right')-1,
                          gpub.searchsorted(np.broadcast_to(decision[:,None],grid.shape)[tail],'right')-1)
        safe = i.clip(0)
        ok = (i>=0)&(last>=i)&(gt[safe]>=np.broadcast_to(lower[:,None],grid.shape)[tail])&(start-gt[safe]<=50000)
        rot = rotation(result[tail])
        for k in range(max(0,int((last-i).max())+1)):
            j = (i+k).clip(0,len(gt)-1)
            stop = np.where(j<last, np.minimum(end,gt[(j+1).clip(max=len(gt)-1)]),end)
            dt = np.maximum(0,stop-np.maximum(start,gt[j]))*(i+k<=last)
            ok &= (dt==0)|(good[j]&(dt<=50000))
            rot = rot*Rotation.from_rotvec(gvalue[j]*dt[:,None]*1e-6)
        result[tail] = rot.as_quat()[:,[3,0,1,2]]
        valid = valid.copy()
        valid[tail] &= ok
    return result,valid


def held(topic, grid, lower, limit):
    t,pub,v,valid = topic
    j = pub.searchsorted(grid,'right')-1
    i = j.clip(0)
    age = grid-t[i]
    ok = (j>=0)&(t[i]>=lower[:,None])&valid[i]&(age>=0)&(age<=limit)
    return v[i],ok,age


def history(topics,w,group,ratio):
    grid = w.decision_us.to_numpy()[:,None]+HGRID
    decision,lower = w.decision_us.to_numpy(),w.support_start_us.to_numpy()
    gap = 40000 if group=='main_50hz' else 160000
    def state(name,limit):
        topic = topics[name]
        ks = knots(topic,grid,decision,lower,limit)
        return topic,ks
    pv,pk = state('vehicle_local_position',gap)
    rate,rk = state('vehicle_angular_velocity',50000)
    att,ak = state('vehicle_attitude',50000 if group=='main_50hz' else 100000)
    pvel = cubic(pv,pk)
    pvel[...,:3] += pvel[...,3:]*(np.where(pk[5],0,pk[7])*1e-6)[...,None]
    omega = cubic(rate,rk)
    quat,aq = attitude(att,ak,rate,grid,decision,lower)
    motor,mo,ma = held(topics['actuator_motors'],grid,lower,160000)
    servo,se,sa = held(topics['actuator_servos'],grid,lower,160000)
    count,ce,ca = held(topics['encoder_count'],grid,lower,50000)
    freq,fe,fa = held(topics['flap_frequency'],grid,lower,120000)
    theta = (count-count[:,-1:])*(2*np.pi/(4096*ratio))
    down = np.zeros_like(omega)
    down[...,2] = 1
    phase_ok = ce & ce[:,-1:]
    masks = np.stack([pk[6],rk[6],aq,mo,se,phase_ok,fe],-1)
    ages = np.stack([pk[7],rk[7],ak[7],ma,sa,ca,fa],-1)
    def masked(value,ok):
        return np.where(ok[...,None],value,0)
    values = [masked(rotate(quat,pvel[...,3:],True),pk[6]&aq), masked(omega,rk[6]),
              masked(rotate(quat,down,True),aq),masked(motor,mo),masked(servo,se),
              masked(np.concatenate([np.sin(theta),np.cos(theta)],-1),phase_ok),masked(freq,fe)]
    features = np.concatenate([*values, masks, np.clip(ages*1e-6,0,.5)],-1)
    initial = np.concatenate([pvel[:,-1],quat[:,-1],omega[:,-1]],-1)
    return features,initial,masks[:,-1,:5].all(-1)


def native_topic(table,name):
    use = table.in_reviewed_flight
    table = table.loc[use]
    t = table.timestamp_sample_us.to_numpy() if name.startswith('vehicle_') else table.timestamp_us.to_numpy()
    pub = table.get('availability_us',table.timestamp_us).to_numpy()
    v = table[TOPICS[name]].to_numpy(dtype=float)
    ok = np.isfinite(v).all(1)
    if name=='vehicle_local_position':
        ok &= table[['xy_valid','z_valid','v_xy_valid','v_z_valid']].all(axis=1).to_numpy()
    if name=='flap_frequency':
        ok &= (v[:,0]>=0)&(v[:,0]<30)
    if name=='vehicle_attitude':
        ok &= abs(np.linalg.norm(v,axis=1)-1)<.01
        v[~ok] = [1,0,0,0]
    if name.startswith('actuator_'):
        ok &= (abs(v)<=1).all(1)
        if name=='actuator_motors':
            ok &= v[:,0]>=0
    if not len(t) or np.any(np.diff(t)<0) or np.any(np.diff(pub)<0) or np.any(t>pub):
        raise ValueError(f'invalid reviewed clocks in {name}')
    return t,pub,np.nan_to_num(v),ok


def targets(samples,w,main):
    # The registered target array contains nearest native measurements on three clocks.
    origin,end = w.origin_sample.to_numpy(),w.end_sample.to_numpy()
    rows = origin[:,None]+np.arange(1,105)
    used = rows<=end[:,None]
    ix = np.minimum(rows,end[:,None])
    assert (end-origin).max()<=ix.shape[1]
    times = samples[['state_measurement_us','target_attitude_measurement_us','target_rate_measurement_us']].to_numpy()[ix]
    times = (times-w.decision_us.to_numpy()[:,None,None])*1e-6
    ok = used[...,None] & (times>0)&(times<=2.04)
    y = samples[LABELS].to_numpy()[ix]
    assert np.isfinite(y[used]).all()
    # Byte-sized masks, two independent quadratures for v and omega.
    weights = np.zeros((*ix.shape,2))
    if main:
        tt = samples.state_measurement_us.to_numpy()
        phase = samples.target_relative_flap_phase_rad.to_numpy()
        good = samples.target_phase_valid.to_numpy()
        wraps = np.flatnonzero((np.diff(phase)<-np.pi)&good[1:]&good[:-1])+1
        if len(wraps)>=2:
            j = np.searchsorted(wraps,origin+1)
            a,b = wraps[j.clip(max=len(wraps)-1)],wraps[(j+1).clip(max=len(wraps)-1)]
            bad = np.r_[0,(~good).cumsum()]
            active = (j+1<len(wraps))&(b<=end)&(b-a>=8)&(bad[b+1]==bad[a])
            for col,clock in enumerate([0,2]):
                x = times[:,:,clock]
                start = (tt[a]-w.decision_us.to_numpy())*1e-6
                stop = (tt[b]-w.decision_us.to_numpy())*1e-6
                mids = (x[:,1:]+x[:,:-1])/2
                bounds = np.concatenate([x[:,:1],mids,x[:,-1:]],axis=1)
                width = np.diff(np.clip(bounds,start[:,None],stop[:,None]),axis=1)
                weights[:,:,col] = width*active[:,None]*ok[:,:,clock]
    return dict(y=y,t=times,valid=ok,beat=weights)


def prepare(root, split, out, registry):
    manifest = json.loads((root/'dataset_manifest.json').read_text())
    assert manifest['dataset_id']==registry['default_dataset_id']=='velocity_rate_multistep_v1'
    assert split in ('train','validation')
    wtab = pd.read_parquet(root/f'windows_{split}.parquet',filters=[('history_us','==',500000),('horizon_us','==',2000000)])
    wtab = wtab.loc[wtab.firmware_verified & ~wtab.review & ~wtab.provenance_review & wtab.complete_state_sequence]
    sample_cols = ['flight_id','sample_in_flight','state_measurement_us','target_attitude_measurement_us',
                   'target_rate_measurement_us','target_relative_flap_phase_rad','target_phase_valid']+LABELS
    samples = pd.read_parquet(root/f'samples_{split}.parquet',columns=sample_cols).groupby('flight_id',sort=False)
    provenance = {r['flight_id']:r for r in json.loads((root/'flight_provenance.json').read_text()) if r['split']==split}
    tables = {name:pd.read_parquet(root/f'native_{name}_{split}.parquet').groupby('flight_id',sort=False) for name in TOPICS}
    data,identities = {},[]
    for fid,w in wtab.groupby('flight_id',sort=False):
        group = w.sampling_group.iloc[0]
        native = {n:native_topic(tbl.get_group(fid),n) for n,tbl in tables.items()}
        s = samples.get_group(fid)
        assert np.array_equal(s.sample_in_flight,np.arange(len(s)))
        features,initial,ok = history(native,w,group,provenance[fid]['transmission_ratio'])
        w = w.loc[ok]
        if w.empty:
            continue
        batch = targets(s,w,group=='main_50hz')
        batch['y'][...,:3] -= initial[ok,None,:3]
        initial[:, :3] = 0.
        batch.update(history=features[ok],initial=initial[ok])
        for key,val in batch.items():
            if not np.isfinite(val).all():
                raise ValueError(f'nonfinite {key} in {fid}')
            data.setdefault(key,[]).append(val.astype(bool if key=='valid' else np.float32))
        identities.append(w[['flight_id','window_id','sampling_group','decision_us']])
    ids = pd.concat(identities,ignore_index=True)
    ids.to_parquet(out/f'{split}_windows.parquet',index=False)
    for key,parts in data.items():
        np.save(out/f'{split}_{key}.npy',np.concatenate(parts))
    print(split,len(ids),ids.flight_id.nunique(),ids.sampling_group.value_counts().to_dict(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('repository',type=Path)
    ap.add_argument('out',type=Path)
    args=ap.parse_args()
    registry=yaml.safe_load((TREE/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
    root=args.repository/registry['datasets'][registry['default_dataset_id']]['repository_relative_root']
    args.out.mkdir(parents=True,exist_ok=False)
    for partition in ('train','validation'):
        prepare(root,partition,args.out,registry)
