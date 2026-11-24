"""Extend the native registry's window constructor, intersect with the original paired starts."""
from collections import Counter
import json
import numpy as np
import pandas as pd
import yaml
from workspace import HERE, ROOT, HISTORY, TAPES, CACHE3
from prepare import LABELS, native_topic, knots, cubic
from controls import native_controls, pack
from system_identification.data.velocity_rate_dataset import build_windows, interval_overlap
from pathlib import Path


def long_control(clock,u,t,stop):
    # Two calls to the archived 20 ms native-event packer cover 3.10 s without a new discretization.
    parts=[pack(clock,u,t,stop)[:,:100],pack(clock,u,t+2000000,stop)[:,:55]]
    width=max(x.shape[2] for x in parts)
    return np.concatenate([np.pad(x,((0,0),(0,0),(0,width-x.shape[2]),(0,0))) for x in parts],axis=1)


def labels(sample,w,origin_p,width):
    start,end=w.origin_sample.to_numpy(),w.end_sample.to_numpy()
    idx=start[:,None]+np.arange(1,width+1)
    exists=idx<=end[:,None]
    ix=np.minimum(idx,end[:,None])
    clocks=sample[['state_measurement_us','target_attitude_measurement_us','target_rate_measurement_us']].to_numpy()
    times=(clocks[ix]-w.decision_us.to_numpy()[:,None,None])*1e-6
    ok=exists[...,None] & (times>0) & (times<=3.04)
    y=sample[LABELS].to_numpy()[ix]
    assert np.isfinite(y[exists]).all()
    y[...,:3]-=origin_p[:,None]
    y[~exists]=0.
    # Masked padding must remain a valid quaternion for finite geodesic derivatives.
    y[:,:,6][~exists]=1.
    weights=np.zeros((*ix.shape,2),np.float32)
    if w.sampling_group.iloc[0]=='main_50hz':
        phase=sample.target_relative_flap_phase_rad.to_numpy()
        good=sample.target_phase_valid.to_numpy()
        wraps=np.flatnonzero((np.diff(phase)<-np.pi)&good[1:]&good[:-1])+1
        if len(wraps)>=2:
            j=np.searchsorted(wraps,start+1)
            a,b=wraps[j.clip(max=len(wraps)-1)],wraps[(j+1).clip(max=len(wraps)-1)]
            bad=np.r_[0,(~good).cumsum()]
            active=(j+1<len(wraps)) & (b<=end) & (b-a>=8) & (bad[b+1]==bad[a])
            for col,ti in enumerate((0,2)):
                x=times[:,:,ti]
                lo=(clocks[a,0]-w.decision_us.to_numpy())*1e-6
                hi=(clocks[b,0]-w.decision_us.to_numpy())*1e-6
                bounds=np.concatenate([x[:,:1],.5*(x[:,1:]+x[:,:-1]),x[:,-1:]],1)
                weights[:,:,col]=np.diff(np.clip(bounds,lo[:,None],hi[:,None]),axis=1)*active[:,None]*ok[:,:,ti]
    return dict(y=y.astype('float32'),t=times.astype('float32'),valid=ok,beat=weights)


def prepare(split):
    tree=HERE.parents[1]
    registry=yaml.safe_load((tree/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
    entry=registry['datasets'][registry['default_dataset_id']]
    dataset=ROOT/entry['repository_relative_root']
    cfg=yaml.safe_load((tree/entry['config_path']).read_text())
    cfg.update(history_us=[500000],horizon_us=[3000000])
    ids=pd.read_parquet(HISTORY/f'{split}_windows.parquet').reset_index(names='history_index')
    tape_ids=pd.read_parquet(TAPES/f'{split}_ids.parquet').reset_index(names='tape_index')
    ids=ids.merge(tape_ids,on='window_id',validate='one_to_one').sort_values('tape_index')
    assert len(ids)==len(tape_ids)
    meta={r['flight_id']:r for r in json.loads((dataset/'flight_provenance.json').read_text()) if r['split']==split}
    cols=['flight_id','sample_in_flight','timestamp_us','support_start_us','support_end_us','segment_id',
          'state_measurement_us','target_attitude_measurement_us','target_rate_measurement_us','mode_change','high_dynamics',
          'valid_encoder','valid_frequency','target_phase_valid','target_relative_flap_phase_rad']+LABELS
    samples=pd.read_parquet(dataset/f'samples_{split}.parquet',columns=cols).groupby('flight_id',sort=False)
    tables={n:pd.read_parquet(dataset/f'native_{n}_{split}.parquet').groupby('flight_id',sort=False)
            for n in ('vehicle_local_position','actuator_motors','actuator_servos')}
    arc={k:np.load(HISTORY/f'{split}_{k}.npy',mmap_mode='r') for k in ('history','initial','y')}
    wins=[]; counts=Counter(); reg_counts=Counter()
    for fid,old in ids.groupby('flight_id',sort=False):
        s=samples.get_group(fid)
        assert np.array_equal(s.sample_in_flight.to_numpy(),np.arange(len(s)))
        longer,byreason=build_windows(s,meta[fid],cfg)
        reg_counts.update(byreason)
        if len(longer):
            w=old.merge(longer.drop(columns='window_id'),on=['flight_id','sampling_group','decision_us'],validate='one_to_one')
        else:
            w=longer
        counts['outside_registered_3s_window']+=len(old)-len(w)
        if len(w):
            counts['extended_review']+=int(w.review.sum())
            w=w[~w.review]
        if len(w):
            wins.append(w)
    width=max(int((w.end_sample-w.origin_sample).max()) for w in wins) if wins else 0
    parts=[]; tapes=[]; identities=[]
    for w in wins:
        fid=w.flight_id.iloc[0]
        rec=meta[fid]
        sample=samples.get_group(fid)
        t=w.decision_us.to_numpy()
        group=w.sampling_group.iloc[0]
        pv=native_topic(tables['vehicle_local_position'].get_group(fid),'vehicle_local_position')
        ks=knots(pv,t[:,None],t,w.support_start_us.to_numpy(),40000 if group=='main_50hz' else 160000)
        pvel=cubic(pv,ks)[:,0]
        origin_p=pvel[:,:3]+pvel[:,3:]*(np.where(ks[5],0,ks[7])[:,0]*1e-6)[:,None]
        batch=labels(sample,w,origin_p,width)
        ix=w.history_index.to_numpy()
        np.testing.assert_allclose(batch['y'][:,0,:3],arc['y'][ix,0,:3],rtol=0,atol=2e-6)
        tol=.035 if group=='main_50hz' else .105
        complete=np.ones(len(w),bool)
        for horizon in (.2,.5,1,2,3):
            complete &= (np.where(batch['valid'],abs(batch['t']-horizon),np.inf).min(axis=1)<=tol).all(1)
        last=np.where(batch['valid'],batch['t'],0).max(axis=(1,2))
        stop=t+np.maximum(np.rint(last*1e6).astype(np.int64),3000000)
        keep=np.ones(len(w),bool)
        def require(ok,reason):
            counts[reason]+=int((keep&~ok).sum())
            keep[:] &= ok
        require(complete,'native_endpoint_outside_tolerance')
        st=sample.state_measurement_us.to_numpy()
        seg=sample.segment_id.to_numpy()
        nxt=st.searchsorted(stop)
        extra=np.array(rec['exclude_intervals_us']+rec['review_intervals_us']+
                       [[z,z] for z in rec['reset_times_us']],dtype=np.int64).reshape(-1,2)
        require((nxt<len(st))&(seg[nxt.clip(max=len(st)-1)]==w.segment_id.to_numpy())&
                (stop<=rec['end_us'])&~interval_overlap(t,stop,extra),'segment_or_overshoot_support')
        clock,u,streams=native_controls(*(tables[n].get_group(fid) for n in ('actuator_motors','actuator_servos')))
        for ts,v in streams:
            gaps=np.flatnonzero(np.diff(ts)>160000)
            first=(ts.searchsorted(t,'right')-1).clip(0)
            good=np.isfinite(v).all(1)&(abs(v)<=1).all(1)
            if v.shape[1]==1: good &= v[:,0]>=0
            invalid=np.r_[0,(~good).cumsum()]
            require((t>=ts[0])&(stop<=ts[-1]+160000)&
                    (np.searchsorted(ts[gaps]+160000,stop,'left')==np.searchsorted(ts[gaps+1],t,'right'))&
                    (invalid[ts.searchsorted(stop,'right')]==invalid[first]),'missing_or_invalid_control')
        if not keep.any(): continue
        batch={k:x[keep] for k,x in batch.items()}
        batch.update(history=arc['history'][ix[keep]],initial=arc['initial'][ix[keep]])
        tape=long_control(clock,u,t[keep],stop[keep])
        np.testing.assert_allclose(tape[:,0,0,:4],batch['history'][:,-1,9:13],rtol=0,atol=2e-7)
        parts.append(batch); tapes.append(tape)
        identities.append(w.loc[keep,['window_id','flight_id','decision_us','sampling_group','history_index','tape_index','origin_sample','end_sample','segment_id']])
    count=sum(map(len,identities))
    assert len(ids)==count+sum(counts.values())
    coverage=dict(split=split,stage1_starts=len(ids),admitted=count,flights=0 if not identities else len(identities),
                  exclusions=counts,native_registry_all_origins=reg_counts,labels_per_window=width)
    (CACHE3/f'{split}_coverage.json').write_text(json.dumps(coverage,indent=2))
    if count:
        index=pd.concat(identities,ignore_index=True)
        assert index.window_id.is_unique and index.tape_index.is_monotonic_increasing
        index.to_parquet(CACHE3/f'{split}_windows.parquet',index=False)
        for k in parts[0]:
            np.save(CACHE3/f'{split}_{k}.npy',np.concatenate([p[k] for p in parts]))
        n=max(a.shape[2] for a in tapes)
        dest=np.lib.format.open_memmap(CACHE3/f'{split}_commands.npy',mode='w+',dtype='float32',shape=(count,155,n,5))
        j=0
        for a in tapes:
            dest[j:j+len(a),:,:a.shape[2]]=a
            j+=len(a)
        del dest
    print(json.dumps(coverage),flush=True)


if __name__=='__main__':
    CACHE3.mkdir(exist_ok=False)
    for split in ('validation','train'):
        prepare(split)
