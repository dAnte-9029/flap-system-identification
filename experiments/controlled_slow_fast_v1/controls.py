"""Time-ordered, publication-time ZOH command segments on the original paired windows."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STEP_US = 20000
STEPS = 105
TREE = Path(__file__).resolve().parents[2]


def native_controls(motor, servo):
    """Union of the two clocks. Retain the last publication at a tied timestamp."""
    streams = []
    for table,n in [(motor,1),(servo,3)]:
        rows = table.loc[table.in_reviewed_flight].drop_duplicates('timestamp_us',keep='last')
        ts = rows.timestamp_us.to_numpy()
        if not len(ts) or not np.all(np.diff(ts)>0):
            raise ValueError('invalid native command publication clock')
        streams.append((ts,rows[[f'control[{i}]' for i in range(n)]].to_numpy()))
    clock = np.union1d(streams[0][0],streams[1][0])
    clock = clock[clock>=max(x[0][0] for x in streams)]
    u = np.column_stack([v[np.searchsorted(t,clock,'right')-1] for t,v in streams])
    changed = np.r_[True,(u[1:]!=u[:-1]).any(1)]
    return clock[changed],u[changed],streams


def pack(clock, u, origins, stops):
    # Every bin starts from its held command; an event exactly at the right boundary
    # belongs to the following bin. Commands beyond the admitted support are held.
    left = origins[:,None]+STEP_US*np.arange(STEPS)
    right = left+STEP_US
    i = clock.searchsorted(np.minimum(left,stops[:,None]),'right')-1
    j = clock.searchsorted(np.minimum(right-1,stops[:,None]),'right')-1
    if (i<0).any():
        raise ValueError('no initial command')
    count = j-i+1
    values = []
    for k in range(count.max()):
        idx = np.minimum(i+k,j)
        start = left if k==0 else np.maximum(left,clock[idx])
        end = np.where(idx<j,clock[(idx+1).clip(max=len(clock)-1)],right)
        duration = np.where(k<count,end-start,0)
        values.append(np.concatenate([u[idx],duration[...,None]],-1).astype(np.float32))
    tape = np.stack(values,axis=2)
    if not (np.isfinite(tape).all() and (abs(tape[...,:4])<=1).all() and (tape[...,0]>=0).all()
            and (tape[...,4]>=0).all() and (tape[...,4].sum(2)==STEP_US).all()):
        raise ValueError(f'invalid ZOH segment: dt {tape[...,4].min()} sum {tape[...,4].sum(2).min()}..{tape[...,4].sum(2).max()} u {np.nanmin(tape[...,:4])}..{np.nanmax(tape[...,:4])} finite {np.isfinite(tape).all()}')
    return tape


def prepare(root,cache,out,split):
    ids = pd.read_parquet(cache/f'{split}_windows.parquet')
    win = pd.read_parquet(root/f'windows_{split}.parquet',columns=['window_id','support_end_us'],
                          filters=[('history_us','==',500000),('horizon_us','==',2000000)])
    ids = ids.join(win.set_index('window_id'),on='window_id',validate='one_to_one')
    history = np.load(cache/f'{split}_history.npy',mmap_mode='r')
    if ids.support_end_us.isna().any():
        raise ValueError('cache window mismatch')
    tables = [pd.read_parquet(root/f'native_actuator_{name}_{split}.parquet').groupby('flight_id',sort=False)
              for name in ['motors','servos']]
    native_t = np.load(cache/f'{split}_t.npy',mmap_mode='r')
    native_ok = np.load(cache/f'{split}_valid.npy',mmap_mode='r')
    parts = []
    for fid,w in ids.groupby('flight_id',sort=False):
        clock,u,streams = native_controls(*(t.get_group(fid) for t in tables))
        origins,stops = w.decision_us.to_numpy(),w.support_end_us.to_numpy()
        last_label = np.where(native_ok[w.index],native_t[w.index],0.).max(axis=(1,2))
        stops = np.minimum(stops,origins+np.rint(last_label*1e6).astype(np.int64))
        keep = origins>=clock[0]
        for ts,values in streams:
            gaps = np.flatnonzero(np.diff(ts)>160000)
            starts,ends = ts[gaps]+160000,ts[gaps+1]
            keep &= np.searchsorted(starts,stops,'left')==np.searchsorted(ends,origins,'right')
            keep &= stops<=ts[-1]+160000
            good = np.isfinite(values).all(1)&(abs(values)<=1).all(1)
            if values.shape[1]==1:
                good &= values[:,0]>=0
            invalid = np.r_[0,(~good).cumsum()]
            first = (ts.searchsorted(origins,'right')-1).clip(0)
            keep &= invalid[ts.searchsorted(stops,'right')]==invalid[first]
        w,origins,stops = w.loc[keep],origins[keep],stops[keep]
        if w.empty:
            continue
        tape = pack(clock,u,origins,stops)
        np.testing.assert_allclose(tape[:,0,0,:4],history[w.index,-1,9:13],rtol=0,atol=2e-7)
        parts.append((w.index.to_numpy(),tape))
    width = max(x.shape[2] for _,x in parts)
    selected = np.sort(np.concatenate([ix for ix,_ in parts]))
    target = np.lib.format.open_memmap(out/f'{split}_commands.npy',mode='w+',dtype='float32',
                                       shape=(len(selected),STEPS,width,5))
    for ix,x in parts:
        target[np.searchsorted(selected,ix),:,:x.shape[2]] = x
    del target
    ids.iloc[selected][['window_id']].to_parquet(out/f'{split}_ids.parquet',index=False)
    print(split,len(selected),'of',len(ids),'maximum segments per 20 ms',width,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('repository','cache','out'):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    reg=yaml.safe_load((TREE/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
    root=a.repository/reg['datasets'][reg['default_dataset_id']]['repository_relative_root']
    a.out.mkdir(parents=True,exist_ok=False)
    for split in ('train','validation'):
        prepare(root,a.cache,a.out,split)
