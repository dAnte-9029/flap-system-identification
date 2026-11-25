"""Causal origin features and staggered supervision on actual native observation clocks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path("/home/zn/flap-system-identification")
CACHE = ROOT / "outputs/autopilot_periodic_motion_20261125"
HISTORY = ROOT / "outputs/history_slow_fast_20261119_cache"
TAPES = ROOT / "outputs/controlled_slow_fast_20261119_tapes"
PREVIOUS = ROOT / "outputs/autopilot_flight_dynamics_20261124"
GRIDS = 6


def harmonic_history(h, main):
    """Fit intercept, linear trend and both relative encoder-phase quadratures to past states."""
    t = np.linspace(-1, 0, h.shape[1])
    co, si = h[..., 14], h[..., 13]
    design = np.stack([np.ones_like(co), np.broadcast_to(t, co.shape), co, si], -1).astype('float64')
    phase_valid = h[..., 21] > 0
    valid = [phase_valid & (h[...,16]>0) & (h[...,18]>0) & main[:,None],
             phase_valid & (h[...,17]>0)]
    coefficients, flags = [], []
    for values, observed in zip((h[...,:3], h[...,3:6]), valid):
        masked = design * observed[...,None]
        gram = masked.transpose(0,2,1) @ masked
        eig = np.linalg.eigvalsh(gram)
        ok = (observed.sum(1)>=12) & (eig[:,0]>.0001*eig[:,-1]) & (h[:,-1,21]>0)
        beta = np.linalg.solve(np.where(ok[:,None,None],gram,np.eye(4)),masked.transpose(0,2,1)@values)
        coefficients.append(np.where(ok[:,None,None],beta[:,2:].transpose(0,2,1),0).reshape(-1,6))
        flags.append(ok[:,None])
    return np.concatenate([*coefficients,*flags],1).astype('float32')


def pool_commands(tape, times):
    """Exact two exponential ZOH command integrals. Accept arbitrary per-window query shapes."""
    dims=times.shape
    b=len(tape)
    q=times.reshape(b,-1).astype('float64')
    cmd=tape[...,:4].reshape(b,-1,4)
    dt=tape[...,4].reshape(b,-1).astype('float64')*1e-6
    ends=dt.cumsum(1)
    starts=ends-dt
    if np.any(q<0) or np.any(q>ends[:,-1:]+1e-6):
        raise ValueError('query outside recorded command interval')
    phi=np.concatenate((cmd,np.tanh(2*cmd)),axis=-1)
    offsets=np.arange(b,dtype='float64')[:,None]*4
    if ends.max()>=4: raise ValueError('unexpected tape duration')
    idx=np.searchsorted((ends+offsets).ravel(),(q+offsets).ravel(),side='left').reshape(q.shape)
    idx=np.clip(idx-np.arange(b)[:,None]*dt.shape[1],0,dt.shape[1]-1)
    row=np.arange(b)[:,None]
    result=[]
    for tau in (.2,1.):
        w=np.exp(starts/tau)*np.expm1(dt/tau)
        sums=np.concatenate((np.zeros((b,1,8)),np.cumsum(w[...,None]*phi,axis=1)),1)
        elapsed=np.clip(q-starts[row,idx],0,dt[row,idx])
        a=sums[row,idx]*np.exp(-q/tau)[...,None]
        a+=(-np.expm1(-elapsed/tau))[...,None]*phi[row,idx]
        result.append(a)
    return np.concatenate(result,-1).reshape(*dims,16).astype('float32')


def staggered(t, y, valid, initial):
    b=len(t)
    idx=np.zeros((b,GRIDS,8,2),dtype=int)
    time=np.zeros(idx.shape,np.float32)
    mask=np.zeros(idx.shape,bool)
    row=np.arange(b)[:,None]
    for j in range(8):
        for k,c in enumerate((0,2)):
            times=t[...,c]
            member=valid[...,c] & (times>j*.25) & (times<=(j+1)*.25)
            n=member.sum(1)
            # Choose six stratified native order statistics, never interpolated labels.
            pos=((np.arange(GRIDS)+.5)*n[:,None]/GRIDS).astype(int)
            ordered=np.sort(np.where(member,np.arange(times.shape[1]),times.shape[1]),axis=1)
            chosen=ordered[row,pos].clip(max=times.shape[1]-1)
            idx[:,:,j,k]=chosen
            time[:,:,j,k]=np.where(n[:,None]>0,times[row,chosen],(j+.5)*.25)
            mask[:,:,j,k]=n[:,None]>0
    rows=np.arange(b)[:,None,None]
    rot=Rotation.from_quat(initial[:,[7,8,9,6]]).as_matrix()
    dv=y[rows,idx[...,0],3:6]-initial[:,None,None,3:6]
    dw=y[rows,idx[...,1],10:13]-initial[:,None,None,10:13]
    target=np.stack([np.einsum('bgqi,bij->bgqj',dv,rot),dw],-2)
    if not np.all(np.diff(time,axis=2)>0): raise ValueError('nonmonotonic query')
    return time,np.where(mask[...,None],target,0).astype('float32'),mask


def prepare(part):
    old=np.load(PREVIOUS/f'{part}.npz',allow_pickle=False)
    meta=pd.read_parquet(HISTORY/f'{part}_windows.parquet').set_index('window_id',verify_integrity=True)
    selected=pd.read_parquet(TAPES/f'{part}_ids.parquet').window_id
    ix=meta.index.get_indexer(selected)
    assert (ix>=0).all() and selected.is_unique
    np.testing.assert_array_equal(old['window'],selected)
    np.testing.assert_array_equal(old['flight'],meta.iloc[ix].flight_id)
    groups=meta.iloc[ix].sampling_group.to_numpy()=='main_50hz'
    def source(name): return np.load(HISTORY/f'{part}_{name}.npy',mmap_mode='r')
    h,t,y,v,init=map(source,['history','t','y','valid','initial'])
    tape=np.load(TAPES/f'{part}_commands.npy',mmap_mode='r')
    n=len(ix)
    def writable(name,shape,dtype):
        return np.lib.format.open_memmap(CACHE/f'{part}_{name}.npy',mode='w+',shape=shape,dtype=dtype)
    hist=writable('features',(n,14),'float32')
    if part=='train':
        times=writable('time',(n,GRIDS,8,2),'float32')
        targets=writable('y',(n,GRIDS,8,2,3),'float32')
        masks=writable('mask',(n,GRIDS,8,2),'bool')
        pools=writable('pooled',(n,GRIDS,8,2,16),'float32')
    # Compare optimized prefix evaluation with the old direct integral on stored inputs.
    check=np.arange(0,n,max(n//64,1))
    np.testing.assert_allclose(pool_commands(tape[check],old['time'][check]),old['pooled'][check],rtol=2e-6,atol=5e-7)
    for start in range(0,n,384):
        sel=slice(start,min(start+384,n)); ids=ix[sel]
        hist[sel]=harmonic_history(h[ids],groups[sel])
        if part=='train':
            a,b,c=staggered(t[ids],y[ids],v[ids],init[ids])
            times[sel],targets[sel],masks[sel],pools[sel]=a,b,c,pool_commands(tape[sel],a)
    result=dict(windows=n,quadrature_available=hist[:,-2:].sum(0).astype(int).tolist())
    if part=='train': result.update(grids=GRIDS,native_counts=masks.sum((0,1)).tolist())
    print(part,result,flush=True)
    return result


if __name__=='__main__':
    CACHE.mkdir(parents=True,exist_ok=False)
    result={part:prepare(part) for part in ('validation','train')}
    (CACHE/'ready.json').write_text(json.dumps(result,indent=2)+'\n')
