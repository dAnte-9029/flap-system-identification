"""Separate changes in start membership from changes in native labels near the 2 s boundary."""
import json
import numpy as np
import pandas as pd
import torch
from workspace import *
from evaluate import restore,tables,H2
from train_three import evaluate


def compare_native(ids):
    met={'overlapping_targets_max_abs':0.,'overlapping_times_max_abs':0.}
    old={k:np.load(HISTORY/f'validation_{k}.npy',mmap_mode='r') for k in ('y','t','valid')}
    new={k:np.load(CACHE3/f'validation_{k}.npy',mmap_mode='r') for k in old}
    tapes=[np.load(p/'validation_commands.npy',mmap_mode='r') for p in (TAPES,CACHE3)]
    clocks=np.array([0]*6+[1]*4+[2]*3)
    changes=np.zeros((4,3),dtype=int)
    before190=before200=0
    for j in range(0,len(ids),512):
        batch=ids.iloc[j:j+512]
        a={k:arr[batch.history_index.to_numpy()] for k,arr in old.items()}
        b={k:arr[j:j+len(batch)] for k,arr in new.items()}
        L=a['t'].shape[1]
        assert (b['valid'][:,:L] | ~a['valid']).all()
        e=max(abs(a['y']-b['y'][:,:L])[a['valid'][:,:,clocks]],default=0.)
        dt=max(abs(a['t']-b['t'][:,:L])[a['valid']],default=0.)
        met['overlapping_targets_max_abs']=max(met['overlapping_targets_max_abs'],float(e))
        met['overlapping_times_max_abs']=max(met['overlapping_times_max_abs'],float(dt))
        for k,h in enumerate(H2):
            def nearest(rec):
                ix=np.where(rec['valid'],abs(rec['t']-h),np.inf).argmin(1)
                return np.take_along_axis(rec['t'],ix[:,None,:],1)[:,0]
            changes[k]+=(nearest(a)!=nearest(b)).sum(0)
        u=tapes[0][batch.tape_index.to_numpy(),:100]
        v=tapes[1][j:j+len(batch),:100]
        width=max(u.shape[2],v.shape[2])
        def pad(x):
            x=np.pad(x,((0,0),(0,0),(0,width-x.shape[2]),(0,0)))
            x[...,:4]*=(x[...,4:]>0)
            return x
        diff=(pad(u)!=pad(v)).any(axis=(2,3))
        before190+=int(diff[:,:95].any(1).sum())
        before200+=int(diff.any(1).sum())
    met.update(changed_endpoints={str(h):dict(zip(('pv','attitude','rate'),row.tolist())) for h,row in zip(H2,changes)},
               different_tapes_through_1p9s=before190,different_tapes_through_2s=before200,windows=len(ids))
    assert met['overlapping_times_max_abs']==0 and met['overlapping_targets_max_abs']<1e-5
    (RUN/'common_prefix_data.json').write_text(json.dumps(met,indent=2))
    print(met,flush=True)


@torch.no_grad()
def main():
    setup()
    ids=pd.read_parquet(CACHE3/'validation_windows.parquet')
    compare_native(ids)
    data,all_ids=conditional_data(HISTORY,'validation',TAPES)
    ix=ids.tape_index.to_numpy()
    assert np.array_equal(all_ids.window_id.iloc[ix],ids.window_id)
    data=select(data,torch.from_numpy(ix.copy()).cuda())
    rep=json.loads((RUN/'freeze.json').read_text())['representative']
    parts=[]
    for name in (rep,*BASELINES):
        model,ck=restore(name)
        _,frame=evaluate(model,data,ids,torch.tensor(ck['config']['units'],device='cuda'),H2)
        parts.append(frame.assign(model=name))
    tables(parts,'matched_original2',rep)


if __name__=='__main__': main()
