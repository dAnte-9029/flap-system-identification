"""Precompute the true endpoint of each partial integration interval on native training beats."""
import numpy as np
import pandas as pd
import torch
from shared import ROOT,PREV,EXTRA,Dynamics,load_tuned,wave

@torch.no_grad()
def prepare():
    d,flight,_=wave.load(ROOT,'train','cpu')
    with np.load(PREV/'train.npz') as z:
        wi,length,t,wt,target,origin,win=[z[k] for k in ('wi','lengths','t','weight','target','origin','window')]
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/train.npz') as z:np.testing.assert_array_equal(win,z['window'])
    cases=np.repeat(wi,length)
    wtab=pd.read_parquet(ROOT/'outputs/history_slow_fast_20261119_cache/train_windows.parquet').set_index('window_id').loc[win]
    ts=wtab.decision_us.to_numpy()[cases]+np.rint(t.astype(float)*1e6).astype(np.int64)
    gyro=pd.read_parquet(ROOT/'dataset/velocity_rate_multistep_v1/native_vehicle_angular_velocity_train.parquet')
    raw=np.empty((len(ts),3),np.float32)
    for fid,rr in pd.Series(flight[cases]).groupby(flight[cases],sort=False).groups.items():
        g=gyro.loc[(gyro.flight_id==fid)&gyro.in_reviewed_flight].drop_duplicates('timestamp_sample_us').sort_values('timestamp_sample_us')
        clock=g.timestamp_sample_us.to_numpy()
        k=clock.searchsorted(ts[rr])
        np.testing.assert_array_equal(clock[k],ts[rr])
        raw[rr]=g[['xyz[0]','xyz[1]','xyz[2]']].to_numpy()[k]
    ends=length.cumsum();starts=ends-length
    np.testing.assert_allclose(np.add.reduceat(wt[:,None]*raw,starts),target,rtol=2e-6,atol=3e-7)
    tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/train_commands.npy',mmap_mode='r')
    orig=load_tuned(17,'cpu');net=Dynamics(orig,False)
    terminal=torch.empty(len(t),1+24+16)
    fixed=torch.empty(len(t),3)
    slot=torch.empty(len(t),dtype=torch.int64)
    for lo in range(0,len(wi),96):
        b=np.arange(lo,min(lo+96,len(wi)));ii=wi[b];r=np.arange(starts[b[0]],ends[b[-1]])
        col=np.arange(length[b].max())
        pos=starts[b,None]+np.minimum(col,length[b,None]-1)
        pp=wave.pool_commands(tape[ii],t[pos])[col<length[b,None]]
        qp=torch.from_numpy(pp)[:,None,None].expand(-1,1,2,-1)
        qt=torch.from_numpy(t[r])[:,None,None].expand(-1,1,2)
        v=cases[r];h,cp,u,ct,co=[d[k][v] for k in ('hist','pooled','u0','time','coef')]
        before=(ct[:,:,1]<qt[:,0,1,None]-1e-7).sum(1)
        slot[r]=before
        for n in torch.unique(before).tolist():
            ids=torch.where(before==n)[0]
            times=torch.cat((ct[ids,:n,1],qt[ids,:,1]),1)[:,:,None].expand(-1,-1,2)
            pools=torch.cat((cp[ids,:n,1],qp[ids,:,1]),1)[:,:,None].expand(-1,-1,2,-1)
            dt,f,mem,_=net.inputs(pools,u[ids],times)
            terminal[r[ids]]=torch.cat((dt[:,-1,1],f[:,-1,1],mem[:,-1,1]),1)
        fixed[r]=orig(h,qp,u,qt,co)[:,0,1]-orig.first_order.core(h[:,:orig.first_order.mean.numel()],qp,u,qt)[:,0,1]+torch.from_numpy(origin[v])
    starts_all=np.zeros(len(flight),np.int64);starts_all[wi]=starts
    lens_all=np.zeros(len(flight),np.int64);lens_all[wi]=length
    EXTRA.mkdir(parents=True,exist_ok=True)
    torch.save(dict(path=terminal,offset=fixed,slot=slot,weight=torch.from_numpy(wt.astype('float32')),
                    raw=torch.from_numpy(raw),start=torch.from_numpy(starts_all),length=torch.from_numpy(lens_all)),EXTRA/'beats.pt')
    print(len(wi),len(t),int(length.max()),flush=True)

if __name__=='__main__':
    torch.set_num_threads(3)
    prepare()
