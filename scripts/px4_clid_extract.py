"""Train-only native 20 ms sequences; raw logs read only, local sample artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
ROOT=Path('/home/zn/flap-system-identification')
DATA=ROOT/'dataset/velocity_rate_multistep_v1'
OUT=Path('experiments/px4_closed_loop_dynamic_id_v2')
META=ROOT/'.worktrees/px4-identifiability-contract-v1/experiments/px4_identifiability_v1/train_flights.json'
DT=20000
GAP=40000
TOL=12000

def zoh_integral(t,v,grid):
    """Exact time average on [grid-DT,grid), using command publication clock."""
    t=np.asarray(t,dtype=np.int64);v=np.asarray(v,float)
    assert np.all(np.diff(t)>0)
    finite=np.isfinite(v).all(1);v=np.where(np.isfinite(v),v,0.)
    area=np.vstack([np.zeros(v.shape[1]),np.cumsum(np.diff(t)[:,None]*v[:-1],axis=0)])
    def integral(q):
        j=np.searchsorted(t,q,side='right')-1;k=np.maximum(j,0)
        return area[k]+(q-t[k])[:,None]*v[k],j
    a,ja=integral(grid);b,jb=integral(grid-DT)
    valid=(jb>=0)&(grid-t[np.maximum(ja,0)]<=GAP)
    # Reject any long publication gap intersecting the interval.
    bad=np.r_[False,np.diff(t)>GAP];cum=np.cumsum(bad)
    valid &= (cum[np.maximum(ja,0)]-cum[np.maximum(jb,0)]==0)
    invalid=np.cumsum(~finite)
    valid &= finite[np.maximum(jb,0)] & (invalid[np.maximum(ja,0)]-invalid[np.maximum(jb,0)]==0)
    return (a-b)/DT,valid

def main():
    meta=json.loads(META.read_text());assert len(meta)==114 and all(m['split']=='train' for m in meta)
    assert len({m['firmware'] for m in meta})==15
    samples=pd.read_parquet(DATA/'samples_train.parquet');assert set(samples.split)=={'train'}
    samples={k:g for k,g in samples.groupby('flight_id',sort=False)}
    sequences=[];coverage=[];timing=[];offsets=[];identities=[]
    for no,m in enumerate(meta):
        names=['vehicle_angular_velocity','actuator_servos','actuator_motors','vehicle_status','wing_phase']
        log=ULog(str(Path('/home/zn/QgcLogs')/m['log_path']),message_name_filter_list=names)
        assert log.msg_info_dict['ver_sw']==m['firmware']
        tabs={d.name:d.data for d in log.data_list if d.multi_id==0}
        a=tabs['vehicle_angular_velocity'];s=tabs['actuator_servos'];motor=tabs['actuator_motors']
        for name,tab in tabs.items():
            pub=tab['timestamp'].astype(np.int64);sel=(pub>=m['start_us'])&(pub<=m['end_us']);p=pub[sel]
            if np.any(np.diff(p)<=0):raise ValueError('time reset or duplicate '+name)
            dd=np.diff(p);delay=(p-tab['timestamp_sample'][sel].astype(np.int64)) if 'timestamp_sample' in tab else np.array([])
            timing.append(dict(date=m['flight_date'],firmware=m['firmware'],topic=name,n=len(p),dt_p50_us=float(np.median(dd)),dt_p99_us=float(np.quantile(dd,.99)),gap_threshold_count=int((dd>GAP).sum()),publication_delay_p50_us=float(np.median(delay)) if len(delay) else None,publication_delay_p99_us=float(np.quantile(delay,.99)) if len(delay) else None))
        grid=np.arange(((m['start_us']+DT-1)//DT)*DT,m['end_us']+1,DT,dtype=np.int64)
        pub=a['timestamp'].astype(np.int64);ts=a['timestamp_sample'].astype(np.int64)
        assert np.all(np.diff(ts)>0)
        j=np.searchsorted(pub,grid,side='right')-1;j=np.maximum(j,0)
        y=np.column_stack([a[f'xyz[{k}]'][j] for k in range(3)])
        off=ts[j]-grid
        ok=(off<=0)&(off>=-TOL)&(pub[j]<=grid)&(np.r_[0,np.diff(ts)][j]<=GAP)
        sv,vs=zoh_integral(s['timestamp'],np.column_stack([s[f'control[{i}]'] for i in range(3)]),grid)
        mv,vm=zoh_integral(motor['timestamp'],motor['control[0]'][:,None],grid)
        u=np.column_stack([mv[:,0],(sv[:,1]-sv[:,0])/2,(sv[:,1]+sv[:,0])/2,sv[:,2]])
        q=samples[m['flight_id']];qt=q.timestamp_us.to_numpy();left=np.searchsorted(qt,grid,side='right')-1;right=left+1
        inrange=(left>=0)&(right<len(q));left=np.clip(left,0,len(q)-1);right=np.clip(right,0,len(q)-1)
        good=(q.valid_core & ~q.excluded & ~q.review & ~q.gap_boundary & ~q.reset_boundary & ~q.mode_change & q.valid_status).to_numpy()
        seg=q.segment_id.to_numpy();mode=q.nav_state.to_numpy();phase=q.relative_flap_phase_rad.to_numpy()
        quality=inrange&good[left]&good[right]&(seg[left]==seg[right])&(mode[left]==mode[right])&(qt[right]-qt[left]<=GAP)
        ok &= vs&vm&quality&np.isfinite(y).all(1)&np.isfinite(u).all(1)
        # Native status transitions inside a grid interval also break the sequence.
        st=tabs['vehicle_status'];sj=np.maximum(np.searchsorted(st['timestamp'],grid,side='right')-1,0)
        md=st['nav_state'][sj];ok &= (md==mode[left])&(st['failsafe'][sj]==0)&(st['arming_state'][sj]==2)
        breaks=np.r_[True,(np.diff(seg[left])!=0)|(np.diff(md)!=0)|(~ok[:-1])|(~ok[1:])]
        group=np.cumsum(breaks);kept=0;nseg=0
        for g in np.unique(group[ok]):
            ix=np.flatnonzero((group==g)&ok)
            if len(ix)<40:continue
            assert np.all(np.diff(grid[ix])==DT)
            sequences.append(dict(y=y[ix],u=u[ix],t=grid[ix],offset=off[ix],phase=phase[left[ix]],date=m['flight_date'],firmware=m['firmware'],flight=m['flight_id'],mode=int(md[ix[0]])))
            offsets.extend(off[ix]);kept+=len(ix);nseg+=1
        coverage.append(dict(date=m['flight_date'],firmware=m['firmware'],grid_candidates=len(grid),quality_pass=int(quality.sum()),aligned_pass=int(ok.sum()),retained=kept,segments=nseg,reviewed_s=m['reviewed_duration_s']))
        identities.append({k:m[k] for k in ['flight_id','log_path','firmware','split','start_us','end_us']})
        print('extract',no+1,kept,flush=True)
    import pickle
    (OUT/'local').mkdir(exist_ok=True,parents=True)
    with (OUT/'local/sequences.pkl').open('wb') as f:pickle.dump(sequences,f)
    (OUT/'local/identities.json').write_text(json.dumps(identities,indent=2))
    pd.DataFrame(coverage).groupby(['date','firmware'],as_index=False).sum(numeric_only=True).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(timing).to_csv(OUT/'native_timing.csv',index=False)
    hist,edges=np.histogram(offsets,bins=np.arange(-TOL,1000,1000));pd.DataFrame(dict(low_us=edges[:-1],high_us=edges[1:],count=hist)).to_csv(OUT/'match_offsets.csv',index=False)
    summary=dict(logs=len(meta),dates=len({x['date'] for x in sequences}),firmware_count=15,dt_us=DT,gap_threshold_us=GAP,match_tolerance_us=TOL,match_rule='last published native rate; sample <= grid; no interpolation',offset_quantiles_us=np.quantile(offsets,[0,.01,.5,.99,1]).tolist(),retained_rows=len(offsets),segments=len(sequences),phase='relative encoder phase diagnostic only; causal Hall phase not established; M2 NO-GO',input=f'exact recorded publication-clock ZOH integral [t_k-{DT}us,t_k); factual replay; unlogged commands unknown',base_commit='c7683865f053ac7aa165a13e7e68641a7198a814')
    (OUT/'time_contract.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--coarse100',action='store_true');args=parser.parse_args()
    if args.coarse100:
        DT=100000;GAP=160000;TOL=30000;OUT=OUT/'coarse100';OUT.mkdir(exist_ok=True)
    main()
