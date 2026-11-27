"""Frozen high-order interventions and matched conditional readout fits."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = Path('/home/zn/flap-system-identification')
ROUND = PROJECT / 'research/rounds/phase-module-evidence-20261125'
sys.path.insert(0, str(ROUND/'combined'))
from predictor import WingbeatPredictor, causal_conditions
sys.path.insert(0, str(ROUND/'waveform'))
import experiment as wave


def oracle(part, flight, window, times, hist):
    data = ROOT/'dataset/velocity_rate_multistep_v1'
    encoder = pd.read_parquet(data/f'native_encoder_count_{part}.parquet',
        columns=['flight_id','timestamp_us','total_count','in_reviewed_flight']).groupby('flight_id')
    meta = {r['flight_id']:r for r in json.loads((data/'flight_provenance.json').read_text()) if r['split']==part}
    wins = pd.read_parquet(ROOT/f'outputs/history_slow_fast_20261119_cache/{part}_windows.parquet',
        columns=['window_id','decision_us','flight_id']).set_index('window_id').loc[window]
    np.testing.assert_array_equal(flight,wins.flight_id)
    output = np.zeros_like(times)
    valid = np.zeros_like(times,dtype=bool)
    for fid, group in pd.Series(np.arange(len(flight))).groupby(flight):
        i = group.to_numpy()
        e = encoder.get_group(fid)
        tm = e.timestamp_us.to_numpy()
        co = e.total_count.to_numpy(dtype='float64')/(4096*meta[fid]['transmission_ratio'])
        ok = e.in_reviewed_flight.to_numpy(bool)
        bad = np.r_[False,(np.diff(tm)>40000)|(~ok[:-1])|(~ok[1:])|(abs(np.diff(co))>.5)].cumsum()
        orig = wins.decision_us.to_numpy()[i]
        end = orig[:,None,None]+times[i].astype('float64')*1e6
        left = tm.searchsorted(orig,'right')-1
        right = tm.searchsorted(end)
        mask = (left[:,None,None]>=0)&(right<len(tm))
        mask &= bad[right.clip(max=len(tm)-1)]==bad[left.clip(0)][:,None,None]
        mask &= (ok[left.clip(0)]&(hist[i,171]>0)&(hist[i,172]>0))[:,None,None]
        for reset in meta[fid]['reset_times_us']:
            mask &= ~((orig[:,None,None]<=reset)&(end>=reset))
        output[i] = 2*np.pi*(np.interp(end,tm,co)-np.interp(orig,tm,co)[:,None,None])
        valid[i] = mask
    return torch.from_numpy(output),torch.from_numpy(valid)


def calculate(model,d,rows,t,pool,phase,offgrid):
    h,u,coef = (d[k][rows] for k in ('hist','u0','coef'))
    m = model.first_order
    core = m.core(h[:,:m.mean.numel()],pool,u,t) if not offgrid else wave.query(
        m.core,h[:,:m.mean.numel()],u,d['time'][rows],d['pooled'][rows],t,pool)
    basis = m.bases(h,pool,u,t)['direction']
    beta = torch.einsum('bh,ckha->bcka',m.context(h),m.weights)
    low = core+torch.einsum('bqck,bcka->bqca',basis,beta)
    theta = m.clocks(h,pool,u,t)[...,-1]
    normal = model.high_order(coef,t,theta,causal_conditions(h,pool,u,t))
    design = {arm:wave.design(coef,t,p,None,'full')[...,4:] for arm,p in [('pred',theta),('oracle',phase)]}
    return low,normal,design


def estimates(model,low,normal,design,fitted):
    out = {'none':low,'normal':low+normal}
    for arm,z in design.items():
        for label,w in [('frozen',model.high_order.readout_coefficients),('refit',fitted[arm])]:
            out[f'{arm}_{label}'] = low+torch.einsum('bqcxm,cxm->bqcx',z,w)
    return out


def load(part):
    d,flights,_ = wave.load(ROOT,part,'cpu')
    with np.load(ROOT/f'outputs/autopilot_flight_dynamics_20261124/{part}.npz') as z:
        ids=z['window']
    for suffix in (['','_early'] if part=='train' else ['']):
        p,valid=oracle(part,flights,ids,d['time'+suffix].numpy(),d['hist'].numpy())
        d['phase'+suffix]=p
        d['valid'+suffix]=valid & d['mask'+suffix].bool()
    return d,flights,ids


def fit(d,m):
    normal={}; rhs={}
    for s in ('','_early'):
        for i in range(0,len(d['hist']),512):
            sl=slice(i,i+512)
            low,_,design=calculate(m,d,sl,d['time'+s][sl],d['pooled'+s][sl],d['phase'+s][sl],bool(s))
            for arm,z in design.items():
                a=(z*d['valid'+s][sl,...,None,None]).double()
                res=(d['y'+s][sl]-low).double()
                normal[arm]=normal.get(arm,0)+torch.einsum('bqcxm,bqcxn->cxmn',a,a)
                rhs[arm]=rhs.get(arm,0)+torch.einsum('bqcxm,bqcx->cxm',a,res)
    return {k:torch.linalg.solve(g+1e-5*torch.diag_embed(g.diagonal(dim1=-2,dim2=-1)),rhs[k][...,None]).squeeze(-1).float() for k,g in normal.items()}


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    train,tr,_=load('train')
    val,flights,ids=load('validation')
    assert len(set(tr))==121 and len(set(flights))==39 and set(tr).isdisjoint(flights)
    cache=wave.native_cache(ROOT,'cpu')
    _,ref,t,pool,bins=cache
    ix=ref['window_idx']; ch=ref['ch']
    p,valid=oracle('validation',flights[ix],ids[ix],t.numpy(),val['hist'][ix,:180].numpy())
    print('native points',len(ix),'phase eligible',int(valid[:,0,0].sum()),flush=True)
    rows=[]; native=[]
    for seed in wave.SEEDS:
        m=WingbeatPredictor.from_seed(seed)
        fitted=fit(train,m)
        np.savez(HERE/f'readouts_{seed}.npz',**{k:v.numpy() for k,v in fitted.items()})
        errors={}; points={}
        for i in range(0,len(ids),512):
            sl=slice(i,i+512)
            parts=calculate(m,val,sl,val['time'][sl],val['pooled'][sl],val['phase'][sl],False)
            for arm,v in estimates(m,*parts,fitted).items():
                errors.setdefault(arm,[]).append((v-val['y'][sl]).square()[:,[1,3,7]].numpy())
        mask=val['valid'][:,[1,3,7]].numpy()
        for arm,parts in errors.items():
            e=np.concatenate(parts)
            for fid in np.unique(flights):
                for j,lead in enumerate((.5,1.,2.)):
                    for c in (0,1):
                        ok=(flights==fid)&mask[:,j,c]
                        if ok.any():
                            rows.append((seed,arm,fid,lead,c,int(ok.sum()),*e[ok,j,c].mean(axis=0,dtype='float64')))
        for i in range(0,len(ix),512):
            sl=slice(i,i+512)
            parts=calculate(m,val,ix[sl],t[sl],pool[sl],p[sl],True)
            for arm,v in estimates(m,*parts,fitted).items():
                points.setdefault(arm,[]).append(v[np.arange(len(ix[sl])),0,ch[sl]].numpy()+ref['origin'][sl])
        for arm,parts in points.items():
            residual=ref['data']-np.concatenate(parts)
            for bid,(jj,w,q,lead) in enumerate(bins):
                if q is None or not valid[jj,0,0].all():continue
                j=jj[0]
                err=residual[jj]*np.sqrt(w[:,None])
                native.append((seed,arm,ref['flight_ids'][ref['flight_index'][j]],int(ix[j]),bid,lead,int(ch[j]),
                               (err**2).sum(),((q.T@err)**2).sum()))
        print('completed',seed,flush=True)
    pd.DataFrame(rows,columns='seed arm flight horizon stream n mse_x mse_y mse_z'.split()).to_csv(HERE/'endpoint.csv',index=False)
    n=pd.DataFrame(native,columns='seed arm flight window beat lead stream full higher'.split())
    n['period']=pd.cut(n.lead,[0,.5,1,np.inf],labels=['early','middle','late'])
    n.groupby(['seed','arm','flight','stream','period','window'],observed=True)[['full','higher']].mean().groupby(
        level=['seed','arm','flight','stream','period'],observed=True).mean().to_csv(HERE/'native.csv')


if __name__=='__main__':
    main()
