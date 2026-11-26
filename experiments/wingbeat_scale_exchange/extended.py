"""Frozen endpoint and within-wingbeat scores for both ablations and the phase incumbent."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from gated import GatedReadout, import_file
from module import ScaleExchange
from run import ARMS, ControlledODE, FROZEN, PROJECT, prepare, score
from train import segment

ROUND = PROJECT/'research/rounds/wingbeat-scale-exchange'
PERIODIC = FROZEN.parent/'periodic-motion-20261125'
AXES = ('vx','vy','vz','p','q','r')


def load_state(path, device):
    return torch.load(path, map_location=device, weights_only=True)


class FrozenModels:
    def __init__(self, device):
        state = load_state(FROZEN/'ode_baseline/best.pt',device)
        self.ode = ControlledODE(state['mean'],state['scale']).to(device)
        self.ode.load_state_dict(state)
        moment = import_file('first_moment',FROZEN/'coherence/module.py')
        s = load_state(FROZEN/'coherence/conditional_seed17.pt',device)
        self.phase = moment.MomentPredictor(s['mean'],s['std']).to(device)
        self.phase.load_state_dict(s)
        with np.load(FROZEN/'coherence/heads_seed17.npz') as heads:
            self.phase.weights.copy_(torch.as_tensor(heads['direction'],device=device))
        s = load_state(PERIODIC/'readout/original_runs/incumbent17/corrected.pt',device)
        self.clock = moment.prior.FlightPredictor(s['mean'],s['std']).to(device)
        self.clock.load_state_dict(s)
        self.wave = import_file('transport',FROZEN/'waveform/coherence/inference.py')
        self.head = self.wave.HarmonicTransport.from_files(17,FROZEN/'waveform/coherence',device)
        self.native_core = import_file('native_core',PERIODIC/'readout/native_path.py').query
        self.first = {k:ScaleExchange(**load_state(ROUND/f'fit/{k}.pt',device),arm=k) for k in ARMS}
        self.gates = {k:GatedReadout(**load_state(ROUND/f'gated/{k}.pt',device)) for k in ('cross','self','deleted')}

    def predict(self, data, base, hist, u0, pool, canonical=None):
        time=data['time']
        co=data['coeff'][...,4:].unflatten(-1,(2,2))
        cond=self.wave.causal_conditions(hist,pool,u0,time)
        angle=self.clock.clocks(hist,pool,u0,time)[...,-1]
        reference=base+self.head(co,time,angle,cond)
        # Phase incumbent was published using IEEE float32; ODE replay uses TF32.
        torch.set_float32_matmul_precision("highest")
        if canonical is None:
            phase_low=self.phase.forecast(hist,pool,u0,time,'direction')
        else:
            core_hist=hist[:,:self.phase.mean.numel()]
            core=self.native_core(self.phase.core,core_hist,u0,canonical[0],canonical[1],time,pool)
            beta=torch.einsum('bh,ckha->bcka',self.phase.context(hist),self.phase.weights)
            basis=self.phase.bases(hist,pool,u0,time)['direction']
            phase_low=core+torch.einsum('bqck,bcka->bqca',basis,beta)
        phase_angle=self.phase.clocks(hist,pool,u0,time)[...,-1]
        phase=phase_low+self.head(co,time,phase_angle,cond)
        torch.set_float32_matmul_precision("high")
        return dict(phase_main=phase,ode=base,reference=reference,
                    **{k:m(base,**data) for k,m in self.first.items()},
                    **{k:m(base,reference,**data) for k,m in self.gates.items()})


def report_endpoints(ids,flights,y,pred,out):
    rows=[]
    for name,a in pred.items():
        diff=abs(a-y).reshape(len(y),8,6)
        for f in np.unique(flights):
            err=diff[flights==f]
            for stat,v in [('MAE',err.mean(0,dtype=np.float64)),('P95',np.quantile(err,.95,axis=0))]:
                for h in (1,3,7):
                    rows.append(dict(arm=name,flight=f,horizon=(h+1)/4,metric=stat,**dict(zip(AXES,v[h]))))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'endpoint_flights.csv',index=False)
    frame.groupby(['arm','horizon','metric'])[list(AXES)].mean().to_csv(out/'endpoints.csv')
    # Validate every route against its previous six-metric published result.
    for folder in ('fit','gated'):
        prior=pd.read_csv(ROUND/folder/'flight_mse.csv')
        for name,sub in prior.groupby('arm'):
            key={'baseline':'ode','prior_waveform':'reference'}.get(name,name)
            computed=score(torch.from_numpy(pred[key]),torch.from_numpy(y),flights)
            np.testing.assert_allclose(computed,sub.set_index('flight').loc[computed.index,computed.columns],atol=1.1e-5,rtol=1e-5)
    old=pd.read_csv(FROZEN/'combined/flight_mse.csv').query('seed==17 and arm=="full"').set_index('flight')
    computed=score(torch.from_numpy(pred['phase_main']),torch.from_numpy(y),flights)
    np.testing.assert_allclose(computed,old.loc[computed.index,computed.columns],atol=2e-5,rtol=3e-5)


def native_batch(tapes,windows,times,canonical,device):
    segments=[]
    for w,t in zip(windows,times):
        both=np.full((1,2),t,dtype=np.float32)
        clocks=np.concatenate((canonical[w],both))
        u,ends,q=segment(tapes[w],clocks)
        segments.append((u,ends,q[-1:]))
    length=max(len(x[0]) for x in segments)
    uu=np.zeros((len(times),length,4),np.float32)
    end=np.zeros(uu.shape[:2],np.float32)
    dt=np.zeros_like(end)
    queries=[]
    for i,(u,e,q) in enumerate(segments):
        uu[i,:len(u)]=u
        end[i,:len(u)]=e
        end[i,len(u):]=e[-1]
        dt[i,:len(u)]=np.diff(np.r_[0.,e])
        queries.append(q)
    return [torch.as_tensor(a,device=device) for a in (uu,end,dt,np.stack(queries))]


def beat_rows(ref,predictions):
    rows=[]
    for ix in np.split(np.arange(len(ref['time'])),1+np.flatnonzero(np.diff(ref['block']))):
        i=ix[0]; w=ref['weight'][ix]; t=ref['time'][ix]; phase=ref['phase'][ix]
        assert np.isclose(w.sum(),1)
        x=np.column_stack([np.ones(len(t)),(t-t.mean())/np.ptp(t),
                          *[fun(k*phase) for k in (1,2,3) for fun in (np.cos,np.sin)]])
        weighted=x*np.sqrt(w)[:,None]
        if len(ix)<10 or np.linalg.cond(weighted)>=100:
            continue
        obs=ref['data'][ix]-ref['origin'][ix]
        values=np.stack([obs,*[v[ix] for v in predictions.values()]])
        beta=np.einsum('hs,msa->mha',np.linalg.pinv(weighted)*np.sqrt(w),values)
        vector=beta[:,2::2]+1j*beta[:,3::2]
        amp=abs(vector)
        delta=values-(values*w[None,:,None]).sum(1,keepdims=True)
        sd=np.sqrt((delta**2*w[None,:,None]).sum(1))
        corr=(w[None,:,None]*delta[:1]*delta).sum(1)/np.maximum(sd[:1]*sd,1e-16)
        lead=w@t
        meta=dict(block=int(ref['block'][i]),flight=ref['flight_ids'][ref['flight_index'][i]],
                  period='early' if lead<=.5 else ('late' if lead>1 else 'middle'))
        for j,axis in enumerate(AXES[int(ref['ch'][i])*3:][:3]):
            for m,name in enumerate(predictions,1):
                if sd[:,j].min()>1e-10:
                    rows.append(dict(**meta,arm=name,axis=axis,order=0,metric='PCC',value=corr[m,j]))
                for h in range(3):
                    args=dict(**meta,arm=name,axis=axis,order=h+1)
                    rows.append(dict(**args,metric='amplitude_MAE',value=abs(amp[m,h,j]-amp[0,h,j])))
                    if amp[0,h,j]>=.1 and np.min(amp[1:,h,j])>1e-10:
                        discrepancy=np.angle(vector[m,h,j]*vector[0,h,j].conjugate(),deg=True)
                        rows.append(dict(**args,metric='phase_MAE_deg',value=abs(discrepancy)))
    return pd.DataFrame(rows)


@torch.inference_mode()
def run(root,out,device):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    models=FrozenModels(device)
    data,base,y,flights,_=prepare(root,'validation',device,models.ode)
    pool_commands=import_file('native_cmd',PROJECT/'experiments/periodic_motion/prepare_native.py').pool_commands
    with np.load(root/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:
        hist=np.c_[z['hist'],np.load(root/'outputs/autopilot_periodic_motion_20261125/validation_features.npy')]
        hist,u,pool=[torch.as_tensor(a,device=device) for a in (hist,z['u0'],z['pooled'])]
        ids=z['window']; canonical=z['time']
        np.testing.assert_array_equal(flights,z['flight'])
    endpoints={}
    for i in range(0,len(y),512):
        s=slice(i,i+512)
        for name,a in models.predict({k:v[s] for k,v in data.items()},base[s],hist[s],u[s],pool[s]).items():
            endpoints.setdefault(name,[]).append(a.cpu().numpy())
    report_endpoints(ids,flights,y.cpu().numpy(),{k:np.concatenate(v) for k,v in endpoints.items()},out)
    del endpoints
    with np.load(root/'research/rounds/periodic-motion-20261125/diagnostics/reference.npz') as z:
        ref=dict(z)
    wi=ref['window_idx']; ts=ref['time'].astype(np.float32)
    np.testing.assert_array_equal(flights[wi],ref['flight_ids'][ref['flight_index']])
    tapes=np.load(root/'outputs/controlled_slow_fast_20261119_tapes/validation_commands.npy',mmap_mode='r')
    pools=np.zeros((len(wi),16),np.float32)
    for win in np.unique(wi):
        idx=np.flatnonzero(wi==win)
        pools[idx]=pool_commands(tapes[win:win+1],ts[None,idx])[0]
    native={}
    for i in range(0,len(wi),384):
        s=slice(i,i+384); ix=wi[s]
        t=torch.as_tensor(ts[s],device=device)[:,None,None].expand(-1,1,2)
        cp=torch.as_tensor(pools[s],device=device)[:,None,None,:].expand(-1,1,2,-1)
        batch={k:a[ix] for k,a in data.items() if k!='time'}
        b=models.ode(hist[ix],*native_batch(tapes,ix,ts[s],canonical,device))
        result=models.predict(dict(batch,time=t),b,hist[ix],u[ix],cp,(data['time'][ix],pool[ix]))
        for name,a in result.items():
            native.setdefault(name,[]).append(a[torch.arange(len(ix)),0,ref['ch'][s]].cpu().numpy())
    rows=beat_rows(ref,{k:np.concatenate(v) for k,v in native.items()})
    keys=['arm','period','axis','order','metric']
    agg=rows.groupby(keys+['flight']).value.agg(['mean','count']).reset_index()
    agg.to_csv(out/'harmonic_flights.csv',index=False)
    agg.groupby(keys).agg(value=('mean','mean'),flights=('flight','size'),beats=('count','sum')).to_csv(out/'harmonics.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--root',type=Path,default=Path('/home/zn/flap-system-identification'))
    p.add_argument('--out',type=Path,default=ROUND/'extended')
    a=p.parse_args()
    run(a.root,a.out,a.device)
