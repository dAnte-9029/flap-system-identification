"""Paired, native-time diagnostics. Observed future values occur only in scoring."""
import json
import numpy as np
from collections import defaultdict
import pandas as pd
from common import *


def pick(x,i):
    return x[torch.arange(len(x),device=x.device),i]


def rotlog(a,b):
    r=unit(qmul(inverse(a),b))
    r=torch.where(r[...,:1]<0,-r,r)
    mag=r[...,1:].norm(dim=-1,keepdim=True)
    return (2*torch.atan2(mag,r[...,:1])/mag.clamp_min(1e-12))*r[...,1:]


def integrate_rate(w,t,mask):
    dt=(t[:,1:]-t[:,:-1])*mask[:,1:]*mask[:,:-1]
    dq=qexp(.5*dt[...,None]*(w[:,1:]+w[:,:-1]))
    return prefix_rotations(dq)[:,-1]


def native_diagnostics(roll,b):
    h,s,y,t,valid,beat,endpoint,_=b
    pv=query(roll,t[:,:,0],'pv'); q=query(roll,t[:,:,1],'q'); w=query(roll,t[:,:,2],'w')
    dv=pv[:,:,3:]-y[:,:,3:6]; dw=w-y[:,:,10:]
    dp=pv[:,:,:3]-y[:,:,:3]; dr=rotlog(y[:,:,6:10],q)
    curves=[]; details={}
    for horizon in np.arange(.1,2.01,.1):
        row=[]
        for err,ti in [(dp,0),(dv,0),(dr,1),(dw,2)]:
            distance=(t[:,:,ti]-horizon).abs().masked_fill(~valid[:,:,ti],float('inf'))
            dist,i=distance.min(1)
            row.append(torch.cat([pick(err,i).square(),dist[:,None]],1))
        curves.append(torch.stack(row,1))
    for label,err,idx in [('v',dv,0),('omega',dw,2)]:
        mask=valid[:,:,idx]
        dt=((t[:,1:,idx]-t[:,:-1,idx])*mask[:,1:]*mask[:,:-1])[...,None]
        integral=(.5*(err[:,1:]+err[:,:-1])*dt).sum(1)
        mu=integral/dt.sum(1).clamp_min(1e-8)
        ms=(.5*(err[:,1:].square()+err[:,:-1].square())*dt).sum(1)/dt.sum(1).clamp_min(1e-8)
        for k in range(3):
            details[f'{label}{k}_bias2']=mu[:,k].square()
            details[f'{label}{k}_ms']=ms[:,k]
        # Signed errors that cancel within the first fully covered wingbeat.
        wt=beat[:,:,0 if idx==0 else 1]
        den=wt.sum(1)
        mean=(wt[...,None]*err).sum(1)/den.clamp_min(.0001)[:,None]
        msbeat=(wt[...,None]*err.square()).sum((1,2))/den.clamp_min(.0001)
        details[label+'_beat_net2']=mean.square().sum(1).masked_fill(den<=.05,np.nan)
        details[label+'_beat_ms']=msbeat.masked_fill(den<=.05,np.nan)
        first=mask.long().argmax(1);last=mask.size(1)-1-mask.flip(1).long().argmax(1)
        if idx==0:
            residual=pick(dp,last)-pick(dp,first)-integral
            details['p_first2']=pick(dp,first).square().sum(1)
            details['p_last2']=pick(dp,last).square().sum(1)
            details['p_integral2']=integral.square().sum(1)
            details['p_closure2']=residual.square().sum(1)
    # Interpolate diagnostic rates to attitude knot times only inside BOTH native supports.
    tw=t[:,:,2].masked_fill(~valid[:,:,2],float('inf'))
    # compact invalid leading/padded rows for monotonic search
    tw,order=tw.sort(1)
    tq=t[:,:,1].contiguous()
    j=torch.searchsorted(tw.contiguous(),tq).clamp(1,tw.size(1)-1)
    lo,hi=tw.gather(1,j-1),tw.gather(1,j)
    support=valid[:,:,1] & hi.isfinite() & (tq>=lo) & (tq<=hi)
    frac=((tq-lo)/(hi-lo).clamp_min(1e-8)).nan_to_num().clamp(0,1)
    def interp(x):
        def at(k):
            return x.gather(1,order.gather(1,k)[...,None].expand(-1,-1,3))
        return torch.lerp(at(j-1),at(j),frac[...,None])
    first=support.long().argmax(1); last=support.size(1)-1-support.flip(1).long().argmax(1)
    qref=y[:,:,6:10]
    qa,qb=pick(qref,first),pick(qref,last)
    int_w=integrate_rate(interp(w),tq,support)
    int_ref=integrate_rate(interp(y[:,:,10:]),tq,support)
    ok=support.sum(1)>=2
    for col,z in [('q_first',geodesic(pick(q,first),qa)),
                  ('q_last',geodesic(pick(q,last),qb)),
                  ('q_integrated_difference',geodesic(int_w,int_ref)),
                  ('q_model_closure',geodesic(qmul(pick(q,first),int_w),pick(q,last))),
                  ('q_observed_closure',geodesic(qmul(qa,int_ref),qb))]:
        details[col+'2']=z.square().masked_fill(~ok,np.nan)
    details['first_q_s']=pick(tq,first)
    return torch.stack(curves,1).cpu().numpy(),pd.DataFrame({k:v.cpu().numpy() for k,v in details.items()})


def harmonic_error(data, ids):
    from dynamics import clock
    totals = defaultdict(lambda: np.zeros(3))
    for start in range(0,len(ids),2048):
        rows = slice(start,start+2048)
        _,dt,_,basis,slope = clock(data[0][rows],data[-1][rows])
        mask = (dt[:,:-1,0]>0) & (basis[:,:-1,:2].square().sum(-1)>.5)
        for k in (1,2):
            pair = slice(2*k-2,2*k)
            exact = (basis[:,1:,pair]-basis[:,:-1,pair])[mask]
            euler = (dt[:,:-1]*slope[:,:-1,pair])[mask]
            totals[k] += [float((euler-exact).square().sum(dtype=torch.float64)),
                          float(exact.square().sum(dtype=torch.float64)), int(mask.sum())]
    result = {k:dict(segments=int(n),relative_rms=float(np.sqrt(err/norm)))
              for k,(err,norm,n) in totals.items()}
    (OUT/'harmonic_quadrature.json').write_text(json.dumps(result,indent=2))


@torch.no_grad()
def main():
    setup()
    data,ids=conditional_data(HISTORY,'validation',TAPES)
    assert len(ids)==44524 and ids.flight_id.nunique()==39
    harmonic_error(data, ids)
    names=['slowfast','transformer','tcn']
    curve_rows=[];pieces=[]
    for name in names:
        model,ck=frozen(name)
        curves=[]; diagnostics=[]
        for j in range(0,len(ids),1024):
            b=select(data,slice(j,j+1024))
            roll=model(b[0],b[1],b[-1])
            c,d=native_diagnostics(roll,b)
            curves.append(c);diagnostics.append(d)
        err=np.concatenate(curves); tab=pd.concat(diagnostics,ignore_index=True)
        pieces.append(tab.groupby(ids.flight_id.to_numpy()).mean().rename_axis('flight_id').reset_index().assign(model=name))
        for k in range(20):
            for m,metric in enumerate(['position_m','velocity_m_s','attitude_rad','rate_rad_s']):
                ok=err[:,k,m,-1]<=np.where(ids.sampling_group=='main_50hz',.035,.105)
                z=pd.DataFrame(err[ok,k,m,:3],columns=['axis0','axis1','axis2'])
                eq=z.groupby(ids.flight_id.to_numpy()[ok]).mean().mean()
                curve_rows.append(dict(model=name,horizon_s=(k+1)/10,metric=metric,rmse=np.sqrt(eq.sum()),**eq.to_dict()))
    pd.DataFrame(curve_rows).to_csv(OUT/'diagnostic_curve.csv',index=False)
    pd.concat(pieces).to_csv(OUT/'diagnostic_flights.csv',index=False)


if __name__=='__main__':
    main()
