"""Causal phase propagation diagnostic, not a six-DoF dynamics experiment.

Fit df/dt=b0+bu*u+bf*f only on train, flight-balanced. Compare with held f.
Future controls are factual ZOH replay, never future frequency or phase inputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import solve
from audit_phase import ROOT, OUT


def load(split):
    return {t:dict(tuple(pd.read_parquet(ROOT/f'native_{t}_{split}.parquet').groupby('flight_id',sort=False)))
            for t in ['encoder_count','flap_frequency','actuator_motors']}


def propagate(f, phi, u, dt, coef):
    b0,bu,bf=coef
    c=b0+bu*u
    if abs(bf)<1e-9: return f+c*dt,phi+2*np.pi*(f*dt+.5*c*dt*dt)
    steady=-c/bf; e=np.exp(bf*dt)
    return steady+(f-steady)*e,phi+2*np.pi*(steady*dt+(f-steady)*np.expm1(bf*dt)/bf)


def run():
    meta={m['flight_id']:m for m in json.loads((ROOT/'flight_provenance.json').read_text()) if m['split'] in ['train','validation']}
    train=load('train');xx=np.zeros((3,3));xy=np.zeros(3);fit_counts={}
    for fid,m in meta.items():
        if m['split']!='train' or not m['firmware']:continue
        f=train['flap_frequency'][fid];u=train['actuator_motors'][fid]
        t=f.timestamp_us.to_numpy();v=f.frequency_hz.to_numpy();dt=np.diff(t)*1e-6
        j=np.searchsorted(u.timestamp_us.to_numpy(),t[:-1],side='right')-1
        uv=u['control[0]'].to_numpy()[np.maximum(0,j)]
        good=(j>=0)&(dt>0)&(dt<=.03)&(v[:-1]>1)&(v[1:]>1)&np.isfinite(uv)&f.in_reviewed_flight.to_numpy()[:-1]&f.in_reviewed_flight.to_numpy()[1:]
        x=np.c_[np.ones(good.sum()),uv[good],v[:-1][good]];y=np.diff(v)[good]/dt[good]
        if len(y)<10:continue
        xx+=x.T@x/len(y);xy+=x.T@y/len(y);fit_counts[fid]=len(y)
    coef=solve(xx+np.eye(3)*1e-6,xy,assume_a='pos')
    (OUT/'frequency_fit.json').write_text(json.dumps(dict(equation='df/dt=b0+bu*u_motor+bf*f',coefficients=coef.tolist(),fit_rows=fit_counts,method='flight-equal derivative least squares, train only; no selection',horizons_s=[.2,.5,1,2]),indent=2))
    rows=[]
    for split in ['train','validation']:
        tables=train if split=='train' else load(split)
        for fid,m in meta.items():
            if m['split']!=split or not m['firmware']:continue
            e=tables['encoder_count'][fid];f=tables['flap_frequency'][fid];u=tables['actuator_motors'][fid]
            et=e.timestamp_us.to_numpy();ec=e.total_count.to_numpy();av=e.availability_us.to_numpy()
            f=f[(f.availability_us>=0)&np.isfinite(f.frequency_hz)].sort_values('availability_us')
            ft=f.availability_us.to_numpy();fv=f.frequency_hz.to_numpy();ut=u.timestamp_us.to_numpy();uv=u['control[0]'].to_numpy()
            if not len(ft):continue
            scale=2*np.pi/(4096*m['transmission_ratio']);next_start=m['start_us']
            for k in range(len(et)):
                t0=int(av[k])
                if t0<next_start or t0+2000000>m['end_us'] or t0<et[k] or t0-et[k]>30000:continue
                next_start=t0+2500000
                j=np.searchsorted(ft,t0,side='right')-1
                if j<0 or fv[j]<=1 or t0-ft[j]>30000:continue
                k2=np.searchsorted(et,t0+2000000)
                if k2>=len(et) or np.any(np.diff(et[k:k2+1])>30000):continue
                initf=float(fv[j]);initphi=2*np.pi*initf*(t0-et[k])*1e-6
                for model,c in [('held_frequency',np.zeros(3)),('motor_first_order',coef)]:
                    cur=t0;fp=initf;pp=initphi
                    for h in [.2,.5,1,2]:
                        target=t0+round(h*1e6)
                        endpoints=np.r_[ut[(ut>cur)&(ut<target)],target]
                        failed=False
                        for end in endpoints:
                            ui=np.searchsorted(ut,cur,side='right')-1
                            if ui<0 or not np.isfinite(uv[ui]):failed=True;break
                            fp,pp=propagate(fp,pp,float(uv[ui]),(end-cur)*1e-6,c);cur=int(end)
                        if failed:break
                        # Offline target interpolation only; bracket gaps checked above.
                        truth=(np.interp(target,et,ec)-ec[k])*scale
                        err=pp-truth
                        rows.append(dict(flight_id=fid,split=split,date=m['flight_date'],model=model,t0_us=t0,horizon_s=h,
                                         phase_error_rad=err,circular_error_rad=float(np.angle(np.exp(1j*err))),predicted_frequency_hz=fp))
        print('forecasted',split,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'phase_forecast_windows.csv.gz',index=False)
    agg=[]
    for (s,model,h),g in df.groupby(['split','model','horizon_s']):
        flight=g.assign(sq=g.circular_error_rad**2,usq=g.phase_error_rad**2).groupby('flight_id')[['sq','usq']].mean()
        agg.append(dict(split=s,model=model,horizon_s=h,flights=len(flight),windows=len(g),
                        circular_rmse_deg=float(np.rad2deg(np.sqrt(flight.sq.mean()))),
                        unwrapped_rmse_deg=float(np.rad2deg(np.sqrt(flight.usq.mean())))))
    pd.DataFrame(agg).to_csv(OUT/'phase_forecast_summary.csv',index=False)
    print(coef);print(pd.DataFrame(agg).to_string(index=False))


if __name__=='__main__':run()
