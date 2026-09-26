"""Exact weighted local-rate MSE decomposition; frozen arrays only."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
import run_model_flight_response_diagnostic as r
m=r.m;ROOT=r.ROOT;OUT=ROOT/'docs/analysis/results/local_rate_error_decomposition_v1'

def decompose(p,t,dt):
    if p.shape!=t.shape or p.shape[:-1]!=dt.shape or not np.all(dt>0):raise ValueError('invalid paired inputs')
    w=dt/dt.sum(-1,keepdims=True);w=w[...,None]
    pm=np.sum(w*p,axis=-2);tm=np.sum(w*t,axis=-2)
    pc=p-pm[...,None,:];tc=t-tm[...,None,:]
    pv=np.sum(w*pc**2,axis=-2);tv=np.sum(w*tc**2,axis=-2);cov=np.sum(w*pc*tc,axis=-2)
    ps=np.sqrt(pv);ts=np.sqrt(tv);denom=ps*ts
    corr=np.full(denom.shape,np.nan);np.divide(cov,denom,out=corr,where=denom>0)
    return dict(mse=np.sum(w*(p-t)**2,axis=-2),bias=pm-tm,bias_mse=(pm-tm)**2,
      amplitude_mse=(ps-ts)**2,shape_mse=2*(denom-cov),pred_var=pv,true_var=tv,corr=corr,endpoint_mse=(p[...,-1,:]-t[...,-1,:])**2)

def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    parent=ROOT/'docs/analysis/results/joint_command_local_reset_v1';old=m.read_json(parent/'protocol.json')
    pins=old['pins'];m.verify_pins(pins)
    groups=pd.read_csv(ROOT/'docs/analysis/results/joint_command_response_v1/origin_groups.csv');n=len(groups)
    b=torch.load(r.ART/'local_batch.pt',map_location='cpu',weights_only=False);truth=b.trajectory.truth.angular_velocity_b.reshape(10,n,6,3)
    dt=b.trajectory.dt_s.reshape(10,n,5)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),source=m.rel(parent/'protocol.json'),pins=pins,
      scope='Frozen local100ms predictions only; no training/inference, heldout or raw logs',
      identity='all10 offsets0..45; same parent groups, labels and actual command tapes',
      formula='For 5 future states, native-dt weights sum1. MSE = (mean(p)-mean(t))^2 + (std(p)-std(t))^2 + 2*(std(p)*std(t)-cov(p,t)). No lag search or filtering; initial zero-error t0 excluded.',
      terms='segment mean error, temporal spread mismatch, centered waveform mismatch; not mutually identifiable physical root causes. Shape includes timing/missed waveform, not proven phase delay.',
      aggregation='average offset MSE within parent for all_offsets; flight equal parents then root for RMSE; equal flights then seed mean/SD. Share uses equal-flight mean squared terms, not squared macro RMSE. Also retain individual offsets.',
      correlation='5 observations only; weighted descriptive Pearson, zero variance undefined and counted. Not statistical confidence or delay estimate.',
      interpretation='Local mean mismatch need not be globally correctable bias; smoothing/noise not inferred. No outputs used to fit corrections.'))
    rows=[];maxres=0.
    for seed in [17,23,42]:
      with np.load(r.ART/f'local_seed{seed}.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['window_ids'],b.trajectory.window_ids);pred=z['angular_velocity_b'].reshape(10,n,6,3)
      np.testing.assert_allclose(pred[:,:,0],truth[:,:,0],rtol=0,atol=1e-6)
      parts=decompose(pred[:,:,1:].astype(float),truth[:,:,1:].astype(float),dt)
      residual=np.max(np.abs(parts['mse']-parts['bias_mse']-parts['amplitude_mse']-parts['shape_mse']));maxres=max(maxres,float(residual));assert residual<1e-10
      for mode in ['Stabilized','Mission']:
       for pattern in ['ALL','sustained']:
        mask=(groups['mode']==mode).to_numpy()&((groups.command_pattern=='sustained').to_numpy() if pattern=='sustained' else np.ones(n,bool))
        for offset in [-1,*range(0,50,5)]:
         values={k:(v.mean(0) if offset<0 and k!='corr' else v[offset//5] if offset>=0 else v) for k,v in parts.items()}
         for flight in sorted(groups.loc[mask,'log_id'].unique()):
          take=mask&(groups.log_id==flight).to_numpy()
          for j,axis in enumerate('pqr'):
           row=dict(seed=seed,mode=mode,pattern=pattern,offset=offset,flight_id=flight,cohort=r.cohort(flight),axis=axis,n_origins=int(take.sum()))
           for k,v in values.items():
            if k=='corr':
             x=v[:,take,j].reshape(-1) if offset<0 else v[take,j];finite=np.isfinite(x)
             row['correlation']=float(x[finite].mean()) if finite.any() else np.nan;row['undefined_correlations']=int((~finite).sum())
            else:row[k]=float(v[take,j].mean())
           for k in ['mse','bias_mse','amplitude_mse','shape_mse','endpoint_mse']:row[k.replace('mse','rmse')]=float(np.sqrt(max(0,row[k])))
           rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_flight.csv',index=False)
    ex=pd.concat([frame,frame.assign(cohort='ALL')]);keys=['mode','pattern','offset','cohort','axis'];cols=['mse','bias','bias_mse','amplitude_mse','shape_mse','pred_var','true_var','correlation','rmse','bias_rmse','amplitude_rmse','shape_rmse','endpoint_rmse']
    per=ex.groupby(keys+['seed'])[cols].mean().reset_index();per.to_csv(OUT/'per_seed.csv',index=False)
    a=per.groupby(keys)[cols].agg(['mean','std']).reset_index();a.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in a.columns]
    for term in ['bias','amplitude','shape']:a[term+'_share_pct']=100*a[term+'_mse_mean']/a.mse_mean.replace(0,np.nan)
    a['centered_amplitude_ratio']=np.sqrt(a.pred_var_mean/a.true_var_mean.replace(0,np.nan));a.to_csv(OUT/'summary.csv',index=False)
    previous=pd.read_csv(parent/'per_flight.csv').query('condition=="local100ms" and metric in ["p_rad_s","q_rad_s","r_rad_s"]');previous['axis']=previous.metric.str[0];previous['offset']=previous.step-5
    match=frame[frame.offset>=0].merge(previous,on=['seed','mode','pattern','offset','flight_id','cohort','axis']);assert len(match)==len(previous)
    parity=float(np.max(np.abs(match.endpoint_rmse-match.value)));assert parity<1e-7
    m.verify_pins(pins);m.write_json(OUT/'checks.json',dict(exact_decomposition_max_residual=maxres,endpoint_parity_cells=len(match),endpoint_parity_max_abs=parity,heldout_accessed=False,training=0,new_predictions=0,source_hashes_unchanged=True))
    print(a.query('pattern=="sustained" and offset==-1 and cohort=="ALL"')[['mode','axis','rmse_mean','bias_mean','bias_share_pct','amplitude_share_pct','shape_share_pct','centered_amplitude_ratio','correlation_mean']].to_string(index=False))

if __name__=='__main__':main()
