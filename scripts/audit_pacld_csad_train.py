"""Read-only exact sampled train coverage and existing A2 excitation audit."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
import torch
from pacld_csad_v1 import OUT,TRAIN,PREV
from pacld_cir_v1 import load_model,audit_identity
from system_identification.data.pacld import open_windows
W=None

def worker(payload):
    global W
    cfg,indices=payload
    if W is None:
        torch.set_num_threads(1);W=open_windows(cfg,'train')
    rows=[]
    for ix in indices:
        w=W[int(ix)];r=W.index.iloc[ix];t=np.array(w.controls.times);u=w.controls.values.numpy();pre=u[np.searchsorted(t,0,side='left')-1];d0=(pre[2]-pre[1])/2
        v={}
        for end in [.05,.1]:
            knots=np.r_[0,t[(t>0)&(t<end)],end];uv=u[np.searchsorted(t,knots[:-1],side='right')-1];d=(uv[:,2]-uv[:,1])/2
            v['d'+str(int(end*1000))]=float(np.dot(np.diff(knots),d-d0)/end)
        events=t[(t>=0)&(t<=.2)];v['control_events_200ms']=len(events)
        differential=(u[:,2]-u[:,1])/2
        jumps=np.diff(differential);jt=t[1:];valid=(jt>=0)&(jt<=.2)&(np.abs(jumps)>1e-12)
        times_changed=jt[valid]
        v['differential_events_200ms']=len(times_changed)
        v['differential_total_variation_200ms']=float(np.abs(jumps[valid]).sum())
        v['first_differential_change_s']=float(times_changed[0]) if len(times_changed) else np.nan
        v['native_gyro_after_first_change_200ms']=sum(e.source=='vehicle_angular_velocity' and float(e.mask[9]*e.quality[9])>0 and times_changed[0]<e.measurement<=.2 for e in w.targets) if len(times_changed) else 0
        for low,high in [(0,.05),(.05,.1),(.1,.15),(.15,.2)]:
            v[f'diff_events_{int(low*1000)}_{int(high*1000)}']=int(((times_changed>=low)&(times_changed<high)).sum())
        for low,high in [(0,.05),(.05,.1),(.1,.15),(.15,.2)]:
            es=[e for e in w.targets if low<e.measurement<=high and e.source=='vehicle_angular_velocity' and float(e.mask[9]*e.quality[9])>0]
            v[f'gyro_{int(low*1000)}_{int(high*1000)}']=len(es)
        rows.append(dict(index=ix,window_id=r.window_id,flight=r.flight_id,t=r.decision_us,horizon_us=r.horizon_us,actual_horizon_s=w.metadata['actual_horizon_s'],**v))
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    _,bundle=load_model('cpu');w=open_windows(bundle['config'],'train');ident=audit_identity(w,bundle,'train')
    log=Path('/home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1/runs/pacld_gpu_b32_seed17_10944windows/training.jsonl')
    records=[json.loads(s) for s in log.read_text().splitlines()]
    assert len(records)==342 and records[-1]['windows_seen']==10944
    draws=[(r['optimizer_steps'],j,int(i)) for r in records for j,i in enumerate(r['global_window_indices'])];assert len(draws)==10944
    sampling=pd.DataFrame(draws,columns=['step','slot','index']);sampling['window_id']=w.index.iloc[sampling['index'].to_numpy()].window_id.to_numpy();sampling.to_csv(OUT/'actual_training_draws.csv',index=False)
    unique=w.index.loc[sorted(sampling['index'].unique())].copy();payloads=[(bundle['config'],g.index.tolist()) for _,g in unique.groupby('flight_id')]
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(worker,payloads):
            rows+=result;print('train sampled coverage',len(rows),'/',len(unique),flush=True)
    result=pd.DataFrame(rows)
    train=pd.read_pickle(TRAIN/'features.pkl');u=np.stack(train.U);res=np.load(PREV/'predictions.npz')['residual']
    excit=train[['flight','date','mode','t']].copy();excit['d50']=u[:,1];excit['d100']=u[:,5];excit['residual100']=res[:,5]
    excit.to_csv(OUT/'train_excitation.csv',index=False)
    dates=train.groupby('flight').date.first();result['date']=result.flight.map(dates)
    threshold=float(np.quantile(abs(u[:,5]),.75));result['rich_train_q75']=abs(result.d100)>=threshold
    result.to_csv(OUT/'sampled_train_coverage.csv',index=False)
    for group in ['flight','date','mode']:
        agg=excit.groupby(group).agg(n=('t','size'),d100_rms=('d100',lambda x:np.sqrt(np.mean(x*x))),residual_rms=('residual100',lambda x:np.sqrt(np.mean(x*x))),residual_energy=('residual100',lambda x:np.sum(x*x)))
        agg.to_csv(OUT/f'excitation_by_{group}.csv')
    x=np.stack(train.X);corr=pd.DataFrame({'feature':np.arange(x.shape[1]),'corr_raw_d100':[np.corrcoef(u[:,5],x[:,j])[0,1] if x[:,j].std()>0 else None for j in range(x.shape[1])],'corr_residual_d100':[np.corrcoef(res[:,5],x[:,j])[0,1] if x[:,j].std()>0 else None for j in range(x.shape[1])]})
    corr.to_csv(OUT/'train_state_control_correlations.csv',index=False)
    counted=sampling.merge(result,on=['index','window_id']);counted.groupby('date').agg(draws=('index','size'),rich_draws=('rich_train_q75','sum')).to_csv(OUT/'sampled_date_coverage.csv')
    flight_energy=excit.groupby('flight').residual100.apply(lambda x:sum(x*x)).sort_values(ascending=False)
    summary=dict(identity=ident,steps=len(records),draws=len(draws),unique_windows=len(result),unique_decisions=len(result.drop_duplicates(['flight','t'])),flights=result.flight.nunique(),dates=result.date.nunique(),rich_cutoff_abs_d100_train_q75=threshold,rich_draws=int(counted.rich_train_q75.sum()),rich_unique_windows=int(result.rich_train_q75.sum()),top5_flight_residual_energy_share=float(flight_energy.head(5).sum()/flight_energy.sum()),raw_d100_rms=float(np.sqrt(np.mean(u[:,5]**2))),residual_d100_rms=float(np.sqrt(np.mean(res[:,5]**2))),lambda_macro=bundle['lambda_macro'],macro_valid_draws=sum(r['macro_valid_windows'] for r in records),gyro_time_bin_coverage={c:dict(windows_with_target=int((result[c]>0).sum()),total_targets=int(result[c].sum())) for c in result if c.startswith('gyro_')})
    (OUT/'training_audit.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':main()
