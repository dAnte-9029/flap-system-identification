from common import *

def measures(g,weights=None):
 w=np.ones(len(g)) if weights is None else np.asarray(weights,float);w/=w.sum();avg=lambda x:float(np.dot(w,np.asarray(x)));out=dict(windows=len(g),flights=g.flight_id.nunique(),accepted_fraction=avg(g.accepted),history_all_C1_fraction=avg(g.history_fallback_rows==0),history_fallback_fraction=float(g.history_fallback_rows.sum()/g.history_rows.sum()),complete_B_fraction=avg(g.complete_B))
 for a in VARS:
  for c in ['C0','C1']:out[f'{c}_{a}_rms']=np.sqrt(avg(g[f'{c}_{a}_error']**2));out[f'{c}_{a}_bias']=avg(g[f'{c}_{a}_error'])
  out[f'{a}_improved_fraction']=avg(abs(g[f'C1_{a}_error'])<abs(g[f'C0_{a}_error']))
 for t in ['T0','T1']:out[f'{t}_attitude_rms_deg']=np.sqrt(avg(g[f'{t}_attitude_mse']))*180/np.pi;out[f'{t}_endpoint_rms_deg']=np.sqrt(avg(g[f'{t}_endpoint_rad']**2))*180/np.pi;out[f'{t}_first_jump_mean']=avg(g[f'{t}_first_jump_norm'])
 out['attitude_improved_fraction']=avg(g.T1_attitude_mse<g.T0_attitude_mse);out['position_rms_m']=np.sqrt(avg(g.position_mse));return out

d=pd.read_csv(OUT/'windows.csv.gz');cfg=json.loads((HERE/'config.json').read_text());summ=[];per=[];strata=[];cover=[]
for split,g in d.groupby('split'):
 for subset,x in [('common_prefix',g),('complete_H1000',g[g.complete_B])]:
  for weighting in ['window','flight']:
   w=np.ones(len(x)) if weighting=='window' else 1/x.groupby('flight_id').flight_id.transform('size').to_numpy();summ.append(dict(split=split,subset=subset,weighting=weighting,**measures(x,w)))
 for fid,x in g.groupby('flight_id'):per.append(dict(split=split,flight_id=fid,**measures(x)))
 for field in ['reason','latest_age_us','observation_change_norm']:
  bins=cfg['age_bins_us'] if field=='latest_age_us' else cfg['change_bins_radps'];group=g[field] if field=='reason' else pd.cut(g[field],bins,right=True,include_lowest=True)
  for key,x in g.groupby(group,observed=True):strata.append(dict(split=split,grouping=field,group=str(key),**measures(x)))
 for reason,x in g.groupby('reason'):cover.append(dict(split=split,reason=reason,windows=len(x),flights=x.flight_id.nunique(),fraction=len(x)/len(g)))
pd.DataFrame(summ).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(per).to_csv(OUT/'per_flight.csv',index=False);pd.DataFrame(strata).to_csv(OUT/'strata.csv',index=False);pd.DataFrame(cover).to_csv(OUT/'coverage.csv',index=False)
h=pd.read_parquet(OUT/'history_states.parquet');h.groupby(['split','reason']).size().rename('unique_history_rows').to_csv(OUT/'history_coverage.csv')
print(pd.DataFrame(summ)[['split','subset','weighting','C0_p_rms','C1_p_rms','T0_attitude_rms_deg','T1_attitude_rms_deg']].to_string(index=False))
print('Flights with lower C1 p RMS',pd.DataFrame(per).assign(better=lambda x:x.C1_p_rms<x.C0_p_rms).groupby('split').better.sum().to_dict())
