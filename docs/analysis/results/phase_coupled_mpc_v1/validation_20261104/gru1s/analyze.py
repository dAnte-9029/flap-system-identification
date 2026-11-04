"""Common-support frozen model comparison: same states, labels, origins and reductions."""
import json,math,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
out=Path(__file__).parent;idx=json.loads((out/'index.json').read_text());prep=json.loads((out/'preparation.json').read_text());old=Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1')
rows=[];offsets=[];regression=[];normal=[]
for item in idx:
 z=np.load(out/f"inputs_{item['i']:03d}.npz");offsets.extend((z['rate_times']-z['times']).tolist());normal.append(dict(i=item['i'],window_id=item['window_id'],flight_id=item['flight_id'],horizon_us=item['horizon_us'],sampling_group='main_50hz',anchor_measurement=(item['origin_rate_time_us']-item['decision_us'])*1e-6))
def error(pred,truth):
 qr=Rotation.from_quat(pred[:,[7,8,9,6]]);qt=Rotation.from_quat(truth[:,[7,8,9,6]])
 return dict(p=np.sum((pred[:,:3]-truth[:,:3])**2,axis=1),v=np.sum((pred[:,3:6]-truth[:,3:6])**2,axis=1),R=(qr.inv()*qt).magnitude()**2,omega=np.sum((pred[:,10:13]-truth[:,10:13])**2,axis=1))
for seed in [17,29]:
 legacy=pd.read_parquet(old/f'experiments/velocity_rate_gru64_horizon_training_v1/metrics/H1000-GRU-Control_seed_{seed}_endpoints.parquet')
 for model in ['A','B','C']:
  name=f'{model}_seed{seed}';dest=out/f'direction-{name}';dest.mkdir(exist_ok=False);verify=json.loads((out/name/'verification.json').read_text());(dest/'verification.json').write_text(json.dumps(verify));(dest/'index.json').write_text(json.dumps(normal))
  for item in idx:
   i=item['i'];z=np.load(out/f'inputs_{i:03d}.npz');pred=np.load(out/name/f'window_{i:03d}.npz')['pred'];truth=z['target'];t=z['times'];er=error(pred,truth)
   np.savez_compressed(dest/f'window_{i:03d}.npz',times=t,truth=truth[:,10:13],pred=pred[:,10:13],origin=z['origin_rate'])
   for h in [.02,.2,.5,1.,2.]:
    j=int(np.argmin(abs(t-h)))
    if abs(t[j]-h)>.03:continue
    for c in ['p','v','R','omega']:rows.append(dict(seed=seed,model=model,flight_id=item['flight_id'],window_id=item['window_id'],horizon_s=h,actual_time_s=t[j],component=c,squared_error=er[c][j]))
   if model=='C' and item['horizon_us']==2000000:
    records=legacy[legacy.window_id==item['window_id']]
    for r in records.itertuples():
     j=int(r.native_step)-1
     for c,col,scale in [('p','position_m',1),('v','velocity_m_s',1),('R','attitude_deg',180/np.pi),('omega','rate_rad_s',1)]:
      current=math.sqrt(er[c][j])*scale;previous=float(getattr(r,col));regression.append(dict(seed=seed,window_id=item['window_id'],horizon_ms=r.horizon_ms,component=c,abs_difference=abs(current-previous),current=current,archived=previous,within_initial_1e4_tolerance=bool(np.isclose(current,previous,atol=1e-4,rtol=1e-4))))
df=pd.DataFrame(rows);df.to_csv(out/'error_windows.csv',index=False);pf=df.groupby(['seed','model','flight_id','horizon_s','component']).squared_error.mean().reset_index();pf.to_csv(out/'error_per_flight.csv',index=False)
ps=pf.groupby(['seed','model','horizon_s','component']).squared_error.mean().reset_index();ps['rmse']=np.sqrt(ps.squared_error);ps.to_csv(out/'error_per_seed.csv',index=False)
mean=ps.groupby(['model','horizon_s','component']).squared_error.mean().reset_index();mean['rmse']=np.sqrt(mean.squared_error);mean.to_csv(out/'error_summary.csv',index=False)
pd.DataFrame(regression).to_csv(out/'gru_archived_regression.csv',index=False)
assert len(regression)>0
(out/'comparison_checks.json').write_text(json.dumps(dict(windows=108,flights=27,seeds=[17,29],models=['A','B','H1000-GRU-Control'],shared_evaluation='same original native VLP state grid and same nearest paired sensor labels; no interpolation; original PAHBD integration queries unchanged',rate_target_offset_seconds=np.quantile(offsets,[0,.5,1]).tolist(),gru_archived_regression_count=len(regression),gru_archived_max_abs_difference=max(r['abs_difference'] for r in regression),error_metric='sqrt(mean_seed(mean_flight(mean_window(vector_error_squared))))'),indent=2))
print(mean.pivot(index=['horizon_s','component'],columns='model',values='rmse').to_string())
