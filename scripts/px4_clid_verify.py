"""Independent checks of delivered artifacts and strict data/recursion boundaries."""
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from px4_clid_models import radius,pulse,rollout,blocks_for,stats,solve_stats,noise
from scipy.signal import lfilter
OUT=Path('experiments/px4_closed_loop_dynamic_id_v2')
def main():
 identities=json.loads((OUT/'local/identities.json').read_text());assert len(identities)==114 and all(x['split']=='train' for x in identities)
 ids={x['flight_id']:x for x in identities};summary={}
 for base in [OUT,OUT/'coarse100']:
  tc=json.loads((base/'time_contract.json').read_text());seq=pickle.load((base/'local/sequences.pkl').open('rb'))
  for s in seq:
   assert s['flight'] in ids and s['firmware']==ids[s['flight']]['firmware']
   assert np.all(np.diff(s['t'])==tc['dt_us']) and np.all(s['offset']<=0) and np.all(s['offset']>=-tc['match_tolerance_us'])
   assert np.all(s['t']+s['offset']>=ids[s['flight']]['start_us']) and np.all(s['t']<=ids[s['flight']]['end_us'])
   assert np.isfinite(s['y']).all() and np.isfinite(s['u']).all() and abs(s['u']).max()<=1.000001
  gradient_checks=[]
  models=json.loads((base/'models.json').read_text());assert len(models)==(15 if base==OUT else 2);checks=0
  for date,mm in models.items():
   for name,m in mm.items():
    m['theta']=np.array(m['theta']);m['c']=np.array(m['c']);assert radius(m['theta'],m['cfg'][0])<1
    assert len(m['c'])<=3 and (len(m['c'])==1 or max(abs(np.roots(m['c'])))<1)
    s=next(x for x in seq if x['date']==date and len(x['y'])>50)
    origin,pred=rollout(m,s);ss=dict(s,y=s['y'].copy());ss['y'][origin[0]+1:]+=100
    np.testing.assert_array_equal(pred,rollout(m,s)[1]);np.testing.assert_allclose(pred[0],rollout(m,ss)[1][0],atol=0,rtol=0)
    if 'profile_target' in m:assert abs(pulse(m)[m.get('profile_horizon_index',9),0]-m['profile_target'])<1e-5
    if name=='M1' or (base.name=='coarse100' and name.startswith('M1_')):
     blocks=blocks_for([ss for ss in seq if ss['date']!=date],m['cfg']);z=np.array(m['noise_reflection'])
     def prof(zz):
      cc=noise(zz);st=stats(blocks,cc);tt,pen=solve_stats(st,m['cfg'][3]);return (sum(np.sum(lfilter([1.],cc,Y-X@tt,axis=0)**2) for X,Y in blocks)+np.sum(pen[:,None]*tt**2))/st[3]
     scale=prof(np.zeros_like(z));gg=[]
     for j in range(len(z)):
      zp=z.copy();zm=z.copy();zp[j]+=1e-4;zm[j]-=1e-4;g=(prof(zp)-prof(zm))/2e-4/scale
      if z[j]>=.95-1e-6:g=max(g,0)
      if z[j]<=-.95+1e-6:g=min(g,0)
      gg.append(abs(g))
     gradient_checks.append(max(gg));assert max(gg)<1e-4,(date,name,gg)
    checks+=1
  f=pd.read_csv(base/'date_metrics.csv');np.testing.assert_allclose(f.rmse,np.sqrt(f.sse/f.n));np.testing.assert_allclose(f.bias,f.error_sum/f.n)
  summary[base.name]=dict(sequence_rows=sum(len(x['y']) for x in seq),sequences=len(seq),retained_flights=len({x['flight'] for x in seq}),retained_dates=len({x['date'] for x in seq}),retained_firmware=len({x['firmware'] for x in seq}),evaluated_dates=len(models),model_checks=checks,max_relative_profile_gradient=max(gradient_checks))
 summary['passed']=True;summary['holdout_data_access']='none: explicit train parquet and enumerated train ULogs only';(OUT/'verification.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
