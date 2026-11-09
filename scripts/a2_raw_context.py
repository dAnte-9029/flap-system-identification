"""Read only train-provenance raw logs; attach causal reference and airspeed candidates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
ROOT=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
OUT=Path('experiments/phase_anchored_coupled_latent_v1/control_identifiability')
def attach(df):
 meta={x['flight_id']:x for x in json.load(open(ROOT/'flight_provenance.json')) if x['split']=='train'}
 fields={'vehicle_rates_setpoint':['roll','pitch','yaw'],'vehicle_attitude_setpoint':['q_d[0]','q_d[1]','q_d[2]','q_d[3]'],'airspeed_validated':['calibrated_airspeed_m_s'],'vehicle_control_mode':['flag_control_auto_enabled','flag_control_manual_enabled','flag_control_rates_enabled','flag_control_attitude_enabled']}
 extras=np.zeros((len(df),24));coverage=[];cache={}
 for fid,g in df.groupby('flight',sort=False):
  path=meta[fid]['log_path']
  if path not in cache:
   cache={path:ULog(str(Path('/home/zn/QgcLogs')/path),message_name_filter_list=list(fields))}
  tabs={x.name:x.data for x in cache[path].data_list};col=0
  for topic,names in fields.items():
   tab=tabs.get(topic);okcount=0
   for name in names:
    if tab is not None and name in tab:
     t=g.t.to_numpy();ts=tab['timestamp'];j=np.searchsorted(ts,t,side='right')-1;safe=np.maximum(j,0);val=tab[name][safe];ok=(j>=0)&(t-ts[safe]<=500000)&np.isfinite(val)
     if topic=='airspeed_validated':ok &= val>0
     extras[g.index,col]=np.where(ok,val,0);extras[g.index,col+12]=ok;okcount+=int(ok.sum())
    col+=1
   coverage.append(dict(flight=fid,topic=topic,present=tab is not None,valid_values=okcount,possible_values=len(g)*len(names)))
 pd.DataFrame(coverage).to_csv(OUT/'raw_context_coverage.csv',index=False);np.save(OUT/'context.npy',extras);return extras
if __name__=='__main__':attach(pd.read_pickle(OUT/'features.pkl'))
