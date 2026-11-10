"""Train-only upstream-reference screening; no IV estimate absent valid source."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
ROOT=Path('/home/zn/flap-system-identification')
HERE=Path(__file__).resolve().parents[1]; OUT=HERE/'experiments/px4_identifiability_a_v1';OUT.mkdir(exist_ok=True)
SHARED=ROOT/'.worktrees/px4-identifiability-contract-v1/experiments/px4_identifiability_v1'
META=json.loads((SHARED/'train_flights.json').read_text()); assert all(m['split']=='train' for m in META)
SRC=ROOT/'.worktrees/pacld-control-identifiability-v1/experiments/phase_anchored_coupled_latent_v1/control_identifiability'
FILES={
 'src/modules/navigator/mission_block.cpp':['get_global_position()', 'get_distance_to_point_global_wgs84','is_mission_item_reached_or_completed'],
 'src/modules/navigator/mission_base.cpp':['is_mission_item_reached_or_completed','advance_mission','seq_current'],
 'src/modules/fw_mode_manager/FixedWingModeManager.cpp':['position_setpoint_triplet','global_position','local_position','navigateWaypoint','curr_pos_local'],
 'src/modules/fw_lateral_longitudinal_control/FwLateralLongitudinalControl.cpp':['navigate','roll','_local_pos','_wind'],
 'src/modules/fw_att_control/FixedwingAttitudeControl.cpp':['control_roll','_manual_control_setpoint.roll','_rates_sp.roll'],
 'src/modules/fw_att_control/fw_roll_controller.cpp':['roll_error','bodyrate_setpoint','euler_rate_setpoint'],
 'src/modules/fw_rate_control/FixedwingRateControl.cpp':['rate_error','_rate_control.update','vehicle_angular_velocity','_rates_sp'],
 'src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessControlSurfaces.cpp':['ROLL','roll','torque'],
 'msg/versioned/VehicleCommand.msg':['= 178 ','= 521 ','= 1000 ','= 1001 ','= 2003 '],
 'src/modules/control_allocator/ControlAllocator.cpp':['allocate()','actuator_servos','_vehicle_torque_setpoint_sub']}
def sources():
 rows=[]
 for sha in sorted(set(m['firmware'] for m in META)):
  for path,patterns in FILES.items():
   r=subprocess.run(['git','-C','/home/zn/PX4-Autopilot','show',sha+':'+path],capture_output=True,text=True)
   hits=[dict(line=i,text=s.strip()) for i,s in enumerate(r.stdout.splitlines(),1) if any(p in s for p in patterns)]
   rows.append(dict(firmware=sha,path=path,available=r.returncode==0,evidence=hits))
 (OUT/'source_evidence.json').write_text(json.dumps(rows,indent=2));return rows

def sample(tab,key,t,maxage=500000):
 if tab is None or key not in tab:return np.full(len(t),np.nan)
 ts=tab['timestamp'].astype(np.int64);j=np.searchsorted(ts,t,side='right')-1;s=np.maximum(j,0)
 v=np.asarray(tab[key])[s].astype(float);v[(j<0)|(t-ts[s]>maxage)]=np.nan;return v

def extract():
 f=pd.read_pickle(SRC/'features.pkl').reset_index(drop=True); assert len(f)==11221 and not f.duplicated(['flight','t']).any()
 md={m['flight_id']:m for m in META};assert set(f.flight)==set(md)
 names=['rate_roll_change','attitude_roll_change','manual_roll_change','waypoint_changed']
 Z=np.full((len(f),4),np.nan);past=np.full(len(f),np.nan);inventory=[];events=[];commands=[]
 topics=['vehicle_rates_setpoint','vehicle_attitude_setpoint','manual_control_setpoint','position_setpoint_triplet','mission_result','vehicle_command','vehicle_angular_velocity']
 for fid,g in f.groupby('flight',sort=False):
  m=md[fid];u=ULog(str(Path('/home/zn/QgcLogs')/m['log_path']),message_name_filter_list=topics);tabs={d.name:d.data for d in u.data_list if d.multi_id==0};t=g.t.to_numpy(np.int64);ix=g.index
  for name in topics:
   a=tabs.get(name);inventory.append(dict(date=m['flight_date'],topic=name,present=a is not None,n_reviewed=0 if a is None else int(np.sum((a['timestamp']>=m['start_us'])&(a['timestamp']<=m['end_us'])))))
  a=tabs.get('vehicle_rates_setpoint');Z[ix,0]=sample(a,'roll',t)-sample(a,'roll',t-100000)
  a=tabs.get('vehicle_attitude_setpoint')
  if a is not None:
   a=dict(a);w,x,y,z=[a[f'q_d[{i}]'] for i in range(4)];a['roll']=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));Z[ix,1]=sample(a,'roll',t)-sample(a,'roll',t-100000)
  a=tabs.get('manual_control_setpoint');Z[ix,2]=sample(a,'roll',t)-sample(a,'roll',t-100000)
  a=tabs.get('position_setpoint_triplet')
  if a is not None:
   v=np.stack([sample(a,k,t,10**12)-sample(a,k,t-100000,10**12) for k in ['current.lat','current.lon','current.alt']]);Z[ix,3]=np.where(np.isfinite(v).all(0),np.any(abs(v)>1e-8,axis=0).astype(float),np.nan)
   ts=a['timestamp'];coords=np.c_[a['current.lat'],a['current.lon'],a['current.alt']];changed=np.r_[False,np.any(abs(np.diff(coords,axis=0))>1e-8,axis=1)];events.append(dict(date=m['flight_date'],kind='waypoint',count=int(np.sum(changed&(ts>=m['start_us'])&(ts<=m['end_us'])))))
  a=tabs.get('vehicle_command')
  if a is not None:
   mask=(a['timestamp']>=m['start_us'])&(a['timestamp']<=m['end_us'])
   for k in np.where(mask)[0]:commands.append(dict(date=m['flight_date'],command=int(a['command'][k]),from_external=bool(a['from_external'][k]),param2=float(a['param2'][k]) if np.isfinite(a['param2'][k]) else None))
   for command in np.unique(a['command'][mask]):events.append(dict(date=m['flight_date'],kind='command_'+str(int(command)),count=int(np.sum(mask&(a['command']==command)))))
  a=tabs.get('vehicle_angular_velocity');past[ix]=sample(a,'xyz[0]',t-100000,50000)-sample(a,'xyz[0]',t-300000,50000)
  print('extract',len(inventory)//len(topics),flush=True)
 pd.DataFrame(inventory).groupby(['date','topic']).agg(logs=('present','size'),present=('present','sum'),n_reviewed=('n_reviewed','sum')).to_csv(OUT/'topic_by_date.csv')
 pd.DataFrame(events).groupby(['date','kind'])['count'].sum().to_csv(OUT/'events_by_date.csv')
 pd.DataFrame(commands).groupby(['date','command','from_external','param2'],dropna=False).size().rename('count').to_csv(OUT/'commands_by_date.csv')
 np.savez(OUT/'private_windows.npz',Z=Z,past=past)
 return f,Z,past,names

def screen(f,Z,past,names):
 X=np.stack(f.X);U=np.stack(f.U)[:,4:8]; dates=f.date.to_numpy();out=[]
 # Fixed linear diagnostic, alpha=10; date-held-out fits only. No tuning or IV assumptions.
 for j,name in enumerate(names):
  keep=np.isfinite(Z[:,j])&np.isfinite(past);xx=X[keep];z=Z[keep,j:j+1];u=U[keep];p=past[keep];ds=dates[keep]
  # Past-motion check is descriptive: X at t contains descendants of earlier motion.
  target=np.c_[u,p];base=np.full_like(target,np.nan);full=base.copy();zr=np.full(len(z),np.nan)
  for date in sorted(set(ds)):
   tr=ds!=date;te=~tr
   def fit(a,b):
    class Model:
     def __init__(self,a,b):
      self.mean=a.mean(0);self.scale=a.std(0);self.scale[self.scale<1e-12]=1;self.ym=b.mean(0);v=(a-self.mean)/self.scale;self.coef=np.linalg.solve(v.T@v+10*np.eye(v.shape[1]),v.T@(b-self.ym))
     def predict(self,a):return (a-self.mean)/self.scale@self.coef+self.ym
    return Model(a,b)
   base[te]=fit(xx[tr],target[tr]).predict(xx[te]);full[te]=fit(np.c_[xx[tr],z[tr]],target[tr]).predict(np.c_[xx[te],z[te]])
   zr[te]=(z[te]-fit(xx[tr],z[tr]).predict(xx[te])).ravel()
  for date in ['ALL']+sorted(set(ds)):
   q=np.ones(len(ds),bool) if date=='ALL' else ds==date
   for k,channel in enumerate(['motor','differential','symmetric','rudder','past_p_change']):
    e=target[q,k]-base[q,k];ef=target[q,k]-full[q,k];zz=zr[q];den=float(np.sum(zz**2));corr=float(np.corrcoef(zz,e)[0,1]) if np.std(zz)>1e-12 and np.std(e)>1e-12 else None
    out.append(dict(candidate=name,date=date,target=channel,n=int(q.sum()),nonzero=int(np.sum(abs(z[q,0])>1e-8)),z_std=float(z[q].std()),raw_corr=float(np.corrcoef(z[q,0],target[q,k])[0,1]) if np.std(z[q,0])>1e-12 and np.std(target[q,k])>1e-12 else None,partial_corr=corr,partial_slope=float(zz@e/den) if den>1e-16 else None,oof_mse_base=float(np.mean(e*e)),oof_mse_plus_z=float(np.mean(ef*ef)),relative_mse_reduction=float(1-np.mean(ef*ef)/np.mean(e*e))))
 pd.DataFrame(out).to_csv(OUT/'screening_by_date.csv',index=False)
 return out
if __name__=='__main__':
 s=sources();f,z,p,n=extract();r=screen(f,z,p,n)
 summary=dict(baseline='9c15fffd24bfcacc7c57cd980cc149098cb83f82',split='train',windows=len(f),flights=f.flight.nunique(),dates=f.date.nunique(),firmwares=len(set(m['firmware'] for m in META)),source_files=len(s),source_available=sum(x['available'] for x in s),decision='NO-GO for causal IV response estimation',reason='No candidate establishes externally scheduled independent excitation and exclusion; feedback reference and state-triggered waypoint timing fail source gate.',response_estimation_performed=False,all_metrics=[x for x in r if x['date']=='ALL'])
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))
