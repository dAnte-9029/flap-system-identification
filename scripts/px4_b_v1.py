"""Train-only layered controller observability, no plant fitting or neural training."""
import json,subprocess,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
ROOT=Path('/home/zn/flap-system-identification'); SH=ROOT/'.worktrees/px4-identifiability-contract-v1/experiments/px4_identifiability_v1'; OUT=Path('experiments/px4_b_v1');OUT.mkdir(exist_ok=True)
meta=json.loads((SH/'train_flights.json').read_text());assert all(m['split']=='train' for m in meta)
paths=['src/modules/fw_rate_control/FixedwingRateControl.cpp','src/modules/fw_att_control/FixedwingAttitudeControl.cpp','src/modules/fw_att_control/fw_roll_controller.cpp','src/lib/rate_control/rate_control.cpp','src/lib/rate_control/gain_compression.cpp','src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessControlSurfaces.cpp','src/lib/control_allocation/control_allocation/ControlAllocationPseudoInverse.cpp','src/modules/control_allocator/ControlAllocator.cpp']
sources=[]
for sha in sorted(set(m['firmware'] for m in meta)):
 for path in paths:
  src=subprocess.check_output(['git','-C','/home/zn/PX4-Autopilot','show',sha+':'+path]).decode(); lines=src.splitlines()
  sources.append(dict(sha=sha,path=path,content_sha256=hashlib.sha256(src.encode()).hexdigest(),anchors=[dict(line=i+1,text=s.strip()) for i,s in enumerate(lines) if any(k in s for k in ['DT_MIN','DT_MAX','rate_error =','const Vector3f torque =','control_roll(','_rate_control.update','_gain_compression','_airspeed_filter','normalizeControlAllocationMatrix','_actuator_sp =','_roll_ctrl.control','_body_rate_setpoint =','flap_aug','flap_b2b','bumpless'])]))
(OUT/'source_evidence.json').write_text(json.dumps(sources,indent=2))
topics=['rate_ctrl_terms','rate_ctrl_status','vehicle_torque_setpoint','vehicle_angular_velocity','actuator_servos','airspeed_validated','vehicle_rates_setpoint','gain_compression','vehicle_control_mode','actuator_outputs','control_allocator_status']
records=[];coverage=[];configs=[]
def align(tab,t,fields,maxage=50000,clock='timestamp'):
 ts=tab[clock];j=np.searchsorted(ts,t,side='right')-1;k=np.maximum(j,0);v=np.column_stack([tab[f][k] for f in fields]);ok=(j>=0)&(t-ts[k]<=maxage)&np.isfinite(v).all(1);return v,ok
for no,m in enumerate(meta):
 u=ULog('/home/zn/QgcLogs/'+m['log_path'],message_name_filter_list=topics);tabs={x.name:x.data for x in u.data_list if x.multi_id==0};p=u.initial_parameters
 configs.append(dict(date=m['flight_date'],firmware=m['firmware'],**{k:v for k,v in p.items() if k.startswith(('FFP','FLAP_SLOW','FLAP_B2B','FW_RR_','IMU_GYRO','FW_GC','FW_USE_AIRSPD','FW_ARSP','CA_METHOD','CA_SV_CS','TRIM_ROLL','FW_DTRIM_R'))}))
 # Restrict diagnostic support to intervals covered by the established eligible common cohort.
 f=pd.read_pickle(ROOT/'.worktrees/pacld-control-identifiability-v1/experiments/phase_anchored_coupled_latent_v1/control_identifiability/features.pkl') if no==0 else f
 decisions=np.sort(f.loc[f.flight==m['flight_id'],'t'].to_numpy())
 def eligible(t):
  j=np.searchsorted(decisions,t,side='right')-1;k=np.maximum(j,0);return (j>=0)&(t<=decisions[k]+200000)
 for topic in topics:
  tab=tabs.get(topic);ts=tab['timestamp'] if tab is not None else np.array([]);ts=ts[eligible(ts)]
  coverage.append(dict(date=m['flight_date'],topic=topic,present=tab is not None,n=len(ts),median_dt_us=float(np.median(np.diff(ts)[np.diff(ts)<200000])) if len(ts)>1 and np.any(np.diff(ts)<200000) else None))
 def metric(name,y,yhat,ok):
  ok=ok&np.isfinite(y)&np.isfinite(yhat)
  if ok.sum():
   a=y[ok];b=yhat[ok];records.append(dict(date=m['flight_date'],firmware=m['firmware'],metric=name,n=len(a),sse=float(np.sum((a-b)**2)),sy=float(a.sum()),sy2=float(a@a),mae=float(np.mean(abs(a-b))),max_error=float(np.max(abs(a-b)))))
 if all(t in tabs for t in ['rate_ctrl_terms','vehicle_angular_velocity','vehicle_rates_setpoint']):
  z=tabs['rate_ctrl_terms'];t=z['timestamp'];v,ok=align(tabs['vehicle_angular_velocity'],z['timestamp_sample'],['xyz[0]','xyz_derivative[0]'],clock='timestamp_sample');r,okr=align(tabs['vehicle_rates_setpoint'],t,['roll']);ok=ok&okr&eligible(t)
  metric('P_from_reference_rate',z['p_term[0]'],p['FW_RR_P']*(r[:,0]-v[:,0]),ok)
  metric('D_from_logged_derivative',z['d_term[0]'],-p['FW_RR_D']*v[:,1],ok)
  metric('PID_sum_identity',z['output[0]'],sum(z[k+'[0]'] for k in ['p_term','i_term','d_term','ff_term']),eligible(t))
  if 'gain_compression' in tabs and 'vehicle_torque_setpoint' in tabs:
   gc,og=align(tabs['gain_compression'],t,['compression_gains[0]'],maxage=200000)
   air,oa=align(tabs['airspeed_validated'],t,['calibrated_airspeed_m_s'],maxage=1000000)
   speed=np.maximum(air[:,0],p['FW_AIRSPD_STALL']);scale=p['FW_AIRSPD_TRIM']/speed if p.get('FW_ARSP_SCALE_EN') and p.get('FW_USE_AIRSPD') else np.ones(len(t))
   # Explicitly approximate: instantaneous logged airspeed replaces unlogged 1 s filter state.
   torque,ot=align(tabs['vehicle_torque_setpoint'],t+2000,['xyz[0]'],maxage=5000)
   pred=np.clip(gc[:,0]*z['output[0]']*scale**2+p.get('TRIM_ROLL',0)*scale**2,-1,1)
   metric('approx_PID_to_torque_oracle_terms',torque[:,0],pred,eligible(t)&og&oa&ot)
 if all(k in tabs for k in ['actuator_servos','vehicle_torque_setpoint']):
  z=tabs['actuator_servos'];t=z['timestamp'];tor,ok=align(tabs['vehicle_torque_setpoint'],t,[f'xyz[{i}]' for i in range(3)],maxage=40000)
  E=np.array([[p[f'CA_SV_CS{i}_TRQ_{ax}'] for i in range(3)] for ax in ['R','P','Y']]);mix=np.linalg.pinv(E);pred=np.clip(tor@mix.T+np.array([p.get(f'CA_SV_CS{i}_TRIM',0) for i in range(3)]),-1,1)
  actual=(z['control[1]']-z['control[0]'])/2
  metric('allocation_differential',actual,(pred[:,1]-pred[:,0])/2,eligible(t)&ok)
 print(no+1,m['flight_date'],flush=True)
pd.DataFrame(configs).to_csv(OUT/'controller_configurations.csv',index=False)
r=pd.DataFrame(records);r.to_csv(OUT/'flight_metrics.csv',index=False);c=pd.DataFrame(coverage);c.to_csv(OUT/'topic_coverage.csv',index=False)
agg=[]
for keys,g in r.groupby(['metric','date']):
 n=int(g.n.sum());sse=g.sse.sum();var=g.sy2.sum()-g.sy.sum()**2/n
 agg.append(dict(metric=keys[0],date=keys[1],n=n,rmse=float(np.sqrt(sse/n)),r2=float(1-sse/var) if var>0 else None,mae=float(np.average(g.mae,weights=g.n))))
pd.DataFrame(agg).to_csv(OUT/'date_metrics.csv',index=False)
summ=[]
for k,g in r.groupby('metric'):
 n=int(g.n.sum());sse=g.sse.sum();var=g.sy2.sum()-g.sy.sum()**2/n
 summ.append(dict(metric=k,n=n,flights=len(g),dates=g.date.nunique(),rmse=float(np.sqrt(sse/n)),r2=float(1-sse/var) if var>0 else None))
(OUT/'summary.json').write_text(json.dumps(dict(split='train',cohort_windows=len(f),flights=len(meta),metrics=summ,plant_identification='NO-GO: controller likelihood does not identify plant when innovation model/exogeneity remain unconstrained',parameter_change_logs=sum(bool(m['parameter_changes']) for m in meta)),indent=2))
print(json.dumps(summ,indent=2))
