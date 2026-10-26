"""Read allowlisted development files and required topics only; no data reconstruction."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
from pyulog import ULog
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SCOPE=HERE.parent/'local_control_response_scope_v1'
spec=importlib.util.spec_from_file_location('scope',SCOPE/'analyze.py');scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
old=scope.old
CFG=json.loads((HERE/'config.json').read_text())
RES=HERE/'results'
COORDS=scope.COORDS
TOPICS={
 'position_setpoint_triplet':['current.lat','current.lon','current.alt','current.valid','current.type','previous.lat','previous.lon'],
 'vehicle_global_position':['lat','lon','alt'],
 'vehicle_local_position_setpoint':['x','y','z','vx','vy','vz'],
 'fixed_wing_lateral_setpoint':['course','airspeed_direction','lateral_acceleration'],
 'fixed_wing_longitudinal_setpoint':['altitude','height_rate','equivalent_airspeed','pitch_direct','throttle_direct'],
 'vehicle_attitude_setpoint':['q_d[0]','q_d[1]','q_d[2]','q_d[3]'],
 'vehicle_rates_setpoint':['roll','pitch','yaw'],
 'rate_ctrl_terms':[f'{k}[{i}]' for k in ['p_term','i_term','d_term','ff_term','output'] for i in range(3)],
 'rate_ctrl_status':['rollspeed_integ','pitchspeed_integ','yawspeed_integ'],
 'vehicle_torque_setpoint':['xyz[0]','xyz[1]','xyz[2]'],
 'vehicle_thrust_setpoint':['xyz[0]','xyz[1]','xyz[2]'],
 'control_allocator_status':['torque_setpoint_achieved','thrust_setpoint_achieved'],
 'vehicle_angular_velocity':['xyz[0]','xyz[1]','xyz[2]'],
}
def write_json(p,obj):
 p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def asof(t,x,at,max_age=None):
 ix=np.searchsorted(t,at,side='right')-1
 if not len(t):return np.full((*np.shape(at),x.shape[-1]),np.nan),np.full(np.shape(at),np.nan)
 v=x[np.clip(ix,0,len(t)-1)].astype(float).copy();age=at-t[np.clip(ix,0,len(t)-1)]
 bad=ix<0
 if max_age is not None:bad|=age>max_age
 v[bad]=np.nan
 return v,np.where(ix>=0,age,np.nan)
def interp(t,x,at):
 """Offline only; explicitly no extrapolation and no bridges >50ms."""
 ix=np.searchsorted(t,at,side='left');cl=ix.clip(1,len(t)-1)
 ok=(at>=t[0])&(at<=t[-1])&((t[cl]-t[cl-1])<=50000)
 y=np.stack([np.interp(at,t,x[:,k]) for k in range(x.shape[1])],axis=-1)
 y[~ok]=np.nan
 return y

def main():
 start=time.time();RES.mkdir(exist_ok=True)
 data=Path(CFG['data_root']);rawroot=Path(CFG['raw_root'])
 reg=yaml.safe_load((ROOT/'configs/data/velocity_rate_dataset_registry.yaml').read_text())['datasets'][CFG['dataset_id']]
 assert old.sha(data/'dataset_manifest.json')==reg['manifest_sha256']
 manifest=json.loads((data/'dataset_manifest.json').read_text());hashes={}
 for n in ['flights.csv','flight_provenance.json','split_manifest.json','build_config.yaml']+[f'{s}_{sp}.parquet' for sp in CFG['partitions'] for s in ['samples','windows','native_actuator_motors','native_actuator_servos']]:
  h=old.sha(data/n);assert h==manifest['artifact_sha256'][n],n;hashes[str(data/n)]=h
 flights=pd.read_csv(data/'flights.csv'); sealed=set(flights.loc[flights.split=='test','sha256'])
 flights=flights[flights.split.isin(CFG['partitions'])&(flights.sampling_group=='main_50hz')].copy()
 assert set(flights.sha256).isdisjoint(sealed)
 assert flights.groupby('sha256').split.nunique().max()==1
 coverage=[];access=[];schemas=[];features=[];metas=[];targets=[];motions=[];controls=[];rows_audit=[];native_records=[];refgrids=[]
 grid=np.arange(-500000,500001,20000,dtype=np.int64);lags=np.array(CFG['history_lags_ms'])*1000
 for split in CFG['partitions']:
  old.permitted(split)
  c=pd.read_csv(SCOPE/f'results/{split}/candidate_index.csv.gz');c=c[c.AUTO_stable].copy();c['nonoverlap']=False
  c.loc[scope.nonoverlap_windows(c),'nonoverlap']=True
  ev=pd.read_csv(SCOPE/f'results/{split}/command_events.csv.gz').rename(columns={'i':'candidate_id'})
  c=c.merge(ev,on='candidate_id',validate='one_to_one')
  cols=list(dict.fromkeys(old.META+old.V+old.Q+old.W+old.U+['rate_measurement_us','relative_flap_phase_sin','relative_flap_phase_cos']))
  s=old.load_parquet(data,'samples',split,cols);m=old.load_parquet(data,'native_actuator_motors',split);v=old.load_parquet(data,'native_actuator_servos',split)
  for row in flights[flights.split==split].itertuples():
   fid=row.flight_id;cg=c[c.flight_id==fid].copy();p=rawroot/row.log_path
   assert row.sha256 not in sealed
   assert old.sha(p)==row.sha256
   ulog=ULog(str(p),message_name_filter_list=list(TOPICS))
   access.append(dict(flight_id=fid,split=split,path=str(p),sha256=row.sha256,changed_parameters=len(ulog.changed_parameters),topics=list(TOPICS)))
   topic={x.name:x.data for x in ulog.data_list if x.multi_id==0}
   decisions=cg.decision_us.to_numpy(np.int64);at=decisions[:,None]-lags
   ref={};internal={};mission={}
   for name,fields in TOPICS.items():
    d=topic.get(name,{});t=np.asarray(d.get('timestamp',[]),dtype=np.int64)
    assert np.all(np.diff(t)>=0),name
    x=np.column_stack([np.asarray(d.get(k,np.full(len(t),np.nan)),float) for k in fields])
    val,age=asof(t,x,at)
    win=(t>=row.start_us)&(t<row.end_us)
    rec=dict(flight_id=fid,split=split,topic=name,records=int(win.sum()),exists=bool(len(t)),auto_windows=len(cg),timestamp_sample='timestamp_sample' in d,age_median_us=float(np.nanmedian(age[:,0])) if len(cg) and np.isfinite(age[:,0]).any() else None)
    for lim in CFG['fresh_sensitivity_us']:rec[f'fresh_{lim}_fraction']=float(np.mean((age[:,0]>=0)&(age[:,0]<=lim))) if len(cg) else None
    for j,k in enumerate(fields):schemas.append(dict(flight_id=fid,split=split,topic=name,field=k,present=k in d,finite_fraction=float(np.isfinite(x[win,j]).mean()) if win.any() else None))
    coverage.append(rec)
    if name in ['position_setpoint_triplet','vehicle_global_position']:
     mission[name]=(val,age)
    elif name in ['vehicle_rates_setpoint','vehicle_attitude_setpoint','fixed_wing_lateral_setpoint','fixed_wing_longitudinal_setpoint']:
     val[age>CFG['reference_fresh_us']]=np.nan
     ref[name]=(val,fields)
    elif name in ['rate_ctrl_terms','rate_ctrl_status']:
     val[age>CFG['reference_fresh_us']]=np.nan;internal[name]=(val,fields)
   if not len(cg):continue
   ss=s[s.flight_id==fid].reset_index(drop=True)
   nat=old.tape(m[m.flight_id==fid],v[v.flight_id==fid]);nt,nu,_=nat
   # Every history feature is selected by actual/conservative publication <= its lag deadline.
   pub=ss.state_publication_us.to_numpy(np.int64);ix=np.searchsorted(pub,at,side='right')-1;assert (ix>=0).all()
   assert (pub[ix]<=at).all()
   origin=cg.origin_sample.to_numpy(int)
   valid=(ss.valid_core.to_numpy()[ix])&(ss.segment_id.to_numpy()[ix]==ss.segment_id.to_numpy()[origin,None])
   xstate=ss[old.V+old.W+old.Q+['flap_frequency_hz']].to_numpy(float)[ix]
   xstate[~valid]=np.nan
   uc=scope.coords(old.zoh(nt,nu,at));u0=uc[:,0]
   feat={}
   for j,lag in enumerate(CFG['history_lags_ms']):
    for k,ch in enumerate(COORDS):feat[f'U{lag}_{ch}']=uc[:,j,k]
    for k,ch in enumerate(['vx','vy','vz','p','q','r','qw','qx','qy','qz','freq']):feat[f'S{lag}_{ch}']=xstate[:,j,k]
   for name,(vals,fields) in {**ref,**internal}.items():
    for j,lag in enumerate(CFG['history_lags_ms']):
     for k,ch in enumerate(fields):feat[f'{"I" if name in internal else "R"}{lag}_{name}_{ch}']=vals[:,j,k]
   wp,wage=mission['position_setpoint_triplet'];gp,gage=mission['vehicle_global_position']
   gp[gage>200000]=np.nan
   wp[wp[:,:,3]!=1]=np.nan
   for j,lag in enumerate(CFG['history_lags_ms']):
    dn=(wp[:,j,0]-gp[:,j,0])*np.pi/180*6371000;de=(wp[:,j,1]-gp[:,j,1])*np.pi/180*6371000*np.cos(gp[:,j,0]*np.pi/180)
    for ch,val in dict(north_m=dn,east_m=de,alt_error=wp[:,j,2]-gp[:,j,2],type=wp[:,j,4],age_s=wage[:,j]/1e6).items():feat[f'M{lag}_{ch}']=val
   feat['mission_changed_500']=np.any(wp[:,0,:3]!=wp[:,-1,:3],axis=1).astype(float)
   feat['mission_changed_500'][~np.isfinite(wp[:,[0,-1],:3]).all(axis=(1,2))]=np.nan
   future=[];commandgrid=scope.coords(old.zoh(nt,nu,decisions[:,None]+grid))
   for r in cg.itertuples():
    yy=[]
    for h in CFG['horizons_ms']:
     tt,uu=scope.cut_tape(nat,int(r.decision_us),int(r.decision_us+h*1000))
     cu=scope.coords(uu);yy.extend(np.average(cu-cu[0],weights=np.diff(tt),axis=0))
    future.append(yy)
   rate=topic['vehicle_angular_velocity'];rt=np.asarray(rate['timestamp_sample'],dtype=np.int64);rv=np.column_stack([rate[f'xyz[{j}]'] for j in range(3)])
   order=np.argsort(rt,kind='stable');rt=rt[order];rv=rv[order];unique=np.r_[np.diff(rt)!=0,True];rt=rt[unique];rv=rv[unique]
   motion=interp(rt,rv,decisions[:,None]+grid)
   diagnostic_refs=[]
   for name in ['vehicle_rates_setpoint','position_setpoint_triplet','vehicle_torque_setpoint']:
    dd=topic.get(name,{});tt=np.asarray(dd.get('timestamp',[]),np.int64)
    ff=TOPICS[name][:3];vv=np.column_stack([np.asarray(dd.get(k,np.full(len(tt),np.nan)),float) for k in ff])
    rr,_=asof(tt,vv,decisions[:,None]+grid,None if name=='position_setpoint_triplet' else 100000)
    diagnostic_refs.append(rr)
   refgrids.append(np.concatenate(diagnostic_refs,axis=-1))
   # Native command tape trimmed to accepted AUTO decision intervals, no overlap double counting.
   intervals=[]
   for r in cg.sort_values('decision_us').itertuples():
    lo,hi=int(r.decision_us),int(r.command_end_us)
    if intervals and lo<=intervals[-1][1]:intervals[-1][1]=max(hi,intervals[-1][1])
    else:intervals.append([lo,hi])
   for seg,(lo,hi) in enumerate(intervals):
    tt,uu=scope.cut_tape(nat,lo,hi);co=scope.coords(uu)
    # Record only compact sufficient native controls, no full raw-log copy.
    rr=pd.DataFrame(co,columns=COORDS);rr['time_us']=tt[:-1]+lo;rr['dt_us']=np.diff(tt);rr['flight_id']=fid;rr['split']=split;rr['interval']=seg
    native_records.append(rr)
   cg['rate_initial_gap']=motion[:,25,0]-xstate[:,0,3]
   cg['rate_age_us']=decisions-ss.rate_measurement_us.to_numpy()[origin]
   cg['reference_complete']=np.isfinite(ref['vehicle_rates_setpoint'][0][:,0]).all(1)
   rows_audit.append(dict(flight_id=fid,split=split,windows=len(cg),history_unavailable=int((~valid).sum()),latest_history_future_us=int(np.max(pub[ix]-at)),min_command_support_margin_us=int(nt[-1]-(decisions+500000).max())))
   metas.append(cg);features.append(pd.DataFrame(feat));targets.append(np.asarray(future));motions.append(motion);controls.append(commandgrid)
   print(split,fid[:10],len(cg),flush=True)
 meta=pd.concat(metas,ignore_index=True);feat=pd.concat(features,ignore_index=True)
 meta.to_csv(RES/'windows.csv.gz',index=False);feat.to_parquet(RES/'causal_features.parquet',index=False)
 np.savez_compressed(RES/'diagnostic_arrays.npz',target=np.concatenate(targets),motion=np.concatenate(motions),control=np.concatenate(controls),grid_us=grid,diagnostic_reference=np.concatenate(refgrids))
 pd.concat(native_records,ignore_index=True).to_csv(RES/'native_auto_controls.csv.gz',index=False)
 pd.DataFrame(coverage).to_csv(RES/'topic_coverage.csv',index=False);pd.DataFrame(schemas).to_csv(RES/'field_coverage.csv',index=False)
 pd.DataFrame(rows_audit).to_csv(RES/'causality_audit.csv',index=False);write_json(RES/'raw_access.json',access)
 # Fold key is original log SHA, never segment id.
 logs=sorted(meta.loc[meta.split=='train','log_sha'].unique());fold={s:i%CFG['crossfit_folds'] for i,s in enumerate(logs)}
 pd.DataFrame([dict(log_sha=k,fold=v) for k,v in fold.items()]).to_csv(RES/'crossfit_groups.csv',index=False)
 write_json(RES/'source_receipt.json',dict(base_sha=CFG['base_sha'],dataset_id=CFG['dataset_id'],manifest_sha256=reg['manifest_sha256'],artifacts=hashes,raw_flights=len(access),partitions=CFG['partitions'],test_read=False,dataset_target_columns_read=False,offline_future_motion_read=True,feature_time='publication<=decision-lag',motion_time='offline native gyro timestamp_sample, no extrapolation, max bracket 50ms',runtime_s=time.time()-start,config_sha256=old.sha(HERE/'config.json')))
if __name__=='__main__':main()
