"""Trace event construction to exact local PX4 objects and authorized train/validation artifacts."""
import json,subprocess,re,urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import yaml
from common import *
from evaluate_merge import redundant
PX4=Path('/home/zn/PX4-Autopilot')
SOURCES=['msg/versioned/ActuatorMotors.msg','msg/versioned/ActuatorServos.msg',
 'src/modules/control_allocator/ControlAllocator.cpp','src/modules/logger/logged_topics.cpp',
 'src/modules/logger/logged_topics.h','src/modules/logger/logger.cpp',
 'platforms/common/uORB/SubscriptionInterval.hpp','src/lib/mixer_module/functions/FunctionServos.hpp','src/lib/mixer_module/functions/FunctionMotors.hpp']

def q(x):
 x=np.asarray(x);return np.quantile(x,[0,.01,.5,.99,1]).tolist() if len(x) else []

def block(text,name):
 match=re.search(r'void\s+LoggedTopics::'+re.escape(name)+r'\(\)\s*\{',text)
 if not match:return ''
 depth=1
 for i in range(match.end(),len(text)):
  depth+=(text[i]=='{')-(text[i]=='}')
  if depth==0:return text[match.start():i+1]
 raise ValueError('unclosed function '+name)

def main():
 OUT.mkdir(exist_ok=True);reg=yaml.safe_load((ROOT/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
 entry=reg['datasets']['velocity_rate_multistep_v1'];mp=ROOT/entry['manifest_path'];assert sha(mp)==entry['manifest_sha256'];manifest=json.loads(mp.read_text())
 flights=pd.read_csv(DATA/'flights.csv');flights=flights[flights.split.isin(['train','validation'])].copy();assert len(flights)==161
 prov={r['flight_id']:r for r in json.loads((DATA/'flight_provenance.json').read_text()) if r['split'] in ['train','validation']}
 checked={};profiles=[]
 for fid,r in prov.items():profiles.append(dict(flight_id=fid,split=r['split'],group=r['sampling_group'],firmware=r['firmware'],SDLOG_PROFILE=r['parameters'].get('SDLOG_PROFILE')))
 pd.DataFrame(profiles).to_csv(OUT/'firmware_flights.csv',index=False)
 snapshots=OUT/'source_objects';snapshots.mkdir(exist_ok=True);records=[];firmware=[]
 commits=sorted(flights.firmware.dropna().unique())
 old=json.loads((OUT/'source_manifest.json').read_text()) if (OUT/'source_manifest.json').exists() else []
 cached={(r['commit'],r['path']):r for r in old if 'artifact' in r}
 def fetch(key):
  commit,path=key
  if key in cached:
   r=cached[key];data=(HERE/r['artifact']).read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256'];return data,r.get('origin',r['status']),r.get('url')
  result=subprocess.run(['git','-C',str(PX4),'show',f'{commit}:{path}'],capture_output=True)
  if result.returncode==0:return result.stdout,'local_git_object',None
  url=f'https://raw.githubusercontent.com/jiashunjiang2022/PX4-Autopilot/{commit}/{path}'
  try:
   with urllib.request.urlopen(url,timeout=30) as response:return response.read(),'exact_commit_https',url
  except Exception as ex:return None,str(ex),url
 keys=[(c,p) for c in commits for p in SOURCES]
 with ThreadPoolExecutor(max_workers=8) as pool:objects=dict(zip(keys,pool.map(fetch,keys)))
 for commit in commits:
  texts={}
  for path in SOURCES:
   data,origin,url=objects[(commit,path)]
   if data is None:records.append(dict(commit=commit,path=path,status='unavailable',error=origin,url=url));continue
   digest=hashlib.sha256(data).hexdigest();dest=snapshots/(digest+'_'+Path(path).name);dest.write_bytes(data);texts[path]=data.decode()
   records.append(dict(commit=commit,path=path,status='verified_exact_commit',origin=origin,url=url,sha256=digest,artifact=str(dest.relative_to(HERE))))
  logger=texts.get('src/modules/logger/logged_topics.cpp','');flap=block(logger,'add_flapping_dataset_topics');default=block(logger,'add_default_topics')
  intervals={}
  for topic in ['actuator_motors','actuator_servos','vehicle_local_position']:
   def interval(s):
    found=re.findall(r'add_topic\("'+topic+r'"(?:,\s*(\d+))?\)',s);return int(found[-1] or 0) if found else None
   intervals[topic+'_default_ms']=interval(default);intervals[topic+'_flapping_ms']=interval(flap)
  firmware.append(dict(commit=commit,flights=int((flights.firmware==commit).sum()),logger_source_available=bool(logger),flapping_profile_exists=bool(flap) if logger else None,**intervals))
 save(OUT/'source_manifest.json',records);pd.DataFrame(firmware).to_csv(OUT/'firmware_logging.csv',index=False)
 raw=[]
 for split in ['train','validation']:
  parts={}
  for kind in ['motors','servos']:
   name=f'native_actuator_{kind}_{split}.parquet';assert sha(DATA/name)==manifest['artifact_sha256'][name];checked[name]=sha(DATA/name)
   parts[kind]=dict(tuple(pd.read_parquet(DATA/name).groupby('flight_id',sort=False)))
  for f in flights[flights.split==split].itertuples():
   motor=parts['motors'][f.flight_id];servo=parts['servos'][f.flight_id]
   motor=motor[motor.in_reviewed_flight];servo=servo[servo.in_reviewed_flight]
   mt=motor.timestamp_us.to_numpy();st=servo.timestamp_us.to_numpy();union=np.union1d(mt,st)
   union=union[union>=max(mt[0],st[0])]
   u=np.c_[motor['control[0]'].to_numpy()[np.searchsorted(mt,union,side='right')-1],servo[[f'control[{i}]' for i in range(3)]].to_numpy()[np.searchsorted(st,union,side='right')-1]]
   equal=np.all(u[1:]==u[:-1],axis=1);dif=np.abs(np.diff(u,axis=0));finite=np.isfinite(u[1:]).all(1)&np.isfinite(u[:-1]).all(1)
   row=dict(flight_id=f.flight_id,split=split,group=f.sampling_group,firmware=f.firmware,union_events=len(union),exact_repeats=int((equal&finite).sum()),
       all_four_delta_le_1e6=int(((dif.max(1)<=1e-6)&finite).sum()),common_motor_servo_timestamps=len(np.intersect1d(mt,st)),motor_records=len(mt),servo_records=len(st))
   for j,name in enumerate(['motor','left_elevon','right_elevon','rudder']):row[name+'_same_at_union_event']=int(((dif[:,j]==0)&finite).sum())
   for kind,frame in [('motor',motor),('servo',servo)]:
    ts=frame.timestamp_us.to_numpy();delta=np.diff(ts);row[kind+'_period_median_ms']=float(np.median(delta)*1e-3);row[kind+'_period_p99_ms']=float(np.quantile(delta,.99)*1e-3)
    row[kind+'_sample_age_median_ms']=float(np.median(ts-frame.timestamp_sample_us.to_numpy())*1e-3)
   raw.append(row)
 pd.DataFrame(raw).to_csv(OUT/'raw_control_per_flight.csv',index=False)
 cache_summaries=[];step_rows=[];redundant_steps=[]
 for split in ['train','validation']:
  digest=cache_hash(split);z=np.load(CACHE/f'{split}.npz');ev=z['events'];le=z['event_length'];valid=z['future_mask'];dt=z['dt']
  meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');same=redundant(ev,le,valid);changed=same.any(2)
  cache_summaries.append(dict(split=split,windows=len(meta),flights=meta.flight_id.nunique(),valid_step_occurrences=int(valid.sum()),tokens=int(le[valid].sum()),
    start_boundary_tokens=int(valid.sum()),redundant_tokens=int(same.sum()),changed_step_occurrences=int(changed.sum()),changed_windows=int(changed.any(1).sum()),
    event_count_distribution={int(v):int(n) for v,n in zip(*np.unique(le[valid],return_counts=True))},cache_sha256=digest))
  for fid,g in meta.groupby('flight_id',sort=False):
   ids=g.index.to_numpy();stepids=g.origin_sample.to_numpy()[:,None]+np.arange(valid.shape[1])[None,:]
   flat_valid=valid[ids].ravel();positions=np.flatnonzero(flat_valid);unique,index=np.unique(stepids.ravel()[positions],return_index=True)
   positions=positions[index];local,kk=np.divmod(positions,valid.shape[1]);ii=ids[local]
   counts=le[ii,kk];r=same[ii,kk].sum(1)
   sf=ev[ii,kk,0,4]/dt[ii,kk]
   step_rows.append(dict(flight_id=fid,split=split,unique_intervals=len(unique),tokens=int(counts.sum()),redundant_tokens=int(r.sum()),changed_intervals=int((r>0).sum()),
       count1=int((counts==1).sum()),count2=int((counts==2).sum()),count3plus=int((counts>=3).sum()),first_duration_fraction_median=float(np.median(sf)),
       first_duration_ms_median=float(np.median(ev[ii,kk,0,4])*1000),state_dt_ms_median=float(np.median(dt[ii,kk])*1000)))
   for source_step,i,k in zip(unique[r>0],ii[r>0],kk[r>0]):
    redundant_steps.append(dict(flight_id=fid,split=split,origin_sample=int(source_step),window_example=meta.iloc[i].window_id,
      event_count=int(le[i,k]),events=ev[i,k,:le[i,k]].tolist()))
 pd.DataFrame(step_rows).to_csv(OUT/'training_intervals_per_flight.csv',index=False);save(OUT/'cache_statistics.json',cache_summaries);save(OUT/'redundant_unique_intervals.json',redundant_steps)
 save(OUT/'provenance_manifest.json',dict(dataset_id='velocity_rate_multistep_v1',registry=entry,manifest_sha256=sha(mp),checked_artifacts=checked,
   authorized_flights=len(flights),sealed_test_not_read=True,raw_log_files_read=False,source_code_sha256=sha(Path(__file__)),
   packing_source_sha256=sha(ROOT/'scripts/prepare_velocity_rate_learning.py'),extractor_sha256=sha(ROOT/'src/system_identification/data/velocity_rate_dataset.py')))
 print('AUDIT COMPLETE',len(records),'source objects,',len(redundant_steps),'unique redundant state intervals')
if __name__=='__main__':main()
