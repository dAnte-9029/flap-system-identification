from pathlib import Path
import os,json,hashlib,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'experiments/local_control_response_rollout_v1';TIMING=ROOT/'experiments/local_control_response_timing_slowfast_v1';SCOPE=ROOT/'experiments/local_control_response_scope_v1'
OUT=Path(os.environ.get('LOCAL_SWAP_OUTPUT',str(HERE/'results')));DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
VARS=['vx','vy','vz','p','q','r'];CHANNELS=[3,4,5,10,11,12]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else str(a))+'\n')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
stats=module(OLD/'analyze.py','old_statistics')
def coords(u):return np.column_stack([u[:,0],(u[:,1]+u[:,2])/2,(u[:,2]-u[:,1])/2,u[:,3]])
def interp(t,y,at):return stats.interpolate(t,y,np.asarray(at,float))[0]
def event_tape(events,lengths,dt):
 t=[];u=[];elapsed=0
 for k,step in enumerate(dt):
  duration=int(round(float(step)*1e6))
  if duration==0:continue
  e=events[k,:int(lengths[k])];du=np.rint(e[:,4].astype(float)*1e6).astype(np.int64)
  assert du.sum()==duration
  for row,span in zip(e,du):
   if span>0:t.append(elapsed);u.append(row[:4]);elapsed+=int(span)
 return np.array(t,np.int64),np.array(u,float),elapsed

def difference(t1,u1,t2,u2,end):
 """Exact time-weighted four-coordinate difference in integer microseconds."""
 knots=np.unique(np.r_[0,t1[(t1>0)&(t1<end)],t2[(t2>0)&(t2<end)],end]).astype(np.int64)
 a=coords(u1[np.searchsorted(t1,knots[:-1],side='right')-1]);b=coords(u2[np.searchsorted(t2,knots[:-1],side='right')-1]);delta=a-b;duration=np.diff(knots)
 mean=(delta*duration[:,None]).sum(0)/end;rms=np.sqrt((delta**2*duration[:,None]).sum(0)/end)
 longest=0;run=0;sign=0
 for val,span in zip(delta[:,2],duration):
  sg=np.sign(val) if abs(val)>=.1 else 0
  run=run+span if sg and sg==sign else span if sg else 0;sign=sg;longest=max(longest,run)
 return mean,rms,int(longest)
