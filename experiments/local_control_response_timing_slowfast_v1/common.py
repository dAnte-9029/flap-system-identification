from pathlib import Path
import hashlib,json,importlib.util,os
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'experiments/local_control_response_rollout_v1';SCOPE=ROOT/'experiments/local_control_response_scope_v1'
RESULTS=Path(os.environ.get('LOCAL_DIAG_OUTPUT',str(HERE/'results')))
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
MODELS=[('GRU-Control',s) for s in [17,29,43]]+[('GRU-NoControl',s) for s in [17,29,43]]+[('Hold',0)]
VARS=['p','q','r']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module(OLD/'analyze.py','old_analysis')
def interp(t,y,x,max_gap=.04):
 t=np.asarray(t,float);y=np.asarray(y,float);x=np.atleast_1d(x).astype(float)
 assert len(t)>=2 and np.all(np.diff(t)>0)
 out=np.full((len(x),y.shape[1]),np.nan)
 for i,q in enumerate(x):
  if q<t[0]-1e-10 or q>t[-1]+1e-10:continue
  q=min(max(q,t[0]),t[-1]) # floating arithmetic only, <=1e-10 s tolerance
  j=np.searchsorted(t,q,side='left')
  if j<len(t) and abs(t[j]-q)<1e-10:out[i]=y[j];continue
  if j==0 or j==len(t) or t[j]-t[j-1]>max_gap+1e-10:continue
  a=(q-t[j-1])/(t[j]-t[j-1]);out[i]=(1-a)*y[j-1]+a*y[j]
 return out

def boxmean(t,y,centers,width,max_gap=.04):
 """Exact integral of piecewise-linear observations, full windows only."""
 centers=np.atleast_1d(centers);out=np.full((len(centers),y.shape[1]),np.nan)
 for i,c in enumerate(centers):
  lo,hi=c-width/2,c+width/2
  if lo<t[0]-1e-10 or hi>t[-1]+1e-10:continue
  knots=np.r_[lo,t[(t>lo)&(t<hi)],hi]
  if np.any(np.diff(knots)>max_gap+1e-10):continue
  vals=interp(t,y,knots,max_gap)
  if np.isfinite(vals).all():out[i]=np.trapz(vals,knots,axis=0)/width
 return out

def weights(p,kind):return old.weights(p,kind)
