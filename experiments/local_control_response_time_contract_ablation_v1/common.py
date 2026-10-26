from pathlib import Path
import os,sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=Path(os.environ.get('LOCAL_ABLATION_OUTPUT',str(HERE/'results')))
SWAP=ROOT/'experiments/local_control_response_swap_sensitivity_v1';OLD=ROOT/'experiments/local_control_response_rollout_v1';TIMING=ROOT/'experiments/local_control_response_timing_slowfast_v1';SCOPE=ROOT/'experiments/local_control_response_scope_v1';DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1');PX4=Path('/home/zn/PX4-Autopilot')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else str(a))+'\n')
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def legacy(p,name):
 old=sys.modules.get('common');sys.modules['common']=module(p.parent/'common.py',name+'_common')
 try:return module(p,name)
 finally:
  if old is not None:sys.modules['common']=old
  else:sys.modules.pop('common',None)
stats=module(OLD/'analyze.py','old_statistics');tc=module(TIMING/'common.py','timing_common');tm=legacy(TIMING/'analyze.py','timing_statistics');sm=legacy(SWAP/'analyze.py','swap_statistics')
VARS=['vx','vy','vz','p','q','r'];CHANNELS=[3,4,5,10,11,12]
