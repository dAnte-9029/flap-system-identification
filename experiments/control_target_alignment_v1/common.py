"""Frozen-reference CPU audit; no model or training module is imported."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
OLD=ROOT/'experiments/local_control_response_rollout_v1'
TIMING=ROOT/'experiments/local_control_response_timing_slowfast_v1'
PILOT=ROOT/'experiments/control_conditioned_training_pilot_v1'
FORMAL=ROOT/'experiments/structured_control_effectiveness_group_holdout_v1'
SWAP=ROOT/'experiments/local_control_response_swap_sensitivity_v1'
ABL=ROOT/'experiments/local_control_response_time_contract_ablation_v1'
TRAIN=ROOT/'experiments/velocity_rate_gru64_horizon_training_v1'
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
CACHE=Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1/artifacts/velocity_rate_gru64_horizon_training_v1')
VARS=['p','q','r'];RATE=['angular_velocity_body_rad_s_'+c for c in 'xyz'];TARGET=['target_'+c for c in RATE]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
tc=module(TIMING/'common.py','frozen_time_common');stats=tc.old
# Analyze module only defines functions; import its common dependency temporarily.
import sys
oldcommon=sys.modules.get('common');sys.modules['common']=tc
metric=module(TIMING/'analyze.py','frozen_time_metrics').metric
if oldcommon is not None:sys.modules['common']=oldcommon
else:del sys.modules['common']
def read_assets():
 return (np.load(OLD/'results/native_rollouts.npz'),np.load(TIMING/'results/native_diagnostic_inputs.npz'),pd.read_csv(OLD/'results/windows.csv'),pd.read_csv(OLD/'results/pairs.csv'),pd.read_csv(OLD/'results/cohorts.csv'))
def tapes(z,n,i):
 count=int(z['future_count'][i]);t=z['native_time_us'][i,:count+1]*1e-6
 a=np.vstack([z['truth_initial'][i,10:13],z['truth'][i,:count,10:13]]).astype(float)
 init=z['initial'][i,10:13].astype(float);anchor=n[f'native_anchor_{i}'];nt=n[f'rate_t_{i}'];ny=n[f'rate_y_{i}']
 end=min(.5,t[-1]);bt=np.r_[0,nt[(nt>0)&(nt<end)],end];by=tc.interp(nt,ny,bt,.02)
 # Native-reference Oracles are continuous native tapes, not a new registered-grid network.
 o0=a.copy();o0[0]=init
 return dict(A=(t,a-a[0]),B=(bt,by-anchor),O0=(t,o0-init),O1=(t,a-a[0]),O2=(bt,by-init),O3=(bt,by-anchor),label=(t,a),native=(bt,by)),init,a[0],anchor

def values(tape,at,slow=False):return tc.boxmean(*tape,at,.27) if slow else tc.interp(*tape,at)
