from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';PREV=ROOT/'experiments/control_target_alignment_v1'
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1');CACHE=Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1/artifacts/velocity_rate_gru64_horizon_training_v1');TRAIN=ROOT/'experiments/velocity_rate_gru64_horizon_training_v1'
sys.path.insert(0,str(ROOT/'src'))
from system_identification.models.velocity_rate_gru import integrate,geodesic
from system_identification.data.velocity_rate_dataset import STATE,TARGET
RATE=STATE[10:13];VARS=['p','q','r'];ACCEPT=['interpolation','forward_extrapolation','backward_extrapolation']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
# Reuse prior frozen support logic without rerunning the Oracle analysis.
spec=importlib.util.spec_from_file_location('previous_full',PREV/'full_audit.py');prev=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
sample=prev.sample

def kinematics(initial,velocity,rates,dt,recorded,mask):
    """Original integrator; future v/r are offline prescribed targets, never GRU inputs."""
    s=torch.as_tensor(initial,dtype=torch.float64).clone();v=torch.as_tensor(velocity,dtype=torch.float64);r=torch.as_tensor(rates,dtype=torch.float64);d=torch.as_tensor(dt,dtype=torch.float64);q=torch.as_tensor(recorded,dtype=torch.float64)
    qs=[];ps=[]
    with torch.inference_mode():
     for k in range(d.shape[1]):
      active=torch.as_tensor(mask[:,k],dtype=torch.bool)
      if k:assert not np.any(mask[:,k]&~mask[:,k-1]),'no restart after unsupported target'
      vk=torch.where(active[:,None],v[:,k],s[:,3:6]);rk=torch.where(active[:,None],r[:,k],s[:,10:13]);dk=torch.where(active,d[:,k],0.)
      s=integrate(s,vk,rk,dk);qs.append(np.where(mask[:,k],geodesic(s[:,6:10],q[:,k,6:10]).numpy(),np.nan));ps.append(np.where(mask[:,k],torch.linalg.vector_norm(s[:,:3]-q[:,k,:3],dim=-1).numpy(),np.nan))
    return np.stack(qs,1),np.stack(ps,1)
