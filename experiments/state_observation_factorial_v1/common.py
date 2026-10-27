from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';PREV=ROOT/'experiments/causal_state_target_contract_v1'
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1');CACHE=Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1/artifacts/velocity_rate_gru64_horizon_training_v1');TRAIN=ROOT/'experiments/velocity_rate_gru64_horizon_training_v1'
sys.path.insert(0,str(ROOT/'src'))
from system_identification.models.velocity_rate_gru import integrate,geodesic
# Unmodified previous helpers, definitions only; do not run previous analysis.
spec=importlib.util.spec_from_file_location('frozen_contract_common',PREV/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
kinematics=old.kinematics;sample=old.sample
VARS=['p','q','r'];COMBOS=['T00','T01','T10','T11']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def contrasts(m00,m01,m10,m11):
 return dict(label_at_C0=m01-m00,label_at_C1=m11-m10,initial_at_A=m10-m00,initial_at_B=m11-m01,interaction=m11-m10-m01+m00,label_main=(m01-m00+m11-m10)/2,initial_main=(m10-m00+m11-m01)/2,joint=m11-m00)
