import importlib.util,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
out=Path(__file__).parent;old=Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1');os.sched_setaffinity(0,{12});torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
prep=json.loads((out/'preparation.json').read_text());idx=json.loads((out/'index.json').read_text());j=json.loads((out/'archived_origin_indices.json').read_text());first=next(i for i,x in enumerate(idx) if x['horizon_us']==2000000);start=j[first]//2048*2048
common=np.load(old/'artifacts/velocity_rate_gru64_long_horizon_v1/common_indices.npy');cache=np.load(old/'artifacts/velocity_rate_gru64_long_horizon_v1/validation.npz');ids=common[start:start+2048];keys=('history','history_mask','history_dt','history_length','initial','dt','events','event_length');x={k:torch.from_numpy(cache[k][ids]).cuda() for k in keys};truth=torch.from_numpy(cache['target'][ids]).cuda()
spec=importlib.util.spec_from_file_location('frozen_gru',prep['source_model']);gru=importlib.util.module_from_spec(spec);spec.loader.exec_module(gru);ck=prep['checkpoints'][0];b=torch.load(ck['checkpoint'],map_location='cpu',weights_only=False);m=gru.VelocityRateGRU(b['normalization'],control=True).cuda();m.load_state_dict(b['model']);m.eval()
with torch.no_grad():pred=m(x);errors=gru.errors(pred,truth).cpu().numpy();errors[...,2]*=180/np.pi
legacy=pd.read_parquet(old/'experiments/velocity_rate_gru64_horizon_training_v1/metrics/H1000-GRU-Control_seed_17_endpoints.parquet');diff=[]
for item,jj in zip(idx,j):
 if item['horizon_us']!=2000000 or jj is None or not start<=jj<start+len(ids):continue
 for r in legacy[legacy.window_id==item['window_id']].itertuples():
  for c,name in enumerate(['position_m','velocity_m_s','attitude_deg','rate_rad_s']):diff.append(float(abs(errors[jj-start,int(r.native_step)-1,c]-getattr(r,name))))
assert diff and max(diff)<1e-5
(out/'archived_batch_check.json').write_text(json.dumps(dict(seed=17,original_batch_start=start,compared_values=len(diff),max_abs_difference=max(diff),conclusion='same archived batch reproduces original metrics; new cohort batch changes float32/TF32 execution numerics'),indent=2));print('archived batch',max(diff),'values',len(diff))
