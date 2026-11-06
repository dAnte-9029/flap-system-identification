#!/usr/bin/env python
"""Frozen PACLD evaluation, never trains or opens sealed partitions."""
import argparse,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from system_identification.models.pacld import PACLD,Config
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.models.pahbd import geometry as so3
from system_identification.data.pacld import open_windows,subset
from system_identification.evaluation.pacld_gpu import predict,factual_predictor
from system_identification.evaluation.pacld import paired_physical_errors,intracycle,latent_diagnostics,hierarchical,summarize_intracycle
from system_identification.evaluation.pahbd import native_horizon_errors,FrozenRollResponse
from system_identification.training.pacld import valid_pairs

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=root/'runs/pacld_frozen_eval_seed17/results';out.mkdir(exist_ok=True)
 bundle=torch.load(root/'runs/pacld_frozen_eval_seed17/model.pt',map_location='cpu',weights_only=False);cfg=bundle['config'];torch.set_num_threads(1)
 val=open_windows(cfg,'validation');ids=bundle['data_identity']['validation_window_ids'];lookup={w:i for i,w in enumerate(val.index.window_id)};val=subset(val,[lookup[w] for w in ids]);assert len(val)==156
 model=BatchedPACLD(Config(**cfg['model']),bundle['precision']).to(a.device).eval();model.load_state_dict(bundle['model'])
 diagnostic=PACLD(Config(**cfg['model'])).double().eval();diagnostic.load_state_dict(bundle['model'])
 native=[];within=[];paired=[];latents=[];phases=[];check=[]
 with torch.no_grad():
  for start in range(0,len(val),8):
   ws=[val[i] for i in range(start,min(start+8,len(val)))];results=predict(model,ws)
   for w,r in zip(ws,results):
    meta={k:w.metadata[k] for k in ('window_id','flight_id','horizon_us','sampling_group')};r['pairs'],_=valid_pairs(w)
    for mode in ('FULL','BASE'):
     for row in native_horizon_errors(r['predictions'][mode],r['times'],w.targets,w.metadata):native.append(dict(row,model=mode,window_horizon_us=meta['horizon_us']))
     for c,v in paired_physical_errors(w,r,mode).items():paired.append(dict(meta,mode=mode,component=c,**v))
    # Constant inertial velocity / constant body rate, from the SAME causal belief.
    x=r['belief'].x;kin=[]
    for t in r['times']:
     y=x.clone();y[:3]=x[:3]+x[3:6]*t;y[6:10]=so3.unit(so3.multiply(x[6:10],so3.exp(x[10:13]*t)));kin.append(r['belief'].with_(x=y))
    for row in native_horizon_errors(kin,r['times'],w.targets,w.metadata):native.append(dict(row,model='kinematic',window_horizon_us=meta['horizon_us']))
    within.extend(dict(meta,**v) for v in intracycle(w,r,cfg));latents.append(dict(meta,**latent_diagnostics(diagnostic,w,r)))
    reference=w.metadata.get('target_phase_reference',[]);by={m:dict(zip(r['times'],r['predictions'][m])) for m in ('FULL','BASE')}
    for j,e in enumerate(w.targets):
     if j>=len(reference) or reference[j] is None:continue
     for name,source,ix,mask in [('q','vehicle_angular_velocity',11,10),('vD','vehicle_local_position',5,5)]:
      if e.source!=source or e.mask[mask]*e.quality[mask]<=0:continue
      for mode in by:phases.append(dict(meta,variable=name,mode=mode,bin=int((reference[j]%(2*math.pi))/(2*math.pi)*16),error=float(by[mode][e.measurement].x[ix]-e.value[ix])))
   print(f'evaluated {start+len(ws)}/156',flush=True)
   # Durable incremental result, no mutation of training artifacts.
   (out/'evaluation.json').write_text(json.dumps(dict(native=native,paired=paired,intracycle=within,latent=latents),indent=2))
 frame=pd.DataFrame(native);frame.to_csv(out/'native_errors.csv',index=False)
 summary=[]
 for (m,c,h),g in frame.groupby(['model','component','horizon_s']):
  mse=g.groupby(['flight_id','window_horizon_us']).squared_error.mean().groupby('flight_id').mean().mean()
  summary.append(dict(model=m,component=c,horizon_s=h,rmse=math.sqrt(mse)*(180/math.pi if c=='R' else 1),flights=g.flight_id.nunique(),windows=g.window_id.nunique(),native_observations=len(g)))
 pd.DataFrame(summary).to_csv(out/'physical_rmse.csv',index=False)
 pd.DataFrame(paired).to_csv(out/'paired_errors.csv',index=False);pd.DataFrame(summarize_intracycle(within)).to_csv(out/'intracycle_summary.csv',index=False)
 pd.DataFrame(phases).to_csv(out/'phase_residuals.csv',index=False)
 source=Path('/home/zn/flap-system-identification/.worktrees/local-control-response-rollout-v1/experiments/local_control_response_rollout_v1')
 response={}
 try:
  protocol=FrozenRollResponse(source)
  for split in ('validation','train'):
   cohort=protocol.pairs[protocol.pairs.split==split];needed=set(cohort[['window_a','window_b']].to_numpy().ravel());ws=open_windows(cfg,split);missing=needed-set(ws.index.window_id)
   response[split]=dict(missing_window_ids=sorted(missing),pairs=len(cohort),flights=len(set(cohort.flight_a)|set(cohort.flight_b)),components=cohort.dependency_component.nunique(),missing_windows=len(missing))
   if not len(cohort) or missing:continue
   result=protocol.evaluate(factual_predictor(model,ws),split);result['rows'].to_csv(out/f'response_{split}.csv',index=False);response[split]['summary']=result['summary']
 except (FileNotFoundError,ValueError) as e:response['unavailable']=str(e)
 (out/'response.json').write_text(json.dumps(response,indent=2,default=int))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(10,7))
 for ax,c in zip(axs.flat,['p','v','R','omega']):
  for m in frame.model.unique():
   g=pd.DataFrame(summary);g=g[(g.model==m)&(g.component==c)];ax.plot(g.horizon_s,g.rmse,'o-',label=m)
  ax.set(title=c,xlabel='Horizon (s)',ylabel='RMSE');ax.legend()
 fig.tight_layout();fig.savefig(out/'physical_rmse.png');plt.close(fig)
 curve=pd.read_csv(out/'validation_curve.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,c in zip(axs,['full','macro']):ax.plot(curve.step,curve[c],'o-');ax.set(title=c,xlabel='Optimizer step')
 fig.tight_layout();fig.savefig(out/'validation_curve.png')
 print(json.dumps(response,default=int),flush=True)
if __name__=='__main__':main()
