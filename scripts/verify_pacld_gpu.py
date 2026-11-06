#!/usr/bin/env python
"""Real native windows: reference parity, precision, Adam, checkpoint continuation."""
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from system_identification.models.pacld import PACLD,Config
from system_identification.models.pacld.batch import BatchedPACLD,FIELDS
from system_identification.models.pahbd import geometry as so3
from system_identification.data.pacld import open_windows
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld import window_objective,seed_all
from system_identification.training.pacld_gpu import BucketSampler,checkpoint_bundle,restore

def grad(m):return {n:p.grad.detach().double().cpu().clone() for n,p in m.named_parameters() if p.grad is not None}
def compare(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)
def main():
 torch.set_num_threads(1);seed_all(17);torch.cuda.set_device(1)
 c=json.load(open('configs/pacld/default.json'));cfg=Config(**c['model']);d=open_windows(c,'train');records=[]
 original=PACLD(cfg).double();state=copy.deepcopy(original.state_dict())
 for horizon in c['data']['horizon_us']:
  indices=d.index[d.index.horizon_us==horizon].groupby('sampling_group',sort=True).head(1).index.tolist();ws=[d[i] for i in indices];batch=compile_windows(ws,cfg)
  ref_values=[];ref_beliefs=[];ref_paths=[];ref_phases=[];original.zero_grad(set_to_none=True)
  for w in ws:
   r=window_objective(original,w,.1,diagnostic_base=True);(r['total']/len(ws)).backward();ref_values.append(float(r['total']));ref_phases.append(torch.stack([s.phi.detach() for s in r['predictions']['FULL']]));ref_beliefs.append({k:getattr(r['belief'],k).detach() for k in FIELDS});ref_paths.append({k:torch.stack([s.x.detach() for s in v]) for k,v in r['predictions'].items()});del r
  rg=grad(original);gpu_reference=None;gpu_grad=None
  for precision in ('fp64','mixed','fp32'):
   m=BatchedPACLD(cfg,precision).cuda(1);m.load_state_dict(state);dt=torch.float32 if precision=='fp32' else torch.float64
   r=m(batch.to('cuda:1',dt),.1);r['total'].backward();g=grad(m);x=r['predictions'].detach().double().cpu();belief={k:getattr(r['belief'],k).detach().double().cpu() for k in FIELDS}
   if precision=='fp64':gpu_reference=x;gpu_grad=g
   belief_error=max(float((belief[k][i]-ref_beliefs[i][k]).abs().max()) for k in FIELDS for i in range(len(ws)))
   reference_error=max(float((x[i+offset,:len(batch.times[i])]-ref_paths[i][mode]).abs().max()) for i in range(len(ws)) for mode,offset in [('FULL',0),('BASE',len(ws))])
   qerror=float((torch.linalg.vector_norm(x[...,6:10],dim=-1)-1).abs().max());angle=float(so3.residual(gpu_reference[...,6:10],x[...,6:10]).abs().max())
   phase_error=max(float((r['phases'][i,:len(batch.times[i])].detach().double().cpu()-ref_phases[i]).abs().max()) for i in range(len(ws)))
   loss=float(r['total']);macro=r['macro'].detach().cpu().tolist();full=r['full'].detach().cpu().tolist();del r
   opt=torch.optim.Adam(m.parameters(),**c['optimizer']);torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step()
   if precision=='fp64': post_reference={k:v.detach().double().cpu().clone() for k,v in m.state_dict().items()}
   post_error=max(float((v.detach().double().cpu()-post_reference[k]).abs().max()) for k,v in m.state_dict().items())
   record=dict(horizon_us=horizon,precision=precision,phase_max_abs=phase_error,post_adam_parameter_max_abs=post_error,reference_belief_max_abs=belief_error,reference_trajectory_max_abs=reference_error,gpu64_trajectory_max_abs=float((x-gpu_reference).abs().max()),attitude_error_rad=angle,quaternion_norm_error=qerror,full=full,macro=macro,loss=loss,reference_loss_error=abs(loss-sum(ref_values)/len(ws)),reference_gradient_max_abs=compare(rg,g),gpu64_gradient_max_abs=compare(gpu_grad,g),gradient_relative_l2=float(torch.cat([(g[k]-gpu_grad[k]).flatten() for k in g]).norm()/torch.cat([v.flatten() for v in gpu_grad.values()]).norm()),post_update_finite=all(bool(torch.isfinite(p).all()) for p in m.parameters()))
   if precision=='fp64':
    assert belief_error<1e-8 and reference_error<1e-8 and compare(rg,g)<1e-7,record
   assert record['post_update_finite'],record
   records.append(record);print(json.dumps(record),flush=True)
   del m,opt;torch.cuda.empty_cache()
 Path('runs/pacld_gpu_batched/numerics.json').write_text(json.dumps(records,indent=2))
 # Exact next draw and Adam update after GPU-schema serialization.
 m=BatchedPACLD(cfg,'mixed').cuda(1);m.load_state_dict(state);opt=torch.optim.Adam(m.parameters(),**c['optimizer']);sampler=BucketSampler(d.index,17,2);sampler.draw()
 batch=compile_windows(ws,cfg).to('cuda:1');r=m(batch,.1);r['total'].backward();opt.step();opt.zero_grad(set_to_none=True);del r
 bundle=checkpoint_bundle(m,opt,sampler,dict(optimizer_steps=1,windows_seen=2),c,d.identity(),.1,1)
 path=Path('runs/pacld_gpu_batched/restore_probe.pt');torch.save(bundle,path)
 expected=sampler.draw();r=m(batch,.1);r['total'].backward();opt.step();expected_state=copy.deepcopy(m.state_dict());del r
 n=BatchedPACLD(cfg,'mixed').cuda(1);o=torch.optim.Adam(n.parameters(),**c['optimizer']);s=BucketSampler(d.index,17,2)
 restore(torch.load(path,weights_only=False),n,o,s,c,d.identity(),.1,1);assert expected==s.draw()
 r=n(batch,.1);r['total'].backward();o.step();error=max(float((v-n.state_dict()[k]).abs().max()) for k,v in expected_state.items());assert error==0,error
 Path('runs/pacld_gpu_batched/restore.json').write_text(json.dumps(dict(next_indices=expected,parameter_max_abs=error)))
if __name__=='__main__':main()
