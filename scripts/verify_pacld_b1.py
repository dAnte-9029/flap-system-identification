#!/usr/bin/env python
"""One native-window CUDA FP64 comparison against the preserved CPU implementation."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from system_identification.models.pacld import PACLD,Config
from system_identification.models.pacld.batch import BatchedPACLD,FIELDS
from system_identification.data.pacld import open_windows
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld import window_objective

def main():
 torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.set_device(1)
 cfg=json.load(open('configs/pacld/default.json'));w=open_windows(cfg,'train')[0];m=PACLD(Config(**cfg['model'])).double();g=BatchedPACLD(m.config).cuda(1);g.load_state_dict(m.state_dict())
 a=window_objective(m,w,.1,diagnostic_base=True);b=g(compile_windows([w],m.config).to('cuda:1'),.1);a['total'].backward();b['total'].backward()
 belief=max(float((getattr(a['belief'],k)-getattr(b['belief'],k)[0].cpu()).abs().max()) for k in FIELDS)
 path=max(float((torch.stack([s.x for s in a['predictions'][mode]])-b['predictions'][i].cpu()).abs().max()) for i,mode in enumerate(('FULL','BASE') if b['predictions'].shape[0]==2 else ('FULL',)))
 gradient=max(float((p.grad-q.grad.cpu()).abs().max()) for p,q in zip(m.parameters(),g.parameters()) if p.grad is not None)
 loss=float((a['total']-b['total'].cpu()).abs());assert max(belief,path,gradient,loss)<1e-8
 Path('runs/pacld_gpu_batched/b1.json').write_text(json.dumps(dict(window_id=w.metadata['window_id'],belief_max_abs=belief,trajectory_max_abs=path,gradient_max_abs=gradient,loss_abs=loss),indent=2))
if __name__=='__main__':main()
