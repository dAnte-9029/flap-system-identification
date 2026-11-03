#!/usr/bin/env python3
"""Numerical check of an actual eight-step region, not a real-window speed claim."""
import copy,json,os,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.models.pahbd.tensor_flow import enable_tensor_engine,unpack
from dataclasses import fields,replace
os.sched_setaffinity(0,{8,10,12,14});torch.set_num_threads(1);torch.cuda.set_device(0)
torch.manual_seed(17);m=PAHBD(Config()).double().cuda();w,_=controlled_periodic_window(m.config,.04)
s=replace(w.initial,**{f.name:getattr(w.initial,f.name).cuda() for f in fields(w.initial) if isinstance(getattr(w.initial,f.name),torch.Tensor)})
u=w.controls.values[0].cuda().requires_grad_();engine=enable_tensor_engine(m,'block8');dt=.003
original=s
for _ in range(8):original=m.flow.step(original,u,dt)
ref=sum(v.square().sum() for v in unpack(original,m.flow));ref.backward()
grad={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None};ugrad=u.grad.clone()
m.zero_grad(set_to_none=True);u.grad=None
start=time.perf_counter();output=engine.block(unpack(s,m.flow),u,*engine.prepare(dt,s.x));loss=sum(v.square().sum() for v in output);loss.backward();torch.cuda.synchronize()
for a,b in zip(unpack(original,m.flow),output):torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-8)
for k,p in m.named_parameters():
 if k in grad:torch.testing.assert_close(grad[k],p.grad,atol=1e-8,rtol=1e-8)
torch.testing.assert_close(ugrad,u.grad,atol=1e-8,rtol=1e-8)
print(json.dumps(dict(block_steps=8,elapsed_with_compilation=time.perf_counter()-start,values_and_gradients_match=True)))
