import copy
import pytest
import torch
from test_pahbd_readiness import model_window
from system_identification.models.pahbd.tensor_flow import enable_tensor_engine,unpack
from system_identification.training.pahbd import window_loss


@pytest.mark.parametrize('mode',['A','B'])
def test_tensor_window_value_schedule_gradient(mode):
    m,w=model_window(.22);m.config.mode=mode;other=copy.deepcopy(m)
    trace=[];original=m.flow.step
    def step(s,u,dt):trace.append(dt);return original(s,u,dt)
    m.flow.step=step
    engine=enable_tensor_engine(other);engine.trace=[]
    loss,parts,belief,pred,_=window_loss(m,w);loss.backward()
    new,nparts,nbelief,npred,_=window_loss(other,w);new.backward()
    assert trace==engine.trace
    for a,b in zip([belief]+pred,[nbelief]+npred):
        for field in ('x','h','f','phi','actuator','variance','phase_quality','phase_covariance'):
            torch.testing.assert_close(getattr(a,field),getattr(b,field),atol=1e-11,rtol=1e-11)
    for key in parts:torch.testing.assert_close(parts[key],nparts[key],atol=1e-11,rtol=1e-11)
    for a,b in zip(m.parameters(),other.parameters()):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-10,rtol=1e-10)


def test_tensor_step_control_gradient():
    m,w=model_window(.02);engine=enable_tensor_engine(m);s=w.initial
    u=w.controls.values[0].clone().requires_grad_();dt=.003
    old=m.flow.step(s,u,dt)
    new=engine.kernel.step(unpack(s,m.flow),u,*engine.prepare(dt,s.x))
    fields=('x','h','f','phi','actuator','variance','phase_quality','phase_covariance')
    for key,value in zip(fields,new):torch.testing.assert_close(getattr(old,key),value,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(old.x.sum()+old.h.sum(),u,retain_graph=True)[0]
    b=torch.autograd.grad(new[0].sum()+new[1].sum(),u)[0]
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('count',[4,8])
def test_tensor_blocks_keep_original_substeps(count):
    m,w=model_window(.04);engine=enable_tensor_engine(m);s=w.initial;u=w.controls.values[0];dt=.003
    original=s
    for _ in range(count):original=m.flow.step(original,u,dt)
    output=getattr(engine.kernel,f'block{count}')(unpack(s,m.flow),u,*engine.prepare(dt,s.x))
    for field,value in zip(('x','h','f','phi','actuator','variance','phase_quality','phase_covariance'),output):
        torch.testing.assert_close(getattr(original,field),value,atol=1e-11,rtol=1e-11)


def test_tensor_step_batched_inputs():
    from dataclasses import fields,replace
    m,w=model_window(.02);engine=enable_tensor_engine(m)
    s=replace(w.initial,**{f.name:torch.stack([getattr(w.initial,f.name)]*2) for f in fields(w.initial) if isinstance(getattr(w.initial,f.name),torch.Tensor)})
    u=torch.stack([w.controls.values[0]]*2).requires_grad_();dt=.002
    ref=m.flow.step(s,u,dt);result=engine.kernel.step(unpack(s,m.flow),u,*engine.prepare(dt,s.x))
    for field,value in zip(('x','h','f','phi','actuator','variance','phase_quality','phase_covariance'),result):
        torch.testing.assert_close(getattr(ref,field),value,atol=1e-11,rtol=1e-11)
