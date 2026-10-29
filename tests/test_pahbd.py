"""Targeted PA-HBD numerical/causality tests; all CPU, no performance comparison."""
import copy
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from system_identification.models.pahbd import PAHBD,Config,ControlTape,Observation
from system_identification.models.pahbd import geometry as so3
from system_identification.models.pahbd.dynamics import PhasePropagator,ControlMixer
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import window_loss,train_steps,save_checkpoint,load_checkpoint,fit_normalization
from system_identification.evaluation.pahbd import periodic_amplitude_phase,control_sensitivity,matched_control_response,section_transitions

torch.set_num_threads(1)


def fixture(horizon=.25):
    torch.manual_seed(17);cfg=Config(width=32,latent=8)
    model=PAHBD(cfg).double();window,truth=controlled_periodic_window(cfg,horizon)
    return model,window,truth


def test_complete_autonomous_gradients_and_checkpoint(tmp_path):
    m,w,_=fixture();loss,parts,b,p,t=window_loss(m,w)
    assert torch.isfinite(loss) and len(p)==len(t)
    loss.backward()
    for name,parameter in m.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
    for module in (m.belief,m.flow.slow.encoder,m.flow.slow.memory,m.flow.mixer.mean,m.flow.mixer.forcing,m.flow.mixer.control):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters())>0
    assert all(torch.isfinite(s.x).all() and torch.allclose(s.x[6:10].norm(),torch.tensor(1.,dtype=torch.float64),atol=1e-12) for s in p)
    assert all(s.h.abs().max()<=m.config.latent_bound for s in p)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    original=m.flow.mixer.mean.weight.detach().clone()
    records=train_steps(m,[w],opt,1)
    assert records[0]['gradient_norm']>0 and not torch.equal(original,m.flow.mixer.mean.weight)
    save_checkpoint(tmp_path/'model.pt',m,opt,{'split':'train'},1)
    restored,_,_=load_checkpoint(tmp_path/'model.pt',optimizer=True)
    assert torch.equal(m(w)[1][-1].x,restored(w)[1][-1].x)
    with pytest.raises(FileExistsError):save_checkpoint(tmp_path/'model.pt',m,opt,{},1)


def test_lie_geometry_identity_antipodes_and_gradcheck():
    q=so3.exp(torch.tensor([.2,-.3,.1],dtype=torch.float64))
    assert torch.allclose(so3.residual(q,-q),torch.zeros(3,dtype=q.dtype),atol=1e-14)
    r=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda r:so3.log(so3.exp(r)),(r,))
    assert torch.autograd.gradcheck(lambda r:so3.rotate(so3.exp(r),torch.tensor([1.,2.,3.],dtype=r.dtype)),(r,))


def test_exact_phase_and_covariance():
    p=PhasePropagator(Config().frequency_coefficients)
    f=torch.tensor([4.],dtype=torch.float64,requires_grad=True);phi=f*0;u=f*0+.6
    a,b=p(f,phi,u,.071);a,b=p(a,b,u,.109)
    c,d=p(f,phi,u,.18)
    assert torch.allclose(a,c,atol=1e-13) and torch.allclose(b,d,atol=1e-13)
    d.sum().backward();assert torch.isfinite(f.grad).all()
    z=PhasePropagator((1.,2.,0.));a,b=z(f.detach(),phi.detach(),u.detach(),.2)
    assert torch.allclose(a,torch.tensor([4.44],dtype=a.dtype))
    cov=p.covariance(torch.eye(2,dtype=f.dtype),.2)
    assert torch.linalg.eigvalsh(cov).min()>0


def test_zoh_equivalent_splits_nonuniform_and_backward_guard():
    m,w,_=fixture();s=w.initial
    constant=ControlTape([w.start],w.controls.values[:1])
    split=ControlTape([w.start,-.21,-.134,-.01],w.controls.values[:1].expand(4,-1).clone())
    a=m.flow.propagate(s,w.start,.123,constant);b=m.flow.propagate(s,w.start,.123,split)
    assert torch.equal(a.x,b.x) and torch.equal(a.phi,b.phi)
    c=m.flow.rollout(s,[-.19,-.083,.017,.123],constant,start=w.start)[-1]
    assert torch.allclose(a.x,c.x,atol=2e-6,rtol=1e-5)
    with pytest.raises(ValueError):m.flow.propagate(s,1.,0.,constant)


def test_no_future_observation_phase_frequency_or_target_leakage():
    m,w,_=fixture();a=m.encode(w)
    poisoned=copy.deepcopy(w)
    e=w.history[-1]
    poisoned.history.append(replace(e,availability=.001,value=torch.full_like(e.value,float('nan'))))
    poisoned.targets=[replace(t,value=torch.full_like(t.value,float('nan'))) for t in w.targets]
    b=m.encode(poisoned)
    for name in ('x','h','f','phi','variance'):assert torch.equal(getattr(a,name),getattr(b,name))
    # Measurement before decision but unavailable until after it must also be ignored.
    poisoned.history[-1]=replace(poisoned.history[-1],measurement=-.1)
    assert torch.equal(a.x,m.encode(poisoned).x)
    bad=replace(e,measurement=e.availability+1)
    with pytest.raises(ValueError):m.belief.update(a,bad,.01)


def test_masked_update_and_variance_bounds():
    m,w,_=fixture();e=w.history[0];s=w.initial
    zero=replace(e,mask=torch.zeros_like(e.mask))
    out=m.belief.update(s,zero,.01)
    assert torch.equal(out.h,s.h) and torch.allclose(out.x,s.x,atol=1e-14)
    for e in w.history:s=m.belief.update(s,e,.01)
    assert s.variance.min()>=m.config.variance_min and s.variance.max()<=m.config.variance_max


def test_phase_gauge_zero_mean_and_branch_isolation():
    m,w,_=fixture();s=w.initial;c,_=m.flow.slow(s)
    a,parts=m.flow.mixer(c,s.actuator,s.phi,s.phase_quality)
    delta=.713;other=copy.deepcopy(m)
    other.flow.mixer.gauge_offset.fill_(delta)
    b,_=other.flow.mixer(c,s.actuator,s.phi+delta,s.phase_quality)
    assert torch.allclose(a,b,atol=1e-12)
    s2=s.with_(phi=s.phi+delta)
    x=m.flow.propagate(s,w.start,.1,w.controls)
    y=other.flow.propagate(s2,w.start,.1,w.controls)
    assert torch.allclose(x.x,y.x,atol=1e-12)
    phases=torch.arange(128,dtype=torch.float64)[:,None]*2*math.pi/128
    _,parts=m.flow.mixer(c.expand(128,-1),s.actuator.expand(128,-1),phases,s.phase_quality.expand(128,-1))
    assert parts['forcing'].mean(0).abs().max()<1e-12
    assert parts['interaction'].mean(0).abs().max()<1e-12
    assert torch.allclose(parts['mean'][0],parts['mean'][-1],atol=1e-14,rtol=1e-14)
    _,changed=m.flow.mixer(c,s.actuator+.2,s.phi,s.phase_quality)
    _,original=m.flow.mixer(c,s.actuator,s.phi,s.phase_quality)
    assert torch.equal(changed['forcing'],original['forcing'])


def test_ab_only_explicit_interaction_and_disabled_phase():
    a,w,_=fixture();b=PAHBD(replace(a.config,mode='B')).double();b.load_state_dict(a.state_dict())
    s=w.initial;c,_=a.flow.slow(s)
    da,parts=a.flow.mixer(c,s.actuator,s.phi,s.phase_quality)
    db,pb=b.flow.mixer(c,s.actuator,s.phi,s.phase_quality)
    assert torch.allclose(da-db,parts['interaction'],atol=1e-12)
    assert torch.equal(parts['mean'],pb['mean']) and torch.equal(parts['forcing'],pb['forcing'])
    assert parts['interaction'].abs().sum()>0 and pb['interaction'].abs().sum()==0
    za,_=a.flow.mixer(c,s.actuator,s.phi,s.phase_quality*0)
    zb,_=b.flow.mixer(c,s.actuator,s.phi+7,s.phase_quality*0)
    assert torch.equal(za,zb)


def test_independent_controlled_truth_integrator_convergence():
    m,w,truth=fixture(.45)
    # Same controlled periodic ODE as independent NumPy/DOP853 truth; no NN label fit.
    def derivative(s):
        p=s.phi[0];a=s.actuator
        body=torch.stack((.2+.5*a[0]+.4*torch.cos(p),.3*a[1]+.1*a[2]*torch.sin(p),
                          -.0-9.80665+.25*torch.sin(p)+.3*a[2]))
        alpha=torch.stack((.2*torch.sin(p)+.6*a[0]*torch.cos(p),.1*a[1],.2*a[2]))
        g=body.new_tensor([0,0,9.80665])
        return so3.rotate(s.x[6:10],body)+g,alpha,torch.zeros_like(s.h)
    m.flow.derivative=derivative
    errors=[]
    for step in (.01,.005,.0025):
        m.config.max_step=step;m.config.phase_step=100.
        pred=m.flow.propagate(w.initial,w.start,.45,w.controls)
        errors.append(float(torch.linalg.vector_norm(pred.x-torch.tensor(truth(.45)[:13]))))
    assert errors[1]<errors[0]*.35 and errors[2]<errors[1]*.35
    assert errors[-1]<2e-5
    print('independent_DOP853_errors',errors)


def test_evaluation_interfaces_and_train_only_normalization():
    m,w,_=fixture();s=m.encode(w)
    jac=control_sensitivity(m,s);assert jac.shape==(6,3) and torch.isfinite(jac).all()
    times=[.02,.1,.2]
    response=matched_control_response(m,s,times,w.controls,w.controls)
    assert response['model_contrast'].abs().max()<1e-12
    phase=torch.arange(100,dtype=torch.float64)*2*math.pi/100
    values=(2+3*torch.cos(phase-.4))[:,None]
    fit=periodic_amplitude_phase(phase,values,1)
    assert torch.allclose(fit['amplitude'],torch.tensor([[3.]],dtype=phase.dtype),atol=1e-12)
    assert torch.allclose(fit['phase'],torch.tensor([[.4]],dtype=phase.dtype),atol=1e-12)
    w.metadata['split']='validation'
    with pytest.raises(ValueError):fit_normalization(m,[w])
    with pytest.raises(ValueError):train_steps(m,[w],torch.optim.Adam(m.parameters()),1)


def test_batched_flow_dimensions():
    m,w,_=fixture();s=w.initial
    from dataclasses import fields
    s=s.with_(**{f.name:getattr(s,f.name).expand(2,-1).clone() for f in fields(s) if getattr(s,f.name) is not None})
    tape=ControlTape(w.controls.times,w.controls.values[:,None,:].expand(-1,2,-1))
    out=m.flow.propagate(s,w.start,.01,tape)
    assert out.x.shape==(2,13) and out.h.shape==(2,8)
    out.x.square().sum().backward()


def test_control_to_full_state_gradient_and_validation_interface():
    from system_identification.training.pahbd import validate
    m,w,_=fixture();s=m.encode(w)
    u=w.controls.values.detach().clone().requires_grad_()
    tape=ControlTape(w.controls.times,u)
    out=m.flow.rollout(s,[.1],tape)[0]
    objective=out.x.square().sum()+out.phi.square().sum()+out.h.square().sum()
    grad=torch.autograd.grad(objective,u)[0]
    assert torch.isfinite(grad).all() and grad[:,0].abs().sum()>0 and grad[:,1:].abs().sum()>0
    report=validate(m,[w]);assert report['windows'] and report['flight_equal']
    assert all(math.isfinite(row['rmse']) for row in report['flight_equal'])


def test_native_availability_masks_and_poisoned_future():
    from pathlib import Path
    from system_identification.data.pahbd import NativeWindows,DEFAULT_ROOT,configure_latent
    if not (Path(DEFAULT_ROOT)/'windows_train.parquet').exists():pytest.skip('local frozen train unavailable')
    d=NativeWindows();m,_,_=fixture();w=configure_latent(d[0],m.config)
    assert w.metadata['split']=='train' and all(e.availability<=0 for e in w.history)
    assert all(e.measurement<=e.availability for e in w.history)
    assert all(e.measurement>0 for e in w.targets)
    assert {e.source for e in w.history}==set(('vehicle_local_position','vehicle_attitude','vehicle_angular_velocity','wing_phase','flap_frequency'))
    assert all(t<=w.start for t in w.metadata['bootstrap_measurements'].values())
    a=m.encode(w)
    decision=w.metadata['decision_us']
    # Poison measurements that existed but were not published by the decision as well.
    for name,tab in d.tables.items():
        if name.startswith('actuator'):continue
        av=tab.get('availability_us',tab.timestamp_us)
        cols={'wing_phase':['phase_rad','encoder_total_count'],'flap_frequency':['frequency_hz'],
              'vehicle_local_position':['x','y','z','vx','vy','vz'],
              'vehicle_attitude':[f'q[{i}]' for i in range(4)],
              'vehicle_angular_velocity':[f'xyz[{i}]' for i in range(3)]}[name]
        tab.loc[av>decision,cols]=123456.
    poisoned=configure_latent(d[0],m.config);b=m.encode(poisoned)
    assert torch.equal(a.x,b.x) and torch.equal(a.phi,b.phi) and torch.equal(a.f,b.f)
    with pytest.raises(ValueError):NativeWindows(split='test')


def test_configured_third_harmonic_resolution():
    m,w,_=fixture();s=w.initial
    x=torch.zeros_like(s.x);x[6]=1
    s=s.with_(x=x,phi=s.phi*0)
    def derivative(s):
        acceleration=torch.stack((torch.cos(3*s.phi[0]),s.phi[0]*0,s.phi[0]*0))
        return acceleration,torch.zeros_like(s.x[10:]),torch.zeros_like(s.h)
    m.flow.derivative=derivative
    tape=ControlTape([0.],w.controls.values[:1]);t=.123
    expected=torch.tensor([(1-math.cos(30*math.pi*t))/(30*math.pi)**2,
                           math.sin(30*math.pi*t)/(30*math.pi)],dtype=torch.float64)
    errors=[]
    for limit in (.15,.075):
        m.config.phase_step=limit
        pred=m.flow.propagate(s,0.,t,tape)
        errors.append(float((pred.x[[0,3]]-expected).norm()))
    assert errors[0]<1e-5 and errors[1]<errors[0]*.3
    print('third_harmonic_phase_step_errors',errors)


def test_exact_autonomous_wingbeat_sections():
    from system_identification.evaluation.pahbd import wingbeat_transitions
    m,w,_=fixture(.45)
    # f=5 and the chosen fixed motor hold constant frequency despite tail events.
    s=w.initial.with_(phi=w.initial.phi*0)
    result=wingbeat_transitions(m,s,w.controls,count=2,max_time=.45)
    assert np.allclose(result['times'],[.2,.4],atol=1e-9)
    assert torch.allclose(result['phase'].flatten(),torch.tensor([2*math.pi,4*math.pi],dtype=torch.float64),atol=1e-8)
    assert result['transitions'].shape==(2,12)
