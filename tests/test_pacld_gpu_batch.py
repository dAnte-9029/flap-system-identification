"""Mathematical contracts for the GPU execution path, CPU-capable FP64 checks."""
from dataclasses import replace
import copy
import torch
from system_identification.models.pacld import PACLD,Config
from system_identification.models.pacld.batch import BatchedPACLD,FIELDS,covariance
from system_identification.models.pahbd import Config as TruthConfig
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld import window_objective

def windows():
    a=controlled_periodic_window(TruthConfig(),.2)[0]
    b=controlled_periodic_window(TruthConfig(),.5)[0]
    # Different event counts and control boundaries, with explicit Hall anchoring.
    e=a.history[0]
    mask=torch.zeros_like(e.mask);mask[13]=1
    hall=replace(e,source='wing_phase',mask=mask)
    a=replace(a,history=[hall]+a.history,initial=replace(a.initial,phase_quality=torch.zeros(1,dtype=torch.float64)))
    return [a,b]

def test_batch_reference_loss_belief_gradient():
    torch.set_num_threads(1);torch.manual_seed(17)
    cfg=Config(width=8);ref=PACLD(cfg).double();model=BatchedPACLD(cfg);model.load_state_dict(ref.state_dict());ws=windows()
    batch=compile_windows(ws,cfg);r=model(batch,.1);r['total'].backward()
    losses=[]
    for i,w in enumerate(ws):
        old=window_objective(ref,w,.1,diagnostic_base=True);losses.append(old['total'].detach());(old['total']/len(ws)).backward()
        for k in FIELDS:torch.testing.assert_close(getattr(r['belief'],k)[i],getattr(old['belief'],k),rtol=1e-8,atol=2e-10)
        for mode,offset in [('FULL',0),('BASE',len(ws))]:
            torch.testing.assert_close(r['predictions'][i+offset,:len(old['times'])],torch.stack([s.x for s in old['predictions'][mode]]),rtol=1e-8,atol=2e-10)
    torch.testing.assert_close(r['total'],torch.stack(losses).mean(),rtol=1e-8,atol=2e-10)
    for p,q in zip(ref.parameters(),model.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-7,atol=2e-9)
    with torch.no_grad():
        alone=model(compile_windows([ws[0]],cfg),.1)
        torch.testing.assert_close(alone['full'][0],r['full'][0],rtol=1e-9,atol=1e-10)
        # Future publication is excluded before inspecting the value (including invalid sentinel).
        future=replace(ws[0].history[0],availability=.01,value=torch.full((15,),float('nan')))
        changed=replace(ws[0],history=ws[0].history+[future])
        noleak=model(compile_windows([changed],cfg),.1)
        torch.testing.assert_close(alone['predictions'],noleak['predictions'],rtol=0,atol=0)

def test_analytic_covariance():
    cfg=Config();m=PACLD(cfg).double();dt=torch.tensor([[0.],[1e-7],[.001],[.005]],dtype=torch.float64)
    cov=torch.tensor([[.2,.03],[.03,.1]],dtype=torch.float64).expand(4,2,2)
    got=covariance(cov,dt,m.flow.phase.coefficients[2],cfg.frequency_noise_psd,cfg.phase_noise_psd)
    noise=torch.diag(torch.tensor([cfg.frequency_noise_psd,cfg.phase_noise_psd],dtype=torch.float64))
    expected=torch.stack([m.flow.phase.covariance(cov[i],float(t),noise) for i,t in enumerate(dt)])
    torch.testing.assert_close(got,expected,rtol=1e-12,atol=1e-14)

def test_bucket_resume_preserves_pending_and_unique_global_batch():
    import pandas as pd
    from system_identification.training.pacld_gpu import BucketSampler
    index=pd.DataFrame([dict(window_id=str(i),flight_id=str(i%3),horizon_us=200000 if i<20 else 500000,split='train') for i in range(40)])
    a=BucketSampler(index,17,8)
    for _ in range(3):assert len(set(a.draw()))==8
    b=BucketSampler(index,17,8);b.load_state_dict(a.state_dict())
    for _ in range(5):assert a.draw()==b.draw()

def test_checkpointed_steps_preserve_gradients():
    torch.set_num_threads(1);torch.manual_seed(17);cfg=Config(width=8)
    a=BatchedPACLD(cfg);b=BatchedPACLD(cfg,checkpoint_steps=True);b.load_state_dict(a.state_dict())
    batch=compile_windows(windows()[:1],cfg)
    ra=a(batch,.1);rb=b(batch,.1);ra['total'].backward();rb['total'].backward()
    torch.testing.assert_close(ra['total'],rb['total'],rtol=0,atol=0)
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
