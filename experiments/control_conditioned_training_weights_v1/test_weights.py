import numpy as np
import pytest
import torch
from weighting import compute_weights,coordinates,canonical,ess
from weighted_loss import weighted_loss,original_window_loss

def control(delta):
 return np.array([.5,-delta,delta,0.])
def run(delta=.1,split=False):
 t=np.array([0,100000,1000000]);u=np.array([control(0),control(delta)])
 if split:t=np.array([0,50000,100000,130000,800000,1000000]);u=np.array([control(0),control(0),control(delta),control(delta),control(delta)])
 out=np.arange(20000,1000001,20000)
 return compute_weights(t,u,out,np.ones(len(out),bool))

def test_zoh_redundancy_and_sign():
 a=run();b=run(split=True);c=run(-.1)
 for key in ['w1','w2','exposure_us','w1_supported','w2_supported']:
  np.testing.assert_array_equal(a[key],b[key]);np.testing.assert_array_equal(a[key],c[key])
 assert a['delta_d']==pytest.approx(2*.1/3)
 assert a['exposure_us'][9]==100000
 assert a['w2'][3]==1

def test_coordinates_and_zero():
 np.testing.assert_allclose(coordinates([.4,.2,.6,-.1]),[.4,.4,.2,-.1])
 a=run(0);assert a['w1']==1 and np.all(a['w2']==1)

def test_no_later_control_access():
 t=np.array([0,100000,200000,1000000]);u=np.array([control(0),control(.1),control(.8)])
 x=compute_weights(t,u,[100000,150000,200000],[True]*3)
 u[-1]=np.nan;y=compute_weights(t,u,[100000,150000,200000],[True]*3)
 np.testing.assert_array_equal(x['w2'],y['w2']);assert y['w1']==1
 # No exposure before the decision, and no extrapolation after valid prefix.
 z=compute_weights(t,u,[-100,150000,250000,1200000],[True]*4)
 assert z['w2'][0]==z['w2'][2]==z['w2'][3]==1

def test_support_padding_and_ineligibility():
 a=compute_weights([0,100000,1000000],[control(0),control(.1)],[200000,400000,700000],[True,True,False],supported_until=300000)
 assert a['w1']>1 and a['w2'][1]==a['w2'][2]==1
 b=compute_weights([0,100000,1000000],[control(0),control(.1)],[200000],[True],auto_stable=False)
 assert b['w1']==1 and b['w2'][0]==1
 c=compute_weights([0,1000000],[control(np.nan)],[200000],[True]);assert c['w1']==c['w2'][0]==1

def tensors(dtype=torch.float64):
 gen=torch.Generator().manual_seed(17)
 p=torch.randn(4,7,13,generator=gen,dtype=dtype);y=torch.randn(4,7,13,generator=gen,dtype=dtype)
 p[:,:,6:10]/=torch.linalg.vector_norm(p[:,:,6:10],dim=-1,keepdim=True);y[:,:,6:10]/=torch.linalg.vector_norm(y[:,:,6:10],dim=-1,keepdim=True)
 mask=torch.arange(7)[None,:]<torch.tensor([7,5,3,6])[:,None]
 return p.requires_grad_(),y,mask,torch.tensor([.3,.4,.2,.6]),torch.tensor([1.,1.,1.,1.])

@pytest.mark.parametrize('scheme',['W0','W1','W2'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_exact_values_and_gradients(scheme,dtype):
 p,y,m,s,c=tensors(dtype);old=original_window_loss(p,y,m,s,c).mean();g=torch.autograd.grad(old,p)[0]
 new=weighted_loss(p,y,m,s,c,scheme=scheme,window_weight=torch.ones(4),step_weight=torch.ones(4,7));gn=torch.autograd.grad(new,p)[0]
 assert torch.equal(old,new) and torch.equal(g,gn)

def independent_errors(p,y):
 # Independent geometric implementation, not calling original errors.
 dot=(p[...,6:10]*y[...,6:10]).sum(-1).abs().clamp(max=1-1e-7)
 return torch.stack([((p[...,sl]-y[...,sl])**2).sum(-1).sqrt() for sl in [slice(0,3),slice(3,6)]]+[2*torch.acos(dot),((p[...,10:13]-y[...,10:13])**2).sum(-1).sqrt()],-1)

@pytest.mark.parametrize('scheme',['W1','W2'])
def test_weighted_formula(scheme):
 p,y,m,s,c=tensors();sw=torch.linspace(1,2,28,dtype=p.dtype).reshape(4,7);ww=torch.tensor([1.,1.2,1.7,2.],dtype=p.dtype)
 values=(independent_errors(p,y)/s).square().sum(-1)
 expected=[]
 for i in range(4):
  selected=values[i,m[i]];weights=sw[i,m[i]] if scheme=='W2' else torch.ones_like(selected)
  expected.append((selected*weights).sum()/weights.sum())
 expected=torch.stack(expected);expected=expected.mean() if scheme=='W2' else (expected*ww).sum()/ww.sum()
 actual=weighted_loss(p,y,m,s,c,scheme=scheme,window_weight=ww,step_weight=sw)
 torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)

def test_padding_never_contributes():
 p,y,m,s,c=tensors();sw=torch.ones(4,7);sw[m]=1.5
 original=weighted_loss(p,y,m,s,c,scheme='W2',step_weight=sw)
 pp=p.detach().clone();yy=y.clone();pp[~m]=float('nan');yy[~m]=float('nan');sw[~m]=float('nan');pp.requires_grad_()
 actual=weighted_loss(pp,yy,m,s,c,scheme='W2',step_weight=sw)
 assert torch.equal(actual,original);g=torch.autograd.grad(actual,pp)[0];assert torch.isfinite(g).all() and (g[~m]==0).all()

def test_ess_bounds():
 w=np.r_[np.ones(50),np.ones(50)*2];assert ess(w)==pytest.approx(90)
 # Any [1,2] weight distribution has ESS/n >= 8/9 > .85.
 rng=np.random.default_rng(17);assert np.min(ess(1+rng.random((100,51)),axis=1)/51)>.85

def test_exact_threshold_and_freshness_prefix():
 from analyze import cut
 t=np.array([0,100000,500000]);u=np.array([control(0),control(.05),control(.8)]);age=np.array([0,0,0])
 tt,uu,support=cut((t,u,age),0,700000,700000)
 assert support==260000
 z=compute_weights(tt,uu,[99999,100000,200000,260000,260001],[True]*5,supported_until=support)
 assert z['w1']==1 and z['w2'][0]==z['w2'][1]==1 and z['w2'][2]==2 and z['w2'][4]==1

def test_batch_permutation_and_detached_weights():
 p,y,m,s,c=tensors();sw=torch.ones(4,7,requires_grad=True)*1.5;ww=torch.tensor([1.,1.2,1.7,2.],requires_grad=True)
 for scheme in ['W0','W1','W2']:
  result=weighted_loss(p,y,m,s,c,scheme=scheme,window_weight=ww,step_weight=sw)
  ix=torch.tensor([2,0,3,1]);permuted=weighted_loss(p[ix],y[ix],m[ix],s,c,scheme=scheme,window_weight=ww[ix],step_weight=sw[ix])
  torch.testing.assert_close(result,permuted)
  gw=torch.autograd.grad(result,ww,allow_unused=True,retain_graph=True)[0];assert gw is None

def test_actual_array_identity_and_folds():
 from pathlib import Path
 import json
 root=Path(__file__).resolve().parent;results=root/'results';a=np.load(results/'full_train/weights.npz')
 assert len(a['w1'])==107920 and len(set(a['flight_id']))==94
 for k,n in enumerate([87239,95627,87750]):
  b=np.load(results/f'fold{k}/weights.npz');assert len(b['w1'])==n
  for key in ['w1','w2','future_mask','window_id','flight_id','output_us']:np.testing.assert_array_equal(b[key],a[key][b['registered_index']])
  iso=json.loads((results/f'fold{k}/isolation.json').read_text());assert not set(iso['train_flights'])&set(iso['held_flights'])
 receipt=json.loads((results/'source_receipt.json').read_text());assert receipt['partitions_read']==['train'] and receipt['npz_members_read']==['dt','future_mask'];assert not receipt['future_targets_read'] and not receipt['test_opened']
 assert not any('target' in c for c in receipt['sample_columns_read'])

def test_weighting_api_excludes_motion_and_errors():
 import inspect
 assert set(inspect.signature(compute_weights).parameters)=={'t','u','outputs','mask','auto_stable','supported_until'}
 from pathlib import Path
 import json
 cfg=json.loads((Path(__file__).resolve().parent/'config.json').read_text())
 assert cfg['w1']==dict(horizon_us=300000,lower=.02,span=.08,max_weight=2)
 assert cfg['w2']==dict(lookback_us=300000,threshold=.05,saturation_us=100000,max_weight=2)
