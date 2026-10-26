from common import *
import torch, pytest
from model import make_model, coordinates
from system_identification.models.velocity_rate_gru import integrate, qmul, qexp, unit

@pytest.fixture
def inputs():
    a=np.load(SWAP/'results/inputs.npz')
    return {k:torch.from_numpy(a[k][:4].copy()) for k in training.INPUT_KEYS}
@pytest.fixture
def norm():return json.loads((HERE/'normalization.json').read_text())

@pytest.mark.parametrize('name',['M1','M2'])
def test_zero_deviation_and_branch_isolation(name,norm,inputs):
    m=make_model(name,norm);last=inputs['history'][torch.arange(4),inputs['history_length']-1]
    ev=inputs['events'].clone();ev[...,:4]=last[:,None,None,13:17]
    moment=m.control_moments(ev,inputs['event_length'],inputs['dt'],last[:,13:17])
    assert torch.count_nonzero(moment)==0
    h=torch.randn(4,64);s=inputs['initial'];phase=torch.tensor([[.3,.9,3.5]]).repeat(4,1);mask=torch.ones(4,2);dt=inputs['dt'][:,0]
    a=m.branch(h,s,phase,mask,dt,torch.zeros(4,8));b=m.branch(h,s,phase,mask,dt,torch.ones(4,8))
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0);torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
    assert torch.count_nonzero(a[2])==0 and b[2].abs().max()>1e-6
    b[2].square().sum().backward();assert m.effect_head.weight.grad.norm()>0 and torch.isfinite(m.effect_head.weight.grad).all()

@pytest.mark.parametrize('name',['M0','M1','M2'])
def test_shape_causality_batch_and_kinematics(name,norm,inputs):
    torch.manual_seed(17);m=make_model(name,norm).eval()
    with torch.no_grad():
        y=m(inputs);assert y.shape==(*inputs['dt'].shape,13) and torch.isfinite(y).all()
        poison=dict(inputs,target=torch.full_like(y,float('nan')),future_phase=torch.full_like(inputs['dt'],float('nan')))
        torch.testing.assert_close(y,m(poison),rtol=0,atol=0)
        single=torch.cat([m({k:v[i:i+1] for k,v in inputs.items()}) for i in range(4)])
        torch.testing.assert_close(y,single,rtol=2e-5,atol=2e-6)
        previous=torch.cat((inputs['initial'][:,None],y[:,:-1]),1)
        position=previous[...,:3]+.5*(previous[...,3:6]+y[...,3:6])*inputs['dt'][...,None]
        attitude=unit(qmul(unit(previous[...,6:10]),qexp(.5*(previous[...,10:13]+y[...,10:13])*inputs['dt'][...,None])))
        torch.testing.assert_close(y[...,:3],position);torch.testing.assert_close(y[...,6:10],attitude)
        torch.testing.assert_close(y[...,6:10].norm(dim=-1),torch.ones_like(inputs['dt']),atol=2e-7,rtol=1e-6)
        cutoff=6;changed={k:v.clone() for k,v in inputs.items()};changed['events'][:,cutoff:,:,:4]+=1
        torch.testing.assert_close(y[:,:cutoff],m(changed)[:,:cutoff],rtol=0,atol=0)

@pytest.mark.parametrize('name',['M1','M2'])
def test_zoh_split_timing_mask(name,norm,inputs):
    m=make_model(name,norm).eval()
    refine=module(ROOT/'experiments/control_event_canonicalization_v1/run.py','canonical_test').refine
    events,lengths=refine(inputs['events'].numpy(),inputs['event_length'].numpy(),4)
    changed=dict(inputs,events=torch.from_numpy(events),event_length=torch.from_numpy(lengths))
    with torch.no_grad():torch.testing.assert_close(m(inputs),m(changed),rtol=0,atol=0)
    # Equal-average commands occurring early vs late must have distinct first moments.
    ev=torch.tensor([[[[.2,.3,.7,.1,.01],[.2,.7,.3,.1,.01]]]])
    dt=torch.tensor([[.02]]);le=torch.tensor([[2]]);ref=torch.tensor([[.2,.5,.5,.1]])
    a=m.control_moments(ev,le,dt,ref);b=m.control_moments(ev.flip(-2),le,dt,ref)
    torch.testing.assert_close(a[...,:4],b[...,:4]);assert (a[...,4:]-b[...,4:]).abs().max()>.01
    x={k:v.clone() for k,v in inputs.items()};x['history_mask'][...,17:20]=False
    changed={k:v.clone() for k,v in x.items()};changed['history'][...,17:20]=123
    with torch.no_grad():torch.testing.assert_close(m(x),m(changed),rtol=0,atol=0)

def test_coordinates_and_parameter_budget(norm):
    raw=torch.randn(20,4);z=coordinates(raw)
    torch.testing.assert_close(torch.stack((z[:,0],z[:,1]-z[:,2],z[:,1]+z[:,2],z[:,3]),-1),raw)
    counts=[sum(p.numel() for p in make_model(n,norm).parameters()) for n in ['M0','M1','M2']]
    assert counts[0]==37590 and all(abs(x/counts[0]-1)<.25 for x in counts)

def test_source_boundary():
    r=json.loads((OUT/'source_receipt.json').read_text());assert not r['test_read'] and not r['repacked']
    assert not any('_test.parquet' in p for p in r['files'])
    assert r['train_windows']==107920 and r['validation_windows']==36332

@pytest.mark.skipif(not (HERE/'training/M2/best.pt').exists(),reason='Post-training regression only')
@pytest.mark.parametrize('name',['M0','M1','M2'])
def test_trained_zoh_and_control_effect(name,inputs):
    ck=torch.load(HERE/'training'/name/'best.pt',map_location='cpu',weights_only=False)
    m=make_model(name,ck['normalization']);m.load_state_dict(ck['model']);m.eval()
    refine=module(ROOT/'experiments/control_event_canonicalization_v1/run.py','trained_canonical_test').refine
    ev,le=refine(inputs['events'].numpy(),inputs['event_length'].numpy(),4)
    with torch.no_grad():
        y=m(inputs);torch.testing.assert_close(y,m(dict(inputs,events=torch.from_numpy(ev),event_length=torch.from_numpy(le))),rtol=0,atol=0)
        assert (y[0]-y[1]).abs().max()>1e-5
    if name!='M0':
        m(inputs)[...,[3,4,5,10,11,12]].square().mean().backward()
        assert m.effect_head.weight.grad.norm()>0 and torch.isfinite(m.effect_head.weight.grad).all()
