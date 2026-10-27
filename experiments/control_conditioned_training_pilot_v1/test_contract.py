from common import *
import torch,pytest
@pytest.mark.parametrize('fold',[0,1,2])
def test_fold_isolation_and_normalization(fold):
    plan=json.loads((OUT/'holdout_protocol.json').read_text())['folds'][fold];receipt=json.loads((HERE/'folds'/f'fold{fold}'/'normalization_receipt.json').read_text())
    assert receipt['passed'] and not receipt['validation_used'] and not receipt['test_read']
    assert set(plan['train_flights']).isdisjoint(plan['held_flights'])
    assert {x.split(':')[0] for x in plan['train_flights']}.isdisjoint(x.split(':')[0] for x in plan['held_flights'])
    assert sha(HERE/'folds'/f'fold{fold}'/'normalization.json')==receipt['normalization_sha256']
    assert max(receipt['independent_float64_max_errors'].values())<2e-5

@pytest.mark.parametrize('name',['M0W0','M1W0'])
def test_frozen_interface_causality_zoh_gradient(name):
    torch.set_num_threads(4);torch.manual_seed(17);norm=json.loads((HERE/'folds/fold0/normalization.json').read_text());m=make_model(name,norm)
    raw=np.load(SWAP/'results/inputs.npz');x={k:torch.from_numpy(raw[k][:4].copy()) for k in training.INPUT_KEYS};y=m(x)
    assert y.shape==(*x['dt'].shape,13)
    torch.testing.assert_close(y,m(dict(x,target=torch.full_like(y,float('nan')),future_phase=torch.full_like(x['dt'],float('nan')))),rtol=0,atol=0)
    refine=module(ROOT/'experiments/control_event_canonicalization_v1/run.py','canonical_test').refine;ev,le=refine(x['events'].numpy(),x['event_length'].numpy(),4)
    torch.testing.assert_close(y,m(dict(x,events=torch.from_numpy(ev),event_length=torch.from_numpy(le))),rtol=0,atol=0)
    y.square().mean().backward()
    if name.startswith('M1'):
        assert m.effect_head.weight.grad.norm()>0 and torch.isfinite(m.effect_head.weight.grad).all()
        last=x['history'][torch.arange(4),x['history_length']-1];events=x['events'].clone();events[...,:4]=last[:,None,None,13:17]
        assert torch.count_nonzero(m.control_moments(events,x['event_length'],x['dt'],last[:,13:17]))==0

def test_main_membership_exactly_once():
    assignments=pd.read_csv(OUT/'holdout_pair_assignments.csv');assert len(assignments)==assignments.pair_uid.nunique()==48
    assert assignments.groupby('fold').size().tolist()==[25,12,11]
    assert assignments.groupby('fold').component_key.nunique().tolist()==[1,4,6]

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_uniform_loss_gradient_and_weighted_formula(dtype):
    torch.manual_seed(17);p=torch.randn(7,11,13,dtype=dtype,requires_grad=True);y=torch.randn_like(p);mask=torch.arange(11)[None,:]<torch.arange(5,12)[:,None];scales=torch.tensor([.3,.4,.5,.6],dtype=dtype);components=torch.ones(4,dtype=dtype)
    orig=training.window_loss(p,y,mask,scales,components).mean();og=torch.autograd.grad(orig,p)[0]
    for scheme in ['W0','W1']:
        new=weighted.weighted_loss(p,y,mask,scales,components,scheme=scheme,window_weight=torch.ones(7,dtype=dtype));ng=torch.autograd.grad(new,p)[0]
        assert torch.equal(orig,new) and torch.equal(og,ng)
    w=torch.linspace(1,2,7,dtype=torch.float64);expected=(training.window_loss(p,y,mask,scales,components)*w).sum()/w.sum()
    actual=weighted.weighted_loss(p,y,mask,scales,components,scheme='W1',window_weight=w)
    assert torch.equal(expected,actual)

def test_true_shuffled_batch_weight_identity():
    records=json.loads((OUT/'batch_identity_checks.json').read_text());assert records['passed'] and len(records['checks'])==27
    meta=pd.read_parquet(TRAIN/'train_windows.parquet');indices=np.load(OUT/'holdout_window_indices.npz')
    for fold in range(3):
        a=np.load(WEIGHTS/f'results/fold{fold}/weights.npz');ids=indices[f'fold{fold}_train_indices'];assert np.array_equal(ids,a['registered_index'])
        lookup=dict(zip(a['window_id'],a['w1']));order=np.random.default_rng(29).permutation(len(ids))[:1024]
        labels=meta.iloc[ids[order]].window_id.astype(str);np.testing.assert_array_equal(a['w1'][order],[lookup[x] for x in labels])
        assert a['w1'].min()>=1 and a['w1'].max()<=2
