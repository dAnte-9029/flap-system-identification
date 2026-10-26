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

@pytest.mark.parametrize('name',['M0','M1','M2'])
def test_frozen_interface_causality_zoh_gradient(name):
    torch.set_num_threads(4);torch.manual_seed(17);norm=json.loads((HERE/'folds/fold0/normalization.json').read_text());m=make_model(name,norm)
    raw=np.load(SWAP/'results/inputs.npz');x={k:torch.from_numpy(raw[k][:4].copy()) for k in training.INPUT_KEYS};y=m(x)
    assert y.shape==(*x['dt'].shape,13)
    torch.testing.assert_close(y,m(dict(x,target=torch.full_like(y,float('nan')),future_phase=torch.full_like(x['dt'],float('nan')))),rtol=0,atol=0)
    refine=module(ROOT/'experiments/control_event_canonicalization_v1/run.py','canonical_test').refine;ev,le=refine(x['events'].numpy(),x['event_length'].numpy(),4)
    torch.testing.assert_close(y,m(dict(x,events=torch.from_numpy(ev),event_length=torch.from_numpy(le))),rtol=0,atol=0)
    y.square().mean().backward()
    if name!='M0':
        assert m.effect_head.weight.grad.norm()>0 and torch.isfinite(m.effect_head.weight.grad).all()
        last=x['history'][torch.arange(4),x['history_length']-1];events=x['events'].clone();events[...,:4]=last[:,None,None,13:17]
        assert torch.count_nonzero(m.control_moments(events,x['event_length'],x['dt'],last[:,13:17]))==0

def test_main_membership_exactly_once():
    assignments=pd.read_csv(OUT/'holdout_pair_assignments.csv');assert len(assignments)==assignments.pair_uid.nunique()==48
    assert assignments.groupby('fold').size().tolist()==[25,12,11]
    assert assignments.groupby('fold').component_key.nunique().tolist()==[1,4,6]
