from common import *
import torch,pytest

@pytest.mark.parametrize('split',['train','validation'])
def test_targets_identity_prefix(split):
 raw=load_numpy(split);pack=target_pack(split);meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');d=with_targets(raw,pack)
 np.testing.assert_array_equal(pack['window_id'],meta.window_id);np.testing.assert_array_equal(pack['flight_id'],meta.flight_id);np.testing.assert_array_equal(pack['registered_index'],meta.index)
 for k in training.INPUT_KEYS:assert d[k] is raw[k]
 np.testing.assert_array_equal(d['target_A'][...,:10],d['target_B'][...,:10]);mask=d['loss_mask'];assert mask.any(1).all() and np.all(~mask|raw['future_mask']) and not np.any(mask[:,1:]&~mask[:,:-1]);assert np.isfinite(d['target_B']).all()
 prev=pd.read_csv(CAUSAL/'results/windows.csv.gz').query('split==@split').sort_values('registered_index');np.testing.assert_array_equal(mask.sum(1),prev.common_prefix_steps)
 # Finite A placeholders outside support are equal and never assigned B values.
 np.testing.assert_array_equal(d['target_A'][~mask],d['target_B'][~mask])

@pytest.mark.parametrize('fold',[0,1,2])
def test_fold_and_scale(fold):
 plan=json.loads((OUT/'holdout_protocol.json').read_text())['folds'][fold];receipt=json.loads((HERE/'folds'/f'fold{fold}'/'normalization_receipt.json').read_text());assert receipt['passed'] and not receipt['validation_used'] and not receipt['test_read'];assert sha(HERE/'folds'/f'fold{fold}'/'normalization.json')==receipt['normalization_sha256']
 for a,b in [('train_flights','held_flights'),('train_flights','validation_flights'),('held_flights','validation_flights')]:assert not {x.split(':')[0] for x in plan[a]}&{x.split(':')[0] for x in plan[b]}

def test_loss_mask_and_gradient():
 torch.manual_seed(17);p=torch.randn(3,7,13,requires_grad=True);y=torch.randn_like(p);mask=torch.arange(7)[None,:]<torch.tensor([3,5,7])[:,None];scales=torch.ones(4);w=torch.ones(4)
 loss=training.window_loss(p,y,mask,scales,w).mean();g=torch.autograd.grad(loss,p)[0];assert torch.count_nonzero(g[~mask])==0
 y2=y.clone();y2[~mask]=100;loss2=training.window_loss(p,y2,mask,scales,w).mean();g2=torch.autograd.grad(loss2,p)[0];assert torch.equal(loss,loss2) and torch.equal(g,g2)

def test_causality_integrator_zoh():
 from system_identification.models.velocity_rate_gru import integrate
 torch.set_num_threads(4);training.set_seed(17);norm=json.loads((HERE/'folds/fold0/normalization.json').read_text());m=make_model('M0A',norm);z=np.load(SWAP/'results/inputs.npz');x={k:torch.from_numpy(z[k][:3].copy()) for k in training.INPUT_KEYS};y=m(x);assert y.shape==(*x['dt'].shape,13)
 torch.testing.assert_close(y,m(dict(x,target=torch.full_like(y,float('nan')),future_phase=torch.full_like(x['dt'],float('nan')),loss_mask=torch.zeros_like(x['dt']))),rtol=0,atol=0)
 first=integrate(x['initial'],y[:,0,3:6],y[:,0,10:13],x['dt'][:,0]);torch.testing.assert_close(y[:,0],first,rtol=0,atol=0)
 refine=module(ROOT/'experiments/control_event_canonicalization_v1/run.py','refine_test').refine;ev,le=refine(x['events'].numpy(),x['event_length'].numpy(),4);torch.testing.assert_close(y,m(dict(x,events=torch.from_numpy(ev),event_length=torch.from_numpy(le))),rtol=0,atol=0)
 training.set_seed(17);m2=make_model('M0B',norm)
 for k,v in m.state_dict().items():assert torch.equal(v,m2.state_dict()[k])

def test_frozen_members():
 a=pd.read_csv(OUT/'holdout_pair_assignments.csv');assert len(a)==a.pair_uid.nunique()==48;assert a.component_key.nunique()==11;assert a.groupby('fold').size().tolist()==[25,12,11]
