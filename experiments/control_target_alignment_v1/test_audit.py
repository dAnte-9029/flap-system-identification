"""Numerical, identity, support and no-training acceptance tests."""
from common import *
from full_audit import sample,average
import pytest

def test_interpolation_no_extrapolation_and_gaps():
 t=np.array([0.,.01,.04,.05]);y=np.c_[t,2*t,3*t]
 v,ok,_,_=sample(t,y,np.array([-.001,0,.005,.025,.05,.051]))
 assert ok.tolist()==[False,True,True,False,True,False]
 np.testing.assert_allclose(v[2],[.005,.01,.015])
 with pytest.raises(AssertionError):sample(t[::-1],y[::-1],[.005])

def test_integral_constant_linear_and_edges():
 t=np.arange(101)*.01;y=np.c_[np.ones(len(t)),t,2*t]
 v,ok=average(t,y,np.array([.1,-.1,.8]),np.array([.37,.17,1.07]))
 assert ok.tolist()==[True,False,False];np.testing.assert_allclose(v[0],[1,.235,.47],atol=1e-14)
 np.testing.assert_allclose(tc.boxmean(t,y,[.235],.27)[0],v[0],atol=1e-14)

def test_gap_inside_mean_rejected():
 t=np.r_[np.arange(20)*.01,np.arange(30,51)*.01];y=np.ones((len(t),3))
 value,ok=average(t,y,np.array([.1]),np.array([.4]));assert not ok[0] and np.isnan(value).all()

def test_period_average_no_fitted_phase():
 t=np.linspace(0,.81,811);y=np.c_[np.sin(2*np.pi*t/.27)]
 np.testing.assert_allclose(tc.boxmean(t,y,[.27,.54],.27),0,atol=1e-14)

def test_initial_bridge_and_future_node_identity():
 t=np.array([0.,.02,.04]);a=np.array([[2.],[3.],[4.]]);initial=np.array([1.]);b=np.array([[.5],[2.],[3.]])
 oracle=a.copy();oracle[0]=initial
 at=np.array([.01,.02,.03]);ef=tc.interp(t,a,at)-tc.interp(t,b,at);eo=initial-b[0];bridge=tc.interp(t,oracle,at)-tc.interp(t,a,at)
 actual=tc.interp(t,oracle,at)-initial-(tc.interp(t,b,at)-b[0])
 np.testing.assert_allclose(actual,ef-eo+bridge);assert bridge[0,0]!=0 and bridge[1,0]==0 and bridge[2,0]==0

def test_oracle_reference_self_consistency():
 d=pd.read_csv(OUT/'oracle_points.csv.gz');o3=d[(d.model=='O3')&d.method.isin(['B','D'])&d.supported];assert abs(o3.error).max()<1e-12
 o1=d[(d.model=='O1')&(d.method=='A')&d.supported];assert abs(o1.error).max()<1e-12

def test_frozen_membership_and_direction_counts():
 d=pd.read_csv(OUT/'all_main_oracle_directions.csv');ass=pd.read_csv(OUT/'component_assignments.csv');assert len(ass)==48 and ass.component_id.nunique()==11
 for (model,method),g in d[d.variable=='p'].groupby(['model','method']):
  assert g.pair_uid.nunique()==48 and len(g)==48
  if method=='B':assert g.direction_active.sum()==40
  if method=='D':assert g.direction_active.sum()==36

def test_support_missing_never_zero_filled():
 d=pd.read_csv(OUT/'oracle_points.csv.gz');missing=d[~d.supported];assert len(missing)>0 and missing[['pred','real','error']].isna().all().all()
 assert not d[(d.method=='D')&(d.time_label=='100')].supported.any()

def test_all_registered_windows_and_split_isolation():
 d=pd.read_csv(OUT/'full_window_audit.csv.gz');assert d.groupby('split').size().to_dict()=={'train':107920,'validation':36332}
 assert d.groupby('split').flight_id.nunique().to_dict()=={'train':94,'validation':27}
 assert not set(d[d.split=='train'].log_sha)&set(d[d.split=='validation'].log_sha)
 assert not d.duplicated(['split','registered_index']).any()
 assert (d.future_steps==d.supported_steps+d.unsupported_steps).all()
 assert (d.D_supported_steps<=d.future_steps).all()

def test_identity_and_causal_available_initial():
 d=pd.read_csv(OUT/'identity_clock_audit.csv');assert len(d)==225 and d.exact_source_identity.all() and d.initial_available_by_decision.all()
 assert (d.initial_publication_us<=d.decision_us).all()
 # Source time can be newer than the nominal state clock and still causally available at decision.
 assert set(d.split)=={'train','validation'}

def test_decomposition_and_candidate_gradient_conflict():
 import torch
 initial=torch.tensor([.2,-.1,.05],dtype=torch.float64);b0=torch.tensor([0.,0.,0.],dtype=torch.float64);b=torch.tensor([.5,.3,-.2],dtype=torch.float64)
 target=torch.tensor([.45,.35,-.15],dtype=torch.float64);pred=target.clone().requires_grad_(True)
 loss=((pred-initial-(b-b0))**2).mean();loss.backward();torch.testing.assert_close(pred.grad,2*(target-initial-b+b0)/3)
 assert pred.grad.abs().max()>0 # At zero original absolute error, increment target is generally not optimal.
 candidate=b+initial-b0;assert ((candidate-initial-(b-b0))**2).max()<1e-30

def test_independent_checks_and_no_training():
 for f in ['independent_fixed_verification.json','independent_full_verification.json','model_regression.json','truth_regression.json']:
  assert json.loads((OUT/f).read_text())['passed']
 receipt=json.loads((OUT/'source_receipt.json').read_text());assert receipt['training_runs']==0 and not receipt['test_read'] and not receipt['checkpoint_loaded']
 assert not list(HERE.rglob('*.pt'))
 for script in ['common.py','oracles.py','full_audit.py','compare_models.py','describe.py']:
  text=(HERE/script).read_text();assert 'torch.load(' not in text and 'optimizer.step(' not in text and '.cuda(' not in text

def test_source_hashes_unchanged_and_only_allowed_splits():
 receipt=json.loads((OUT/'source_receipt.json').read_text())
 for path,digest in receipt['files'].items():
  assert 'samples_test' not in path and 'native_vehicle_angular_velocity_test' not in path and '/test.npz' not in path
  assert sha(path)==digest,path

def test_full_summary_independent_equal_flight_reduction():
 d=pd.read_csv(OUT/'full_window_audit.csv.gz');s=pd.read_csv(OUT/'full_summary.csv')
 for r in s.itertuples():
  g=d[d.split==r.split];v=r.axis
  parts=pd.DataFrame({'flight':g.flight_id,'origin':g[f'{v}_origin_error']**2,'future':g[f'{v}_future_mse'],'increment':g[f'{v}_increment_mse'],'cross':g[f'{v}_cross_moment']})
  if r.weighting=='flight':parts=parts.groupby('flight').mean()
  mean=parts.mean(numeric_only=True)
  np.testing.assert_allclose([np.sqrt(mean.origin),np.sqrt(mean.future),np.sqrt(mean.increment),mean.cross],[r.origin_rms,r.future_rms,r.increment_rms,r.cross_moment],atol=1e-13,rtol=0)
