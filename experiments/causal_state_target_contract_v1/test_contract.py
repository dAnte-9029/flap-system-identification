from common import *
from state import causal_rate,apply_rates

def test_publication_deadline_not_measurement_only():
 tm=np.array([10000,15000,18000]);pub=np.array([11000,16000,30000]);y=np.c_[tm,tm*2,tm*3]/1e4
 a,i=causal_rate(tm,pub,y,[18000],[19000],[[1.5,3,4.5]])
 np.testing.assert_allclose(a,[[1.8,3.6,5.4]]);assert i['reason'][0]=='forward_extrapolation'
 y[-1]=1e8;b,_=causal_rate(tm,pub,y,[18000],[19000],[[1.5,3,4.5]]);np.testing.assert_array_equal(a,b)

def test_interpolation_and_backward_extrapolation():
 y=np.array([[1.,2,3],[2.,4,6]])
 a,i=causal_rate([10000,20000],[11000,21000],y,[15000,5000],[22000,22000],[[2,4,6],[2,4,6]])
 np.testing.assert_allclose(a,[[1.5,3,4.5],[.5,1,1.5]])
 assert i['reason'].tolist()==['interpolation','backward_extrapolation']

def test_two_independent_extrapolation_limits():
 base=np.array([[7.,8,9]])
 for tm,t in [([10000,15000],21000),([10000,30000],46000)]:
  a,i=causal_rate(tm,np.array(tm)+100,[[1,2,3],[2,4,6]],[t],[50000],base)
  np.testing.assert_array_equal(a,base);assert i['reason'][0]=='extrapolation_limit'

def test_gap_invalid_and_insufficient_fallback():
 base=np.array([[7.,8,9]])
 for tm,pub,y in [([10000,31000],[11000,32000],[[1,2,3],[2,4,6]]),([10000,15000],[11000,16000],[[1,2,float('nan')],[2,4,6]]),([15000,10000],[16000,17000],[[1,2,3],[2,4,6]])]:
  a,i=causal_rate(tm,pub,y,[18000],[35000],base);np.testing.assert_array_equal(a,base);assert i['reason'][0] not in ACCEPT

def test_history_and_initial_same_contract_padding_unchanged():
 h=np.arange(2*4*20,dtype=np.float32).reshape(2,4,20);s=h[:,0,:13].copy();rates=np.ones((2,4,3))*-2
 hc,ic=apply_rates(h,s,rates,[2,4]);np.testing.assert_array_equal(hc[:,:,:10],h[:,:,:10]);np.testing.assert_array_equal(hc[:,:,13:],h[:,:,13:]);np.testing.assert_array_equal(hc[0,2:],h[0,2:]);np.testing.assert_array_equal(ic[:,:10],s[:,:10]);np.testing.assert_array_equal(ic[:,10:],-2)

def test_original_integrator_constant_rate_and_velocity():
 s=torch.zeros((1,13),dtype=torch.float64);s[:,6]=1;s[:,3]=2;s[:,12]=.7
 for _ in range(50):s=integrate(s,s[:,3:6],s[:,10:],torch.tensor([.02],dtype=torch.float64))
 np.testing.assert_allclose(s[0,:3],[2,0,0],atol=1e-13);np.testing.assert_allclose(s[0,6:10],[np.cos(.35),0,0,np.sin(.35)],atol=1e-13)
 assert abs(torch.linalg.vector_norm(s[:,6:10]).item()-1)<1e-14

def test_result_causality_and_fallback():
 d=pd.read_parquet(OUT/'history_states.parquet');g=d[d.accepted]
 assert (g.right_publication_us<=g.decision_us).all() and (g.left_publication_us<=g.decision_us).all()
 assert g.gap_us.between(1,20000).all() and (g.extrapolation_us<=15000).all() and (g.extrapolation_us<=g.gap_us).all()
 for v in VARS:np.testing.assert_array_equal(d.loc[~d.accepted,'C0_'+v],d.loc[~d.accepted,'C1_'+v])

def test_result_support_and_partition_contract():
 d=pd.read_csv(OUT/'windows.csv.gz');assert d.groupby('split').size().to_dict()=={'train':107920,'validation':36332}
 assert not set(d[d.split=='train'].log_sha)&set(d[d.split=='validation'].log_sha)
 assert (d.common_prefix_steps<=d.B_supported_steps).all() and (d.B_supported_steps<=d.future_steps).all()
 assert (d.loc[d.complete_B,'common_prefix_steps']==d.loc[d.complete_B,'future_steps']).all()
 r=json.loads((OUT/'receipt.json').read_text());assert r['training_runs']==0 and not r['test_read'] and r['history_initial_identity_checked']==len(d)
 assert not list(HERE.rglob('*.pt'))

def test_kinematic_missing_targets_remain_missing():
 initial=np.zeros((1,13));initial[:,6]=1
 target=np.zeros((1,3,13));target[:,:,6]=1;rates=np.zeros((1,3,3));rates[:,1:]=np.nan
 q,p=kinematics(initial,np.zeros((1,3,3)),rates,np.ones((1,3))*.02,target,np.array([[True,False,False]]))
 assert q[0,0]==0 and p[0,0]==0 and np.isnan(q[0,1:]).all() and np.isnan(p[0,1:]).all()
