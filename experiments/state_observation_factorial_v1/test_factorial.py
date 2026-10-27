from common import *

def test_two_factor_algebra():
 rng=np.random.default_rng(17);a,b,c,d=rng.random((4,25));r=contrasts(a,b,c,d)
 np.testing.assert_allclose(r['joint'],r['label_main']+r['initial_main'],atol=1e-15)
 np.testing.assert_allclose(r['interaction'],r['label_at_C1']-r['label_at_C0'],atol=1e-15)
 np.testing.assert_allclose(r['joint'],r['label_at_C0']+r['initial_at_A']+r['interaction'],atol=1e-15)

def test_same_integrator_and_missing_support():
 from system_identification.models.velocity_rate_gru import integrate as original
 assert integrate is original and old.integrate is original
 initial=np.zeros((1,13));initial[:,6]=1;target=np.zeros((1,3,13));target[:,:,6]=1;mask=np.array([[True,True,False]]);rates=np.zeros((1,3,3));rates[:,2]=np.nan
 a,p=kinematics(initial,np.zeros((1,3,3)),rates,np.ones((1,3))*.02,target,mask)
 assert np.all(a[:,:2]==0) and np.isnan(a[:,2]).all();assert np.all(p[:,:2]==0) and np.isnan(p[:,2]).all()

def test_reused_endpoints_and_exact_window_support():
 d=pd.read_csv(OUT/'factorial_windows.csv.gz');p=pd.read_csv(PREV/'results/windows.csv.gz');m=d.merge(p,on=['split','registered_index','window_id','flight_id'],validate='one_to_one',suffixes=('','_old'))
 assert len(m)==144252
 for col in ['common_prefix_steps','future_steps','complete_B','common_duration_s']:np.testing.assert_array_equal(m[col],m[col+'_old'])
 for new,oldtag in [('T00','T0'),('T11','T1')]:
  np.testing.assert_allclose(m[new+'_path_mse'],m[oldtag+'_attitude_mse'],atol=1e-16,rtol=0)
  np.testing.assert_allclose(m[new+'_endpoint_mse'],m[oldtag+'_endpoint_rad']**2,atol=1e-16,rtol=0)
 r=json.loads((OUT/'run_receipt.json').read_text());assert r['training_runs']==0 and not r['test_read'];assert all(x['max_error']<1e-11 for x in r['regression']);assert all(x['same_initial_position_velocity_attitude'] and x['same_dt_velocity_target_support'] for x in r['checks'])
 assert d.groupby('split').size().to_dict()=={'train':107920,'validation':36332}
 assert not set(d[d.split=='train'].flight_id)&set(d[d.split=='validation'].flight_id)

def test_saved_factors_and_aggregation():
 d=pd.read_csv(OUT/'factorial_windows.csv.gz');s=pd.read_csv(OUT/'summary.csv')
 for measure in ['path','endpoint']:
  r=contrasts(*[d[f'{tag}_{measure}_mse'].to_numpy() for tag in COMBOS])
  for term,expected in r.items():np.testing.assert_allclose(d[f'{measure}_{term}'],expected,atol=5e-16,rtol=0)
 for row in s.itertuples():
  g=d[d.split==row.split]
  if row.subset=='complete_H1000':g=g[g.complete_B]
  for tag in COMBOS:
   values=g[f'{tag}_path_mse'] if row.weighting=='window' else g.groupby('flight_id')[f'{tag}_path_mse'].mean()
   assert abs(np.sqrt(values.mean())*180/np.pi-getattr(row,f'{tag}_path_rms_deg'))<1e-11
