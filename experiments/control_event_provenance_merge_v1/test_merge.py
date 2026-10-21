import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from common import *
from evaluate_merge import merged_events

def test_source_function_parser_stops_at_matching_brace():
 from audit_source import block
 text='void LoggedTopics::a() { if (true) { f(); } }\nvoid\nLoggedTopics::b() { g(); }'
 assert 'f();' in block(text,'a') and 'g();' not in block(text,'a')
 f=pd.read_csv(OUT/'firmware_logging.csv');assert f.logger_source_available.all()
 assert (f.actuator_motors_default_ms==100).all()
 assert (f.loc[f.flapping_profile_exists,'actuator_motors_flapping_ms']==20).all()

def test_exact_merge_preserves_real_changes_and_padding():
 ev=np.zeros((1,2,4,5),np.float32)
 ev[0,0,:3]=[[.5,0,0,0,.005],[.5,0,0,0,.005],[.5,.1,0,0,.01]]
 d=dict(events=ev,event_length=np.array([[3,1]]),future_mask=np.array([[True,False]]))
 merged,length,changed=merged_events(d)
 np.testing.assert_array_equal(length,[[2,1]]);np.testing.assert_array_equal(changed,[[True,False]])
 np.testing.assert_allclose(merged[0,0,:2],[[.5,0,0,0,.01],[.5,.1,0,0,.01]])
 np.testing.assert_array_equal(merged[0,1],ev[0,1])

def test_no_tolerance_merge():
 ev=np.array([[[[.5,0,0,0,.01],[.5000001,0,0,0,.01]]]],np.float32)
 d=dict(events=ev,event_length=np.array([[2]]),future_mask=np.array([[True]]))
 result,length,changed=merged_events(d);assert not changed.any();np.testing.assert_array_equal(result,ev)

def test_all_real_merged_intervals_have_identical_zoh():
 z=np.load(OUT/'merge_changes.npz');assert len(z['i'])==728
 for a,b,na,nb in zip(z['before'],z['after'],z['before_length'],z['after_length']):
  a=a[:na];b=b[:nb];ea=np.cumsum(np.rint(a[:,4].astype(float)*1e6).astype(int));eb=np.cumsum(np.rint(b[:,4].astype(float)*1e6).astype(int))
  assert ea[-1]==eb[-1];starts=np.unique(np.r_[0,ea[:-1],eb[:-1]])
  np.testing.assert_array_equal(a[np.searchsorted(ea,starts,side='right'),:4],b[np.searchsorted(eb,starts,side='right'),:4])

def test_split_hashes_and_untouched_predictions():
 m=json.loads((OUT/'evaluation_manifest.json').read_text());assert m['windows']==36332 and m['flights']==27 and not m['test_opened'] and not m['retrained']
 assert all(r['unchanged_prediction_max_difference']==0 for r in m['checks'])
 w=pd.read_parquet(OUT/'validation_windows.parquet');assert w.split.eq('validation').all();assert w.sampling_group.eq('main_50hz').all()
 for path,digest in m['checkpoints'].items():assert sha(ROOT/path)==digest
 assert sha(ROOT/'src/system_identification/models/velocity_rate_gru.py')==m['model_code_sha256']

def test_independent_metrics_all_windows_and_horizons():
 z=np.load(OUT/'predictions.npz');d=pd.read_parquet(OUT/'endpoint_errors.parquet')
 for seed in [17,29,43]:
  for h in [500,1000]:
   truth=z[f'{h}_target'].astype(float)
   for variant in ['native','merged']:
    p=z[f'{seed}_{h}_{variant}'].astype(float);g=d[(d.seed==seed)&(d.horizon_ms==h)&(d.variant==variant)]
    for name,sl in [('position_m',slice(0,3)),('velocity_m_s',slice(3,6)),('rate_rad_s',slice(10,13))]:
     np.testing.assert_allclose(np.sqrt(np.sum((p[:,sl]-truth[:,sl])**2,axis=1)),g[name],atol=1e-12,rtol=0)
    rp=Rotation.from_quat(p[:,[7,8,9,6]]);rt=Rotation.from_quat(truth[:,[7,8,9,6]])
    np.testing.assert_allclose(np.rad2deg((rp.inv()*rt).magnitude()),g.attitude_deg,atol=1e-10,rtol=0)

def test_independent_flight_equal_aggregation():
 d=pd.read_parquet(OUT/'endpoint_errors.parquet');s=pd.read_csv(OUT/'summary.csv')
 for row in s.itertuples():
  x=d[(d.horizon_ms==row.horizon_ms)&(d.variant==row.variant)]
  if row.scope=='affected':x=x[x.affected]
  if row.seed!='all':x=x[x.seed==int(row.seed)]
  mse=[]
  for fid in sorted(x.flight_id.unique()):mse.append(np.mean(x.loc[x.flight_id==fid,row.metric].to_numpy()**2))
  assert abs(np.sqrt(np.mean(mse))-row.flight_rmse)<1e-12

def test_provenance_sources_and_unique_interval_statistics():
 p=json.loads((OUT/'provenance_manifest.json').read_text());assert p['authorized_flights']==161 and p['sealed_test_not_read']
 sources=json.loads((OUT/'source_manifest.json').read_text())
 for r in sources:
  if r['status']=='verified_exact_commit':assert sha(HERE/r['artifact'])==r['sha256']
 c=pd.read_csv(OUT/'training_intervals_per_flight.csv');assert set(c.split)=={'train','validation'}
 assert c.groupby('split').redundant_tokens.sum().to_dict()=={'train':175,'validation':88}
