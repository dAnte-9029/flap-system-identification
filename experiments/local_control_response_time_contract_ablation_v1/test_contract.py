from common import *
from prepare import nearest_other
from scipy.spatial import cKDTree

def test_nearest_excludes_receiver_exactly():
 points=np.arange(30.).reshape(-1,1);flights=np.array(['a']*20+['b']*10);query=np.array([[1.2],[18.7],[25.1]])
 got=nearest_other(cKDTree(points),flights,query,'a');expected=np.min(abs(query-points[flights!='a'].T),axis=1);np.testing.assert_allclose(got,expected)

def test_past_only_and_phase_rotation():
 rows=pd.read_csv(OUT/'input_changes.csv.gz');assert (rows.encoder_available_upper_us<=rows.decision_us).all();assert (rows.frequency_available_upper_us<=rows.decision_us).all();assert (rows.rate_available_us<=rows.decision_us).all()
 delta=2*np.pi*rows.frequency*(rows.state_us-rows.encoder_source_us)*1e-6;np.testing.assert_allclose(delta,rows.phase_delta_rad,atol=1e-14)
 np.testing.assert_allclose(rows.new_sin**2+rows.new_cos**2,1,atol=1e-7)

def test_only_phase_changed_and_initial_consistent():
 a=dict(np.load(SWAP/'results/inputs.npz'));b=dict(np.load(OUT/'V1_inputs.npz'))
 for k in a:
  if k=='history':np.testing.assert_array_equal(a[k][...,:17],b[k][...,:17]);np.testing.assert_array_equal(a[k][...,19],b[k][...,19])
  else:np.testing.assert_array_equal(a[k],b[k])
 for i,n in enumerate(b['history_length']):np.testing.assert_array_equal(b['history'][i,n-1,:13],b['initial'][i])

def test_same_receiver_and_nocontrol():
 x=dict(np.load(OUT/'V1_inputs.npz'));p=dict(np.load(OUT/'V1_predictions.npz'))
 for k in ['history','history_mask','history_dt','history_length','initial','dt']:
  np.testing.assert_array_equal(x[k][::4],x[k][1::4]);np.testing.assert_array_equal(x[k][2::4],x[k][3::4])
 for k,v in p.items():
  if k.startswith('GRU-NoControl'):np.testing.assert_array_equal(v[::4],v[1::4]);np.testing.assert_array_equal(v[2::4],v[3::4])

def test_no_extrapolation_or_filter_padding():
 t=np.array([0,.1,.2,.3,.4,.5]);y=np.column_stack([t,t*0+3]);assert np.isnan(tc.interp(t,y,[.51],.11)).all();assert np.isnan(tc.boxmean(t,y,[.1],.27,.11)).all();np.testing.assert_allclose(tc.boxmean(t,y,[.3],.27,.11),[[.3,3]])

def test_synthetic_period_box_integral():
 t=np.arange(0,.501,.0005);y=np.column_stack([np.sin(2*np.pi*t/.27),np.ones(len(t))]);got=tc.boxmean(t,y,[.2,.3],.27,.01);np.testing.assert_allclose(got[:,0],0,atol=1e-10);np.testing.assert_allclose(got[:,1],1,atol=1e-12)

def test_truth_and_support_identical_between_variants():
 f=pd.read_csv(OUT/'factual_points.csv.gz');keys=['pair_uid','model','seed','variable','method','time_label'];a=f[f.variant=='V0'].set_index(keys).sort_index();b=f[f.variant!='V0'].set_index(keys).sort_index();np.testing.assert_array_equal(a.supported,b.supported);np.testing.assert_allclose(a.real,b.real,atol=0,equal_nan=True);assert not f[f.time_label=='500'].supported.any()
 mem=pd.read_csv(SWAP/'results/cohorts.csv');ids=mem.query("split=='train' and cohort=='main'").pair_uid
 assert f[f.pair_uid.isin(ids)&(f.time_label=='300')].supported.all()

def test_frozen_split_and_admission():
 r=json.loads((OUT/'input_receipt.json').read_text());assert not set(r['train_flights'])&set(r['validation_flights']);assert not r['test_read'] and not r['reference_uses_validation'];ad=json.loads((OUT/'admission.json').read_text());assert ad['origin_causal_brackets']==65 and ad['origins']==225;assert ad['V2'].startswith('not admitted')

def test_baseline_and_future_input_gates():
 r=json.loads((OUT/'inference_checks.json').read_text());assert r['passed'];checks=[x for x in r['checks'] if 'V0_max_error' in x];assert len(checks)==6 and max(c['V0_max_error'] for c in checks)<2e-5
 for c in r['checks']:
  if 'variant' in c:assert c['extra_future_truth_phase_excluded'] and c['redundant_events_invariant']
 assert json.loads((OUT/'baseline_statistics.json').read_text())['passed']

def test_internal_closure():
 d=pd.read_csv(OUT/'internal_points.csv.gz');assert abs(d.F-d.C-d.H).max()<1e-12;assert abs(d.CA-d.CB-d.HA+d.HB).max()<1e-12
