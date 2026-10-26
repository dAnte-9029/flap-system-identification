from common import *
import pytest
replay=module(OLD/'run.py','test_replay')

def test_shift_zoh_preserves_prefix_and_duration():
 own=(np.array([0,100,150,500]),np.array([[.1,0,0,0],[.2,0,0,0],[.3,0,0,0],[.4,0,0,0]]))
 donor=(np.array([1000,1100,1130,1200,1500]),np.array([[.5,0,0,0],[.6,0,0,0],[.7,0,0,0],[.8,0,0,0],[.9,0,0,0]]))
 t,u=replay.response.splice_tape(own,donor,100,1100,200)
 q=np.array([90,100,129,130,199,200,299]);got=u[np.searchsorted(t,q,side='right')-1,0]
 np.testing.assert_allclose(got,[.1,.6,.6,.7,.7,.8,.8]);assert 130 in t and 200 in t

def test_event_duration_and_exact_mean():
 ev=np.array([[[0.,-.2,.2,0.,.01],[0.,-.4,.4,0.,.01]],[[0.,-.6,.6,0.,.02],[0,0,0,0,0]]],np.float32);t,u,end=event_tape(ev,np.array([2,1]),np.array([.02,.02]));assert end==40000
 mean,rms,longest=difference(t,u,np.array([0]),np.zeros((1,4)),40000)
 np.testing.assert_allclose(mean[2],.45,atol=1e-7);np.testing.assert_allclose(rms[2],np.sqrt(.23),atol=1e-7);assert longest==40000

def test_no_extrapolation_and_native_prefix():
 t=np.array([0,.02,.04]);y=np.column_stack([t,t*2]);got=interp(t,y,[0,.01,.04,.041]);assert np.isnan(got[-1]).all();np.testing.assert_allclose(got[1],[.01,.02])
 x=dict(np.load(OUT/'inputs.npz'));cases=pd.read_csv(OUT/'cases.csv');w=pd.read_csv(OUT/'windows.csv');old=dict(np.load(OLD/'results/native_rollouts.npz'))
 for r in cases.itertuples():
  n=int(x['valid_steps'][r.case_id]);np.testing.assert_array_equal(x['dt'][r.case_id,:n],old['dt'][r.receiver,:n]);assert (x['dt'][r.case_id,n:]==0).all();assert x['native_time_us'][r.case_id,n]+w.initial_measurement_us.iloc[r.receiver]<=w.decision_us.iloc[r.receiver]+500000

def test_four_cell_initial_history_and_split():
 x=dict(np.load(OUT/'inputs.npz'));cases=pd.read_csv(OUT/'cases.csv');w=pd.read_csv(OUT/'windows.csv')
 for k in ['history','history_mask','history_dt','history_length','initial','dt']:
  np.testing.assert_array_equal(x[k][::4],x[k][1::4]);np.testing.assert_array_equal(x[k][2::4],x[k][3::4])
 assert not set(w.query("split=='train'").flight_id)&set(w.query("split=='validation'").flight_id)
 for r in cases.itertuples():assert w.split.iloc[r.receiver]==w.split.iloc[r.donor]==r.split

def test_nocontrol_exact_and_factual_gate():
 p=dict(np.load(OUT/'predictions.npz'));checks=json.loads((OUT/'inference_checks.json').read_text());assert checks['passed']
 for k,v in p.items():
  if k.startswith('GRU-NoControl'):np.testing.assert_array_equal(v[::4],v[1::4]);np.testing.assert_array_equal(v[2::4],v[3::4])
 assert len([r for r in checks['checks'] if r['check']=='full factual archive replay'])==6
 for r in checks['checks']:
  if r['check']=='full factual archive replay':assert r['max_abs_error']==0
  else:assert r['factual_prefix_max_abs_error']<2e-5 and r['history_hidden_identical'] and r['phase_and_dt_identical'] and r['redundant_events_identical']

def test_future_truth_and_phase_excluded():
 assert set(replay.KEYS)=={'history','history_mask','history_dt','history_length','initial','dt','events','event_length'}
 for r in json.loads((OUT/'inference_checks.json').read_text())['checks']:
  if r['check']=='four-cell invariants':assert r['future_labels_excluded']
 c=pd.read_csv(OUT/'state_phase_clock_audit.csv')
 for source in ['encoder','frequency','rate']:assert (c[source+'_publication_us']<=c.decision_us).all()
 assert not c.encoder_measurement_known.any() and not c.frequency_measurement_known.any()

def test_closure_all_curves_and_zero_denominator():
 d=pd.read_csv(OUT/'decomposition_curves.csv.gz');d=d[d.supported];assert abs(d.F-d.C-d.H).max()<1e-12;assert abs((d.CA-d.CB)-(d.HA-d.HB)).max()<1e-12
 from analyze import metrics
 g=d[(d.time_label=='0')&(d.variable=='p')].head(3);m=metrics(g,np.ones(len(g)));assert m['C_gain_valid_n']==0 and np.isnan(m['C_signed_gain'])

def test_frozen_membership_main_and_reference():
 old=pd.read_csv(OLD/'results/cohorts.csv');new=pd.read_csv(OUT/'cohorts.csv');pd.testing.assert_frame_equal(old,new)
 assert len(new.query("cohort=='main' and split=='train'"))==48;assert len(new.query("cohort=='flight_disjoint' and split=='train'"))==17
 assert len(new.query("cohort=='main' and split=='validation'"))==1
 receipt=json.loads((OUT/'support_receipt.json').read_text());assert not receipt['test_read'] and not receipt['reference_uses_validation'];assert not set(receipt['train_flights'])&set(receipt['validation_flights'])


def test_padding_is_not_control_range_evidence():
 x=dict(np.load(OUT/'inputs.npz'));support=pd.read_csv(OUT/'swap_support.csv');r=json.loads((OUT/'support_receipt.json').read_text());lo=np.array(r['command_min']);hi=np.array(r['command_max'])
 for i in range(len(support)):
  values=[]
  for k,n in enumerate(x['event_length'][i]):
   for e in x['events'][i,k,:int(n)]:
    if e[4]>0:values.append(e[:4])
  values=np.array(values);flag=bool(np.any((values<lo-1e-6)|(values>hi+1e-6)))
  assert flag==support.control_outside_train_marginal.iloc[i]
