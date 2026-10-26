"""Independent joint-endpoint, source immutability and phase formula checks."""
from common import *
from verify import sample
OUT=RESULTS;z=dict(np.load(OLD/'results/native_rollouts.npz'));n=dict(np.load(OUT/'native_diagnostic_inputs.npz'));p=pd.read_csv(OLD/'results/pairs.csv').set_index('pair_uid')
d=pd.read_csv(OUT/'matched_endpoint_points.csv');maximum=0.
for (uid,model,seed,method),g in d.groupby(['pair_uid','model','seed','method'],sort=False):
 pair=p.loc[uid];ra=[];pr=[];time=float(g.time_s.iloc[0])
 for i in [int(pair.a),int(pair.b)]:
  count=int(z['future_count'][i]);t=z['native_time_us'][i,:count+1]*1e-6
  if method=='A':truth=np.vstack([z['truth_initial'][i,10:],z['truth'][i,:count,10:]]);val=sample(t,truth,[time])[0]-truth[0]
  else:nt=n[f'rate_t_{i}'];ny=n[f'rate_y_{i}'];val=sample(nt,ny,[time])[0]-sample(nt,ny,[0])[0]
  pred=np.vstack([z['initial'][i,10:],z[f'{model}_{seed}'][i,:count,10:]]).astype(float)
  ra.append(val);pr.append(sample(t,pred,[time])[0]-pred[0])
 for col,val in [('real',ra[0]-ra[1]),('pred',pr[0]-pr[1])]:
  got=g[col].to_numpy();np.testing.assert_allclose(got,val,atol=2e-12);maximum=max(maximum,float(max(abs(got-val))))
ph=pd.read_csv(OUT/'phase_forecast_audit.csv');np.testing.assert_allclose(ph.total_phase_error,2*np.pi*ph.last_frequency_hz*ph.time_s-ph.observed_relative_phase-ph.encoder_staleness_phase_rad,atol=2e-13)
checks=json.loads((OUT/'baseline_check.json').read_text())
for path,h in checks['original_files'].items():assert sha(ROOT/path)==h,path
rec=json.loads((OUT/'source_receipt.json').read_text())
for k,h in rec['input_keys_hashes'].items():assert hashlib.sha256(z[k].tobytes()).hexdigest()==h
assert pd.read_csv(OUT/'origin_clock_audit.csv').initial_causal.all()
# Data gaps are below both operators' specified thresholds; no hidden long-gap bridging.
assert pd.read_csv(OUT/'origin_clock_audit.csv').native_gap_max_us.max()<20000
assert np.max(z['dt'])<.04
save(OUT/'extra_verification.json',dict(passed=True,matched_endpoint_rows=len(d),maximum_abs_difference=maximum,old_files_unchanged=len(checks['original_files']),eight_model_inputs_unchanged=True,phase_formula_checked=True,all_initial_rates_published_by_decision=True,native_gap_below_limit=True,model_inference=False,test_read=False))
print(json.loads((OUT/'extra_verification.json').read_text()))
