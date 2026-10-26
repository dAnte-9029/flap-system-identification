"""Gate: reproduce the unchanged archived response before new diagnostics."""
from common import *
OUT=RESULTS;OUT.mkdir(exist_ok=True)
manifest=json.loads((OLD/'archive_manifest.json').read_text())
checked={}
for x in manifest['files']:
 p=OLD/x['path'];assert sha(p)==x['sha256'],p;checked[str(p.relative_to(ROOT))]=x['sha256']
z=np.load(OLD/'results/native_rollouts.npz');w=pd.read_csv(OLD/'results/windows.csv');p=pd.read_csv(OLD/'results/pairs.csv')
cfg=json.loads((OLD/'config.json').read_text());d,support,c=old.response_rows(z,w,p,cfg)
saved=pd.read_csv(OLD/'results/response_curves.csv.gz')
keys=['pair_uid','model','seed','variable','time_kind','time_ms']
assert d[keys].drop(columns='time_ms').equals(saved[keys].drop(columns='time_ms'))
err=0.
for col in d.select_dtypes(include='number'):
 a=d[col].to_numpy();b=saved[col].to_numpy();np.testing.assert_allclose(a,b,atol=2e-12,rtol=1e-10,equal_nan=True)
 err=max(err,float(np.nanmax(abs(a-b))))
summary=pd.read_csv(OLD/'results/summary.csv');members=pd.read_csv(OLD/'results/cohorts.csv');records=[]
# All cohorts, seeds, weightings and times for primary p, independently rebuilt values.
for r in summary[summary.variable=='p'].itertuples():
 ids=members[(members.split==r.split)&(members.cohort==r.cohort)].pair_uid
 q=p[p.pair_uid.isin(ids)];g=d[(d.pair_uid.isin(ids))&(d.variable=='p')&(d.model==r.model)&d.supported]
 g=g[g.time_kind=='supported_endpoint'] if r.time_label=='supported_endpoint' else g[(g.time_kind=='grid')&np.isclose(g.time_ms,float(r.time_label))]
 if r.seed!='pooled':g=g[g.seed==int(r.seed)]
 q=q[q.pair_uid.isin(g.pair_uid)];wt=g.pair_uid.map(weights(q,r.weighting)).to_numpy();v=old.metrics(g,wt,.1)
 for k in ['rmse','direction_accuracy','scale_ratio','signed_gain','near_zero_fraction']:
  assert np.isclose(v[k],getattr(r,k),atol=2e-12,equal_nan=True),(r.Index,k)
 records.append(dict(split=r.split,cohort=r.cohort,model=r.model,seed=r.seed,time_label=r.time_label,weighting=r.weighting,**v))
pd.DataFrame(records).to_csv(OUT/'baseline_reproduction.csv',index=False)
save(OUT/'baseline_check.json',dict(passed=True,curve_rows=len(d),maximum_numeric_difference=err,p_summary_rows=len(records),old_archive_hash=sha(OLD/'archive_manifest.json'),original_files=checked,model_inference=False,test_read=False))
print('Baseline reproduced',len(d),'response rows;',len(records),'p summary rows; max difference',err)
# Frequency-only design, BEFORE new outcome metrics or validation tuning.
m=members[(members.split=='train')&(members.cohort=='main')];q=p[p.pair_uid.isin(m.pair_uid)];ix=np.unique(q[['a','b']].to_numpy());f=w.iloc[ix].history_frequency.to_numpy()
width=round(1/np.median(f),2);sensitivity=round(1/f.min(),2)
config=dict(base_sha='c6580f7a9369db2de721cfd3e31b339c95d317f2',dataset='velocity_rate_multistep_v1',sampling_group='main_50hz',models=MODELS,variables=VARS,nominal_horizon_s=.5,
 main_width_s=width,sensitivity_width_s=sensitivity,filter='centered box mean, exact piecewise-linear integral, future-only full support; no padding, no extrapolation',
 width_rule='nearest 10 ms to 1/median history frequency of 96 frozen train-main windows; sensitivity nearest 10 ms to 1/min frequency. No prediction metrics used.',
 train_frequency_quantiles=np.quantile(f,[0,.1,.5,.9,1]).tolist(),grid_s=.02,report_times_s=[.1,.2,.3,.4,.5],max_rate_gap_s=.02,max_model_gap_s=.04,
 direction_floors=[.05,.1,.2],main_direction_floor=.1,
 methods={'A':'original nominal position-clock targets minus original instantaneous truth_initial; predictions minus initial',
 'B':'native measured-rate interpolation at nominal model physical times, minus native interpolated value at t0; model unchanged',
 'C':'box mean of A response, preserving original instantaneous anchor; can be nonzero at first valid center',
 'D':'box mean of B response, preserving physical instantaneous anchor',
 'C_cycle':'C minus C at first full-window center W/2; later cycle-average change, not response from original t0',
 'D_cycle':'D minus D at first full-window center W/2; later cycle-average change, not response from original t0'},
 alignment='No time shift fitted. Model output time remains its native dt/position-clock contract; stale initial is not repaired. Native truth at the same nominal physical time is an offline reference.',
 test_read=False,model_inference=False,training=False,control_swap=False,
 fixed_examples=['train:124','train:336','train:412','validation:44','train:105','train:183'])
config=json.loads(json.dumps(config))
if (HERE/'config.json').exists():assert json.loads((HERE/'config.json').read_text())==config
else:save(HERE/'config.json',config)
print('Frequency-only widths',width,sensitivity)
