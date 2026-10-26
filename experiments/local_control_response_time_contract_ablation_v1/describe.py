from common import *

def main():
 mem=pd.read_csv(SWAP/'results/cohorts.csv');pairs=pd.read_csv(SWAP/'results/pairs.csv');f=pd.read_csv(OUT/'factual_points.csv.gz');e=pd.read_csv(OUT/'internal_points.csv.gz');delta=pd.read_csv(OUT/'paired_factual_changes.csv.gz');di=pd.read_csv(OUT/'paired_internal_changes.csv.gz');wp=pd.read_csv(OUT/'window_points.csv.gz');keys=['case_id','pair_uid','split','cell','receiver_window','donor_window','model','seed','variable','time_ms','time_kind'];dw=wp[wp.variant!='V0'].merge(wp[wp.variant=='V0'],on=keys,suffixes=('_new','_V0'),validate='one_to_one');dw['prediction_change']=dw.increment_new-dw.increment_V0;dw.to_csv(OUT/'paired_window_changes.csv.gz',index=False)
 rows=[];sens=[];seedrows=[]
 for (split,cohort),mm in mem.groupby(['split','cohort']):
  pp=pairs[pairs.pair_uid.isin(mm.pair_uid)];dd=delta[delta.pair_uid.isin(mm.pair_uid)&(delta.time_label=='300')&delta.supported_new]
  for (model,var,method),g in dd.groupby(['model','variable','method']):
   for weighting in ['pair','flight','component']:
    weights=g.pair_uid.map(stats.weights(pp,weighting)).to_numpy();weights/=weights.sum();avg=lambda v:float(np.dot(weights,np.asarray(v)));active=abs(g.real_new)>.1
    rows.append(dict(split=split,cohort=cohort,model=model,variable=var,method=method,weighting=weighting,pairs=g.pair_uid.nunique(),seed_rows=len(g),prediction_change_rms=np.sqrt(avg(g.prediction_change**2)),prediction_change_max=float(abs(g.prediction_change).max()),mae_change=avg(g.absolute_error_change),amplitude_mae_change=avg(g.amplitude_error_change),direction_gained=int((active&~g.direction_V0&g.direction_new).sum()),direction_lost=int((active&g.direction_V0&~g.direction_new).sum()),amplitude_improved_rows=int((g.amplitude_error_change<0).sum())))
  ee=e[e.pair_uid.isin(mm.pair_uid)&(e.time_label=='300')&(e.variable=='p')&e.supported]
  for (variant,model),g in ee.groupby(['variant','model']):
   for floor in [.05,.1,.2]:sens.append(dict(split=split,cohort=cohort,variant=variant,model=model,floor=floor,**sm.metrics(g,g.pair_uid.map(stats.weights(pp,'flight')),floor)))
 for (variant,uid,model,var),g in e[(e.time_label=='300')&e.supported].groupby(['variant','pair_uid','model','variable']):
  for effect in ['C','CA','CB','H']:
   v=g[effect].to_numpy();seedrows.append(dict(variant=variant,pair_uid=uid,model=model,variable=var,effect=effect,minimum=v.min(),maximum=v.max(),all_active=bool((abs(v)>.1).all()),all_near=bool((abs(v)<=.1).all()),same_sign=bool((v>1e-8).all() or (v< -1e-8).all()),mixed_sign=bool((v>1e-8).any() and (v< -1e-8).any())))
 pd.DataFrame(rows).to_csv(OUT/'paired_change_summary.csv',index=False);pd.DataFrame(sens).to_csv(OUT/'internal_direction_sensitivity.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'internal_seed_consistency.csv',index=False)
 s=pd.read_csv(OUT/'factual_summary.csv');s=s[(s.time_label=='300')&(s.variable=='p')&(s.weighting=='flight')];save(OUT/'factual_300_summary.json',json.loads(s.to_json(orient='records')))
 ids=set(mem.query("split=='train' and cohort=='main'").pair_uid);ep=di[di.pair_uid.isin(ids)&(di.time_label=='300')&(di.variable=='p')&(di.model=='GRU-Control')];pp=pairs[pairs.pair_uid.isin(ids)];weights=ep.pair_uid.map(stats.weights(pp,'flight')).to_numpy();weights/=weights.sum();scale={k:dict(change_rms=float(np.sqrt(np.dot(weights,ep[k+'_change']**2))),maximum=float(abs(ep[k+'_change']).max())) for k in ['CA','CB','C','H','F']};save(OUT/'main_internal_change.json',scale)
 a=pd.read_csv(OUT/'feature_distribution.csv').sort_values('window_id');b=pd.read_csv(HERE/'results/feature_distribution_reference.csv').sort_values('window_id');pd.testing.assert_frame_equal(a,b,atol=1e-12,rtol=0);save(OUT/'distribution_verification.json',dict(passed=True,windows=len(a),adaptive_neighbor_search_equals_independent_per_flight_tree=True))
 print(pd.DataFrame(rows).query("split=='train' and cohort=='main' and variable=='p' and weighting=='flight'").to_string(index=False));print(json.dumps(scale,indent=2))
if __name__=='__main__':main()
