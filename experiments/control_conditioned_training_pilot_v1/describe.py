"""Paired descriptive diagnostics; no selection, tuning or model inference."""
from common import *
from evaluate import old

def main():
 pairs=pd.read_csv(OUT/'pairs.csv');assign=pd.read_csv(OUT/'component_assignments.csv');mainids=set(assign.pair_uid);d=pd.read_csv(OUT/'paired_changes.csv.gz');primary=d[d.pair_uid.isin(mainids)&(d.variable=='p')&(d.time_label=='300')&d.method.isin(['B','D'])].copy()
 primary['new_wrong']=(primary.real*primary.pred<0)&primary.direction_active;primary['old_wrong']=(primary.real*primary.pred_W0<0)&primary.direction_active;primary['wrong_response_amplified']=primary.new_wrong&primary.old_wrong&(abs(primary.pred)>abs(primary.pred_W0))
 primary=primary.merge(assign[['pair_uid','component_id']],on='pair_uid',validate='many_to_one');primary.to_csv(OUT/'all_main_direction_changes.csv',index=False)
 rows=[]
 for (model,method),g in primary.groupby(['model','method']):
  pp=pairs[pairs.pair_uid.isin(g.pair_uid)]
  for weighting in ['pair','flight','component']:
   w=g.pair_uid.map(old.stats.weights(pp,weighting)).to_numpy()
   for sign in ['positive','negative']:
    mask=g.real.to_numpy()>.1 if sign=='positive' else g.real.to_numpy()<-.1;z=g[mask];ww=w[mask];ww=ww/ww.sum();r=z.real.to_numpy();a=z.pred_W0.to_numpy();b=z.pred.to_numpy()
    rows.append(dict(model=model,method=method,weighting=weighting,sign=sign,pairs=len(z),direction_W0=float(np.dot(ww,(r*a>0)&(abs(a)>1e-8))),direction_W1=float(np.dot(ww,(r*b>0)&(abs(b)>1e-8))),gained=int(z.gained.sum()),lost=int(z.lost.sum()),amplitude_bias_W0=float(np.dot(ww,abs(a)-abs(r))),amplitude_bias_W1=float(np.dot(ww,abs(b)-abs(r))),signed_bias_W0=float(np.dot(ww,a-r)),signed_bias_W1=float(np.dot(ww,b-r)),wrong_response_amplified_pairs=int(z.wrong_response_amplified.sum())))
 pd.DataFrame(rows).to_csv(OUT/'sign_comparison.csv',index=False)
 # Prediction-zero versus real-near-zero are distinct definitions, frozen at original 1e-8 and 0.1.
 points=pd.read_csv(OUT/'factual_points.csv.gz');rows=[]
 for key,g in points[(points.role=='heldout')&points.pair_uid.isin(mainids)&(points.time_label=='300')&points.supported&points.variable.isin(['p','q','r'])].groupby(['model','variable','method']):
  active=abs(g.real)>.1;rows.append(dict(zip(['model','variable','method'],key),pairs=len(g),real_near_zero=int((~active).sum()),pred_zero_active=int((active&(abs(g.pred)<=1e-8)).sum()),pred_abs_under_01=int((abs(g.pred)<=.1).sum())))
 pd.DataFrame(rows).to_csv(OUT/'direction_abstention_counts.csv',index=False)
 effects=pd.read_csv(OUT/'internal_points.csv.gz');effects['architecture']=effects.model.str[:2];keys=['architecture','fold','pair_uid','seed','variable','time_label'];b=effects[effects.model.str.endswith('W0')];z=effects[effects.model.str.endswith('W1')].merge(b,on=keys,suffixes=('','_W0'),validate='one_to_one')
 for key in ['CA','CB','C','H','interaction']:z[key+'_change']=z[key]-z[key+'_W0']
 z.to_csv(OUT/'internal_paired_changes.csv.gz',index=False)
 traj=pd.read_csv(OUT/'trajectory_per_flight.csv');traj['architecture']=traj.model.str[:2];base=traj[traj.model.str.endswith('W0')];joined=traj[traj.model.str.endswith('W1')].merge(base,on=['fold','architecture','seed','flight_id'],suffixes=('','_W0'),validate='one_to_one')
 cols=[c for c in traj if c.endswith(('_path_rms','_endpoint_norm'))]
 for c in cols:joined[c+'_change']=joined[c]-joined[c+'_W0']
 joined.to_csv(OUT/'trajectory_paired_per_flight.csv',index=False)
 selected=primary[primary.pair_uid.isin(json.loads((HERE/'config.json').read_text())['examples'])];selected.to_csv(OUT/'prespecified_cases_300.csv',index=False)
 print('Flight-equal p at 300 ms:')
 s=pd.read_csv(OUT/'factual_summary.csv');print(s[(s.scope=='all_heldout')&(s.cohort=='main')&(s.variable=='p')&(s.time_label=='300')&(s.weighting=='flight')][['model','method','active_pairs','direction_accuracy','scale_ratio','signed_gain','mae']].to_string(index=False))
 print('Same-architecture p direction transitions:')
 p=pd.read_csv(OUT/'paired_summary.csv');print(p[(p.scope=='all_heldout')&(p.variable=='p')&(p.weighting=='pair')][['model','method','gained_pair_seed_rows','lost_pair_seed_rows','direction_difference','mean_amplitude_error_change']].to_string(index=False))
 print('Physical-unit held-flight trajectory summary:');print(pd.read_csv(OUT/'trajectory_summary.csv').to_string(index=False))
if __name__=='__main__':main()
