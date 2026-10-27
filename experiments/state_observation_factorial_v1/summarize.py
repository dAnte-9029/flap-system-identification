from common import *

def summarize(g,w=None):
 w=np.ones(len(g)) if w is None else np.asarray(w,float);w=w/w.sum();avg=lambda x:float(np.dot(w,np.asarray(x)));r=dict(windows=len(g),flights=g.flight_id.nunique(),complete_fraction=avg(g.complete_B),mean_duration_s=avg(g.common_duration_s))
 for tag in COMBOS:
  for measure in ['path','endpoint']:
   mse=avg(g[f'{tag}_{measure}_mse']);r[f'{tag}_{measure}_mse_rad2']=mse;r[f'{tag}_{measure}_rms_deg']=np.sqrt(mse)*180/np.pi
  r[f'{tag}_first_jump_norm_mean']=avg(g[f'{tag}_first_jump_norm'])
  for axis in VARS:r[f'{tag}_first_jump_{axis}_rms']=np.sqrt(avg(g[f'{tag}_first_jump_{axis}']**2))
 for c in ['C0','C1']:
  for axis in VARS:r[f'{c}_{axis}_rms']=np.sqrt(avg(g[f'{c}_{axis}_error']**2))
 for measure in ['path','endpoint']:
  for term in ['label_at_C0','label_at_C1','initial_at_A','initial_at_B','interaction','label_main','initial_main','joint']:
   r[f'{measure}_{term}_rad2']=avg(g[f'{measure}_{term}']);r[f'{measure}_{term}_deg2']=r[f'{measure}_{term}_rad2']*(180/np.pi)**2
  for name,a,b in [('label_at_C0','T01','T00'),('label_at_C1','T11','T10'),('initial_at_A','T10','T00'),('initial_at_B','T11','T01')]:r[f'{measure}_{name}_rms_change_deg']=r[f'{a}_{measure}_rms_deg']-r[f'{b}_{measure}_rms_deg'];r[f'{measure}_{name}_improved_fraction']=avg(g[f'{a}_{measure}_mse']<g[f'{b}_{measure}_mse'])
 return r

def main():
 d=pd.read_csv(OUT/'factorial_windows.csv.gz');summary=[];per=[]
 for split,all_ in d.groupby('split'):
  for subset,g in [('common_prefix',all_),('complete_H1000',all_[all_.complete_B])]:
   for weighting in ['window','flight']:
    w=np.ones(len(g)) if weighting=='window' else 1/g.groupby('flight_id').flight_id.transform('size').to_numpy();summary.append(dict(split=split,subset=subset,weighting=weighting,**summarize(g,w)))
   for fid,x in g.groupby('flight_id'):per.append(dict(split=split,subset=subset,flight_id=fid,**summarize(x)))
 s=pd.DataFrame(summary);p=pd.DataFrame(per);s.to_csv(OUT/'summary.csv',index=False);p.to_csv(OUT/'per_flight.csv',index=False)
 cols=['split','registered_index','window_id','flight_id','complete_B','common_prefix_steps']+[c for c in d if c.startswith('path_') or c.startswith('endpoint_')];d[cols].to_csv(OUT/'factor_differences.csv.gz',index=False)
 s[['split','subset','weighting']+[c for c in s if '_at_' in c or 'interaction' in c or '_main_' in c or '_joint_' in c]].to_csv(OUT/'factor_summary.csv',index=False)
 print(s.query("weighting=='window'")[[ 'split','subset']+[f'{tag}_path_rms_deg' for tag in COMBOS]+['path_label_at_C0_deg2','path_initial_at_A_deg2','path_interaction_deg2']].to_string(index=False))
 print(p.query("subset=='common_prefix'").groupby('split')[[f'path_{term}_deg2' for term in ['label_at_C0','label_at_C1','initial_at_A','initial_at_B']]].agg(lambda x:int((x<0).sum())).to_string())
if __name__=='__main__':main()
