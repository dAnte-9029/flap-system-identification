"""Compact reproducible report facts and saved prior-evaluation context (no refiltering)."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 cfg=json.loads((HERE/'config.json').read_text());s=pd.read_csv(OUT/'summary.csv');mem=pd.read_csv(OUT/'cohorts.csv');pairs=pd.read_csv(OUT/'pairs.csv');ids=set(mem.query("split=='train' and cohort=='main'").pair_uid);pp=pairs[pairs.pair_uid.isin(ids)];d=pd.read_csv(OUT/'point_decomposition.csv.gz');m=d[d.pair_uid.isin(ids)&(d.variable=='p')&(d.model=='GRU-Control')&(d.time_label=='300')];g=m.groupby('pair_uid').agg(Cmin=('C','min'),Cmax=('C','max'),g=('g','first'),Cmean=('C','mean'),Hmean=('H','mean'));subset={}
 for label,ss in [('negative_internal_gain_all_seeds',g[(abs(g.g)>=.02)&(g.Cmin*g.g<0)&(g.Cmax*g.g<0)]),('near_C_all_seeds',g[np.maximum(abs(g.Cmin),abs(g.Cmax))<=.1])]:
  p=pp[pp.pair_uid.isin(ss.index)];subset[label]=dict(pairs=len(p),flights=len(set(p.flight_a)|set(p.flight_b)),dates=sorted(set(p.date_a)|set(p.date_b)),controller_count=len(set(p.controller_a)|set(p.controller_b)),pair_ids=p.pair_uid.tolist())
 summary=s[(s.variable=='p')&(s.time_label=='300')];save(OUT/'summary_300ms.json',json.loads(summary.to_json(orient='records')))
 support=pd.read_csv(OUT/'swap_support.csv');c=pd.read_csv(OUT/'control_differences.csv');c=c[c.pair_uid.isin(ids)&(c.time_kind=='grid')&(c.time_ms.round(6)==300)];clock=pd.read_csv(OUT/'state_phase_clock_audit.csv');ts=pd.read_csv(OUT/'time_support.csv');flags=m.assign(both_near=(abs(m.CA)<=.1)&(abs(m.CB)<=.1),both_active=(abs(m.CA)>.1)&(abs(m.CB)>.1),history_dominates=abs(m.H)>abs(m.C),negative_internal_gain=(abs(m.g)>=.02)&(m.C*m.g<0));graph=stats.labels(pp);component_counts=pd.Series(list(graph.values())).value_counts();degree=pd.Series(list(pp.flight_a)+list(pp.flight_b)).value_counts()
 save(OUT/'report_facts.json',dict(main_train_pairs=len(pp),flights=len(degree),components=len(component_counts),largest_component_pairs=int(component_counts.max()),max_flight_degree=int(degree.max()),subsets=subset,counts_per_seed=flags.groupby('seed')[['both_near','both_active','history_dominates','negative_internal_gain']].sum().to_dict(),control_describe=json.loads(c[['g','A_differential_rms','A_differential_longest_ms','A_motor_rms','A_common_rms','A_rudder_rms']].describe().to_json()),clock_describe=json.loads(clock.filter(regex='age_.*us$').describe().to_json()),main_endpoint_ms=json.loads(ts[ts.pair_uid.isin(ids)].common_end_ms.describe().to_json()),support_main=json.loads(support[support.pair_uid.isin(ids)].groupby('cell')[['conditional_warning','conditional_unknown','control_outside_train_marginal','history_outside_train_marginal']].sum().to_json()),factual_only_comparison=True,internal_effects_have_no_swapped_truth=True))
 prior=pd.read_csv(TIMING/'results/points.csv.gz');prior=prior[(prior.time_s.round(6)==.3)&(prior.point=='300')&(prior.variable=='p')&(((prior.method.isin(['A','B']))&(prior.domain=='full'))|((prior.method=='D')&(prior.width_s==.27)))];prior.to_csv(OUT/'prior_factual_context_300ms.csv',index=False)
 # Raw logged command shapes, all channels, used solely to explain internal effects.
 x=dict(np.load(OUT/'inputs.npz'));cases=pd.read_csv(OUT/'cases.csv');fig,axes=plt.subplots(4,4,figsize=(14,10),sharex=True)
 for ri,uid in enumerate(cfg['examples']):
  for label,color in [('AA','#0072B2'),('AB','#D55E00')]:
   ci=int(cases[(cases.pair_uid==uid)&(cases.cell==label)].case_id.iloc[0]);t,u,end=event_tape(x['events'][ci],x['event_length'][ci],x['dt'][ci]);v=coords(u)
   for j,name in enumerate(['motor','common','differential','rudder']):axes[ri,j].step(np.r_[t,end]*.001,np.r_[v[:,j],v[-1,j]],where='post',label=label,color=color);axes[ri,j].set_title(uid+' '+name);axes[ri,j].axvline(300,c='gray',ls=':',lw=.6)
 for a in axes[-1]:a.set_xlabel('Receiver A model time (ms)')
 axes[0,0].legend();fig.tight_layout();dest=HERE/'figures' if OUT==HERE/'results' else OUT.parent/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'fig6_control_waveforms.png',dpi=180);plt.close(fig)
 print('Saved report facts and previous-stage factual context; no new slow/fast filtering')
if __name__=='__main__':main()
