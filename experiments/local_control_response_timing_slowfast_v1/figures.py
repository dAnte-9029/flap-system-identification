from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=RESULTS;FIG=Path(os.environ.get('LOCAL_DIAG_FIGURES',str(HERE/'figures')));FIG.mkdir(parents=True,exist_ok=True)
cfg=json.loads((HERE/'config.json').read_text());d=pd.read_csv(OUT/'responses.csv.gz');s=pd.read_csv(OUT/'summary.csv');p=pd.read_csv(OLD/'results/pairs.csv');m=pd.read_csv(OLD/'results/cohorts.csv');w=pd.read_csv(OLD/'results/windows.csv')
main=set(m.query("cohort=='main' and split=='train'").pair_uid);cols={'GRU-Control':'#1565c0','GRU-NoControl':'#e67e22','Hold':'#777777'}
plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
def savefig(fig,name):fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def curve(ax,uid,slow=False):
 q=d[(d.pair_uid==uid)&(d.variable=='p')&(d.width_s==(.27 if slow else 0))]
 for method,color,style in [('C' if slow else 'A','#888888','--'),('D' if slow else 'B','black','-')]:
  g=q[(q.method==method)&(q.model=='Hold')].sort_values('time_s');ax.plot(g.time_s*1000,g.real,style,c=color,lw=2,label='Recorded '+method)
 for model in ['GRU-Control','GRU-NoControl']:
  for seed,g in q[(q.method==('D' if slow else 'B'))&(q.model==model)].groupby('seed'):
   g=g.sort_values('time_s');ax.plot(g.time_s*1000,g.pred,c=cols[model],alpha=.65,lw=1,label=f'{model} {seed}')
 ax.axhline(0,color='gray',lw=.6);ax.set(xlabel='Physical nominal time (ms)',ylabel='D p (rad/s)',title=uid+(' | 270 ms centered mean' if slow else ' | raw'))
fig,axes=plt.subplots(2,4,figsize=(17,8))
for j,uid in enumerate(cfg['fixed_examples'][:4]):curve(axes[0,j],uid);curve(axes[1,j],uid,True)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.04),ncol=4,fontsize=8);savefig(fig,'fig1_fixed_cases')
fig,axes=plt.subplots(2,2,figsize=(12,8))
for j,uid in enumerate(cfg['fixed_examples'][4:]):curve(axes[0,j],uid);curve(axes[1,j],uid,True)
axes[0,0].legend(fontsize=7);savefig(fig,'fig2_prior_median_cases')
# Same 300ms and same 48 pairs, with active denominator visibly labelled.
base=s[(s.split=='train')&(s.cohort=='main')&(s.variable=='p')&(s.seed=='pooled')&(s.weighting=='flight')&(s.point=='300')&(s.width_s==.27)]
fig,axes=plt.subplots(1,3,figsize=(15,4.6));methods=['A','B','C','D']
for ax,metric,title in zip(axes,['direction_accuracy','scale_ratio','signed_gain'],['Direction agreement (active truth only)','Predicted / recorded RMS','Signed gain']):
 for j,model in enumerate(cols):
  q=base[base.model==model].set_index('method').loc[methods];ax.bar(np.arange(4)+(j-1)*.24,q[metric],.23,color=cols[model],label=model)
 ax.set(xticks=range(4),xticklabels=[f'{mm}\nactive {int(base[(base.method==mm)&(base.model=="Hold")].active_pairs.iloc[0])}/48' for mm in methods],title=title)
 if metric!='direction_accuracy':ax.axhline(1,c='gray',ls='--',lw=.8)
axes[0].set_ylim(0,1);axes[0].legend(fontsize=8);savefig(fig,'fig3_direction_amplitude_gain')
fig,axes=plt.subplots(1,4,figsize=(16,4))
for ax,method in zip(axes,methods):
 q=d[(d.pair_uid.isin(main))&(d.variable=='p')&(d.width_s==.27)&(d.method==method)&(d.model=='GRU-Control')&(d.point=='300')]
 for seed,g in q.groupby('seed'):ax.scatter(g.real,g.pred,s=15,alpha=.6,label=str(seed))
 lim=max(q.real.abs().max(),q.pred.abs().max())*1.05;ax.plot([-lim,lim],[-lim,lim],'k--',lw=.8);ax.axvspan(-.1,.1,color='gray',alpha=.2)
 ax.set(xlabel='Recorded D p',ylabel='Predicted D p',title=method+' | 300ms',xlim=(-lim,lim),ylim=(-lim,lim))
axes[0].legend(title='Seed',fontsize=8);savefig(fig,'fig4_scatter')
# Clock source audit, showing recorded timestamps rather than shifting predictions.
n=np.load(OUT/'native_diagnostic_inputs.npz');o=pd.read_csv(OUT/'origin_clock_audit.csv').set_index('rollout_index');fig,axes=plt.subplots(2,4,figsize=(17,7))
for j,uid in enumerate(cfg['fixed_examples'][:4]):
 pair=p.set_index('pair_uid').loc[uid]
 for k,side in enumerate(['a','b']):
  i=int(pair[side]);row=o.loc[i];t=n[f'rate_t_{i}'];y=n[f'rate_y_{i}'][:,0];mask=abs(t)<=.035;ax=axes[k,j];ax.plot(t[mask]*1000,y[mask],'-o',ms=3,c='black',label='Native rate')
  for x,y,c,label in [((row.initial_rate_measurement_us-row.state_measurement_us)/1000,row.initial_p,'#1565c0','Causal initial sample'),((row.truth_rate_measurement_us-row.state_measurement_us)/1000,row.truth_initial_p,'#e67e22','Old target sample'),(0,row.physical_initial_p,'#c62828','Interpolated t0')]:ax.scatter(x,y,c=c,s=50,label=label,zorder=4)
  ax.axvline(0,color='gray',ls='--',lw=.7);ax.set(title=f'{uid} {side.upper()}',xlabel='Time relative to position t0 (ms)',ylabel='p (rad/s)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.04),ncol=4,fontsize=8);savefig(fig,'fig5_origin_clocks')
fig,axes=plt.subplots(1,3,figsize=(15,4.5));subs=['main','flight_disjoint','same_firmware','same_controller','strict','small_pair']
for ax,metric in zip(axes,['direction_accuracy','scale_ratio','signed_gain']):
 for model in cols:
  q=s[(s.split=='train')&(s.variable=='p')&(s.seed=='pooled')&(s.weighting=='flight')&(s.point=='300')&(s.width_s==.27)&(s.method=='D')&(s.model==model)].set_index('cohort').loc[subs]
  ax.plot(range(len(subs)),q[metric],'-o',color=cols[model],label=model)
 ax.set(xticks=range(len(subs)),xticklabels=['All\n48/36','Disjoint\n17/10','Firmware\n38/28','Config\n11/7','Strict\n2/1','Background\n29/23'],title=metric+' | D, 300ms');ax.tick_params(axis='x',labelsize=8)
axes[0].legend(fontsize=8);fig.suptitle('Labels: total / active pairs; overlapping subsets',y=1.02);savefig(fig,'fig6_subsets')
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
ph=pd.read_csv(OUT/'phase_forecast_audit.csv');pp=p[p.pair_uid.isin(main)];ix=np.unique(pp[['a','b']].to_numpy());ids=set(w.iloc[ix].window_id)
for _,g in ph[ph.window_id.isin(ids)].groupby('window_id'):axes[0].plot(g.time_s*1000,g.total_phase_error,alpha=.22,color='#1565c0')
axes[0].set(xlabel='Time (ms)',ylabel='Forecast - encoder phase (rad)',title='Within-flight relative phase; no fitted shift')
a=pd.read_csv(OUT/'one_period_repeatability.csv');models=['Recorded','GRU-Control','GRU-NoControl'];values=[a[(a.window_id.isin(ids))&(a.model==mo)].one_period_fast_correlation.dropna() for mo in models]
axes[1].boxplot(values,tick_labels=['Recorded','Control','NoControl']);axes[1].set(ylabel='One-period lag correlation',title='History + future, centered residual')
for method,c in [('B','black'),('D','#1565c0'),('D_cycle','#e67e22')]:
 q=d[(d.pair_uid=='validation:44')&(d.model=='Hold')&(d.width_s==.27)&(d.variable=='p')&(d.method==method)].sort_values('time_s');axes[2].plot(q.time_s*1000,q.real,label=method,c=c)
axes[2].set(xlabel='Time (ms)',ylabel='Recorded D p (rad/s)',title='Different response definitions: validation');axes[2].legend();savefig(fig,'fig7_phase_and_estimand')
# Ready-to-read tables, without dropping any seed or reference class.
s[(s.point=='300')&(s.width_s==.27)&(s.variable=='p')].to_csv(OUT/'comparison_300ms.csv',index=False)
print('Saved 7 figures')
