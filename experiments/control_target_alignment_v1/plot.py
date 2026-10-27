"""Fixed-case, whole-cohort and full-data figures; no case selection by score."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
def done(fig,name):fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=165,bbox_inches='tight');plt.close(fig)
summary=pd.read_csv(OUT/'main_p_300.csv');colors=['#2369a4','#e68a2e','#6d4894','#319468']
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for ax,col,title,ref in zip(axs,['direction_accuracy','scale_ratio','signed_gain'],['Direction agreement','Predicted / real RMS','Signed response gain'],[1,1,1]):
 for j,m in enumerate(['B','D']):
  g=summary[summary.method==m].set_index('model').loc[['O0','O1','O2','O3']];ax.bar(np.arange(4)+(j-.5)*.34,g[col],width=.34,label=m)
 ax.set_xticks(range(4),['O0','O1','O2','O3']);ax.axhline(ref,color='gray',lw=.8,ls='--');ax.set_title(title);ax.legend();ax.set_ylim(0,1.35 if col!='direction_accuracy' else 1.09)
fig.suptitle('Frozen 48 pairs, 300 ms p; equal-flight descriptive weights');done(fig,'oracle_direction_amplitude_gain')
dec=pd.read_csv(OUT/'decomposition_summary.csv').query("split=='train' and cohort=='main' and variable=='p' and weighting=='flight'")
fig,ax=plt.subplots(figsize=(7,4))
for j,m in enumerate(['B','D']):
 r=dec[dec.method==m].iloc[0];ax.bar(np.arange(3)+(j-.5)*.3,[r.future_rms,r.origin_rms,r.increment_rms],width=.3,label=m)
ax.set_xticks(range(3),['Future sampling error','Causal origin error','Combined increment error']);ax.set_ylabel('rad/s RMS');ax.legend();ax.set_title('Terms are correlated: not additive independent variance shares');done(fig,'error_decomposition')
full=pd.read_csv(OUT/'full_window_audit.csv.gz');fig,axs=plt.subplots(1,3,figsize=(13,3.7))
for ax,var in zip(axs,VARS):
 for split,color in [('train','#2369a4'),('validation','#e68a2e')]:
  vals=np.sort(abs(full.loc[full.split==split,f'{var}_origin_error']));ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=split,color=color)
 ax.set_xlabel(f'|{var} causal initial - B origin| [rad/s]');ax.set_ylabel('Window empirical CDF');ax.set_xscale('symlog',linthresh=.01);ax.legend()
done(fig,'full_origin_error_distribution')
comp=pd.read_csv(OUT/'oracle_per_group.csv').query("split=='train' and grouping=='component' and variable=='p' and method=='D'");fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=True);names=sorted(comp.group.unique())
for j,model in enumerate(['O0','O1','O2','O3']):
 g=comp[comp.model==model].set_index('group').reindex(names)
 for ax,col in zip(axs,['direction_accuracy','scale_ratio']):ax.plot(range(len(names)),g[col],marker='o',label=model,color=colors[j])
axs[0].set_ylabel('Direction agreement');axs[1].set_ylabel('Pred / real RMS');axs[1].set_xticks(range(len(names)),[f'{x}\nn={int(comp[(comp.group==x)&(comp.model=="O0")].pairs.iloc[0])}' for x in names]);axs[0].legend(ncol=4);axs[0].set_title('All 11 dependence components; C03 is the 25-pair Fold0 component');done(fig,'oracle_components_D')
curves=pd.read_csv(OUT/'oracle_curves.csv.gz');models=pd.read_csv(PILOT/'results/factual_curves.csv.gz');cfg=json.loads((HERE/'config.json').read_text())
for uid in cfg['cases']:
 fig,axs=plt.subplots(2,2,figsize=(13,7));case=curves[(curves.pair_uid==uid)&(curves.variable=='p')&curves.supported];mm=models[(models.pair_uid==uid)&(models.variable=='p')&models.model.isin(['M0W0','M1W0'])&models.supported]
 for col,method in enumerate(['B','D']):
  for row in [0,1]:
   ax=axs[row,col];g=case[(case.method==method)&(case.model=='O0')].sort_values('time_ms').drop_duplicates('time_ms');ax.plot(g.time_ms,g.real,'k',lw=2,label='Recorded reference');ax.plot(g.time_ms,g.pred,color=colors[0],lw=1.8,label='O0 original-label oracle');ax.axhline(0,color='gray',lw=.7);ax.set_title(f'{uid} | {method}');ax.set_ylabel('p increment difference [rad/s]');ax.set_xlabel('Nominal output time [ms]')
   if row==0:
    for oracle,color in zip(['O1','O2'],colors[1:3]):
     gg=case[(case.method==method)&(case.model==oracle)].sort_values('time_ms').drop_duplicates('time_ms');ax.plot(gg.time_ms,gg.pred,color=color,ls='--',label=oracle)
   else:
    for (model,fold),gg in mm[mm.method==method].groupby(['model','fold']):
     gg=gg.sort_values('time_ms').drop_duplicates('time_ms');ax.plot(gg.time_ms,gg.pred,ls='--',color='#ba3f3f' if model=='M0W0' else '#935bb5',alpha=.7,label=f'{model} 6ep fold{fold}')
   ax.legend(fontsize=8)
 done(fig,'case_'+uid.replace(':','_'))
model=pd.read_csv(OUT/'model_comparison_300.csv').query("variable=='p' and weighting=='flight' and method in ['B','D']");fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,method in zip(axs,['B','D']):
 g=model[model.method==method];labels=[f'{r.budget_epochs}ep {r.model}\ns{r.seed}' for r in g.itertuples()];x=np.arange(len(g));ax.bar(x-.22,g.label_error_rms,width=.22,label='O0-reference');ax.bar(x,g.learning_error_rms,width=.22,label='Model-O0');ax.bar(x+.22,g.total_error_rms,width=.22,label='Model-reference');ax.set_xticks(x,labels,rotation=45,ha='right',fontsize=8);ax.set_ylabel('rad/s RMS');ax.set_title(method+' | correlated terms, not variance shares');ax.legend(fontsize=8)
done(fig,'model_label_error_comparison')
print('Figures',len(list(FIG.glob('*.png'))))
