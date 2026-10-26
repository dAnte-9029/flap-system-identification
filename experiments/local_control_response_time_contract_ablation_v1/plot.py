from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 dest=HERE/'figures' if OUT==HERE/'results' else OUT.parent/'figures';dest.mkdir(exist_ok=True);s=pd.read_csv(OUT/'factual_summary.csv');inte=pd.read_csv(OUT/'internal_summary.csv');cfg=json.loads((HERE/'config.json').read_text());curves=pd.read_csv(OUT/'factual_curves.csv.gz');ic=pd.read_csv(OUT/'internal_curves.csv.gz');colors={17:'#0072B2',29:'#D55E00',43:'#009E73'};versions=['V0','V1_source_assumption']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 def finish(fig,name):fig.tight_layout();fig.savefig(dest/name,dpi=180,bbox_inches='tight');plt.close(fig)
 main=s[(s.split=='train')&(s.cohort=='main')&(s.variable=='p')&(s.time_label=='300')&(s.weighting=='flight')]
 fig,axes=plt.subplots(1,3,figsize=(13,4));positions=np.arange(3)
 for model,color in [('GRU-Control','#0072B2'),('GRU-NoControl','#D55E00')]:
  for version,style in zip(versions,['--o','-s']):
   g=main[(main.model==model)&(main.variant==version)&(main.seed=='pooled')].set_index('method').loc[['A','B','D']]
   for ax,key in zip(axes,['direction_accuracy','scale_ratio','signed_gain']):ax.plot(positions,g[key],style,color=color,label=model+' '+version.replace('_source_assumption','*'));ax.set_xticks(positions,['A raw','B physical reference','D slow 270 ms']);ax.set_ylabel(key);ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);axes[0].set_ylim(0,1);fig.suptitle('300 ms, 48 frozen train pairs; * source-time hypothesis, not verified physical alignment',y=1.02);finish(fig,'fig1_direction_amplitude_gain.png')
 fig,axes=plt.subplots(1,3,figsize=(14,4));g=inte.query("split=='train' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight' and seed=='pooled'")
 for j,key in enumerate(['C_rms','H_rms','both_near_fraction']):
  for mi,model in enumerate(['GRU-Control','GRU-NoControl']):
   vals=[g[(g.model==model)&(g.variant==v)][key].iloc[0] for v in versions];axes[j].bar(np.arange(2)+mi*.32,vals,.3,label=model)
  axes[j].set_xticks([.16,1.16],['V0','V1 source assumption']);axes[j].set_ylabel(key);axes[j].legend(fontsize=8)
 finish(fig,'fig2_internal_control_history.png')
 for uid in cfg['examples']:
  fig,axes=plt.subplots(2,3,figsize=(13,7));sub=curves[(curves.pair_uid==uid)&(curves.variable=='p')&curves.supported];inn=ic[(ic.pair_uid==uid)&(ic.variable=='p')&ic.supported]
  for ri,model in enumerate(['GRU-Control','GRU-NoControl']):
   for j,method in enumerate(['B','D']):
    d=sub[(sub.model==model)&(sub.method==method)]
    for (version,seed),g in d.groupby(['variant','seed']):g=g.sort_values('time_ms').drop_duplicates('time_ms');axes[ri,j].plot(g.time_ms,g.pred,'-' if version!= 'V0' else '--',color=colors[seed],label=version.replace('_source_assumption','*')+' '+str(seed))
    g=d[(d.variant=='V0')&(d.seed==17)].sort_values('time_ms').drop_duplicates('time_ms');axes[ri,j].plot(g.time_ms,g.real,'k',lw=2,label='Recorded '+method);axes[ri,j].set_title(model+' '+method)
   for (version,seed),g in inn[inn.model==model].groupby(['variant','seed']):g=g.sort_values('time_ms').drop_duplicates('time_ms');axes[ri,2].plot(g.time_ms,g.C,'-' if version!='V0' else '--',color=colors[seed],label=version.replace('_source_assumption','*')+' '+str(seed))
   axes[ri,2].set_title(model+' internal C (no swapped truth)')
   for ax in axes[ri]:ax.axhline(0,c='gray',lw=.6);ax.axvline(300,c='gray',ls=':',lw=.6);ax.set(xlabel='Elapsed receiver time (ms)',ylabel='p difference (rad/s)')
  axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle(uid+' — dashed V0, solid source-time hypothesis',y=1.02);finish(fig,'case_'+uid.replace(':','_')+'.png')
 cohorts=['main','flight_disjoint','same_firmware_controller','strict','small_pair','both_future_quiet'];fig,axes=plt.subplots(1,3,figsize=(14,5));g=s.query("split=='train' and variable=='p' and time_label=='300' and weighting=='flight' and seed=='pooled' and method=='D' and model=='GRU-Control'")
 for vi,v in enumerate(versions):
  gg=g[g.variant==v].set_index('cohort').loc[cohorts]
  for ax,key in zip(axes,['direction_accuracy','scale_ratio','signed_gain']):ax.barh(np.arange(len(cohorts))+vi*.35,gg[key],.32,label=v.replace('_source_assumption','*'));ax.set_yticks(np.arange(len(cohorts))+.17,[f'{c} n={int(gg.loc[c,"pairs"])}' for c in cohorts]);ax.set_xlabel(key);ax.invert_yaxis()
 axes[0].legend(fontsize=8);finish(fig,'fig3_fixed_cohorts_slow.png')
 changes=pd.read_csv(OUT/'paired_factual_changes.csv.gz');mem=pd.read_csv(SWAP/'results/cohorts.csv');ids=set(mem.query("split=='train' and cohort=='main'").pair_uid);dd=changes[changes.pair_uid.isin(ids)&(changes.time_label=='300')&(changes.variable=='p')&(changes.model=='GRU-Control')];fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,method in zip(axes,['A','B','D']):
  g=dd[dd.method==method]
  for seed,z in g.groupby('seed'):ax.scatter(z.pred_V0,z.pred_new,c=colors[seed],label=str(seed),s=20,alpha=.7)
  limit=max(abs(g.pred_V0).max(),abs(g.pred_new).max())*1.1;ax.plot([-limit,limit],[-limit,limit],'k:');ax.set(xlabel='V0 factual p difference',ylabel='V1 hypothesis factual p difference',title=method+' all 48 pairs × 3 seeds')
 axes[0].legend(title='Seed');finish(fig,'fig4_paired_prediction_changes.png')
 fig,axes=plt.subplots(1,3,figsize=(12,4));g=s.query("split=='train' and cohort=='main' and time_label=='300' and weighting=='flight' and seed=='pooled' and method=='D'")
 for ax,var in zip(axes,['p','q','r']):
  for model,color in [('GRU-Control','#0072B2'),('GRU-NoControl','#D55E00')]:
   gg=g[(g.variable==var)&(g.model==model)].set_index('variant').loc[versions];ax.plot([0,1],gg.scale_ratio,'o-',color=color,label=model)
  ax.set(xticks=[0,1],xticklabels=['V0','V1 assumption'],ylabel='Prediction / observed RMS',title=var+' slow response');ax.axhline(1,c='gray',ls=':')
 axes[0].legend(fontsize=8);finish(fig,'fig5_pqr_coupling.png')
 print('Saved',len(list(dest.glob('*.png'))),'figures')
if __name__=='__main__':main()
