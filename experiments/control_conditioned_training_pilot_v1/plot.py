from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'M0W0':'#4169a1','M0W1':'#22a6b3','M1W0':'#d49a32','M1W1':'#b94749'}
def finish(fig,name):
 fig.tight_layout();fig.savefig(HERE/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig)
def main():
 s=pd.read_csv(OUT/'factual_summary.csv');delta=pd.read_csv(OUT/'paired_summary.csv');components=pd.read_csv(OUT/'component_assignments.csv');internal=pd.read_csv(OUT/'internal_summary.csv');curves=pd.read_csv(OUT/'factual_curves.csv.gz');effects=pd.read_csv(OUT/'internal_curves.csv.gz');points=pd.read_csv(OUT/'factual_points.csv.gz');mainids=set(components.pair_uid)
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for row,method in enumerate(['B','D']):
  d=s[(s.cohort=='main')&(s.scope=='all_heldout')&(s.variable=='p')&(s.time_label=='300')&(s.method==method)&(s.weighting=='flight')].set_index('model')
  for ax,metric in zip(axes[row],['direction_accuracy','scale_ratio','signed_gain']):
   ax.bar(range(4),[d.loc[m,metric] for m in COMBOS],color=list(COLORS.values()));ax.set_xticks(range(4),COMBOS);ax.set_title(f'{method}: {metric}');ax.axhline(1,color='gray',ls=':',lw=.7)
   if metric=='direction_accuracy':ax.set_ylim(0,1.05);ax.set_ylabel(f'{int(d.iloc[0].active_pairs)}/48 active pairs')
 fig.suptitle('Held flights; 300 ms roll response; flight equal; seed 17',y=1.02);finish(fig,'direction_amplitude_gain.png')
 fig,axes=plt.subplots(2,2,figsize=(10,7))
 for row,method in enumerate(['B','D']):
  for col,weighting in enumerate(['pair','component']):
   d=delta[(delta.scope=='all_heldout')&(delta.variable=='p')&(delta.method==method)&(delta.weighting==weighting)].set_index('model')
   if weighting=='pair':
    axes[row,col].bar(np.arange(2)-.17,[d.loc[m,'gained_pair_seed_rows'] for m in ['M0W1','M1W1']],.34,label='wrong to correct',color='#379a73');axes[row,col].bar(np.arange(2)+.17,[-d.loc[m,'lost_pair_seed_rows'] for m in ['M0W1','M1W1']],.34,label='correct to wrong',color='#b94749')
   else:axes[row,col].bar(range(2),[d.loc[m,'direction_difference']*100 for m in ['M0W1','M1W1']],color=[COLORS[m] for m in ['M0W1','M1W1']]);axes[row,col].set_ylabel('percentage points')
   axes[row,col].axhline(0,c='k',lw=.7);axes[row,col].set_xticks([0,1],['M0: W1−W0','M1: W1−W0']);axes[row,col].set_title(f'{method}; {weighting}');
 axes[0,0].legend();finish(fig,'paired_direction_wins_losses.png')
 sizes=components.groupby('component_id').size();giant=components[components.fold==0].component_id.iloc[0];fig,axes=plt.subplots(2,2,figsize=(13,7));x=np.arange(len(sizes))
 for row,method in enumerate(['B','D']):
  for ax,metric in zip(axes[row],['direction_difference','mean_amplitude_error_change']):
   for j,model in enumerate(['M0W1','M1W1']):
    d=delta[(delta.scope.isin(sizes.index))&(delta.variable=='p')&(delta.method==method)&(delta.weighting=='pair')&(delta.model==model)].set_index('scope').reindex(sizes.index);ax.bar(x+(j-.5)*.36,d[metric],.36,label=model[:2]+' W1−W0',color=COLORS[model])
    for i,n in enumerate(d.active_pairs):
     if n==0 and metric=='direction_difference':ax.text(i,.03,'NA',transform=ax.get_xaxis_transform(),ha='center',fontsize=7)
   ax.set_xticks(x,[f'{c}\nn={n}' for c,n in sizes.items()],fontsize=8);ax.axhline(0,c='gray',lw=.7);ax.set_title(f'{method}: {metric}');ax.legend(fontsize=8);i=list(sizes.index).index(giant);ax.axvspan(i-.49,i+.49,color='#e0cc77',alpha=.25)
 fig.suptitle('All 11 shared-flight components; shaded = Fold 0 giant component',y=1.01);finish(fig,'dependence_components.png')
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for row,method in enumerate(['B','D']):
  for j,var in enumerate(['p','q','r']):
   d=s[(s.cohort=='main')&(s.scope=='all_heldout')&(s.variable==var)&(s.time_label=='300')&(s.method==method)&(s.weighting=='flight')].set_index('model');axes[row,j].bar(range(4),[d.loc[m,'direction_accuracy'] for m in COMBOS],color=list(COLORS.values()));axes[row,j].set_xticks(range(4),COMBOS,rotation=20);axes[row,j].set_ylim(0,1.05);axes[row,j].set_title(f'{method} {var}; active {int(d.iloc[0].active_pairs)}/48')
 finish(fig,'coupled_axes.png')
 fig,axes=plt.subplots(1,3,figsize=(12,4));d=internal[(internal.cohort=='main')&(internal.scope=='all_heldout')&(internal.variable=='p')&(internal.time_label=='300')&(internal.weighting=='flight')].set_index('model')
 for i,m in enumerate(COMBOS):
  axes[0].bar(i-.16,d.loc[m,'C_rms'],.32,color=COLORS[m]);axes[0].bar(i+.16,d.loc[m,'H_rms'],.32,color=COLORS[m],alpha=.4,hatch='//')
  axes[1].bar(i,d.loc[m,'same_direction'],color=COLORS[m]);axes[2].bar(i,d.loc[m,'abstention_fraction'],color=COLORS[m])
 for ax,title in zip(axes,['C solid / H hatched RMS (rad/s)','CA/CB agreement when both active','CA/CB abstention fraction']):ax.set_xticks(range(4),COMBOS,rotation=20);ax.set_title(title)
 finish(fig,'internal_control_history.png')
 traj=pd.read_csv(OUT/'trajectory_summary.csv').set_index('model');fig,axes=plt.subplots(1,4,figsize=(13,4))
 for ax,key in zip(axes,['position_m_path_rms','velocity_mps_path_rms','attitude_rad_path_rms','rate_radps_path_rms']):ax.bar(range(4),[traj.loc[m,key] for m in COMBOS],color=list(COLORS.values()));ax.set_xticks(range(4),COMBOS,rotation=20);ax.set_title(key.replace('_path_rms',''))
 fig.suptitle('All held-out 1s windows; per-window RMS then flight-equal mean',y=1.02);finish(fig,'held_trajectory.png')
 for uid in json.loads((HERE/'config.json').read_text())['examples']:
  g=curves[(curves.pair_uid==uid)&(curves.variable=='p')&curves.supported];fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
  for col,arch in enumerate(['M0','M1']):
   for row,method in enumerate(['B','D']):
    real=g[(g.model==arch+'W0')&(g.method==method)].sort_values('time_ms').drop_duplicates('time_ms');axes[row,col].plot(real.time_ms,real.real,'k',lw=2,label='Recorded')
    for m in [arch+'W0',arch+'W1']:
     d=g[(g.model==m)&(g.method==method)].sort_values('time_ms').drop_duplicates('time_ms');axes[row,col].plot(d.time_ms,d.pred,color=COLORS[m],label=m)
    axes[row,col].set_ylabel(f'{method} p (rad/s)');axes[row,col].axhline(0,c='gray',lw=.7);axes[row,col].legend();axes[0,col].set_title(arch)
   for m in [arch+'W0',arch+'W1']:
    d=effects[(effects.pair_uid==uid)&(effects.model==m)&(effects.variable=='p')&effects.supported].sort_values('time_ms').drop_duplicates('time_ms');axes[2,col].plot(d.time_ms,d.C,c=COLORS[m],label=m+' C');axes[2,col].plot(d.time_ms,d.H,c=COLORS[m],ls='--',label=m+' H')
   axes[2,col].legend(fontsize=8);axes[2,col].set_xlabel('ms');axes[2,col].set_ylabel('Internal C/H (rad/s)');axes[2,col].axhline(0,c='gray',lw=.7)
  fig.suptitle(uid+' — prespecified case; held-out flights',y=1.01);finish(fig,'case_'+uid.replace(':','_')+'.png')
 fig,axes=plt.subplots(2,4,figsize=(14,7))
 for row,method in enumerate(['B','D']):
  for col,m in enumerate(COMBOS):
   d=points[points.pair_uid.isin(mainids)&(points.variable=='p')&(points.method==method)&(points.model==m)&(points.time_label=='300')];ax=axes[row,col];ax.scatter(d.real,d.pred,c=COLORS[m],s=20);lim=max(abs(d.real).max(),abs(d.pred).max())*1.05;ax.plot([-lim,lim],[-lim,lim],'k:',lw=.7);ax.axhline(0,c='gray',lw=.7);ax.axvline(0,c='gray',lw=.7);ax.axvspan(-.1,.1,color='gray',alpha=.15);ax.set(title=f'{method} {m}',xlabel='Recorded rad/s',ylabel='Predicted rad/s')
 finish(fig,'all_main_pairs.png')
 print('Saved figures for all combinations, 11 components and all five fixed cases')
if __name__=='__main__':main()
