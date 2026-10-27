from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLOR={'M0A':'#4069a3','M0B':'#df843a'}
def finish(fig,name):
 fig.tight_layout();fig.savefig(HERE/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig)
def main():
 s=pd.read_csv(OUT/'factual_summary.csv');delta=pd.read_csv(OUT/'paired_summary.csv');ass=pd.read_csv(OUT/'component_assignments.csv');curves=pd.read_csv(OUT/'factual_curves.csv.gz')
 fig,axs=plt.subplots(2,3,figsize=(10,6))
 for row,method in enumerate(['B','D']):
  g=s[(s.scope=='all_heldout')&(s.cohort=='main')&(s.variable=='p')&(s.time_label=='300')&(s.weighting=='flight')&(s.method==method)].set_index('model')
  for ax,metric in zip(axs[row],['direction_accuracy','scale_ratio','signed_gain']):
   ax.bar([0,1],[g.loc[m,metric] for m in COMBOS],color=list(COLOR.values()));ax.set(xticks=[0,1],xticklabels=['A target','B target'],title=method+': '+metric);ax.axhline(1,c='grey',ls=':',lw=.6)
   if metric=='direction_accuracy':ax.set_ylim(0,1.05)
 finish(fig,'response_direction_amplitude.png')
 fig,axs=plt.subplots(2,2,figsize=(12,6));sizes=ass.groupby('component_id').size()
 for row,method in enumerate(['B','D']):
  g=delta[(delta.scope.isin(sizes.index))&(delta.method==method)&(delta.variable=='p')&(delta.weighting=='pair')].set_index('scope').reindex(sizes.index)
  for ax,metric in zip(axs[row],['direction_difference','mean_amplitude_error_change']):
   ax.bar(np.arange(len(g)),g[metric],color='#568c98');ax.axhline(0,c='black',lw=.6);ax.set(xticks=np.arange(len(g)),xticklabels=[f'{k}\nn={v}' for k,v in sizes.items()],title=method+': B target minus A target '+metric);ax.tick_params(axis='x',labelsize=7)
   for i,k in enumerate(sizes.index):
    if (ass[ass.component_id==k].fold==0).all():ax.axvspan(i-.45,i+.45,color='gold',alpha=.15)
 finish(fig,'component_changes.png')
 t=pd.read_csv(OUT/'trajectory_summary.csv');fig,axs=plt.subplots(2,3,figsize=(11,6))
 for ax,var in zip(axs.ravel(),['position_m_rms','velocity_mps_rms','attitude_rad_rms','p_rms','q_rms','r_rms']):
  for model in COMBOS:
   for ref,style in [('A','-'),('B','--')]:
    g=t[(t.model==model)&(t.reference==ref)&(t.support=='common_B_prefix')].sort_values('horizon_ms');factor=180/np.pi if var=='attitude_rad_rms' else 1
    ax.plot(g.horizon_ms,g[var]*factor,style+'o',color=COLOR[model],label=model[-1]+' target / '+ref+' reference')
  ax.set(title=var.replace('attitude_rad','attitude_deg'),xlabel='Path horizon (ms)')
 axs[0,0].legend(fontsize=7);finish(fig,'heldout_trajectory.png')
 ids=json.loads((HERE/'config.json').read_text())['examples'];fig,axs=plt.subplots(len(ids),2,figsize=(11,12))
 for row,uid in enumerate(ids):
  for col,method in enumerate(['B','D']):
   ax=axs[row,col];d=curves[(curves.pair_uid==uid)&(curves.variable=='p')&(curves.method==method)&curves.supported]
   for j,model in enumerate(COMBOS):
    g=d[d.model==model].sort_values('time_ms').drop_duplicates('time_ms');ax.plot(g.time_ms,g.pred,label=model[-1]+' target',color=COLOR[model])
    if j==0:ax.plot(g.time_ms,g.real,'k--',label='Recorded response')
   ax.axhline(0,c='gray',lw=.5);ax.set(title=uid+' '+method,ylabel='p increment difference (rad/s)',xlabel='ms')
 axs[0,0].legend();finish(fig,'fixed_cases.png')
 logs=[]
 for p in (HERE/'folds').glob('fold*/training/*/seed17/training.jsonl'):logs.extend(json.loads(x) for x in p.read_text().splitlines())
 logs=pd.DataFrame(logs);fig,axs=plt.subplots(1,3,figsize=(12,3.5))
 for fold,ax in enumerate(axs):
  for model in COMBOS:
   g=logs[(logs.fold==fold)&(logs.model==model)];ax.plot(g.epoch,g.validation_A_flight_loss,'-',color=COLOR[model],label=model[-1]+' target/A ref');ax.plot(g.epoch,g.validation_B_flight_loss,'--',color=COLOR[model],label=model[-1]+' target/B ref')
  ax.set(title='Fold '+str(fold),xlabel='Epoch',ylabel='Validation flight-equal original scaled loss')
 axs[0].legend(fontsize=7);finish(fig,'training_curves.png')
if __name__=='__main__':main()
