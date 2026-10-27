from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=pd.read_csv(OUT/'summary.csv');f=pd.read_csv(OUT/'per_flight.csv');g=s.query("subset=='common_prefix' and weighting=='window'").set_index('split')
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,measure in zip(axs,['path','endpoint']):
 for i,split in enumerate(['train','validation']):ax.bar(np.arange(4)+(i-.5)*.34,[g.loc[split,f'{t}_{measure}_rms_deg'] for t in COMBOS],width=.34,label=split)
 ax.set(xticks=np.arange(4),xticklabels=['C0+A','C0+B','C1+A','C1+B'],ylabel='SO(3) RMS (degrees)',title=measure+' error / common prefix');ax.legend()
fig.tight_layout();fig.savefig(HERE/'figures/four_combinations.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[3,1]})
terms=['label_at_C0','label_at_C1','initial_at_A','initial_at_B']
for i,split in enumerate(['train','validation']):
 axs[0].bar(np.arange(4)+(i-.5)*.34,[g.loc[split,f'path_{t}_deg2'] for t in terms],width=.34,label=split)
 axs[1].bar(i,g.loc[split,'path_interaction_deg2'])
axs[0].set(xticks=np.arange(4),xticklabels=['B-A | C0','B-A | C1','C1-C0 | A','C1-C0 | B'],ylabel='Path MSE change (degree squared)');axs[0].legend();axs[1].set(xticks=[0,1],xticklabels=['train','validation'],title='Interaction (own scale)')
for ax in axs:ax.axhline(0,color='black',lw=.7)
fig.tight_layout();fig.savefig(HERE/'figures/factor_effects.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,split in zip(axs,['train','validation']):
 x=f.query("subset=='common_prefix' and split==@split").reset_index(drop=True)
 ax.plot(x.index,x.path_label_at_C0_deg2,'o',ms=3,label='B-A at C0');ax.plot(x.index,x.path_initial_at_B_deg2,'x',ms=3,label='C1-C0 at B');ax.axhline(0,color='black',lw=.7);ax.set(title=split,xlabel='Flight index (CSV order)',ylabel='Path MSE change (degree squared)');ax.legend()
fig.tight_layout();fig.savefig(HERE/'figures/per_flight_effects.png',dpi=180);plt.close(fig)
