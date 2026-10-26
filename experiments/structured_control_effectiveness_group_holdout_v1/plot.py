"""Descriptive figures; all seeds visible, no confidence intervals or model selection."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'M0':'#536477','M1':'#d87924','M2':'#168d91'}
FIG=HERE/'figures'
def finish(fig,name):
    fig.tight_layout();fig.savefig(FIG/name,dpi=160,bbox_inches='tight');plt.close(fig)
def main():
    FIG.mkdir(exist_ok=True)
    hist=pd.read_csv(OUT/'training_history.csv');summary=pd.read_csv(OUT/'factual_summary.csv');internal=pd.read_csv(OUT/'internal_summary.csv');changes=pd.read_csv(OUT/'paired_summary.csv');components=pd.read_csv(OUT/'component_assignments.csv');points=pd.read_csv(OUT/'factual_points.csv.gz');curves=pd.read_csv(OUT/'factual_curves.csv.gz');decomp=pd.read_csv(OUT/'internal_curves.csv.gz')
    primary=summary.query("cohort=='main' and time_label=='300' and variable=='p' and weighting=='flight'")
    fig,ax=plt.subplots(2,3,figsize=(12,6),sharex=True)
    for fold in range(3):
        for model in COLORS:
            for seed in [17,29,43]:
                g=hist.query('fold==@fold and model==@model and seed==@seed')
                for row,key in enumerate(['train_window_loss','validation_flight_loss']):ax[row,fold].plot(g.epoch,g[key],color=COLORS[model],alpha=.7,label=model if seed==17 else None)
        ax[0,fold].set_title(f'Fold {fold} (own loss scales)');ax[1,fold].set_xlabel('Epoch');ax[0,fold].legend()
    ax[0,0].set_ylabel('Train window loss');ax[1,0].set_ylabel('Validation flight loss');finish(fig,'fig1_training.png')
    fig,ax=plt.subplots(2,3,figsize=(12,6))
    for row,method in enumerate(['B','D']):
        d=primary.query("scope=='all_heldout' and method==@method")
        for j,key in enumerate(['direction_accuracy','scale_ratio','signed_gain']):
            for k,model in enumerate(COLORS):
                g=d[d.model==model];pooled=g[g.seed=='pooled'].iloc[0];v=g[g.seed!='pooled'][key].to_numpy();ax[row,j].bar(k,pooled[key],color=COLORS[model],alpha=.65);ax[row,j].scatter(k+np.array([-.12,0,.12]),v,c='k',s=20)
            ax[row,j].set_xticks(range(3),COLORS);ax[row,j].set_title(f'{method}: {key}');ax[row,j].axhline(1,color='k',ls=':',lw=.7)
            if j==0:ax[row,j].set_ylim(0,1.06);ax[row,j].set_ylabel(f"{method}: {int(d.iloc[0].active_pairs)}/48 active pairs")
    fig.suptitle('300 ms p; flight equal; bars pool seed scores, dots are seeds',y=1.02);finish(fig,'fig2_direction_amplitude_gain.png')
    fig,ax=plt.subplots(2,3,figsize=(12,7))
    for row,method in enumerate(['B','D']):
        for col,seed in enumerate([17,29,43]):
            d=changes[(changes.scope=='all_heldout')&(changes.variable=='p')&(changes.method==method)&(changes.weighting=='pair')&(changes.seed==str(seed))]
            for k,model in enumerate(['M1','M2']):
                r=d[d.model==model].iloc[0];ax[row,col].bar(k-.15,r.gained_pair_seed_rows,.3,color=COLORS[model],label='wrong to right' if k==0 else None);ax[row,col].bar(k+.15,-r.lost_pair_seed_rows,.3,color=COLORS[model],hatch='//',alpha=.5,label='right to wrong' if k==0 else None)
            ax[row,col].axhline(0,color='k',lw=.7);ax[row,col].set_xticks([0,1],['M1−M0','M2−M0']);ax[row,col].set_title(f'{method}; seed {seed}');ax[row,col].legend(fontsize=8)
    finish(fig,'fig3_paired_direction_changes.png')
    fig,ax=plt.subplots(2,2,figsize=(13,7))
    sizes=components.groupby('component_id').size();labels=[f'{c}\nn={n}' for c,n in sizes.items()];xx=np.arange(len(sizes))
    for row,method in enumerate(['B','D']):
        for j,key in enumerate(['direction_difference','mean_absolute_error_change']):
            for k,model in enumerate(['M1','M2']):
                d=changes[(changes.scope.isin(sizes.index))&(changes.variable=='p')&(changes.method==method)&(changes.weighting=='pair')&(changes.seed=='pooled')&(changes.model==model)].set_index('scope').reindex(sizes.index)
                ax[row,j].bar(xx+(k-.5)*.36,d[key],.36,label=model+'−M0',color=COLORS[model])
            ax[row,j].set_xticks(xx,labels,fontsize=8);ax[row,j].axhline(0,color='k',lw=.7);ax[row,j].set_title(f'{method}: {key}');ax[row,j].legend(fontsize=8)
            if j==0:
                for ii,n in enumerate(d.active_pairs):
                    if n==0:ax[row,j].text(xx[ii],.04,'NA',transform=ax[row,j].get_xaxis_transform(),ha='center',fontsize=8)
            giant=components.query('fold==0').component_id.unique()[0];gi=list(sizes.index).index(giant);ax[row,j].axvspan(gi-.48,gi+.48,color='#e3bf5d',alpha=.18)
            ax[row,j].set_xlim(-.6,len(sizes)-.4)
    fig.suptitle('All 11 dependence components; shaded = Fold 0 giant component; no independence claim',y=1.01);finish(fig,'fig4_components.png')
    fig,ax=plt.subplots(1,3,figsize=(12,3.5))
    d=internal.query("cohort=='main' and scope=='all_heldout' and time_label=='300' and variable=='p' and weighting=='flight' and seed=='pooled'").set_index('model')
    for k,model in enumerate(COLORS):
        for j,key in enumerate(['C_rms','H_rms']):ax[0].bar(k+(j-.5)*.32,d.loc[model,key],.32,color=COLORS[model],alpha=1 if j==0 else .4,hatch=None if j==0 else '//',label=key if k==0 else None)
        ax[1].bar(k,d.loc[model,'same_direction'],color=COLORS[model]);ax[1].text(k,.03,f"active {int(d.loc[model,'both_active_pairs'])}/48",rotation=90,ha='center',color='white')
        ax[2].bar(k,d.loc[model,'abstention_fraction'],color=COLORS[model])
    for a in ax:a.set_xticks(range(3),COLORS)
    ax[0].set_title('Internal C/H RMS (rad/s)');ax[0].legend();ax[1].set_title('CA/CB sign agreement (both > 0.1)');ax[1].set_ylim(0,1.05);ax[2].set_title('CA/CB abstention fraction');ax[2].set_ylim(0,1.05);finish(fig,'fig5_internal_sensitivity.png')
    fig,ax=plt.subplots(2,3,figsize=(12,6))
    for row,method in enumerate(['B','D']):
        for j,var in enumerate(['p','q','r']):
            d=summary[(summary.cohort=='main')&(summary.scope=='all_heldout')&(summary.time_label=='300')&(summary.variable==var)&(summary.method==method)&(summary.weighting=='flight')&(summary.seed=='pooled')].set_index('model')
            ax[row,j].bar(range(3),[d.loc[m,'direction_accuracy'] for m in COLORS],color=list(COLORS.values()));ax[row,j].set_xticks(range(3),COLORS);ax[row,j].set_ylim(0,1.05);ax[row,j].set_title(f'{method} {var}: {int(d.iloc[0].active_pairs)}/48 active')
    finish(fig,'fig6_coupled_direction.png')
    traj=pd.read_csv(OUT/'trajectory_summary.csv');fig,ax=plt.subplots(1,4,figsize=(13,3.5))
    for j,key in enumerate(['position_m_path_rms','velocity_mps_path_rms','attitude_rad_path_rms','rate_radps_path_rms']):
        for k,model in enumerate(COLORS):
            v=traj[traj.model==model][key].to_numpy();ax[j].bar(k,v.mean(),color=COLORS[model],alpha=.65);ax[j].scatter(k+np.array([-.12,0,.12]),v,c='k',s=18)
        ax[j].set_xticks(range(3),COLORS);ax[j].set_title(key.replace('_path_rms','').replace('_',' '))
    fig.suptitle('Unseen-flight 1s trajectories: per-window RMS, then equal-flight mean; dots = seeds',y=1.02);finish(fig,'fig7_held_trajectory.png')
    mainids=set(components.pair_uid);pairchange=pd.read_csv(OUT/'paired_changes.csv.gz');d=pairchange[(pairchange.pair_uid.isin(mainids))&(pairchange.time_label=='300')&(pairchange.variable=='p')&(pairchange.method=='D')].groupby('pair_uid').absolute_error_change.mean().sort_values()
    examples=list(dict.fromkeys(['train:124','train:336','train:412','train:99','train:203',d.index[0],d.index[-1]]));save(OUT/'illustration_selection.json',dict(fixed=['train:124','train:336','train:412','validation:44'],posthoc_direction_change_cases=['train:99','train:203'],direction_case_reason='99 gains D direction in all seeds for both new models; 203 loses it in two seeds for both. No cases removed from scoring.',posthoc_best_absolute_error_change=d.index[0],posthoc_worst_absolute_error_change=d.index[-1],rule='D p 300ms mean paired absolute-error change over both new models and all seeds; illustrations only, no filtering or model selection'))
    for uid in examples+['validation:44']:
        folds=sorted(curves[curves.pair_uid==uid].fold.unique())
        for fold in folds:
            fig,ax=plt.subplots(3,3,figsize=(13,9),sharex=True)
            for col,seed in enumerate([17,29,43]):
                g=curves[(curves.pair_uid==uid)&(curves.fold==fold)&(curves.seed==seed)&(curves.variable=='p')&curves.supported]
                for row,method in enumerate(['B','D']):
                    real=g[(g.model=='M0')&(g.method==method)].sort_values('time_ms').drop_duplicates('time_ms');ax[row,col].plot(real.time_ms,real.real,'k',lw=2,label='Recorded')
                    for model in COLORS:
                        q=g[(g.model==model)&(g.method==method)].sort_values('time_ms').drop_duplicates('time_ms');ax[row,col].plot(q.time_ms,q.pred,color=COLORS[model],label=model)
                    ax[row,col].set_ylabel(f'{method} p difference (rad/s)');ax[row,col].axhline(0,color='k',lw=.5);ax[row,col].legend(fontsize=8);ax[0,col].set_title(f'seed {seed}')
                for model in COLORS:
                    q=decomp[(decomp.pair_uid==uid)&(decomp.fold==fold)&(decomp.seed==seed)&(decomp.variable=='p')&(decomp.model==model)&decomp.supported].sort_values('time_ms').drop_duplicates('time_ms');ax[2,col].plot(q.time_ms,q.C,color=COLORS[model],label=model+' C');ax[2,col].plot(q.time_ms,q.H,color=COLORS[model],ls='--',alpha=.6,label=model+' H')
                ax[2,col].axhline(0,color='k',lw=.5);ax[2,col].set_ylabel('Internal C / H (rad/s)');ax[2,col].set_xlabel('ms');ax[2,col].legend(fontsize=7,ncol=2)
            fig.suptitle(f'{uid}; fold {fold}'+(' — selection-exposed validation case' if uid.startswith('validation') else ' — held-out flights'),y=1.01);finish(fig,f'case_{uid.replace(":","_")}_fold{fold}.png')
    # Every main pair is shown, retaining all seeds and correct/incorrect quadrants.
    fig,ax=plt.subplots(2,3,figsize=(12,8))
    for row,method in enumerate(['B','D']):
        for col,model in enumerate(COLORS):
            d=points[(points.pair_uid.isin(mainids))&(points.variable=='p')&(points.method==method)&(points.model==model)&(points.time_label=='300')]
            for seed,mark in zip([17,29,43],['o','s','^']):
                g=d[d.seed==seed];ax[row,col].scatter(g.real,g.pred,s=18,marker=mark,alpha=.6,label=str(seed),color=COLORS[model])
            extent=max(abs(d.real).max(),abs(d.pred).max())*1.05;ax[row,col].plot([-extent,extent],[-extent,extent],'k:',lw=.7);ax[row,col].axhline(0,color='gray',lw=.5);ax[row,col].axvline(0,color='gray',lw=.5);ax[row,col].axvspan(-.1,.1,color='gray',alpha=.1);ax[row,col].set_title(f'{method}: {model}, 48 pairs × 3 seeds');ax[row,col].set_xlabel('Recorded (rad/s)');ax[row,col].set_ylabel('Predicted (rad/s)');ax[row,col].legend(fontsize=8)
    finish(fig,'fig8_all_pairs_scatter.png')

if __name__=='__main__':main()
