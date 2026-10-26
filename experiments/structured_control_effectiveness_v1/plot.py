from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    dest=HERE/'figures';dest.mkdir(exist_ok=True);colors={'M0':'#0072B2','M1':'#D55E00','M2':'#009E73'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def finish(fig,name):fig.tight_layout();fig.savefig(dest/name,dpi=180,bbox_inches='tight');plt.close(fig)
    h=pd.read_csv(OUT/'training_history.csv');s=pd.read_csv(OUT/'factual_summary.csv');ins=pd.read_csv(OUT/'internal_summary.csv');f=pd.read_csv(OUT/'factual_curves.csv.gz');e=pd.read_csv(OUT/'internal_curves.csv.gz')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name,g in h.groupby('model'):
        for ax,key in zip(axes,['train_window_loss','validation_flight_loss','gradient_norm_mean']):ax.plot(g.epoch,g[key],'o-',color=colors[name],label=name);ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend();finish(fig,'fig1_training.png')
    main=s.query("split=='train' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight'")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for i,name in enumerate(colors):
        g=main[main.model==name].set_index('method').loc[['A','B','D']]
        for ax,key in zip(axes,['direction_accuracy','scale_ratio','signed_gain']):ax.bar(np.arange(3)+(i-1)*.24,g[key],.23,color=colors[name],label=name);ax.set_xticks(range(3),['A raw','B physical-time','D slow']);ax.set_ylabel(key)
    axes[0].set_ylim(0,1);axes[0].legend();fig.suptitle('Seed 17, 6 epochs each; 48 seen train pairs; 300 ms');finish(fig,'fig2_direction_amplitude.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4));g=ins.query("split=='train' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight'").set_index('model')
    for i,name in enumerate(colors):
        axes[0].bar(np.arange(2)+(i-1)*.24,g.loc[name,['C_rms','H_rms']],.23,color=colors[name],label=name)
    axes[0].set(xticks=[0,1],xticklabels=['C','H'],ylabel='Internal prediction RMS (rad/s)');axes[0].legend()
    ids=set(pd.read_csv(SWAP/'results/cohorts.csv').query("split=='train' and cohort=='main'").pair_uid)
    d=e[(e.time_label=='300')&(e.variable=='p')&e.pair_uid.isin(ids)]
    for name,z in d.groupby('model'):axes[1].scatter(z.CA,z.CB,s=15,c=colors[name],label=name);axes[2].scatter(z.C,z.H,s=15,c=colors[name],label=name)
    axes[1].set(xlabel='CA (rad/s)',ylabel='CB (rad/s)');axes[2].set(xlabel='C (rad/s)',ylabel='H (rad/s)')
    for ax in axes[1:]:ax.axhline(0,c='gray',lw=.7);ax.axvline(0,c='gray',lw=.7)
    fig.suptitle('Model-internal decomposition; no observed swapped truth');finish(fig,'fig3_internal_response.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,var in zip(axes,['p','q','r']):
        g=s.query("split=='train' and cohort=='main' and time_label=='300' and weighting=='flight' and method=='D'");g=g[g.variable==var].set_index('model')
        ax.bar(list(colors),g.loc[list(colors)].scale_ratio,color=list(colors.values()));ax.axhline(1,c='gray',ls=':');ax.set(title=var,ylabel='Predicted / observed RMS')
    finish(fig,'fig4_coupled_axes.png')
    cohorts=['main','flight_disjoint','same_firmware_controller','strict','small_pair','both_future_quiet'];fig,axes=plt.subplots(1,3,figsize=(15,5))
    g=s.query("split=='train' and variable=='p' and time_label=='300' and weighting=='flight' and method=='D'")
    for i,name in enumerate(colors):
        z=g[g.model==name].set_index('cohort').loc[cohorts]
        for ax,key in zip(axes,['direction_accuracy','scale_ratio','signed_gain']):ax.barh(np.arange(len(cohorts))+(i-1)*.24,z[key],.23,color=colors[name],label=name);ax.set_yticks(range(len(cohorts)),[f'{c}: {int(z.loc[c,"active_pairs"])}/{int(z.loc[c,"pairs"])} active' for c in cohorts]);ax.set_xlabel(key);ax.invert_yaxis()
    axes[0].legend();finish(fig,'fig5_subsets.png')
    tr=pd.read_csv(OUT/'validation_trajectory_summary.csv').set_index('model');fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,var in zip(axes,['position_m','velocity_mps','attitude_rad','rate_radps']):ax.bar(list(colors),tr.loc[list(colors),var+'_endpoint_norm'],color=list(colors.values()));ax.set_title(var);ax.set_ylabel('Mean native 1 s endpoint norm')
    fig.suptitle('27 validation flights equally weighted; checkpoint-selection data');finish(fig,'fig6_trajectory.png')
    for uid in json.loads((HERE/'config.json').read_text())['examples']:
        fig,axes=plt.subplots(2,3,figsize=(13,7));sub=f[(f.pair_uid==uid)&f.supported];inter=e[(e.pair_uid==uid)&e.supported]
        for j,var in enumerate(['p','q','r']):
            for row,method in enumerate(['B','D']):
                dd=sub[(sub.variable==var)&(sub.method==method)]
                for name,z in dd.groupby('model'):
                    z=z.sort_values('time_ms').drop_duplicates('time_ms');axes[row,j].plot(z.time_ms,z.pred,c=colors[name],label=name)
                z=dd[dd.model=='M0'].sort_values('time_ms').drop_duplicates('time_ms');axes[row,j].plot(z.time_ms,z.real,c='black',lw=2,label='Recorded');axes[row,j].axhline(0,c='gray',lw=.6);axes[row,j].axvline(300,c='gray',ls=':',lw=.6);axes[row,j].set(title=var+' '+method,xlabel='Time (ms)',ylabel='Response difference (rad/s)')
        axes[0,0].legend();fig.suptitle(uid+' — all models seed 17 / six epochs');finish(fig,'case_'+uid.replace(':','_')+'.png')
    print('Saved',len(list(dest.glob('*.png'))),'PNG figures')
if __name__=='__main__':main()
