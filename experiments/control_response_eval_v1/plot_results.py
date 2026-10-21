"""Frozen cohort visualizations; success/failure cases chosen after evaluation for illustration only."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import HERE,OUT,write_json,file_hash
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COLORS={'GRU-Control':'tab:blue','GRU-NoControl':'tab:orange'}

def main():
    summary=pd.read_csv(OUT/'response_summary.csv');metrics=pd.read_parquet(OUT/'response_metrics.parquet');swaps=pd.read_parquet(OUT/'swap_metrics.parquet')
    comparisons=pd.read_csv(OUT/'paired_comparisons.csv');cases=[]
    val=metrics.query('cohort=="signal" and split=="validation"')
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for row,h in enumerate([500,1000]):
        for col,metric in enumerate(['velocity','rate']):
            ax=axs[row,col];s=summary.query('cohort=="signal" and split=="validation" and horizon_ms==@h and metric==@metric and seed=="all"')
            for model,g in s.groupby('model'):ax.plot(g.point_ms,g.response_rmse,'o-',label=model,color=COLORS[model])
            truth=s[s.model=='GRU-Control'];ax.plot(truth.point_ms,truth.true_response_rms,'k--',label='Observed contrast RMS (zero-response error)')
            n=summary.query('cohort=="small_control" and split=="train" and horizon_ms==@h and metric==@metric and seed=="all" and model=="GRU-Control"')
            ax.plot(n.point_ms,n.true_response_rms,':',color='gray',label=f'Train small-command reference (n={int(n.pairs.iloc[0])})')
            ax.set(title=f'{h} ms cohort: {metric}',xlabel='Time (ms)',ylabel='m/s' if metric=='velocity' else 'rad/s');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIG/'response_error_growth.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for row,h in enumerate([500,1000]):
        for col,metric in enumerate(['velocity','rate']):
            ax=axs[row,col];d=val.query('horizon_ms==@h and point_ms==@h and metric==@metric')
            per=np.sqrt(d.groupby(['pair_uid','model']).squared_error.mean().unstack())
            ax.scatter(per['GRU-NoControl'],per['GRU-Control'],s=35,alpha=.8)
            lim=1.05*per.to_numpy().max();ax.plot([0,lim],[0,lim],'k--');ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='NoControl response error',ylabel='Control response error',title=f'{h} ms {metric}: {len(per)} pairs');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'paired_error_distribution.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for row,h in enumerate([500,1000]):
        for col,metric in enumerate(['velocity','rate']):
            ax=axs[row,col];d=val.query('horizon_ms==@h and point_ms==@h and metric==@metric')
            ds=pd.concat([d.assign(flight_id=d.flight_a),d.assign(flight_id=d.flight_b)])
            f=np.sqrt(ds.groupby(['flight_id','model']).squared_error.mean().unstack())
            gain=(f['GRU-NoControl']-f['GRU-Control']).sort_values()
            ax.bar(np.arange(len(gain)),gain,color=np.where(gain>0,'tab:blue','tab:red'))
            ax.axhline(0,color='k',lw=.8);ax.set(title=f'{h} ms {metric}',xlabel='Flight rank (different order per panel)',ylabel='RMSE reduction');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'flight_error_gains.png',dpi=170);plt.close(fig)
    # Rank illustrative cases by mean normalized MSE improvement, using frozen train loss scales.
    norm=json.loads((HERE.parents[1]/'experiments/velocity_rate_gru64_horizon_training_v1/train_1000ms/normalization.json').read_text())
    scales={'velocity':norm['loss_scales'][1],'rate':norm['loss_scales'][3]}
    for h in [500,1000]:
        d=val.query('horizon_ms==@h and point_ms==@h and metric in ["velocity","rate"]').copy()
        d['scaled_mse']=d.squared_error/d.metric.map(scales)**2
        score=d.groupby(['pair_uid','model']).scaled_mse.mean().unstack();gain=score['GRU-NoControl']-score['GRU-Control']
        for label,uid in [('largest_improvement',gain.idxmax()),('worst_relative_case',gain.idxmin())]:
            part=val[val.pair_uid==uid];info=part.iloc[0]
            fig,axs=plt.subplots(2,3,figsize=(12,6),sharex=True)
            for row,metric in enumerate(['velocity','rate']):
                for c,axis in enumerate('xyz'):
                    ax=axs[row,c];d=part[part.metric==metric]
                    true=d.drop_duplicates('point_ms').sort_values('point_ms');t=np.r_[0,true.point_ms/1000]
                    ax.plot(t,np.r_[0,true['true_'+axis]],'k-',lw=2,label='Observed increment contrast')
                    for model,g in d.groupby('model'):
                        pivot=g.pivot(index='point_ms',columns='seed',values='pred_'+axis).sort_index()
                        ax.plot(t,np.r_[0,pivot.mean(1)],color=COLORS[model],label=model)
                        ax.fill_between(t,np.r_[0,pivot.min(1)],np.r_[0,pivot.max(1)],color=COLORS[model],alpha=.15)
                    ax.set(title=f'{metric} {axis}',xlabel='Time (s)',ylabel='m/s' if metric=='velocity' else 'rad/s');ax.grid(alpha=.2)
            axs[0,0].legend(fontsize=7);fig.suptitle(f'A: {h} ms {label}; {uid}; historical v/R/w gaps={info.v_rmse:.2f}/{info.R_rmse_deg:.1f}deg/{info.w_rmse:.2f}\nPost-hoc illustration only; cohort and metrics unchanged; bands=three-seed range',fontsize=10)
            fig.tight_layout();name=f'A_{h}ms_{label}.png';fig.savefig(FIG/name,dpi=160);plt.close(fig)
            # Fixed history A/B sensitivity has no counterfactual observed label.
            d=swaps.query('cohort=="signal" and split=="validation" and model=="GRU-Control" and pair_uid==@uid')
            fig,axs=plt.subplots(2,3,figsize=(12,6),sharex=True)
            for row,metric in enumerate(['velocity','rate']):
                for c,axis in enumerate('xyz'):
                    ax=axs[row,c];g=d[d.metric==metric]
                    for side,color in [('a','tab:blue'),('b','tab:green')]:
                        pivot=g.pivot(index='point_ms',columns='seed',values=f'swap_{side}_{axis}').sort_index();t=np.r_[0,pivot.index/1000]
                        ax.plot(t,np.r_[0,pivot.mean(1)],label=f'Fixed history {side.upper()}',color=color)
                        ax.fill_between(t,np.r_[0,pivot.min(1)],np.r_[0,pivot.max(1)],color=color,alpha=.15)
                    ax.axhline(0,color='k',lw=.6);ax.set(title=f'{metric} {axis}',xlabel='Time (s)',ylabel='m/s' if metric=='velocity' else 'rad/s');ax.grid(alpha=.2)
            axs[0,0].legend(fontsize=8);fig.suptitle(f'B: command A minus command B; {uid}\nModel sensitivity only: no counterfactual truth; bands=three-seed range',fontsize=10)
            fig.tight_layout();bname=f'B_{h}ms_{label}.png';fig.savefig(FIG/bname,dpi=160);plt.close(fig)
            cases.append(dict(pair_uid=uid,horizon_ms=h,label=label,flight_a=info.flight_a,flight_b=info.flight_b,
                selection='post-hoc rank of frozen-cohort normalized MSE gain; not used for cohort, metrics or model selection',
                gain_score=float(gain.loc[uid]),experiment_A='results/figures/'+name,experiment_B='results/figures/'+bname))
            z=np.load(OUT/f'predictions_{h}ms.npz');pi=int(np.flatnonzero(z['pair_index']==int(uid.split(':')[1]))[0])
            indices=[int(z['i'][pi]),int(z['j'][pi])]
            fig,axs=plt.subplots(2,3,figsize=(13,7))
            for idx,side,color in zip(indices,['A','B'],['tab:blue','tab:green']):
                times=[];commands=[];elapsed=0.
                for k,dt in enumerate(z['dt'][idx]):
                    if dt<=0:continue
                    ev=z['events'][idx,k,:z['event_length'][idx,k]]
                    times.extend(elapsed+np.r_[0,np.cumsum(ev[:,4])[:-1]])
                    commands.extend(ev[:,:4]);elapsed+=float(dt)
                times.append(elapsed);commands.append(commands[-1]);commands=np.array(commands)
                for c,ax in enumerate([axs[0,0],axs[0,1],axs[1,0],axs[1,1]]):
                    ax.step(times,commands[:,c],where='post',label=f'Recorded command {side}',color=color)
                    ax.set(title=['Motor','Left elevon','Right elevon','Rudder'][c],xlabel='Native elapsed time (s)',ylabel='Normalized command');ax.grid(alpha=.2)
                ax=axs[indices.index(idx),2]
                truth=z['truth'][idx,:,:3]-z['truth_initial'][idx,:3]
                ax.plot(np.r_[0,truth[:,1]],np.r_[0,truth[:,0]],'k-o',label='Observed')
                for model in ['GRU-Control','GRU-NoControl']:
                    pred=np.mean([z[f'{model}_{seed}'][idx,:,:3]-z['initial'][idx,:3] for seed in [17,29,43]],axis=0)
                    ax.plot(np.r_[0,pred[:,1]],np.r_[0,pred[:,0]],'-o',label=model,color=COLORS[model])
                ax.set(title=f'History {side}: factual relative path',xlabel='East displacement (m)',ylabel='North displacement (m)');ax.axis('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
            axs[0,0].legend(fontsize=7);fig.suptitle(f'{uid}: recorded command tapes and factual paths; post-hoc illustration only')
            fig.tight_layout();cname=f'controls_paths_{h}ms_{label}.png';fig.savefig(FIG/cname,dpi=150);plt.close(fig)
            cases[-1]['commands_and_paths']='results/figures/'+cname
    bs=pd.read_csv(OUT/'swap_summary.csv');fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axs,['velocity','rate']):
        d=bs.query('cohort=="signal" and split=="validation" and seed=="all" and model=="GRU-Control" and horizon_ms==1000 and metric==@metric')
        for col,label in [('swap_rms','Native tape swap'),('common_grid_response_rms','Common event-grid swap'),('timing_refinement_rms','Same command, extra event subdivision')]:ax.plot(d.point_ms,d[col],'o-',label=label)
        ax.set(title=metric,xlabel='Time (ms)',ylabel='Response norm');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIG/'swap_and_event_encoding.png',dpi=170);plt.close(fig)
    write_json(OUT/'cases.json',cases)
    print('Saved',len(list(FIG.glob('*.png'))),'figures and post-hoc case selection records')

if __name__=='__main__':main()
