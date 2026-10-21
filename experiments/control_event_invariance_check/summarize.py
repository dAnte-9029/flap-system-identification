"""Descriptive small-sample summaries; selected anomaly cohort is not a population estimate."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check import OUT,HERE

def main():
    d=pd.read_csv(OUT/'metrics.csv');rows=[]
    for (variant,h,metric),g in d.groupby(['variant','horizon_ms','metric']):
        for seed in [17,29,43,'all']:
            x=g if seed=='all' else g[g.seed==seed];threshold=json.loads((HERE/'config.json').read_text())['state_difference_thresholds'][metric]
            row=dict(variant=variant,horizon_ms=h,metric=metric,seed=seed,n_windows=x.window.nunique(),n_predictions=len(x),
                mean_difference=x.difference.mean(),max_difference=x.difference.max(),rms_difference=np.sqrt(np.mean(x.difference**2)),
                max_transient=x.max_through_horizon.max(),affected_windows=x.loc[x.difference>threshold,'window'].nunique(),
                first_step_min=x.first_changed_step.min(),first_step_max=x.first_changed_step.max())
            for ref in ['donor','fixed_donor']:
                y=d[(d.variant==ref)&(d.horizon_ms==h)&(d.metric==metric)]
                y=x[['window','seed','difference']].merge(y[['window','seed','difference']],on=['window','seed'],suffixes=('_artifact','_control'))
                ratio=y.difference_artifact/y.difference_control.replace(0,np.nan)
                row[ref+'_ratio_of_means']=y.difference_artifact.mean()/y.difference_control.mean()
                row[ref+'_ratio_median']=ratio.median();row[ref+'_ratio_max']=ratio.max()
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,unit in zip(axs,['velocity','rate','attitude_deg'],['m/s','rad/s','degrees']):
        sub=d[(d.metric==metric)&(d.horizon_ms==500)]
        variants=['repeat','prior_refined','split2','split4','donor','fixed_split4'];groups=[sub[sub.variant==v].difference.to_numpy() for v in variants]
        ax.boxplot(groups,tick_labels=variants,showfliers=True);ax.tick_params(axis='x',rotation=35)
        ax.set(title=f'500 ms: {metric}',ylabel=unit);ax.grid(axis='y',alpha=.2)
    fig.suptitle('8 selected histories x 3 seeds; descriptive dependence, not independent replicates')
    fig.tight_layout();fig.savefig(OUT/'differences_500ms.png',dpi=150);plt.close(fig)
    layers=pd.read_csv(OUT/'layers.csv');fig,axs=plt.subplots(1,2,figsize=(11,4))
    z=np.load(OUT/'traces.npz');uid=pd.read_csv(OUT/'windows.csv').window.iloc[0];times=np.cumsum(z[f'{uid}__native__input_dt'][0]);seed=17
    for variant in ['split2','split4','prior_refined']:
        e=np.max(np.abs(z[f'{uid}__{seed}__{variant}__event_encoded']-z[f'{uid}__{seed}__native__event_encoded']),axis=-1)
        axs[0].plot(times,e,label=variant)
        p=z[f'{uid}__{seed}__{variant}__pred'][0];q=z[f'{uid}__{seed}__native__pred'][0]
        axs[1].plot(times,np.linalg.norm(p[:,3:6]-q[:,3:6],axis=-1),label=variant)
    axs[0].set(title='Event embedding difference',ylabel='Max absolute feature difference');axs[1].set(title='Velocity prediction difference',ylabel='m/s')
    for ax in axs:ax.set_xlabel('Time (s)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'Illustration: {uid}, seed {seed}; no future truth used')
    fig.tight_layout();fig.savefig(OUT/'layer_trace.png',dpi=150);plt.close(fig)
    print('Summaries and two figures saved')

if __name__=='__main__':main()
