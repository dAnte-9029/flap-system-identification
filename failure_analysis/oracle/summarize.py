"""Flight-balanced paired RMSE differences with resampling across entire flights."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HERE=Path(__file__).resolve().parent
e=pd.read_csv(HERE/'endpoint.csv')
e['mse']=e[['mse_x','mse_y','mse_z']].sum(axis=1)
n=pd.read_csv(HERE/'native.csv')
scores=[]; contrasts=[]
for frame,groups,metrics in [(e,['stream','horizon'],['mse']), (n,['stream','period'],['full','higher'])]:
    for (c,lead),g in frame.groupby(groups):
        for metric in metrics:
            by={a:k.pivot(index='seed',columns='flight',values=metric) for a,k in g.groupby('arm')}
            for a,v in by.items():
                scores.append([c,lead,metric,a,v.shape[1],*np.sqrt(v.mean(axis=1)),np.sqrt(v.mean(axis=1)).mean()])
            idx=np.random.default_rng(17).integers(0,v.shape[1],(5000,v.shape[1]))
            for baseline,arm in [('none','normal'),('normal','pred_frozen'),('normal','oracle_frozen'),('pred_frozen','oracle_frozen'),('pred_refit','oracle_refit')]:
                b=by[baseline];a=by[arm]
                assert b.columns.equals(a.columns)
                delta=np.sqrt(b.mean(axis=1))-np.sqrt(a.mean(axis=1))
                boot=(np.sqrt(b.values[:,idx].mean(axis=2))-np.sqrt(a.values[:,idx].mean(axis=2))).mean(axis=0)
                contrasts.append([c,lead,metric,baseline,arm,b.shape[1],int((b.mean()>a.mean()).sum()),*delta,delta.mean(),*np.quantile(boot,[.025,.975])])
pd.DataFrame(scores,columns='stream horizon metric arm flights seed17 seed29 seed43 mean'.split()).to_csv(HERE/'scores.csv',index=False)
p=pd.DataFrame(contrasts,columns='stream horizon metric baseline arm flights wins seed17 seed29 seed43 gain lo hi'.split())
p.to_csv(HERE/'contrasts.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(8,3),layout='constrained')
for c,ax in enumerate(axs):
    for k,arm in enumerate(['oracle_frozen','oracle_refit']):
        b='pred_'+arm.split('_')[1]
        sub=p[(p.stream==c)&(p.metric=='mse')&(p.arm==arm)&(p.baseline==b)].sort_values('horizon')
        y=sub.gain.to_numpy()
        ax.errorbar(sub.horizon.astype(float)+(k-.5)*.04,y,yerr=np.maximum(0,[y-sub.lo,sub.hi-y]),marker='o',capsize=3,label=arm.split('_')[1])
    ax.axhline(0,color='grey',lw=.7)
    ax.set(xlabel='Horizon (s)',ylabel='Paired oracle RMSE gain ('+('m/s' if c==0 else 'rad/s')+')')
    ax.legend()
fig.savefig(HERE/'oracle_gain.png',dpi=150)

archived=pd.read_csv(HERE.parents[1]/'research/rounds/phase-module-evidence-20261125/combined/flight_mse.csv')
for arm,reference in [('none','direction'),('normal','full')]:
    a=archived[archived.arm==reference].set_index(['seed','flight']).sort_index()
    for c,unit in enumerate(['v','w']):
        for h,tag in zip([.5,1,2],['05','1','2']):
            measured=e[(e.arm==arm)&(e.stream==c)&(e.horizon==h)].set_index(['seed','flight']).sort_index()
            assert measured.index.equals(a.index)
            np.testing.assert_allclose(measured.mse,a[unit+tag],rtol=2e-6,atol=2e-6)
print('Archived endpoint parity passed for two arms, three seeds, 39 flights and six outcomes.')
