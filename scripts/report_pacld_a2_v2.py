"""Date-clustered stage-one summaries; no data-dependent tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_pacld_a2_v2 import OUT

def main():
    errors=pd.read_csv(OUT/'date_errors.csv');g=pd.read_csv(OUT/'local_g.csv');le=pd.read_csv(OUT/'local_errors.csv');coverage=pd.read_csv(OUT/'local_coverage.csv');identity=pd.read_csv(OUT/'identity.csv');rng=np.random.default_rng(17);summary=[]
    for axis,a in errors.groupby('axis'):
        ix=rng.integers(0,len(a),(4000,len(a)))
        for model,base in [('M1','M0'),('negative','M0'),('M1','negative'),('M1','without_differential'),('M1','without_rudder')]:
            x=a[base].to_numpy();y=a[model].to_numpy();delta=np.sqrt(x[ix].mean(1))-np.sqrt(y[ix].mean(1))
            summary.append(dict(axis=axis,model=model,baseline=base,rmse=float(np.sqrt(y.mean())),baseline_rmse=float(np.sqrt(x.mean())),gain=float(np.sqrt(x.mean())-np.sqrt(y.mean())),low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),positive_dates=int((x>y).sum()),dates=len(a)))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'prediction_summary.csv',index=False)
    # Flight/mode first, then date aggregation; no pair/window bootstrap.
    ld=le.groupby(['date','width','axis'])[['state_mse','full_mse']].mean().reset_index();ld.to_csv(OUT/'local_date_errors.csv',index=False)
    ls=[]
    for (width,axis),a in ld.groupby(['width','axis']):
        ix=rng.integers(0,len(a),(4000,len(a)));b=a.state_mse.to_numpy();v=a.full_mse.to_numpy();d=np.sqrt(b[ix].mean(1))-np.sqrt(v[ix].mean(1))
        ls.append(dict(width=width,axis=axis,state_rmse=float(np.sqrt(b.mean())),full_rmse=float(np.sqrt(v.mean())),gain=float(np.sqrt(b.mean())-np.sqrt(v.mean())),low=float(np.quantile(d,.025)),high=float(np.quantile(d,.975)),positive_dates=int((b>v).sum()),dates=len(a)))
    pd.DataFrame(ls).to_csv(OUT/'local_summary.csv',index=False)
    gd=g.groupby(['date','width','axis','channel']).G.median().reset_index();gd.to_csv(OUT/'local_date_g.csv',index=False)
    gs=gd.groupby(['width','axis','channel']).G.agg(['median','min','max',lambda v:int((v>0).sum()),'count']).reset_index();gs.columns=['width','axis','channel','median','min','max','positive_dates','dates'];gs.to_csv(OUT/'gain_summary.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for k,axis in enumerate('pr'):
        a=errors[errors.axis==axis];x=np.arange(len(a))
        for name in ['M1','negative']:ax[k].plot(x,np.sqrt(a.M0)-np.sqrt(a[name]),'o-',label=name)
        ax[k].axhline(0,color='gray');ax[k].set_xticks(x,a.date.str[5:],rotation=90);ax[k].set_ylabel('RMSE improvement (rad/s)');ax[k].set_title(axis);ax[k].legend()
    fig.tight_layout();fig.savefig(OUT/'date_prediction.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,7))
    for ai,axis in enumerate('pr'):
        for ci,c in enumerate(['differential','rudder']):
            for width in [.5,1.,2.]:
                a=gd[(gd.axis==axis)&(gd.channel==c)&(gd.width==width)];ax[ai,ci].plot(np.arange(len(a)),a.G,'o-',label=str(width))
            ax[ai,ci].axhline(0,color='gray');ax[ai,ci].set_title(c+' -> '+axis);ax[ai,ci].set_xlabel('held-out date index');ax[ai,ci].legend()
    fig.tight_layout();fig.savefig(OUT/'independent_local_g.png',dpi=160);plt.close(fig)
    pred=np.load(OUT/'predictions.npz');corr=np.corrcoef(np.c_[pred['U'][:,4:],pred['N'][:,4:]].T)[:4,4:]
    result=dict(samples=len(identity),flights=identity.flight.nunique(),dates=identity.date.nunique(),date_bootstrap_replicates=4000,seed=17,negative_normal_100ms_same_channel_correlation=np.diag(corr).tolist(),prediction=summary.to_dict('records'),local_prediction=ls,gain=gs.to_dict('records'),local_evaluation_pairs=int(le[le.width==1].query("axis=='p'").pairs.sum()),local_fit_rows=len(coverage),local_fit_identified=int(coverage.identified.fillna(False).sum()),overlap_min=float(coverage.overlap_fraction.min()),overlap_median=float(coverage.overlap_fraction.median()),local_condition_quantiles=coverage.condition.quantile([0,.5,.95,1]).to_dict())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(summary.to_string(index=False));print(pd.DataFrame(ls).to_string(index=False));print(gs.to_string(index=False))
if __name__=='__main__':main()
