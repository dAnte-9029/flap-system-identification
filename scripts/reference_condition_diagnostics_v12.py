"""History-only strata and waveform diagnostics; no outcome-based selection."""
import numpy as np
import pandas as pd
from reference_error_structure_v9 import decompose

FEATURES=['body_speed','frequency_mean','frequency_std','rate_std','rate_abs_mean','reference_std','reference_step_rms','actuator_step_rms']


def covariates(x,axis):
    h=np.asarray(x['history_state'],dtype=float);r=np.asarray(x['history_reference'],dtype=float);a=np.asarray(x['history_actuator'],dtype=float)
    return pd.DataFrame(dict(body_speed=np.linalg.norm(h[:,:,:3],axis=2).mean(1),frequency_mean=h[:,:,11].mean(1),
        frequency_std=h[:,:,11].std(1),rate_std=h[:,:,3+axis].std(1),rate_abs_mean=abs(h[:,:,3+axis]).mean(1),
        reference_std=r[:,:,axis].std(1),reference_step_rms=np.sqrt(np.mean(np.diff(r[:,:,axis],axis=1)**2,1)),
        actuator_step_rms=np.sqrt(np.mean(np.diff(a,axis=1)**2,axis=(1,2)))))


def weighted_terciles(values,logs):
    """Empirical inverse-CDF quantiles, each fit flight carries equal total weight."""
    v=np.asarray(values,dtype=float)
    if not np.isfinite(v).all():raise ValueError('nonfinite conditioning values')
    _,code,count=np.unique(logs,return_inverse=True,return_counts=True)
    w=1/count[code];order=np.argsort(v,kind='stable');c=np.cumsum(w[order]);c/=c[-1]
    return v[order[np.searchsorted(c,[1/3,2/3])]]


def bins(values,cuts):
    return np.where(values<=cuts[0],'low',np.where(values<=cuts[1],'mid','high'))


def waveform(pred,truth,held):
    d=decompose(pred[:,:,None],truth[:,:,None]);d={k:v[:,0] for k,v in d.items()}
    p=pred-pred.mean(1,keepdims=True);t=truth-truth.mean(1,keepdims=True)
    cov=np.mean(p*t,1);product=d['predicted_std']*d['truth_std']
    d.update(covariance=cov,variance_product=product,pred_variance=d['predicted_std']**2,truth_variance=d['truth_std']**2,
        correlation=np.divide(cov,product,out=np.full_like(cov,np.nan),where=product>1e-10),
        held_mse=np.mean((held-truth)**2,1),reference_effect_mse=np.mean((pred-held)**2,1))
    return d


def aggregate(g):
    m=g.mean(numeric_only=True);tv=m.truth_variance;pv=m.pred_variance
    return pd.Series(dict(n=len(g),rmse=np.sqrt(m.total_mse),centered_rmse=np.sqrt(m.centered_mse),
        normalized_centered=np.sqrt(m.centered_mse/tv) if tv>1e-10 else np.nan,
        amplitude_ratio=np.sqrt(pv/tv) if tv>1e-10 else np.nan,
        mean_correlation=g.correlation.mean(),valid_correlations=g.correlation.notna().sum(),
        reference_effect_rms=np.sqrt(m.reference_effect_mse),reference_utility=1-np.sqrt(m.total_mse/m.held_mse) if m.held_mse>1e-10 else np.nan,
        bias_share=m.bias_mse/m.total_mse,amplitude_share=m.amplitude_mse/m.total_mse,shape_share=m.shape_mse/m.total_mse))


def repeatability(perflight):
    """Prespecified descriptive screen, not multiple-testing-adjusted evidence."""
    records=[];pairs=[]
    for (axis,feature),g in perflight[(perflight.cohort=='oof')&(perflight.horizon_s==1)&(perflight.condition!='mode')].groupby(['axis','condition']):
        v=g[g.stratum.isin(['low','high'])].pivot(index=['seed','log_id'],columns='stratum',values=['n','normalized_centered','rmse','mean_correlation'])
        if not all((k,s) in v for k in ['n','normalized_centered','rmse','mean_correlation'] for s in ['low','high']):continue
        keep=(v['n']['low']>=8)&(v['n']['high']>=8);v=v[keep].dropna()
        ratio=v['normalized_centered']['high']/v['normalized_centered']['low']
        corr=v['mean_correlation']['high']-v['mean_correlation']['low'];rr=v['rmse']['high']/v['rmse']['low']
        d=pd.DataFrame(dict(normalized_ratio=ratio,corr_difference=corr,rmse_ratio=rr)).reset_index()
        if d.empty:continue
        d['axis']=axis;d['condition']=feature;pairs.append(d)
        overall=float(np.exp(np.log(ratio).groupby('log_id').mean().mean()));direction=1 if overall>=1 else -1
        flight=np.log(ratio).groupby('log_id').mean();seed=np.log(ratio).groupby('seed').mean()
        cd=corr.groupby('log_id').mean();supported=bool(len(flight)>=4)
        passed=bool(supported and max(overall,1/overall)>=1.2 and (direction*flight>0).sum()>=4 and (direction*seed>0).all() and (direction*cd<0).sum()>=4)
        records.append(dict(axis=axis,condition=feature,paired_flights=len(flight),high_low_normalized_ratio=overall,
            high_low_rmse_ratio=float(np.exp(np.log(rr).groupby('log_id').mean().mean())),mean_correlation_difference=float(cd.mean()),
            consistent_flights=int((direction*flight>0).sum()),consistent_seeds=int((direction*seed>0).sum()),correlation_agree_flights=int((direction*cd<0).sum()),
            supported=supported,repeatable_screen=passed))
    return pd.DataFrame(records),pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame()
