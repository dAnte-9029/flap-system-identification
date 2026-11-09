"""Authorized descriptive validation residual regression; never feeds score C.

All nuisance predictions, scaling and bandwidths are frozen from train.
Only residual-response moments are estimated on validation for A/B/D.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from diagnose_pacld_a2 import StandardScaler
from diagnose_pacld_a2_v2 import OUT,CHANNELS
from score_pacld_a2_v2 import read

def ridge_statistics(gram,cross,counts,alpha=10):
    return np.linalg.solve(np.einsum('bd,dij->bij',counts,gram)+alpha*np.eye(gram.shape[-1]),np.einsum('bd,dij->bij',counts,cross))

def main():
    train,X,U,Y,P=read('train');val,V,W,T,Q=read('validation')
    r=np.load(OUT/'validation_responses.npz');ur=r['U_res'];yr=r['Y_res'];pr=r['P_res'];valid=r['local_valid']
    # Use train residual scaling saved by the train-only scoring stage.
    scales=np.load(OUT/'train_control_residual_scale.npy');sc=StandardScaler().fit(X);z=sc.transform(V);us=ur/scales
    coverage=pd.read_csv(OUT/'validation_local_coverage.csv');dates=val.date.to_numpy();unique_dates=np.unique(dates);rng=np.random.default_rng(17)
    draws=rng.integers(0,len(unique_dates),(4000,len(unique_dates)));counts=np.stack([(draws==k).sum(1) for k in range(len(unique_dates))],axis=1)
    rows=[];date_rows=[]
    for c in coverage.itertuples():
        query=np.flatnonzero((val.flight.to_numpy()==c.flight)&(val['mode'].to_numpy()==c.mode)&valid)
        pool=np.flatnonzero((val['mode'].to_numpy()==c.mode)&valid)
        if not len(query) or len(pool)<4:continue
        distance=cKDTree(z[query]).query(z[pool])[0];weights=np.exp(-.5*(distance/(c.width*c.bandwidth))**2)
        a=us[pool];b=np.c_[yr[pool],pr[pool]];gram=[];cross=[];date_n=[]
        for date in unique_dates:
            ix=dates[pool]==date;w=weights[ix];aa=a[ix];bb=b[ix]
            gram.append(aa.T@(aa*w[:,None]));cross.append(aa.T@(bb*w[:,None]));date_n.append(int(ix.sum()))
        gram=np.asarray(gram);cross=np.asarray(cross);rank=int(np.linalg.matrix_rank(gram.sum(0)))
        if rank<4:continue
        beta=ridge_statistics(gram,cross,np.ones((1,len(unique_dates))))[0]/scales[:,None]
        boot=ridge_statistics(gram,cross,counts)/scales[None,:,None]
        for axis in range(2):
            for channel in [1,3]:
                obs=beta[channel,axis];pred=beta[channel,axis+2];obsboot=boot[:,channel,axis];preboot=boot[:,channel,axis+2];lo,hi=np.quantile(obsboot,[.025,.975]);delta=preboot-obsboot
                # No arbitrary sign-rate threshold: CI must exclude zero, and stage-one date evidence must support direction.
                eligible=channel==1 and axis==0 and lo>0
                rows.append(dict(flight=c.flight,date=c.date,mode=c.mode,width=c.width,axis='pr'[axis],channel=CHANNELS[channel],G_obs=obs,G_PACLD=pred,obs_low=lo,obs_high=hi,pacld_low=np.quantile(preboot,.025),pacld_high=np.quantile(preboot,.975),absolute_error=abs(pred-obs),signed_difference=pred-obs,difference_low=np.quantile(delta,.025),difference_high=np.quantile(delta,.975),relative_bias=(pred-obs)/abs(obs) if eligible else np.nan,direction_eligible=eligible,direction_agreement=bool(obs*pred>0) if eligible else None,validation_samples=len(pool),validation_dates=sum(n>0 for n in date_n),rank=rank,condition=float(np.linalg.cond(gram.sum(0))),effective_weight_n=float(weights.sum()**2/(weights@weights))))
        # Date-specific descriptive matrices, using the identical fixed penalty.
        if c.width==1:
            for j,date in enumerate(unique_dates):
                if np.linalg.matrix_rank(gram[j])<4:continue
                bd=np.linalg.solve(gram[j]+10*np.eye(4),cross[j])/scales[:,None]
                for axis in range(2):
                    for channel in [1,3]:date_rows.append(dict(query_flight=c.flight,query_mode=c.mode,date=date,axis='pr'[axis],channel=CHANNELS[channel],G_obs=bd[channel,axis],G_PACLD=bd[channel,axis+2],n=date_n[j]))
    out=pd.DataFrame(rows);out.to_csv(OUT/'validation_evaluation_gains.csv',index=False);pd.DataFrame(date_rows).to_csv(OUT/'validation_evaluation_date_gains.csv',index=False)
    # Query flights are summaries of repeated localizations, not independent gain observations.
    summarized=out.groupby(['width','axis','channel'])[['G_obs','G_PACLD','absolute_error','signed_difference','relative_bias','obs_low','obs_high','pacld_low','pacld_high']].median().reset_index();summarized.to_csv(OUT/'validation_evaluation_gain_summary.csv',index=False)
    eligible=out[out.direction_eligible];direction=eligible.groupby(['width','axis','channel']).agg(same_direction_rate=('direction_agreement','mean'),eligible_localizations=('flight','count'),query_dates=('date','nunique')).reset_index();direction.to_csv(OUT/'validation_evaluation_direction.csv',index=False)
    (OUT/'gain_evaluation_contract.json').write_text(json.dumps(dict(user_authorized_validation_residual_regression=True,nuisance_fitting='train only',normalization='train crossfit control residual scale',bandwidth='train-only local pair bandwidth at validation X',regularization=10,bootstrap='4000 draws of 4 validation date clusters; nuisance fits held fixed',direction_gate='primary differential-roll only, stage-one sign stable and local observed bootstrap interval excludes zero; cross terms and rudder auxiliary',use_for_score_C=False,units='rad/s per normalized control command',notes='descriptive validation association, not independent local-gain generalization or causal derivative'),indent=2))
    print(summarized.to_string(index=False));print(direction.to_string(index=False))
if __name__=='__main__':main()
