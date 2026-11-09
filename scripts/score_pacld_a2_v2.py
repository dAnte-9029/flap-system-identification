"""Train-only nuisance fits and local gains; final validation scores are read-only."""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from diagnose_pacld_a2 import fit, StandardScaler
from diagnose_pacld_a2_v2 import OUT,SOURCE,CHANNELS,crossflight_pairs,local_fit

def response_score(y,p):
    denominator=np.mean(y*y,axis=0)
    return np.where(denominator>0,1-np.mean((y-p)**2,axis=0)/denominator,np.nan)

def read(split):
    source=SOURCE if split=='train' else OUT/'validation_features'
    df=pd.read_pickle(source/'features.pkl').reset_index(drop=True)
    context=np.load(source/'context.npy')
    X=np.c_[np.stack(df.X),context,np.stack([df['mode'].to_numpy()==m for m in [3,15]],axis=1)]
    U=np.stack(df.U)[:,4:];Y=np.stack(df.Y)[:,[0,2]]
    pred=pd.read_csv(OUT/f'pacld_{split}.csv');assert not pred.duplicated(['flight','t']).any()
    joined=df[['flight','t']].merge(pred,on=['flight','t'],how='left',validate='one_to_one');P=joined[['delta_p','delta_r']].to_numpy()
    assert len(pred)==len(df) and np.isfinite(P).all()
    keep=df['mode'].isin([3,15]).to_numpy()
    if split=='train':assert keep.all()
    else:df.loc[~keep,['flight','date','mode','t']].assign(reason='mode_absent_from_train').to_csv(OUT/'validation_unevaluable_modes.csv',index=False)
    return df.loc[keep].reset_index(drop=True),X[keep],U[keep],Y[keep],P[keep]

def main():
    train,X,U,Y,P=read('train');val,V,W,T,Q=read('validation')
    assert not set(train.flight)&set(val.flight) and not set(train.date)&set(val.date)
    dates=train.date.to_numpy();flights=train.flight.to_numpy();modes=train['mode'].to_numpy()
    # Separate baselines for observed and PACLD response; identical model family.
    residuals=np.full((len(train),8),np.nan);alphas=[]
    target=np.c_[U,Y,P]
    for date in np.unique(dates):
        tr=np.flatnonzero(dates!=date);te=np.flatnonzero(dates==date)
        for start,end,name in [(0,4,'control'),(4,6,'observed'),(6,8,'pacld')]:
            model=fit(X[tr],target[tr,start:end],dates[tr]);residuals[te,start:end]=target[te,start:end]-model.predict(X[te]);alphas.append(dict(date=date,target=name,alpha=model.alpha))
    e=fit(X,U,dates);m=fit(X,Y,dates);mp=fit(X,P,dates);m1=fit(np.c_[X,U],Y,dates)
    ur=W-e.predict(V);yr=T-m.predict(V);pr=Q-mp.predict(V);lr=m1.predict(np.c_[V,W])-m.predict(V)
    rows=[]
    for f,ix in val.groupby('flight').groups.items():
        date=val.loc[ix,'date'].iloc[0]
        for axis in range(2):
            y=yr[ix,axis]
            rows.append(dict(flight=f,date=date,axis='pr'[axis],n=len(ix),zero_response_mse=float(np.mean(y*y)),M1_response_mse=float(np.mean((y-lr[ix,axis])**2)),PACLD_response_mse=float(np.mean((y-pr[ix,axis])**2)),M0_raw_mse=float(np.mean((T[ix,axis]-m.predict(V[ix])[:,axis])**2)),M1_raw_mse=float(np.mean((T[ix,axis]-m1.predict(np.c_[V[ix],W[ix]])[:,axis])**2)),PACLD_raw_mse=float(np.mean((T[ix,axis]-Q[ix,axis])**2))))
    errors=pd.DataFrame(rows);errors.to_csv(OUT/'validation_flight_scores.csv',index=False)
    numeric=[c for c in errors if c.endswith('mse')];bydate=errors.groupby(['date','axis'])[numeric].mean().reset_index();bydate.to_csv(OUT/'validation_date_scores.csv',index=False)
    rng=np.random.default_rng(17);scores=[]
    for axis,a in bydate.groupby('axis'):
        ix=rng.integers(0,len(a),(4000,len(a)));den=a.zero_response_mse.to_numpy()
        for name in ['M1','PACLD']:
            err=a[name+'_response_mse'].to_numpy();boot=1-err[ix].mean(1)/den[ix].mean(1)
            scores.append(dict(axis=axis,model=name,response_score=float(1-err.mean()/den.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),dates=len(a),zero_baseline_score=0.,raw_rmse=float(np.sqrt(a[name+'_raw_mse'].mean())),M0_rmse=float(np.sqrt(a.M0_raw_mse.mean()))))
    pd.DataFrame(scores).to_csv(OUT/'validation_summary.csv',index=False)
    sc=StandardScaler().fit(X);z=sc.transform(X);zv=sc.transform(V);su=StandardScaler().fit(residuals[:,:4]);u=su.transform(residuals[:,:4]);np.save(OUT/'train_control_residual_scale.npy',su.scale_);observed=residuals[:,4:6];modeled=residuals[:,6:8]
    gains=[];coverage=[];conditional_local=np.zeros_like(T);conditional_obs=np.zeros_like(T);valid=np.zeros(len(val),bool)
    for mode in np.unique(val['mode']):
        b=np.flatnonzero(modes==mode);pairs=crossflight_pairs(z,flights,b)
        for f,ix in val[val['mode']==mode].groupby('flight').groups.items():
            ix=np.asarray(ix)
            for width in [.5,1.,2.]:
                mo,so,info,go,overlap=local_fit(z,u,observed,pairs,zv[ix],width,su.scale_)
                mm,sm,info2,gp,overlap2=local_fit(z,u,modeled,pairs,zv[ix],width,su.scale_)
                assert info==info2 and np.array_equal(overlap,overlap2)
                coverage.append(dict(flight=f,date=val.loc[ix[0],'date'],mode=int(mode),width=width,supported_queries=int(overlap.sum()),unsupported_queries=int((~overlap).sum()),support_status='full' if overlap.all() else ('partial' if overlap.any() else 'unevaluable'),**info))
                if not info['identified'] or not overlap.any():continue
                if width==1:
                    good=ix[overlap];conditional_local[good]=ur[good]@gp.T;conditional_obs[good]=ur[good]@go.T;valid[good]=True
                for axis in range(2):
                    for c in [1,3]:
                        # Primary evidence only differential -> p; differential -> r is supported cross-coupling.
                        direction_eligible=c==1
                        gains.append(dict(flight=f,date=val.loc[ix[0],'date'],mode=int(mode),width=width,axis='pr'[axis],channel=CHANNELS[c],G_obs=float(go[axis,c]),G_PACLD=float(gp[axis,c]),absolute_error=float(abs(go[axis,c]-gp[axis,c])),signed_difference=float(gp[axis,c]-go[axis,c]),relative_bias=float((gp[axis,c]-go[axis,c])/abs(go[axis,c])) if direction_eligible and go[axis,c]>0 else np.nan,direction_eligible=direction_eligible,direction_agreement=bool(go[axis,c]*gp[axis,c]>0) if direction_eligible and go[axis,c]>0 else None,n=int(overlap.sum())))
    pd.DataFrame(gains).to_csv(OUT/'validation_gains.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'validation_local_coverage.csv',index=False);pd.DataFrame(alphas).to_csv(OUT/'nuisance_alphas.csv',index=False)
    supported=[]
    for f,indices in val.groupby('flight').groups.items():
        ix=np.asarray(indices);ix=ix[valid[ix]]
        if not len(ix):continue
        for axis in range(2):
            supported.append(dict(flight=f,date=val.loc[ix[0],'date'],axis='pr'[axis],n=len(ix),zero_mse=float(np.mean(yr[ix,axis]**2)),M1_mse=float(np.mean((yr[ix,axis]-lr[ix,axis])**2)),PACLD_mse=float(np.mean((yr[ix,axis]-pr[ix,axis])**2))))
    sf=pd.DataFrame(supported);sf.to_csv(OUT/'validation_supported_flight_scores.csv',index=False)
    sd=sf.groupby(['date','axis'])[['zero_mse','M1_mse','PACLD_mse']].mean().reset_index();sd.to_csv(OUT/'validation_supported_date_scores.csv',index=False)
    ss=[]
    for axis,a in sd.groupby('axis'):
        ids=rng.integers(0,len(a),(4000,len(a)));den=a.zero_mse.to_numpy()
        for name in ['M1','PACLD']:
            err=a[name+'_mse'].to_numpy();boot=1-err[ids].mean(1)/den[ids].mean(1)
            ss.append(dict(axis=axis,model=name,score=float(1-err.mean()/den.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),dates=len(a),supported_samples=int(valid.sum())))
    pd.DataFrame(ss).to_csv(OUT/'validation_supported_summary.csv',index=False)
    np.savez_compressed(OUT/'validation_responses.npz',Y=T,P=Q,U=W,U_res=ur,Y_res=yr,P_res=pr,M1_res=lr,local_P_response=conditional_local,local_obs_response=conditional_obs,local_valid=valid)
    g=pd.DataFrame(gains);date_g=g.groupby(['date','width','axis','channel'])[['G_obs','G_PACLD','absolute_error','signed_difference','relative_bias']].median().reset_index();date_g.to_csv(OUT/'validation_date_gains.csv',index=False)
    gs=date_g.groupby(['width','axis','channel'])[['G_obs','G_PACLD','absolute_error','signed_difference','relative_bias']].median().reset_index();gs.to_csv(OUT/'validation_gain_summary.csv',index=False)
    (OUT/'scoring_summary.json').write_text(json.dumps(dict(samples=len(val),flights=val.flight.nunique(),dates=val.date.nunique(),train_samples=len(train),train_flights=train.flight.nunique(),train_dates=train.date.nunique(),valid_local_samples=int(valid.sum()),scores=scores,gains=gs.astype(object).where(pd.notnull(gs),None).to_dict('records'),scope='differential roll primary; rudder yaw auxiliary; gains trained on train residuals, localized by validation X only; no validation parameter fit'),indent=2))
    print(pd.DataFrame(scores).to_string(index=False));print(gs.to_string(index=False))
if __name__=='__main__':main()
