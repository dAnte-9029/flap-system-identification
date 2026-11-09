"""PACLD-CRM-v1: train-only observational moment diagnostics; never optimize."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diagnose_pacld_a2 import fit

BASE = Path('experiments/phase_anchored_coupled_latent_v1')
OUT = BASE/'control_residual_moment_v1'
A2 = Path('/home/zn/flap-system-identification/.worktrees/pacld-control-effectiveness-v2')/BASE/'control_effectiveness_a2_v2'
SOURCE = Path('/home/zn/flap-system-identification/.worktrees/pacld-control-identifiability-v1')/BASE/'control_identifiability'
CHECKPOINT = Path('/home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1/runs/pacld_frozen_eval_seed17/model.pt')
SHA = '9d82640692c0baec476e276696be4f89fd89a58854361bbcfdf3dbb84cb466c9'
SEED = 17

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))

def weights(df):
    n=df.groupby(['date','flight']).flight.transform('size').to_numpy()
    nf=df.groupby('date').flight.transform('nunique').to_numpy()
    return 1/(df.date.nunique()*nf*n)

def aggregate(df, columns):
    flights=df.groupby(['date','flight'])[columns].mean().reset_index()
    dates=flights.groupby('date')[columns].mean().reset_index()
    return flights, dates

def summary(x):
    x=np.asarray(x)
    return dict(n=len(x),mean=float(x.mean()),rms=float(np.sqrt(np.mean(x*x))),**{f'q{q}':float(np.quantile(x,q/100)) for q in [0,1,5,25,50,75,95,99,100]})

def statistics():
    OUT.mkdir(exist_ok=True)
    contract=json.loads((A2/'contract.json').read_text())
    for file,key in [('features.pkl','source_feature_sha256'),('context.npy','source_context_sha256')]:
        assert hashlib.sha256((SOURCE/file).read_bytes()).hexdigest()==contract[key]
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA
    frame=pd.read_pickle(SOURCE/'features.pkl')
    identity=pd.read_csv(A2/'identity.csv')
    assert not identity.duplicated(['flight','t']).any()
    # Keyed join to the authenticated source, never implicit CSV order alignment.
    df=identity.merge(frame,on=['flight','t','date','mode'],validate='one_to_one',how='left',indicator=True)
    assert (df['_merge']=='both').all()
    df=df.drop(columns='_merge')
    source_keys=pd.MultiIndex.from_frame(frame[['flight','t']])
    positions=source_keys.get_indexer(pd.MultiIndex.from_frame(df[['flight','t']]))
    assert (positions>=0).all()
    context=np.load(SOURCE/'context.npy')[positions]
    raw=np.stack(df.X); X=np.c_[raw,context,pd.get_dummies(df['mode']).to_numpy()]
    arrays=np.load(A2/'features.npz'); U=np.stack(df.U); Y=np.stack(df.Y)[:,[0,2]]
    # A2 archive is accepted only after every source feature and target matches its keyed identity.
    for key,value in [('X',X),('U',U),('Y',Y)]:np.testing.assert_allclose(arrays[key],value,rtol=0,atol=1e-12)
    N=arrays['N']; assert np.isfinite(N).all()
    pac=pd.read_csv(A2/'pacld_train.csv')
    meta=json.loads((A2/'pacld_train_identity.json').read_text())
    assert meta['split']=='train' and meta['checkpoint_sha256']==SHA and meta['no_future_observation_assimilation']
    df=df.merge(pac,on=['flight','date','t'],validate='one_to_one',how='left')
    assert df.delta_p.notna().all() and len(df)==11221 and df.flight.nunique()==114 and df.date.nunique()==17
    assert df.sort_values('t').groupby('flight').t.diff().dropna().min()>=1000000
    assert ((df.query_s>0)&(df.query_s<.25)).all()
    np.testing.assert_allclose(df.query_s,.2+np.stack(df.offset)[:,0]*1e-6,rtol=0,atol=1e-10)
    # Recompute nuisances: old NPZ predictions do not carry independent per-row keys.
    dates=df.date.to_numpy(); residual=np.zeros((len(df),3));pred=np.zeros((len(df),2));folds=[]
    controls=np.c_[U[:,5],N[:,1],N[:,5]]
    for date in sorted(set(dates)):
        tr=dates!=date;te=~tr
        # Fit all 8 primary controls jointly to retain the original A2 alpha selection criterion.
        e=fit(X[tr],U[tr],dates[tr]);residual[te,0]=U[te,5]-e.predict(X[te])[:,5]
        en=fit(X[tr],N[tr],dates[tr]);residual[te,1:]=N[te][:,[1,5]]-en.predict(X[te])[:,[1,5]]
        row=dict(date=date,train_dates=len(set(dates[tr])),evaluation_dates=1,control_alpha=e.alpha,future_alpha=en.alpha)
        for j,design in enumerate([X,np.c_[X,U]]):
            model=fit(design[tr],Y[tr],dates[tr]);pred[te,j]=model.predict(design[te])[:,0];row[f'M{j}_alpha']=model.alpha
        folds.append(row);print('crossfit',date,flush=True)
    pd.DataFrame(folds).to_csv(OUT/'folds.csv',index=False)
    df['Y']=Y[:,0];df['Y_hat']=df.delta_p;df['E_increment']=df.Y-df.Y_hat
    df['p_observed_initial']=raw[:,3];df['E_initial']=raw[:,3]-df.belief_p
    df['p_target']=raw[:,3]+df.Y;df['p_predicted_target']=df.belief_p+df.Y_hat
    df['E_endpoint']=df.p_target-df.p_predicted_target
    identity_err=float(np.max(abs(df.E_endpoint-df.E_increment-df.E_initial)))
    assert identity_err<1e-12
    su=np.sqrt(np.mean(residual**2,axis=0));sp=float(np.sqrt(np.mean(Y[:,0]**2)))
    for j,name in enumerate(['normal','future50','future100']):
        df['U_'+name]=controls[:,j];df['R_'+name]=residual[:,j]
        df['moment_'+name]=residual[:,j]/su[j]*df.E_increment/sp
    r=residual[:,0]/su[0]
    for name,value in [('obs',df.Y),('model',df.Y_hat),('initial',df.E_initial),('endpoint',df.E_endpoint),('M0',df.Y-pred[:,0]),('M1',df.Y-pred[:,1])]:df['moment_'+name]=r*value/sp
    df['M0_prediction']=pred[:,0];df['M1_prediction']=pred[:,1]
    df['phase_group']=np.floor(np.mod(np.arctan2(raw[:,10],raw[:,11]),2*np.pi)/(np.pi/2)).astype(int)
    boundaries=np.quantile(raw[:,0],[1/3,2/3]);df['speed_group']=np.digitize(raw[:,0],boundaries)
    df['forward_speed']=raw[:,0];df['p_initial']=raw[:,3]
    columns=[c for c in df if c.startswith('moment_')]
    flights,day=aggregate(df,columns); flights.to_csv(OUT/'flight_moments.csv',index=False);day.to_csv(OUT/'date_moments.csv',index=False)
    rng=np.random.default_rng(SEED);boot=day[columns].to_numpy()[rng.integers(0,len(day),(20000,len(day)))].mean(1)
    totals={c:dict(M=float(day[c].mean()),ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),positive_dates=int((day[c]>0).sum()),negative_dates=int((day[c]<0).sum()),date_min=float(day[c].min()),date_max=float(day[c].max())) for j,c in enumerate(columns)}
    assert abs(totals['moment_obs']['M']-totals['moment_model']['M']-totals['moment_normal']['M'])<1e-12
    save('moments.json',dict(scales=dict(s_u=su.tolist(),s_p=sp,scale_reduction='window RMS over common train cohort'),moments=totals,L_moment=totals['moment_normal']['M']**2,bootstrap=dict(seed=SEED,draws=20000,cluster='date',interpretation='descriptive percentile interval; 17 dates')))
    residual_stats=[]
    for group in ['date','flight','mode']:
        for key,g in df.groupby(group):
            for name in ['normal','future50','future100']:
                residual_stats.append(dict(group=group,key=str(key),control=name,**summary(g['R_'+name]),energy_share=float(np.sum(g['R_'+name]**2)/np.sum(df['R_'+name]**2))))
    pd.DataFrame(residual_stats).to_csv(OUT/'residual_distributions.csv',index=False)
    correlations=[]
    for j in range(X.shape[1]):
        for k,name in enumerate(['normal','future50','future100']):
            corr=np.corrcoef(residual[:,k],X[:,j])[0,1] if np.std(X[:,j]) else None
            correlations.append(dict(feature=j,control=name,correlation=corr))
    pd.DataFrame(correlations).to_csv(OUT/'residual_feature_correlations.csv',index=False)
    save('residual_summary.json',dict(raw={n:summary(controls[:,j]) for j,n in enumerate(['normal','future50','future100'])},residual={n:summary(residual[:,j]) for j,n in enumerate(['normal','future50','future100'])},residual_correlations=np.corrcoef(residual.T).tolist(),feature_order='0:3 body velocity; 3:6 FRD rates; 6:9 gravity; 9 frequency; 10:12 phase sin/cos; 12:24 control baseline/old/recent history; 24:36 references/airspeed/mode flags; 36:48 availability; 48+ mode dummies'))
    groups=[]
    for group in ['mode','phase_group','speed_group']:
        for key,g in df.groupby(group):
            f,d=aggregate(g,columns)
            for c in columns:groups.append(dict(group=group,key=int(key),metric=c,M=float(d[c].mean()),dates=len(d),flights=len(f),windows=len(g),positive_dates=int((d[c]>0).sum()),negative_dates=int((d[c]<0).sum())))
            d.to_csv(OUT/f'{group}_{key}_date_moments.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'condition_moments.csv',index=False)
    # Diagnostic influence only; primary cohort and normalization are never changed.
    sensitivity=[]
    for fraction in [.01,.05]:
        take=np.abs(residual[:,0])<=np.quantile(abs(residual[:,0]),1-fraction)
        _,d=aggregate(df[take],['moment_normal'])
        sensitivity.append(dict(excluded_top_fraction=fraction,M=float(d.moment_normal.mean()),removed_windows=int((~take).sum())))
    save('influence.json',dict(tail=sensitivity,leave_one_date_out={str(d):float(day.loc[day.date!=d,'moment_normal'].mean()) for d in day.date}))
    # Monte Carlo: iid windows versus date/flight-balanced iid draws (weights target M).
    w=weights(df);m=df.moment_normal.to_numpy();M=float(w@m);records=[]
    for scheme,p in [('window_uniform',None),('date_flight_balanced',w)]:
        target=float(m.mean()) if p is None else M
        for B in [32,128,256,512]:
            ids=rng.choice(len(df),(10000,B),replace=True,p=p);est=m[ids].mean(1);sq=est**2
            records.append(dict(scheme=scheme,B=B,repeats=len(est),M_global=M,design_expectation=target,mean=float(est.mean()),bias_to_global=float(est.mean()-M),sd=float(est.std()),sign_flip=float(np.mean(est*M<0)),M2_mean=float(sq.mean()),M2_q05=float(np.quantile(sq,.05)),M2_q50=float(np.median(sq)),M2_q95=float(np.quantile(sq,.95)),variance_identity_error=float(abs(sq.mean()-est.mean()**2-est.var())),mean_unique_flights=float(np.mean([len(set(df.flight.to_numpy()[ix])) for ix in ids[:1000]]))))
    pd.DataFrame(records).to_csv(OUT/'batch_statistics.csv',index=False)
    # Fixed gradient cohort before any gradient measurements: each date, four phase groups, both residual signs where available.
    selected=[]
    for _,g in df.assign(residual_sign=(df.R_normal>=0).astype(int)).groupby(['date','phase_group','residual_sign']):
        selected.append(int(rng.choice(g.index)))
    selected=np.array(selected);rng.shuffle(selected)
    df.loc[selected,['flight','t','date','phase_group','R_normal']].assign(batch=np.arange(len(selected))%4).to_csv(OUT/'gradient_cohort.csv',index=False)
    df.drop(columns=['X','U','offset','saturation']).to_csv(OUT/'windows.csv',index=False)
    save('audit.json',dict(baseline='7746e96d456f25a33e60b074e7f8f47aedb62d26',split='train',windows=len(df),flights=df.flight.nunique(),dates=df.date.nunique(),checkpoint_sha256=SHA,source=str(SOURCE),a2=str(A2),source_feature_sha256=contract['source_feature_sha256'],source_context_sha256=contract['source_context_sha256'],identity_join=['flight','t'],minimum_spacing_us=int(df.sort_values('t').groupby('flight').t.diff().dropna().min()),endpoint_identity_max_error=identity_err,speed_tertiles=boundaries.tolist(),gradient_windows=len(selected),seed=SEED,nuisance='refit original A2 LODO with inner 3 date folds and original 8-control/2-response alpha criterion',no_parameter_updates=True))
    print(json.dumps(totals,indent=2),flush=True)

if __name__=='__main__':
    statistics()
