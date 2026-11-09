"""A2-v2: date-held-out observational validation, train only.

The stage-one entry point never opens validation or a dynamics checkpoint.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from diagnose_pacld_a2 import StandardScaler, Ridge, fit, transform

OUT = Path('experiments/phase_anchored_coupled_latent_v1/control_effectiveness_a2_v2')
SOURCE = Path('/home/zn/flap-system-identification/.worktrees/pacld-control-identifiability-v1/experiments/phase_anchored_coupled_latent_v1/control_identifiability')
ROOT = Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
CHANNELS = ['motor', 'differential', 'symmetric', 'rudder']

def zoh_mean(ts, values, start, end):
    edges = np.r_[start, ts[(ts > start) & (ts < end)], end]
    ix = np.searchsorted(ts, edges[:-1], side='right') - 1
    if (ix < 0).any() or np.max(edges[1:] - ts[ix]) > 160000:
        raise ValueError('missing or stale control support')
    return np.sum(values[ix] * np.diff(edges)[:, None], axis=0) / (end-start)

def load_features(source):
    df = pd.read_pickle(source/'features.pkl').reset_index(drop=True)
    meta = {m['flight_id']: m for m in json.loads((ROOT/'flight_provenance.json').read_text()) if m['split']=='train'}
    assert set(df.flight) <= set(meta) and df.flight.nunique()==114 and df.date.nunique()==17
    context = np.load(source/'context.npy')
    assert context.shape == (len(df),24)
    X = np.c_[np.stack(df.X), context, pd.get_dummies(df['mode']).to_numpy()]
    U = np.stack(df.U); Y = np.stack(df.Y)[:,[0,2]]
    neg = np.full_like(U,np.nan); checks=[]
    for fid,g in df.groupby('flight',sort=False):
        motor = pd.read_parquet(ROOT/'native_actuator_motors_train.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
        servo = pd.read_parquet(ROOT/'native_actuator_servos_train.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
        status = pd.read_parquet(ROOT/'native_vehicle_status_train.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
        for i,row in g.iterrows():
            t=int(row.t); outputs=[]; ok=True
            for tab,fields in [(motor,['control[0]']),(servo,[f'control[{k}]' for k in range(3)])]:
                ts=tab.timestamp_us.to_numpy();v=tab[fields].to_numpy(float);j=np.searchsorted(ts,t,side='left')-1
                try:
                    if j<0 or t-ts[j]>160000: raise ValueError('base')
                    # Same dimensionality and duration as factual 0-50/0-100ms.
                    outputs.append(np.r_[zoh_mean(ts,v,t+250000,t+300000)-v[j],zoh_mean(ts,v,t+250000,t+350000)-v[j]])
                except ValueError:ok=False
            after=status[(status.timestamp_us>t)&(status.timestamp_us<=t+350000)]
            if (after.nav_state!=row['mode']).any():ok=False
            if ok:
                raw=np.r_[outputs[0][:1],outputs[1][:3],outputs[0][1:],outputs[1][3:]]
                neg[i]=np.r_[transform(raw[:4]),transform(raw[4:])]
        checks.append(dict(flight=fid,date=meta[fid]['flight_date'],n=len(g),negative_valid=int(np.isfinite(neg[g.index]).all(1).sum())))
    keep=np.isfinite(neg).all(1)
    pd.DataFrame(checks).to_csv(OUT/'negative_coverage.csv',index=False)
    assert np.all(np.stack(df.offset)[:,0] < 50000) # all Y precedes negative-control start
    np.savez_compressed(OUT/'features.npz',X=X[keep],U=U[keep],N=neg[keep],Y=Y[keep])
    df.loc[keep,['flight','date','mode','t']].to_csv(OUT/'identity.csv',index=False)
    return df.loc[keep].reset_index(drop=True),X[keep],U[keep],neg[keep],Y[keep]

def crossflight_pairs(z, flight, indices, k=4):
    """Bounded nearest-neighbor pairs with distinct flights; no response access."""
    indices=np.asarray(indices); tree=cKDTree(z[indices]);pairs=set()
    if len(indices)<2:return np.empty((0,2),int)
    # Query by flight to guarantee candidates are from a different flight.
    for f in np.unique(flight[indices]):
        a=indices[flight[indices]==f];b=indices[flight[indices]!=f]
        if not len(b):continue
        _,nn=cKDTree(z[b]).query(z[a],k=min(k,len(b)))
        nn=np.asarray(nn).reshape(len(a),-1)
        for i,jj in zip(a,b[nn]):
            for j in jj:pairs.add(tuple(sorted((int(i),int(j)))))
    return np.asarray(sorted(pairs),int).reshape(-1,2)

def local_fit(z, u, y_train, pairs, query_x, scale, uscale):
    """All labels supplied here belong to training dates. Query is state-only."""
    i,j=pairs.T
    center=(z[i]+z[j])/2
    distance=cKDTree(query_x).query(center)[0]
    # Training neighborhood: closest 24 pair centers per evaluation state.
    _,near=cKDTree(center).query(query_x,k=min(24,len(center)))
    selected=np.unique(near); bandwidth=max(float(np.median(distance[selected])),1e-12)
    weights=np.exp(-.5*(distance[selected]/(scale*bandwidth))**2)
    i=i[selected];j=j[selected];dx=z[j]-z[i];du=u[j]-u[i];dy=y_train[j]-y_train[i]
    m=Ridge(10,False).fit(np.c_[dx,du],dy,sample_weight=weights)
    state=Ridge(10,False).fit(dx,dy,sample_weight=weights)
    ur=du-Ridge(10,False).fit(dx,du,sample_weight=weights).predict(dx)
    gram=(ur*np.sqrt(weights[:,None])).T@(ur*np.sqrt(weights[:,None]))
    rank=int(np.linalg.matrix_rank(gram));condition=float(np.linalg.cond(gram))
    # Extrapolation is diagnosed against observed train cross-flight support radius.
    train_dist=np.linalg.norm(dx,axis=1)
    qdistance=cKDTree(np.r_[z[i],z[j]]).query(query_x)[0]
    support_radius=float(np.max(train_dist));m.support_indices=np.unique(np.r_[i,j])
    overlap=qdistance<=support_radius
    identified=rank==4 and np.isfinite(condition) and len(selected)>dx.shape[1]+4
    return m,state,dict(pairs=len(selected),pair_effective_n=float(weights.sum()**2/(weights@weights)),condition=condition,rank=rank,overlap_fraction=float(overlap.mean()),query_n=len(query_x),identified=bool(identified),bandwidth=bandwidth,support_radius=support_radius),m.coef_[:,-4:]/uscale[None,:],overlap

def stage_one(df,X,U,N,Y):
    dates=df.date.to_numpy();flights=df.flight.to_numpy();modes=df['mode'].to_numpy()
    predictions={name:np.full_like(Y,np.nan) for name in ['M0','M1','negative','without_differential','without_rudder']}
    residual=np.zeros_like(U);folds=[];local=[];local_errors=[];coverage=[]
    for date in np.unique(dates):
        tr=np.flatnonzero(dates!=date);te=np.flatnonzero(dates==date)
        assert not set(dates[tr])&set(dates[te])
        print('date',date,'train',len(tr),'evaluation',len(te),flush=True)
        designs={'M0':X,'M1':np.c_[X,U],'negative':np.c_[X,N]}
        for c in [1,3]:designs['without_'+CHANNELS[c]]=np.c_[X,U[:,[j for j in range(8) if j%4!=c]]]
        for name,design in designs.items():
            model=fit(design[tr],Y[tr],dates[tr]);predictions[name][te]=model.predict(design[te])
            folds.append(dict(date=date,model=name,alpha=model.alpha,train_dates=len(set(dates[tr])),train_flights=len(set(flights[tr])),evaluation_flights=len(set(flights[te]))))
        residual[te]=U[te]-fit(X[tr],U[tr],dates[tr]).predict(X[te])
        sc=StandardScaler().fit(X[tr]);z=sc.transform(X);su=StandardScaler().fit(U[tr,4:]);u=su.transform(U[:,4:])
        # Local fitter cannot index evaluation outcomes: they are replaced with NaN.
        ytrain=Y.copy();ytrain[te]=np.nan
        for mode in np.unique(modes[te]):
            b=tr[modes[tr]==mode];a=te[modes[te]==mode]
            pairs=crossflight_pairs(z,flights,b)
            evalpairs=crossflight_pairs(z,flights,a)
            if not len(pairs) or not len(evalpairs):
                coverage.append(dict(date=date,mode=int(mode),reason='no_crossflight_pairs',query_n=len(a)));continue
            assert np.isfinite(ytrain[pairs]).all() and not np.isin(pairs,te).any()
            for f in np.unique(flights[a]):
                query=a[flights[a]==f];ep=evalpairs[np.isin(evalpairs[:,0],query)|np.isin(evalpairs[:,1],query)]
                for width in [.5,1.,2.]:
                    model,state,info,gain,overlap=local_fit(z,u,ytrain,pairs,z[query],width,su.scale_)
                    coverage.append(dict(date=date,mode=int(mode),flight=f,width=width,supported_queries=int(overlap.sum()),unsupported_queries=int((~overlap).sum()),support_status='full' if overlap.all() else ('partial' if overlap.any() else 'unevaluable'),**info))
                    if not info['identified'] or not overlap.any():continue
                    # Evaluate only pairs with both states inside observed training radius.
                    i,j=ep.T
                    train_radius=info['support_radius']
                    distances=cKDTree(z[model.support_indices]).query(z[np.r_[i,j]])[0]
                    good=(distances[:len(i)]<=train_radius)&(distances[len(i):]<=train_radius)
                    i=i[good];j=j[good]
                    if not len(i):continue
                    dx=z[j]-z[i];du=u[j]-u[i];dy=Y[j]-Y[i]
                    p0=state.predict(dx);p1=model.predict(np.c_[dx,du])
                    for axis in range(2):
                        local_errors.append(dict(date=date,flight=f,mode=int(mode),width=width,axis='pr'[axis],pairs=len(i),state_mse=float(np.mean((dy[:,axis]-p0[:,axis])**2)),full_mse=float(np.mean((dy[:,axis]-p1[:,axis])**2))))
                        for c in range(4):local.append(dict(date=date,flight=f,mode=int(mode),width=width,axis='pr'[axis],channel=CHANNELS[c],G=float(gain[axis,c]),**info))
    records=[];exc=[]
    for date in np.unique(dates):
        ix=np.flatnonzero(dates==date)
        for axis in range(2):
            vals={name:float(np.mean([np.mean((Y[ix[flights[ix]==f],axis]-p[ix[flights[ix]==f],axis])**2) for f in np.unique(flights[ix])])) for name,p in predictions.items()}
            records.append(dict(date=date,axis='pr'[axis],flights=len(set(flights[ix])),n=len(ix),**vals))
        for mode in np.unique(modes[ix]):
            take=ix[modes[ix]==mode];r=residual[take,4:];cov=np.cov(r.T)
            for c in range(4):exc.append(dict(date=date,mode=int(mode),channel=CHANNELS[c],n=len(take),flights=len(set(flights[take])),raw_rms=float(np.sqrt(np.mean(U[take,4+c]**2))),residual_rms=float(np.sqrt(np.mean(r[:,c]**2))),condition=float(np.linalg.cond(cov))))
    for name,rows in [('date_errors',records),('excitation',exc),('local_g',local),('local_errors',local_errors),('local_coverage',coverage),('folds',folds)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',Y=Y,U=U,N=N,residual=residual,**predictions)
    np.savetxt(OUT/'normal_negative_correlation.csv',np.corrcoef(np.c_[U,N].T),delimiter=',')
    np.savetxt(OUT/'residual_covariance.csv',np.cov(residual[:,4:].T),delimiter=',')
    np.savetxt(OUT/'residual_correlation.csv',np.corrcoef(residual[:,4:].T),delimiter=',')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--reuse-extracted',action='store_true');args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.reuse_extracted:
        df=pd.read_csv(OUT/'identity.csv');a=np.load(OUT/'features.npz');X,U,N,Y=[a[k] for k in ['X','U','N','Y']]
    else:df,X,U,N,Y=load_features(args.source)
    stage_one(df,X,U,N,Y)
