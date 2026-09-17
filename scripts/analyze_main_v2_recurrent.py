"""Train-only reference fitting and reporting for Step 7; no test input."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
class PCA:
    """Small deterministic covariance PCA; avoids an optional dependency."""
    def __init__(self,n_components,svd_solver=None): self.n_components=n_components
    def fit(self,x):
        self.mean_=x.mean(0);z=x-self.mean_
        values,vectors=np.linalg.eigh(z.T@z/max(len(z)-1,1))
        order=np.argsort(values)[::-1]
        self.components_=vectors[:,order[:self.n_components]].T
        self.explained_variance_ratio_=values[order[:self.n_components]]/values.sum()
        return self
    def transform(self,x): return (x-self.mean_)@self.components_.T

from run_main_v2_recurrent_audit import (ROOT,OLD,DATA,SELECT,HORIZONS,PHYSICAL_FIELDS,histories,
    features_at,tensor,npy,save_json,log)
from system_identification.models.rolling_history_simulator import state_features


def target(p,c,k,anchor):
    phase=p['relative_phase_rad'][:,k]-anchor
    return torch.cat((torch.sin(phase)[:,None],torch.cos(phase)[:,None],
        p['flap_frequency_hz'][:,k,None],p['angular_velocity_b'][:,k],
        p['angular_velocity_b'][:,k]-p['angular_velocity_b'][:,k-2],
        c[:,k-25:k+1].mean(1),torch.sin(p['relative_phase_rad'][:,k,None]),torch.cos(p['relative_phase_rad'][:,k,None])),1)


def bank(sim,device,out,art,smoke):
    metadata=pd.read_csv(OLD/'main_v2_dynamics_observability_phase/train_points.csv').iloc[::100 if smoke else 10].copy()
    metadata['start_sample_in_segment']=metadata.sample_in_segment
    samples=pd.read_parquet(DATA/'samples_train.parquet')
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert set(samples.log_id)==set(manifest['split_contract']['assignments']['train'])
    p,c,_=histories(samples,metadata,26,0,device)
    old=np.load(OLD/'main_v2_dynamics_observability_phase/train_diagnostic_arrays.npz')
    proxy=tensor(old['proxies'][metadata.index],device)
    hs=[];xs=[];ys=[];rows=[];base=sim.model.base_model
    with torch.no_grad():
        for gauge in np.arange(8)*np.pi/4:
            anchor=p['relative_phase_rad'][:,-1]-gauge
            f=features_at(p,slice(None),anchor)
            h=[]
            for start in range(0,len(metadata),512):
                v=f[start:start+512]
                h.append(base._encode_history(v,c[start:start+512],torch.ones(v.shape[:2],device=device,dtype=torch.bool)))
            hs.append(npy(torch.cat(h)));xs.append(npy(torch.cat((f[:,-1],proxy),1)));ys.append(npy(target(p,c,25,anchor)))
            rows.append(metadata[['log_id','segment_id','sample_in_segment','timestamp_s']].assign(gauge_rad=gauge))
    pd.concat(rows,ignore_index=True).to_csv(out/'train_bank_rows.csv',index=False)
    h,x,y=map(np.concatenate,(hs,xs,ys))
    np.savez_compressed(art/'train_bank.npz',hidden=h,physical=x,target=y)
    return h,x,y


class Reference:
    def __init__(self,x,device,groups=None):
        self.mean=x.mean(0);self.std=np.maximum(x.std(0),1e-3)
        z=(x-self.mean)/self.std
        self.z=tensor(z,device);self.device=device
        self.groups=np.asarray(groups) if groups is not None else None
        cov=z.T@z/len(z)+np.eye(z.shape[1])*1e-3
        self.inv=tensor(np.linalg.inv(cov),device)
        self.pca=PCA(n_components=3,svd_solver='full').fit(z)
        # Train calibration includes exclusion of exact self (first nearest result).
        ids=np.linspace(0,len(z)-1,min(1024,len(z)),dtype=int)
        kn,ma=self.score(x[ids],exclude_self=True)
        self.threshold_kn=float(np.quantile(kn,.99));self.threshold_ma=float(np.quantile(ma,.99))
        self.threshold_leave_log=None
        if self.groups is not None:
            kk,_=self.score(x[ids],excluded_groups=self.groups[ids])
            self.threshold_leave_log=float(np.quantile(kk,.99))
    def score(self,x,exclude_self=False,excluded_groups=None):
        z=tensor((x-self.mean)/self.std,self.device);ks=[];ms=[]
        with torch.no_grad():
            for lo in range(0,len(z),256):
                a=z[lo:lo+256]
                d=torch.cdist(a,self.z)/np.sqrt(a.shape[1])
                if excluded_groups is not None:
                    exclude=excluded_groups[lo:lo+len(a),None]==self.groups[None,:]
                    d=d.masked_fill(torch.as_tensor(exclude,device=self.device),float('inf'))
                v=d.topk(6 if exclude_self else 5,largest=False).values
                if exclude_self:v=v[:,1:]
                ks.append(npy(v.mean(1)));ms.append(npy(torch.sqrt(torch.clamp(torch.einsum('bi,ij,bj->b',a,self.inv,a)/a.shape[1],min=0))))
        return np.concatenate(ks),np.concatenate(ms)
    def project(self,x):return self.pca.transform((x-self.mean)/self.std)


def summarize(frame,keys,values):
    rows=[]
    for key,g in frame.groupby(keys):
        key=key if isinstance(key,tuple) else (key,)
        r=dict(zip(keys,key));r['n']=len(g)
        for v in values:
            a=g[v].to_numpy(dtype=float);r[v+'_mean']=np.mean(a);r[v+'_median']=np.median(a)
            r[v+'_p90']=np.quantile(a,.9);r[v+'_p95']=np.quantile(a,.95);r[v+'_max']=np.max(a)
        rows.append(r)
    return pd.DataFrame(rows)


def analyze(sim,batch,p,c,windows,traces,out,art,device,smoke):
    z=np.load(art/'interventions.npz');teacher=z['teacher'];free=traces['D'][0]['gru_hidden'];b=traces['B'][0]['gru_hidden']
    steps=SELECT;N=len(windows);origins=np.repeat(np.arange(N),len(steps));ss=np.tile(steps,N)
    actual_time=np.c_[np.zeros(N),np.cumsum(batch.trajectory.dt_s,axis=1)]
    baseframe=pd.DataFrame(dict(origin_index=origins,step=ss,horizon_s=ss*.02,actual_duration_s=actual_time[origins,ss],log_id=windows.log_id.to_numpy()[origins],window_id=windows.window_id.to_numpy()[origins]))
    hrows=[]
    for name,h in [('D',free),('B',b),('R13',traces['R13'][0]['gru_hidden']),('R26',traces['R26'][0]['gru_hidden'])]:
        t=teacher[:,steps];a=h[:,steps];df=baseframe.copy();df['mode']=name
        df['l2']=np.linalg.norm(a-t,axis=-1).ravel();df['dimension_rmse']=np.sqrt(np.mean((a-t)**2,axis=-1)).ravel()
        df['cosine']=(np.sum(a*t,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(t,axis=-1)+1e-12)).ravel()
        hrows.append(df)
    hdf=pd.concat(hrows);hdf.to_csv(out/'hidden_distance_per_origin.csv',index=False)
    summarize(hdf,['mode','horizon_s'],['l2','dimension_rmse','cosine']).to_csv(out/'hidden_distance_vs_horizon.csv',index=False)
    pd.DataFrame(np.sqrt(np.mean((free-teacher)**2,axis=0)),columns=[f'h{i}_rmse' for i in range(free.shape[-1])]).assign(step=np.arange(251)).to_csv(out/'hidden_dimension_rmse.csv',index=False)
    for name,key in [('hidden','hidden_effect'),('state','state_effect')]:
        d=z[key];rows=[]
        for h,k in HORIZONS.items():
            e=d[:,max(0,k-10):k]
            for i,r in enumerate(windows.itertuples()):
                rows.append(dict(log_id=r.log_id,window_id=r.window_id,horizon_s=h,
                    linear=np.sqrt(np.mean(np.sum(e[i,:,:3]**2,axis=-1))),angular=np.sqrt(np.mean(np.sum(e[i,:,3:6]**2,axis=-1))),frequency=np.sqrt(np.mean(e[i,:,6]**2)),linear_body=np.sqrt(np.mean(np.sum(e[i,:,7:10]**2,axis=-1))) if e.shape[-1]>=10 else np.nan))
        df=pd.DataFrame(rows);df.to_csv(out/f'{name}_output_sensitivity_per_origin.csv',index=False)
        summarize(df,['horizon_s'],['linear','angular','frequency','linear_body']).to_csv(out/f'{name}_output_sensitivity.csv',index=False)
    log('Fit train-only hidden/physical reference bank, PCA and ridge probe')
    htrain,xtrain,ytrain=bank(sim,device,out,art,smoke)
    groups=pd.read_csv(out/'train_bank_rows.csv').log_id.to_numpy()
    rh=Reference(htrain,device,groups);rx=Reference(xtrain,device,groups)
    # Actual reset/training gauge is zero at the current point. Keep this strict
    # bank separately: gauge augmentation is a control, not the training distribution.
    n_native=len(htrain)//8
    native_refs={'hidden':Reference(htrain[:n_native],device),'physical':Reference(xtrain[:n_native],device)}

    save_json(out/'reference_contract.json',dict(n_bank=len(htrain),gauges=8,knn_k=5,calibration='1024 linspace train-bank rows; self excluded; secondary excludes all same-log rows',native_thresholds={name:dict(knn_p99=ref.threshold_kn,mahalanobis_p99=ref.threshold_ma) for name,ref in native_refs.items()},covariance_ridge=.001,
        hidden_leave_log_p99=rh.threshold_leave_log,physical_leave_log_p99=rx.threshold_leave_log,hidden_knn_p99=rh.threshold_kn,hidden_mahalanobis_p99=rh.threshold_ma,physical_knn_p99=rx.threshold_kn,
        physical_mahalanobis_p99=rx.threshold_ma,hidden_pca_explained_ratio=rh.pca.explained_variance_ratio_.tolist()))
    np.savez_compressed(art/'reference_fit.npz',hidden_mean=rh.mean,hidden_std=rh.std,physical_mean=rx.mean,physical_std=rx.std,
        hidden_pca_components=rh.pca.components_,hidden_pca_mean=rh.pca.mean_,hidden_covariance_inverse=npy(rh.inv),physical_covariance_inverse=npy(rx.inv))
    hframes=[];xframes=[]
    anchor=p['relative_phase_rad'][:,25]
    truth_feature=features_at(p,slice(25,None),anchor)
    dfree=traces['D'][0]
    with torch.no_grad():
        predphysical={name:tensor(dfree[name],device) for name in PHYSICAL_FIELDS}
        freefeature=features_at(predphysical,slice(None),anchor)
    proxy=np.concatenate((dfree['drive_state'][...,None],dfree['tail_state']),axis=-1)
    # Actuator proxy is identical between interventions, driven by shared command tape.
    physical_teacher=np.concatenate((npy(truth_feature),proxy),axis=-1)
    physical_free=np.concatenate((npy(freefeature),proxy),axis=-1)
    for name,h in [('teacher',teacher),('B',b),('D',free),('R13',traces['R13'][0]['gru_hidden']),('R26',traces['R26'][0]['gru_hidden'])]:
        log('Hidden OOD '+name)
        kn,ma=rh.score(h[:,steps].reshape(-1,h.shape[-1]));df=baseframe.copy();df['mode']=name;df['knn']=kn;df['mahalanobis']=ma
        df['knn_outside_leave_log_p99']=kn>rh.threshold_leave_log;df['knn_outside_train_p99']=kn>rh.threshold_kn;df['mahalanobis_outside_train_p99']=ma>rh.threshold_ma;hframes.append(df)
    for name,x in [('teacher',physical_teacher),('D',physical_free)]:
        kn,ma=rx.score(x[:,steps].reshape(-1,x.shape[-1]));df=baseframe.copy();df['mode']=name;df['knn']=kn;df['mahalanobis']=ma
        df['knn_outside_leave_log_p99']=kn>rx.threshold_leave_log;df['knn_outside_train_p99']=kn>rx.threshold_kn;df['mahalanobis_outside_train_p99']=ma>rx.threshold_ma;xframes.append(df)
    physical_native=physical_teacher.copy();physical_native[:,:,9]=0;physical_native[:,:,10]=1
    native_frames=[]
    for space,ref,queries in [('hidden',native_refs['hidden'],[('teacher_native',z['native']),('teacher_fixed',teacher),('D',free)]),
                               ('physical',native_refs['physical'],[('teacher_native',physical_native),('teacher_fixed',physical_teacher),('D',physical_free)])]:
        local=[]
        for name,x in queries:
            kn,ma=ref.score(x[:,steps].reshape(-1,x.shape[-1]));df=baseframe.copy();df['mode']=name
            df['knn']=kn;df['mahalanobis']=ma;df['knn_outside_train_p99']=kn>ref.threshold_kn
            local.append(df)
        df=pd.concat(local);df.to_csv(out/f'native_{space}_ood_per_origin.csv',index=False)
        summarize(df,['mode','horizon_s'],['knn','mahalanobis','knn_outside_train_p99']).to_csv(out/f'native_{space}_ood.csv',index=False)
    hd=pd.concat(hframes);xd=pd.concat(xframes)
    hd.to_csv(out/'hidden_ood_per_origin.csv',index=False);xd.to_csv(out/'physical_ood_per_origin.csv',index=False)
    for label,df in [('hidden',hd),('physical',xd)]:
        summarize(df,['mode','horizon_s'],['knn','mahalanobis','knn_outside_train_p99','knn_outside_leave_log_p99','mahalanobis_outside_train_p99']).to_csv(out/f'{label}_ood.csv',index=False)
    # Correlation at each fixed horizon avoids interpreting common time trends as causality.
    corr=[];per=pd.read_csv(out/'per_rollout.csv');var=pd.read_csv(out/'variation_per_rollout.csv')
    for h in HORIZONS:
        e=per[(per['mode']=='D')&(per.horizon_s==h)].set_index('window_id')
        v=var[(var['mode']=='D')&(var.horizon_s==h)&(var.scope=='prefix')].set_index('window_id')
        for label,df in [('hidden',hd),('physical',xd)]:
            g=df[(df['mode']=='D')&np.isclose(df.horizon_s,h)].set_index('window_id')
            for metric in ['angular_acceleration','linear_acceleration','attitude_deg','velocity_m_s','variation']:
                y=v.loc[g.index,'ratio'] if metric=='variation' else e.loc[g.index,metric]
                corr.append(dict(space=label,horizon_s=h,metric=metric,spearman=spearmanr(g.knn,y).statistic,n=len(g)))
    pd.DataFrame(corr).to_csv(out/'ood_error_correlations.csv',index=False)
    # Onset compared per origin, requiring three consecutive sampled checkpoints above p99.
    onset=[]
    for i in range(N):
        rr={'window_id':windows.iloc[i].window_id,'log_id':windows.iloc[i].log_id}
        for label,df in [('hidden',hd),('physical',xd)]:
            g=df[(df.origin_index==i)&(df['mode']=='D')].sort_values('step');a=g.knn_outside_train_p99.to_numpy()
            ids=np.where(np.convolve(a.astype(int),np.ones(3,dtype=int),'valid')==3)[0]
            rr[label+'_onset_s']=g.iloc[ids[0]].actual_duration_s if len(ids) else np.nan
        onset.append(rr)
    pd.DataFrame(onset).to_csv(out/'ood_onset.csv',index=False)
    # Linear information decoder, fitted only on training real-history encodings.
    hx=(htrain-rh.mean)/rh.std;ym=ytrain.mean(0);ys=np.maximum(ytrain.std(0),1e-3)
    design=np.c_[np.ones(len(hx)),hx];pen=np.eye(design.shape[1]);pen[0,0]=0
    coef=np.linalg.solve(design.T@design+pen,design.T@((ytrain-ym)/ys))
    np.savez_compressed(art/'hidden_probe.npz',coef=coef,target_mean=ym,target_std=ys)
    truth_y=np.stack([npy(target(p,c,int(k)+25,anchor)) for k in steps],1)
    names=['phase_relative','frequency','body_rate','recent_delta_omega','recent_command_mean','within_log_phase']
    slices=[slice(0,2),slice(2,3),slice(3,6),slice(6,9),slice(9,13),slice(13,15)]
    probes=[]
    for name,hh in [('teacher',teacher),('B',b),('D',free)]:
        v=hh[:,steps];xx=(v-rh.mean)/rh.std;yp=(np.c_[np.ones(N*len(steps)),xx.reshape(-1,xx.shape[-1])]@coef).reshape(N,len(steps),-1)*ys+ym
        for j,k in enumerate(steps):
            for label,sl in zip(names,slices):
                e=(yp[:,j,sl]-truth_y[:,j,sl]);yn=truth_y[:,j,sl]
                probes.append(dict(mode=name,horizon_s=k*.02,target=label,rmse=np.sqrt(np.mean(e**2)),normalized_rmse=np.sqrt(np.mean((e/ys[sl])**2)),
                    r2=1-np.sum(e**2)/max(np.sum((yn-yn.mean(0))**2),1e-12)))
    pd.DataFrame(probes).to_csv(out/'hidden_probe_metrics.csv',index=False)
    jacobians(sim,htrain,xtrain,windows,art,out,device)
    plots(out,rh,teacher,free,windows)
    temporal_and_uncertainty(batch,windows,traces,out)
    # Candidate comparisons preserve equal-flight aggregation and all horizons.
    summary=pd.read_csv(out/'rollout_mode_comparison.csv');ref=summary[summary['mode']=='D'].set_index('horizon_s')
    comparisons=[]
    for r in summary[summary['mode'].isin(['R13','R26','C','C_native'])].to_dict('records'):
        row={k:r[k] for k in ('mode','horizon_s','n_rollouts','failure_pct')}
        for metric in ['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad']:
            key=metric+'_equal_log_rmse';row[metric+'_change_pct']=100*(r[key]/ref.loc[r['horizon_s'],key]-1)
        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(out/'reencode_comparison.csv',index=False)


def jacobians(sim,htrain,xtrain,windows,art,out,device):
    points=[];base=sim.model.base_model
    ids=np.linspace(0,len(htrain)-1,48,dtype=int)
    for i in ids:points.append(('train_teacher',int(i),-1,htrain[i],npy(base._normalize_features(tensor(xtrain[i,:12],device)))))
    ops=torch.load(art/'operating_points.pt',weights_only=False)
    oi=windows.groupby('log_id',sort=False).head(2).index.to_numpy()
    for op in ops:
        if op['step'] not in (0,50,250-1):continue
        for name,hkey,zkey in [('validation_teacher','teacher','z_teacher'),('validation_free','free','z_free'),('validation_B','b','z_teacher')]:
            for i in oi:points.append((name,int(i),op['step'],op[hkey][i],op[zkey][i]))
    rows=[]
    for mode,i,step,h,z in points:
        ht=tensor(h,device).requires_grad_(True);zt=tensor(z,device)
        jac=torch.autograd.functional.jacobian(lambda v:base.recurrent_cell(zt[None],v[None])[0],ht,vectorize=True)
        j=npy(jac).astype(float);sv=np.linalg.svd(j,compute_uv=False);ev=np.linalg.eigvals(j)
        rows.append(dict(mode=mode,point_index=i,step=step,horizon_s=step*.02,largest_singular=sv[0],median_singular=np.median(sv),spectral_radius=np.abs(ev).max(),frobenius=np.linalg.norm(j)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'recurrent_jacobian_points.csv',index=False)
    summarize(frame,['mode','horizon_s'],['largest_singular','median_singular','spectral_radius']).to_csv(out/'recurrent_jacobian_summary.csv',index=False)


def plots(out,reference,teacher,free,windows):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    ids=windows.groupby('log_id',sort=False).head(1).index
    for i in ids:
        for ax,h,label in zip(axes,[teacher,free],['teacher fixed-anchor','full free-run']):
            xy=reference.project(h[i]);ax.plot(xy[:,0],xy[:,1],alpha=.7,label=str(i));ax.set(title=label,xlabel='train PCA 1',ylabel='train PCA 2')
    fig.tight_layout();fig.savefig(out/'hidden_pca.png',dpi=150);plt.close(fig)
    for filename,y in [('hidden_ood','knn_median'),('hidden_distance_vs_horizon','l2_median')]:
        df=pd.read_csv(out/f'{filename}.csv');fig,ax=plt.subplots(figsize=(7,4))
        for mode,g in df.groupby('mode'):ax.plot(g.horizon_s,g[y],label=mode)
        ax.legend();ax.set(xlabel='Nominal horizon (s)',ylabel=y);fig.tight_layout();fig.savefig(out/(filename+'_vs_horizon.png' if filename=='hidden_ood' else filename+'.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,['physical','hidden']):
        df=pd.read_csv(out/f'{name}_ood.csv')
        for m in ['teacher','D']:
            g=df[df['mode']==m];ax.plot(g.horizon_s,g.knn_outside_train_p99_mean,label=m)
        ax.set(title=name,ylabel='Outside train-reference p99 fraction',xlabel='Nominal horizon (s)');ax.legend()
    fig.tight_layout();fig.savefig(out/'physical_vs_hidden_ood.png',dpi=150);plt.close(fig)
    df=pd.read_csv(out/'rollout_mode_comparison.csv')
    for filename,modes in [('mode_error_vs_horizon',['A_native','A_fixed','B','C','D']),('reencode_free_running',['D','R13','R26'])]:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,metric in zip(axes.flat,['velocity_m_s','attitude_deg','body_rate_rad_s','linear_acceleration','angular_acceleration','position_m']):
            for m in modes:
                g=df[df['mode']==m];ax.plot(g.horizon_s,g[metric+'_equal_log_rmse'],label=m)
            ax.set(title=metric,xlabel='Nominal horizon (s)');ax.legend(fontsize=7)
        fig.suptitle('A/B/C: NOT DEPLOYABLE; A/B one-step endpoints' if filename=='mode_error_vs_horizon' else 'Autonomous predicted-history controls')
        fig.tight_layout();fig.savefig(out/f'{filename}.png',dpi=150);plt.close(fig)


def temporal_and_uncertainty(batch,windows,traces,out):
    """Descriptive flight-time stability; exact paired five-flight bootstrap."""
    import itertools
    truth=batch.trajectory.truth;dt=batch.trajectory.dt_s;n=len(windows)
    from scipy.spatial.transform import Rotation
    q=truth.quaternion_nb[:,:-1].reshape(-1,4)
    rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix().reshape(n,250,3,3)
    raw_v=np.diff(truth.velocity_n,axis=1)/dt[...,None]
    raw_w=np.diff(truth.angular_velocity_b,axis=1)/dt[...,None]
    pred,diag=traces['A_native'];a=np.einsum('btij,btj->bti',rot,diag['acceleration_b'])
    timestamp=windows.start_timestamp_us.to_numpy()[:,None]*1e-6+np.c_[np.zeros(n),np.cumsum(dt,axis=1)][:,:-1]
    frame=pd.DataFrame(dict(log_id=np.repeat(windows.log_id.to_numpy(),250),segment_id=np.repeat(windows.segment_id.to_numpy(),250),
        sample=(windows.start_sample_in_segment.to_numpy()[:,None]+np.arange(250)).ravel(),timestamp_s=timestamp.ravel(),
        linear_squared=np.sum((a-raw_v)**2,axis=-1).ravel(),angular_squared=np.sum((diag['angular_acceleration_b']-raw_w)**2,axis=-1).ravel()))
    frame=frame.drop_duplicates(['log_id','segment_id','sample'],keep='first')
    frame['flight_time_bin']=frame.groupby('log_id').timestamp_s.transform(lambda x:pd.cut(x,5,labels=False,include_lowest=True))
    rows=[]
    for (logid,binid),g in frame.groupby(['log_id','flight_time_bin']):
        rows.append(dict(log_id=logid,flight_time_bin=binid,n_unique_points=len(g),start_s=g.timestamp_s.min(),end_s=g.timestamp_s.max(),linear_rmse=np.sqrt(g.linear_squared.mean()),angular_rmse=np.sqrt(g.angular_squared.mean())))
    pd.DataFrame(rows).to_csv(out/'teacher_flight_time_bins.csv',index=False)
    pf=pd.read_csv(out/'per_flight.csv');rows=[]
    logs=sorted(pf.log_id.unique());draw=np.array(list(itertools.product(range(len(logs)),repeat=len(logs))))
    for mode in ['C','C_native','R13','R26']:
        for horizon in HORIZONS:
            a=pf[(pf['mode']==mode)&(pf.horizon_s==horizon)].set_index('log_id').loc[logs]
            b=pf[(pf['mode']=='D')&(pf.horizon_s==horizon)].set_index('log_id').loc[logs]
            for metric in ['velocity_m_s','attitude_deg','body_rate_rad_s']:
                x=a[metric+'_rmse'].to_numpy();y=b[metric+'_rmse'].to_numpy()
                ratios=100*(x[draw].mean(1)/y[draw].mean(1)-1)
                rows.append(dict(mode=mode,horizon_s=horizon,metric=metric,change_pct=100*(x.mean()/y.mean()-1),
                    ci95_low=np.quantile(ratios,.025),ci95_high=np.quantile(ratios,.975),improved_flights=int((x<y).sum())))
    pd.DataFrame(rows).to_csv(out/'paired_flight_changes.csv',index=False)
    per=pd.read_csv(out/'per_rollout.csv')
    per[per.horizon_s==5].groupby('mode')[['numerical_failed','support_failed','clipping_failed']].sum().reset_index().to_csv(out/'failures.csv',index=False)
