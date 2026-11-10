"""Held-date innovation diagnostics, finite dynamic information and date-cluster uncertainty."""
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.optimize._numdiff import approx_derivative
from px4_clid_models import *
from px4_clid_run import save_model
OUT=Path('experiments/px4_closed_loop_dynamic_id_v2')

def corr(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    return (a*b).sum(0)/np.maximum(np.sqrt((a*a).sum(0)*(b*b).sum(0)),1e-20)

def information(seq,model):
    cfg=model['cfg'];blocks=blocks_for(seq,cfg);G,R,YY,n=stats(blocks,model['c']);na,nb,d,_=cfg
    ids=np.array([1+na*3+4*j+1 for j in range(nb)]);rest=np.setdiff1d(np.arange(len(G)),ids)
    cond=G[np.ix_(ids,ids)]-G[np.ix_(ids,rest)]@np.linalg.pinv(G[np.ix_(rest,rest)])@G[np.ix_(rest,ids)]
    eig,vec=np.linalg.eigh(cond);scale=np.sqrt(np.maximum(np.diag(G),1e-12));sv=np.sqrt(np.maximum(np.linalg.eigvalsh(G/scale[:,None]/scale[None,:]),0))[::-1]
    return dict(n=n,lag_ms=[20*(d+j) for j in range(nb)],conditional_gram=cond.tolist(),conditional_rms=np.sqrt(np.maximum(np.diag(cond),0)/n).tolist(),conditional_eigenvalues=eig.tolist(),weakest_direction=vec[:,0].tolist(),strongest_direction=vec[:,-1].tolist(),scaled_singular_values=sv.tolist(),scaled_condition=float(sv[0]/max(sv[-1],1e-15)))

def cluster_uncertainty(seq,model):
    # Joint Gauss-Newton information for actual causal PEM residuals, including C.
    t=model['theta'];p=len(t);nc=len(model['noise_reflection']);dim=t.size+nc;H=np.zeros((dim,dim));scores={};n=0
    for s in seq:
        X,Y,_=design(s,model['cfg']);xf=lfilter([1.],model['c'],X,axis=0);raw=Y-X@t;e=lfilter([1.],model['c'],raw,axis=0)
        n+=len(e);h=np.zeros_like(H);g=np.zeros(dim)
        # Flatten theta in row-major order, residual rows likewise.
        gg=xf.T@xf
        for axis in range(3):
            ix=np.arange(p)*3+axis;h[np.ix_(ix,ix)]=gg;g[ix]=-xf.T@e[:,axis]
        if nc:
            deriv=[]
            for j in range(nc):
                zp=model['noise_reflection'].copy();zm=zp.copy();zp[j]+=1e-5;zm[j]-=1e-5
                deriv.append((lfilter([1.],noise(zp),raw,axis=0)-lfilter([1.],noise(zm),raw,axis=0))/2e-5)
            D=np.stack(deriv,axis=-1).reshape(-1,nc);h[-nc:,-nc:]=D.T@D;g[-nc:]=D.T@e.ravel()
            for axis in range(3):
                ix=np.arange(p)*3+axis;cross=-xf.T@D.reshape(len(e),3,nc)[:,axis];h[np.ix_(ix,np.arange(dim-nc,dim))]=cross;h[np.ix_(np.arange(dim-nc,dim),ix)]=cross.T
        H+=h;scores[s['date']]=scores.get(s['date'],np.zeros(dim))+g
    V=sum(np.outer(g,g) for g in scores.values());inv=np.linalg.pinv(H,rcond=1e-10);cov=inv@V@inv*len(scores)/max(len(scores)-1,1)
    def response(v):return pulse(dict(model,theta=v[:t.size].reshape(t.shape)))[:,0]
    v=np.r_[t.ravel(),model['noise_reflection']];J=approx_derivative(response,v,method='3-point');se=np.sqrt(np.maximum(np.diag(J@cov@J.T),0))
    diag=np.sqrt(np.maximum(np.diag(H),1e-20));ev=np.linalg.eigvalsh(H/diag[:,None]/diag[None,:]);
    return dict(cluster_dates=len(scores),rows=n,parameter_count=dim,information_eigenvalues=ev.tolist(),information_condition=float(ev[-1]/max(ev[0],1e-15)),gain=pulse(model)[:,0].tolist(),cluster_se=se.tolist(),method='joint causal PEM Gauss-Newton plus date-cluster sandwich; local approximation, not a physical confidence bound; ridge omitted from information')

def main():
    seq=pickle.load((OUT/'local/sequences.pkl').open('rb'));dates=sorted({s['date'] for s in seq});correlations=[];distributions=[];info={};uncert={};allmodels={}
    for date in dates:
        r=pickle.load((OUT/'local'/f'fold_{date}.pkl').open('rb'));test=[s for s in seq if s['date']==date];train=[s for s in seq if s['date']!=date]
        for name in ['M0','M1']:
            m=r['models'][name];allmodels[(date,name)]=m;pieces=[]
            for s in test:
                e,start=innovations(m,s);e=e[10:];start+=10
                pieces.append((s,e,start))
            for lag in range(1,11):
                a=np.concatenate([e[lag:] for s,e,start in pieces if len(e)>lag]);b=np.concatenate([e[:-lag] for s,e,start in pieces if len(e)>lag])
                for ax,val in zip('pqr',corr(a,b)):correlations.append(dict(date=date,model=name,kind='innovation_acf',channel=ax,lag_ms=lag*20,correlation=float(val),n=len(a)))
            for lag in [-10,-5,-2,-1,0,1,2,5,10]:
                aa=[];bb=[];zz=[]
                for s,e,start in pieces:
                    indices=np.arange(start,len(s['y']));good=(indices-lag>=0)&(indices-lag<len(s['y']));idx=indices[good]
                    aa.append(e[good,0]);bb.append(np.column_stack([s['u'][idx-lag],s['y'][idx-lag]]));zz.append(np.column_stack([np.ones(len(idx)),s['y'][idx-1],s['u'][idx-1][:,[0,2,3]]]))
                a=np.concatenate(aa);b=np.concatenate(bb);z=np.concatenate(zz)
                for j,ch in enumerate(['motor','differential','symmetric','rudder','p','q','r']):
                    correlations.append(dict(date=date,model=name,kind='innovation_vs_history',channel=ch,lag_ms=lag*20,correlation=float(corr(a[:,None],b[:,j:j+1])[0]),n=len(a)))
                ar=a-z@np.linalg.lstsq(z,a,rcond=None)[0];br=b[:,1]-z@np.linalg.lstsq(z,b[:,1],rcond=None)[0]
                correlations.append(dict(date=date,model=name,kind='partial_diff_given_previous_state_other_control',channel='differential',lag_ms=lag*20,correlation=float(corr(ar[:,None],br[:,None])[0]),n=len(a)))
            # Mode and relative encoder phase distributions; no claim of Hall phase.
            for s,e,start in pieces:
                ph=(s['phase'][start:]%(2*np.pi));bins=np.minimum((ph/(2*np.pi)*8).astype(int),7)
                for b in range(8):
                    v=e[bins==b]
                    if len(v):
                        for axis,ch in enumerate('pqr'):distributions.append(dict(date=date,model=name,mode=s['mode'],phase_bin=b,axis=ch,n=len(v),sum=float(v[:,axis].sum()),sum_sq=float((v[:,axis]**2).sum())))
            info[date+'_'+name]=information(test,m)
            uncert[date+'_'+name]=cluster_uncertainty(train,m)
            print('diagnostic',date,name,flush=True)
    pd.DataFrame(correlations).to_csv(OUT/'innovation_correlations.csv',index=False)
    df=pd.DataFrame(distributions).groupby(['date','model','mode','phase_bin','axis'],as_index=False)[['n','sum','sum_sq']].sum();df['mean']=df['sum']/df.n;df['std']=np.sqrt(np.maximum(df.sum_sq/df.n-df['mean']**2,0));df.to_csv(OUT/'innovation_distribution.csv',index=False)
    # Final all-train model uses the modal nested-training choice, predeclared here.
    sels=json.loads((OUT/'selection.json').read_text());from collections import Counter
    cfg=Counter(tuple(s['cfg']) for s in sels).most_common(1)[0][0];nc=Counter(s['chosen_nc'] for s in sels).most_common(1)[0][0]
    final={'M0':fit(blocks_for(seq,cfg),cfg),'M1':fit(blocks_for(seq,cfg),cfg,nc)}
    for name,m in final.items():
        info['alltrain_'+name]=information(seq,m);uncert['alltrain_'+name]=cluster_uncertainty(seq,m)
        for firmware in sorted({s['firmware'] for s in seq}):info['firmware_'+firmware+'_'+name]=information([s for s in seq if s['firmware']==firmware],m)
    (OUT/'final_models.json').write_text(json.dumps({k:save_model(v) for k,v in final.items()},indent=2));(OUT/'dynamic_information.json').write_text(json.dumps(info,indent=2));(OUT/'response_uncertainty.json').write_text(json.dumps(uncert,indent=2))
if __name__=='__main__':main()
