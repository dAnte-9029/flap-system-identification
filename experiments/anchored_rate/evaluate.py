"""Paired native endpoint and independently referenced beat scoring."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train import data,args
from anchored_model import AnchoredRate,wave,ROOT
from search import old_model,causal_conditions

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def grouped(frame,cols):
    return frame.groupby(['flight','window'])[cols].mean().groupby('flight').mean()

def changes(frame,pred,reference,group):
    s=frame.copy()
    s['p2']=(pred**2).sum(1)
    s['r2']=(reference**2).sum(1)
    s['dot']=(pred*reference).sum(1)
    s['error']=((pred-reference)**2).sum(1)
    by=grouped(s,['p2','r2','dot','error'])
    m=by.mean()
    alignment=m['dot']/np.sqrt(m.p2*m.r2) if m.p2>0 else None
    return dict(rmse=float(np.sqrt(m.error)),hold=float(np.sqrt(m.r2)),predicted_rms=float(np.sqrt(m.p2)),
                alignment=alignment,flights=len(by),windows=len(s.groupby(['flight','window'])),beats=len(s)),by.assign(group=group)

def native(net,use,d,b,out):
    directory,ref,times,pool,bins=wave.native_cache(ROOT,'cpu')
    old=old_model(17,'cpu')
    wi,ch=ref['window_idx'],ref['ch']
    anchor=torch.from_numpy(b['anchor']);origin=torch.from_numpy(b['origin']).float()
    with np.load(directory/'original_corrected17_points.npz') as z:
        cached=torch.from_numpy(z['slow_abs']+z['fast_abs'])
    full=[];branch=[]
    for j in range(0,len(ch),512):
        sl=slice(j,j+512);ii=wi[sl];cc=ch[sl];t=times[sl];p=pool[sl]
        h,cp,u,ct,co=args(d,ii)
        ac=net.ac
        inc=ac(h,p,u,t,co)-old(h,p,u,t)
        base=cached[sl]+inc[np.arange(len(ii)),0,cc]
        slow=anchor[ii]
        if use:
            states=net.states(h,cp,u,ct)
            feat=net.features(states,ct[:,:,1],t[:,:,1],p[:,:,1],anchor[ii])
            drift=(feat@net.weights)[:,0]
            slow=anchor[ii]+drift
            absolute=origin[ii]+net.offsets(h,p,u,t,co,anchor[ii],origin[ii])[:,0]+drift
            valid=b['ok'][ii]&(cc==1)
            base[valid]=absolute[valid].to(base.dtype)
        full.append(base.numpy());branch.append(slow.numpy())
    full=np.concatenate(full);branch=np.concatenate(branch)
    first=np.r_[0,np.flatnonzero(np.diff(ref['block']))+1]
    def mean(v):
        return np.add.reduceat(ref['weight'][:,None]*v,first)
    df=pd.DataFrame(dict(flight=ref['flight_ids'][ref['flight_index'][first]],window=wi[first],ch=ch[first],ok=b['ok'][wi[first]]))
    for name,v in [('p',mean(full)),('r',mean(ref['slow'])),('s',mean(branch)),('a',b['anchor'][wi[first]])]:
        df[[name+str(i) for i in range(3)]]=v
    details={};tables=[]
    for channel,label in enumerate(('speed','rate')):
        sub=df.loc[df.ch==channel]
        columns=['p0','p1','p2','r0','r1','r2']
        grouped_win=sub.groupby(['flight','window'])
        delta=(sub[columns]-grouped_win[columns].transform('first')).to_numpy()
        later=grouped_win.cumcount().to_numpy()>0
        key='retrospective_'+label
        details[key],table=changes(sub[later],delta[later,:3],delta[later,3:],key)
        tables.append(table)
    sub=df.loc[(df.ch==1)&df.ok]
    a=sub[['a0','a1','a2']].to_numpy()
    refdelta=sub[['r0','r1','r2']].to_numpy()-a
    mag=np.linalg.norm(refdelta,axis=1)
    threshold=np.quantile(mag,.75)
    details['high_change_threshold']=threshold
    for quantity,col in [('full','p')]+([('slow_readout','s')] if use else []):
        delta=sub[[col+str(i) for i in range(3)]].to_numpy()-a
        for subset,mask in [('all',np.ones(len(a),bool)),('high',mag>=threshold)]:
            key=quantity+'_'+subset
            details[key],table=changes(sub[mask],delta[mask],refdelta[mask],key)
            tables.append(table)
    pd.concat(tables).reset_index().to_csv(out/'beats_by_flight.csv',index=False)
    write(out/'beats.json',details)
    if not use:
        np.testing.assert_allclose(details['retrospective_rate']['rmse'],.58587812,atol=2e-6)
    return details

@torch.inference_mode()
def run():
    torch.set_num_threads(3)
    out=Path(os.environ['RA_OUTPUT'])
    pack=torch.load(os.environ['RA_MODEL'],weights_only=True,map_location='cpu')
    net=AnchoredRate();net.weights.copy_(pack['weights'])
    use=pack['use_b']
    d,flight,b=data('validation')
    anchor=torch.from_numpy(b['anchor']);origin=torch.from_numpy(b['origin']).float();ok=torch.from_numpy(b['ok'])
    err=[];same=0.
    for j in range(0,len(flight),768):
        sl=slice(j,j+768)
        argv=args(d,sl)
        ref=net.ac(*argv)
        if use:
            full,_=net(*argv,anchor[sl],origin[sl],ok[sl])
            same=max(same,float((full[:,:,0]-ref[:,:,0]).abs().max()))
            torch.testing.assert_close(full[~ok[sl]],ref[~ok[sl]],rtol=0,atol=0)
        else:
            full=ref
        error=(full-d['y'][sl]).square().sum(-1)
        error=error.masked_fill(~d['mask'][sl],float('nan'))
        err.append(error[:,[1,3,7]].reshape(-1,6).numpy())
    assert same==0.
    score=pd.DataFrame(np.concatenate(err).astype('float64'),columns=wave.METRICS)
    score['flight']=flight
    per=score.groupby('flight').mean()
    per.to_csv(out/'endpoints_by_flight.csv')
    score[ok.numpy()].groupby('flight').mean().to_csv(out/'endpoints_anchored_by_flight.csv')
    if not use:
        past=pd.read_csv(ROOT/'research/rounds/wingbeat-hyperparameter-search-20261126/results/endpoints_by_flight.csv')
        past=past[(past.arm=='full')&(past.seed==17)].set_index('flight').loc[per.index,list(wave.METRICS)]
        np.testing.assert_allclose(per,past,rtol=2e-6,atol=2e-7)
    details=native(net,use,d,b,out)
    metrics={**np.sqrt(per.mean()).to_dict(),'slowrate':details['full_all']['rmse'],'direction_high':details['full_high']['alignment']}
    write(Path(os.environ['RA_METRICS']),metrics)
    write(out/'coverage.json',dict(windows=len(flight),flights=len(per),anchored=int(ok.sum()),anchored_flights=len(set(flight[ok.numpy()]))))
    print(metrics,flush=True)

if __name__=='__main__': run()
