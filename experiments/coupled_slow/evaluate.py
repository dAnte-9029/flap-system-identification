import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from network import ROOT,Dynamics,wave,load_tuned
from dataset import PREV,anchor_input
from fit_coupled import call,read
from shifted import observations

def mse(p,y):
    return np.square(p.astype('float64')-y).sum(-1)

def average(df,cols):
    return df.groupby(['flight','window'])[cols].mean().groupby('flight').mean()

def trend(frame,p,r):
    df=frame.copy();df['error']=mse(p,r);df['reference']=(r*r).sum(1)
    df['pred2']=(p*p).sum(1);df['dot']=(p*r).sum(1)
    by=average(df,['error','reference','pred2','dot']);a=by.mean()
    norm=np.sqrt(a.reference*a.pred2)
    return dict(rmse=np.sqrt(a.error),hold=np.sqrt(a.reference),direction=float(a['dot']/norm) if norm>0 else None,
                predicted_rms=np.sqrt(a.pred2),flights=len(by),beats=len(df)),by

def exact_points(net,orig,d,b,wi,t,p,freeze,slow=False):
    outputs=[];reads=[]
    for j in range(0,len(wi),512):
        ids=wi[j:j+512];qt=t[j:j+512];qp=p[j:j+512]
        h,u,co,ct,cp=[d[k][ids] for k in ('hist','u0','coef','time','pooled')]
        corr=orig(h,qp,u,qt,co)[:,0,1]-orig.first_order.core(h[:,:orig.first_order.mean.numel()],qp,u,qt)[:,0,1]
        change,ss=net.point(h,cp,u,ct,b['a'][ids],qt,qp,freeze,slow)
        outputs.append((b['origin_tensor'][ids]+corr+change).numpy())
        if ss is not None:reads.append(ss.numpy())
    return np.concatenate(outputs),np.concatenate(reads) if reads else None

def beats(net,orig,d,b,flight,freeze,cache,out):
    _,ref,times,pool,_=cache
    ids=np.flatnonzero(ref['ch']==1)
    wi=ref['window_idx'][ids]
    np.testing.assert_array_equal(flight[wi],ref['flight_ids'][ref['flight_index'][ids]])
    p,s=exact_points(net,orig,d,b,wi,times[ids],pool[ids],freeze,True)
    first=np.r_[0,np.flatnonzero(np.diff(ref['block'][ids]))+1]
    w=ref['weight'][ids,None]
    def mean(x):return np.add.reduceat(w*x,first)
    np.testing.assert_allclose(mean(np.ones((len(wi),1))),1,atol=1e-8)
    mp,r,raw=mean(p),mean(ref['slow'][ids]),mean(ref['data'][ids])
    df=pd.DataFrame(dict(flight=flight[wi[first]],window=wi[first]))
    take=b['ok'][wi[first]]
    ah=b['anchor'][wi[first]]
    high=np.linalg.norm((r-ah)[take],axis=1)
    cutoff=np.quantile(high,.75)
    scores={};tables=[]
    for name,v in [('full',mp)]+([('readout',mean(s))] if s is not None else []):
        for key,mask in [('all',take),('high',take&(np.linalg.norm(r-ah,axis=1)>=cutoff))]:
            tag=name+'_'+key
            scores[tag],table=trend(df[mask],(v-ah)[mask],(r-ah)[mask])
            tables.append(table.assign(group=tag))
    for cols,x in [('p',mp),('r',r)]:
        df[[cols+str(i) for i in range(3)]]=x
    col=['p0','p1','p2','r0','r1','r2'];groups=df.groupby(['flight','window'])
    delta=(df[col]-groups[col].transform('first')).to_numpy();later=groups.cumcount()>0
    scores['future_first'],table=trend(df[later],delta[later,:3],delta[later,3:])
    tables.append(table.assign(group='future_first'))
    v=df.loc[take,['flight','window']].copy()
    v['native']=mean(mse(p,ref['data'][ids])[:,None])[take,0]
    v['beat_mean']=mse(mp,raw)[take];v['centered']=v.native-v.beat_mean
    if s is not None:v['readout_gap']=mse(mp[take],mean(s)[take])
    by=average(v,[k for k in ('native','beat_mean','centered','readout_gap') if k in v])
    scores['measured']={k:float(np.sqrt(v)) for k,v in by.mean().items()}
    by.to_csv(out/'native_by_flight.csv')
    pd.concat(tables).reset_index().to_csv(out/'trends_by_flight.csv',index=False)
    return scores

def shifted(net,orig,d,b,flight,freeze,inputs,out):
    times,ys,masks=inputs; tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/validation_commands.npy',mmap_mode='r')
    tables=[]
    for j in range(3):
        ix=np.flatnonzero(masks[:,j].all(1))
        t=times[ix,j];parts=[]
        for sub in np.array_split(np.arange(len(ix)),(len(ix)+511)//512):
            parts.append(wave.pool_commands(tape[ix[sub]],t[sub]))
        qt=torch.from_numpy(t.reshape(-1))[:,None,None].expand(-1,1,2)
        qp=torch.from_numpy(np.concatenate(parts).reshape(-1,16))[:,None,None,:].expand(-1,1,2,-1)
        pred,_=exact_points(net,orig,d,b,np.repeat(ix,3),qt,qp,freeze)
        err=mse(pred.reshape(len(ix),3,3),ys[ix,j])
        fr=pd.DataFrame(err,columns=['minus125','minus20','registered']).assign(flight=flight[ix]).groupby('flight').mean()
        tables.append(fr.assign(horizon=(.5,1.,2.)[j]))
    result=pd.concat(tables).reset_index();result.to_csv(out/'shift_by_flight.csv',index=False)
    return {str(k):v.to_dict() for k,v in result.groupby('horizon')[['minus125','minus20','registered']].mean().pow(.5).iterrows()}

@torch.inference_mode()
def evaluate():
    torch.set_num_threads(3)
    out=Path(os.environ['RA_OUTPUT'])
    pack=torch.load(os.environ['RA_MODEL'],weights_only=True,map_location='cpu')
    orig=load_tuned(17,'cpu').eval()
    net=Dynamics(orig,pack['coupled']).eval();net.load_state_dict(pack['state'])
    d,flight,_=wave.load(ROOT,'validation','cpu')
    with np.load(PREV/'validation.npz') as z:b={k:z[k] for k in ('ok','origin','anchor','window')}
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:np.testing.assert_array_equal(z['window'],b['window'])
    b['a']=anchor_input(b);b['origin_tensor']=torch.from_numpy(b['origin']).float()
    cache=wave.native_cache(ROOT,'cpu')
    points=observations(d,flight,b)
    fast=read('validation','cpu')
    modes={'normal':(net,False)}
    if net.coupled:modes['held_s0']=(net,True)
    else:modes['original']=(Dynamics(orig,False).eval(),False)
    results={}
    for tag,(model,freeze) in modes.items():
        sub=out/tag;sub.mkdir(exist_ok=True)
        errors=[]
        for ix in torch.arange(len(flight)).split(768):
            p,_=call(model,fast,ix,freeze)
            errors.append(mse(p[:,[1,3,7]].numpy(),fast['y'][ix][:,[1,3,7]].numpy()).reshape(len(ix),6))
        err=pd.DataFrame(np.concatenate(errors),columns=wave.METRICS).assign(flight=flight)
        table=err.groupby('flight').mean();table.to_csv(sub/'endpoints.csv')
        ep=table.mean().pow(.5).to_dict()
        beat=beats(model,orig,d,b,flight,freeze,cache,sub)
        shift=shifted(model,orig,d,b,flight,freeze,points,sub)
        results[tag]=dict(endpoint=ep,beats=beat,shifted=shift)
        if tag=='original':
            np.testing.assert_allclose([ep['w2'],beat['full_all']['rmse'],shift['2.0']['minus125']],
                [.53528346,.53108657,.92473039],atol=7e-5)
    (out/'validation.json').write_text(json.dumps(results,allow_nan=False,indent=2)+'\n')
    m=results['normal'];metrics={**m['endpoint'],'slowrate':m['beats']['full_all']['rmse'],'offgrid2':m['shifted']['2.0']['minus125']}
    Path(os.environ['RA_METRICS']).write_text(json.dumps(metrics,allow_nan=False)+'\n')
    print(metrics,flush=True)
if __name__=='__main__':evaluate()
