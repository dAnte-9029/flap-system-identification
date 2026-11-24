"""Compare observed attitude/gyro closure with model attitude residual, on common native support."""
import pandas as pd
import numpy as np
import torch
from workspace import *
from evaluate import restore
from diagnose import pick, rotlog, integrate_rate
from dynamics import qmul, inverse, unit


@torch.no_grad()
def main():
    setup()
    data,ids=conditional_data(HISTORY,'validation',TAPES)
    model,_=restore('s2b')
    parts=[]
    for j in range(0,len(ids),1024):
        b=select(data,slice(j,j+1024)); t,mask=b[3],b[4]; y=b[2]
        qpred=query(model(b[0],b[1],b[-1]),t[:,:,1],'q')
        tw,order=t[:,:,2].masked_fill(~mask[:,:,2],float('inf')).sort(1)
        tq=t[:,:,1].contiguous()
        ix=torch.searchsorted(tw.contiguous(),tq).clamp(1,tw.size(1)-1)
        lo,hi=tw.gather(1,ix-1),tw.gather(1,ix)
        valid=mask[:,:,1]&hi.isfinite()&(tq>=lo)&(tq<=hi)
        def at(k):
            return y[:,:,10:].gather(1,order.gather(1,k)[...,None].expand(-1,-1,3))
        w=torch.lerp(at(ix-1),at(ix),((tq-lo)/(hi-lo).clamp_min(1e-8)).nan_to_num().clamp(0,1)[...,None])
        first=valid.long().argmax(1);last=valid.size(1)-1-valid.flip(1).long().argmax(1)
        q0=pick(y[:,:,6:10],first); qn=pick(y[:,:,6:10],last)
        integrated=unit(qmul(q0,integrate_rate(w,tq,valid)))
        obs=rotlog(qn,integrated); err=rotlog(qn,pick(qpred,last))
        parts.append(torch.stack([(obs*err).sum(-1),obs.square().sum(-1),err.square().sum(-1)],-1)
                     .masked_fill((valid.sum(1)<2)[:,None],np.nan).cpu().numpy())
    z=pd.DataFrame(np.concatenate(parts),columns=['dot','closure_sq','pred_error_sq']).groupby(ids.flight_id.to_numpy()).mean()
    z.to_csv(RUN/'s2b_gyro_closure.csv')
    a=z.mean();print('cosine',a['dot']/np.sqrt(a.closure_sq*a.pred_error_sq),'positive flights',int((z['dot']>0).sum()),'norms',np.sqrt(a.iloc[1:]).tolist())


if __name__=='__main__': main()
