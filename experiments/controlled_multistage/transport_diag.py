"""Test whether the body-to-NED transport term aligns with the remaining horizontal error."""
import numpy as np
import pandas as pd
import torch
from workspace import *
from evaluate import restore
from dynamics import rotate,inverse


@torch.no_grad()
def main():
    setup()
    d,ids=conditional_data(HISTORY,'validation',TAPES)
    m,_=restore('s2b')
    parts=[]
    for j in range(0,len(ids),1024):
        b=select(d,slice(j,j+1024))
        r=m(b[0],b[1],b[-1])
        time=(torch.arange(100,device='cuda')+.5)*.02
        t=time.expand(len(b[0]),-1)
        v=query(r,t,'slow_v'); w=query(r,t,'slow_w'); q=query(r,t,'q')
        tr=rotate(q,torch.linalg.cross(w,rotate(inverse(q),v)))
        rows=torch.arange(len(t),device='cuda'); ix=b[6][:,-1,0]
        end=b[3][rows,ix,0]
        # Native end times, with midpoint quadrature for the diagnostic perturbation.
        dp=(b[2][rows,ix,:3]-query(r,end[:,None],'pv')[:,0,:3])[:,:2]
        candidate=(.02*((end[:,None]-t).clamp_min(0)[:,:,None]*tr)).sum(1)[:,:2]
        parts.append(torch.stack([(dp*candidate).sum(-1),candidate.square().sum(-1),
                                  dp.square().sum(-1),tr[:,:,:2].square().sum(-1).mean(1)],-1).cpu().numpy())
    f=pd.DataFrame(np.concatenate(parts),columns=['error_dot_term','term_sq','error_sq','horizontal_acc_sq'])
    by=f.groupby(ids.flight_id.to_numpy()).mean()
    by.to_csv(RUN/'s2b_transport_diagnostic.csv')
    z=by.mean()
    print('cosine',z.error_dot_term/np.sqrt(z.term_sq*z.error_sq),'horizontal_term_rms',np.sqrt(z.horizontal_acc_sq),
          'positive_flights',int((by.error_dot_term>0).sum()))


if __name__=='__main__': main()
