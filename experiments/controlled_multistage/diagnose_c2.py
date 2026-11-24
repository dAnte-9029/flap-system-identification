import numpy as np
import pandas as pd
import torch
from workspace import *
from diagnose import native_diagnostics


@torch.no_grad()
def main():
    setup()
    data,ids=conditional_data(HISTORY,'validation',TAPES)
    model,_=checkpoint('c2')
    pieces=[]
    for j in range(0,len(ids),1024):
        b=select(data,slice(j,j+1024))
        roll=model(b[0],b[1],b[-1])
        _,diag=native_diagnostics(roll,b)
        for i,(component,slow,clock,axis) in enumerate([('pv','slow_v',0,slice(3,6)),('w','slow_w',2,slice(None))]):
            total=query(roll,b[3][:,:,clock],component)[...,axis]
            residual=total-query(roll,b[3][:,:,clock],slow)
            weights=b[5][:,:,i]
            den=weights.sum(1)
            net=(residual*weights[...,None]).sum(1)/den.clamp_min(1e-8)[:,None]
            diag[component+'_fast_beat_net2']=net.square().sum(1).masked_fill(den<=.05,np.nan).cpu().numpy()
            diag[component+'_fast_beat_ms']=((residual.square().sum(-1)*weights).sum(1)/den.clamp_min(1e-8)).masked_fill(den<=.05,np.nan).cpu().numpy()
        pieces.append(diag)
    tab=pd.concat(pieces,ignore_index=True).groupby(ids.flight_id.to_numpy()).mean().rename_axis('flight_id')
    tab.to_csv(RUN/'c2_diagnostic_flights.csv')
    print(tab.mean().drop('first_q_s').pow(.5).round(4).to_string())


if __name__=='__main__':
    main()
