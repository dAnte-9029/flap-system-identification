"""Score both validation-selected checkpoints on the same flights and native targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fit import make_model, conditional_data, conditional_loss
from train import HORIZONS,METRICS,select


@torch.no_grad()
def evaluate(name,run,data,ids):
    directory = run/name
    ck = torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    cfg = ck['config']
    sd = ck['model']
    model = make_model(name,sd['mean'],sd['scale']).cuda().eval()
    model.load_state_dict(sd)
    units = torch.tensor(cfg['units'],device='cuda')
    rows,summary = [],{}
    for policy in ('logged','held'):
        sums,masks,scores = [],[],[]
        for j in range(0,len(ids),cfg['batch']):
            batch = select(data,slice(j,j+cfg['batch']))
            if policy=='held':
                # Same integration partition, change only the four command values.
                tape = batch[-1]
                fixed = batch[0][:,-1,9:13][:,None,None,:].expand_as(tape[...,:4])
                batch[-1] = torch.cat([fixed,tape[...,4:]],-1)
            _,shared,sq,mask = conditional_loss(model,batch,units)
            sums.append(sq.cpu().numpy())
            masks.append(mask.cpu().numpy())
            scores.append(shared.cpu().numpy())
        sq,mask = np.concatenate(sums),np.concatenate(masks)
        flight = ids.flight_id.to_numpy()
        score = float(pd.Series(np.concatenate(scores)).groupby(flight).mean().mean())
        if policy=='logged':
            np.testing.assert_allclose(score,ck['validation_score'],atol=1e-5,rtol=1e-5)
        summary[policy] = score
        for k,h in enumerate(HORIZONS):
            for m,metric in enumerate(METRICS):
                valid=mask[:,k,m]
                grouped=pd.Series(sq[valid,k,m].astype(np.float64)).groupby(flight[valid]).agg(['mean','size'])
                for fid,item in grouped.iterrows():
                    rows.append(dict(model=name,control=policy,horizon_s=h,metric=metric,flight_id=fid,
                                     mse=item['mean'],windows=int(item['size'])))
    summary.update(best_epoch=ck['epoch'],params=cfg['params'],updates=json.loads((directory/'completed.json').read_text())['updates'])
    return pd.DataFrame(rows),summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','cache','tapes'):
        p.add_argument(key,type=Path)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    torch.cuda.set_device(a.gpu)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    data,ids=conditional_data(a.cache,'validation',a.tapes)
    parts,summary=[],{}
    for name in ('slowfast','gru'):
        frame,summary[name]=evaluate(name,a.run,data,ids)
        parts.append(frame)
    flights=pd.concat(parts,ignore_index=True)
    flights.to_csv(a.run/'flight_metrics.csv',index=False)
    by=['control','horizon_s','metric','flight_id']
    paired=flights.pivot(index=by,columns='model',values='mse')
    support=flights.pivot(index=by,columns='model',values='windows')
    assert not paired.isna().any().any() and (support.slowfast==support.gru).all()
    grouped=paired.groupby(level=by[:-1])
    rmse=grouped.mean().pow(.5)
    rmse['reduction_pct']=100*(1-rmse.slowfast/rmse.gru)
    rmse['better_flights']=(paired.slowfast<paired.gru).groupby(level=by[:-1]).sum()
    rmse['flights']=grouped.size()
    rmse['windows']=support.gru.groupby(level=by[:-1]).sum()
    rmse.to_csv(a.run/'comparison.csv')
    (a.run/'scores.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(rmse.loc['logged'].round(4))
