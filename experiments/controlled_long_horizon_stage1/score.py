"""Frozen flight-paired table for the four baselines, original model and both candidates."""
import json
import numpy as np
import pandas as pd
from common import OUT, SUITE, HISTORY, TAPES, setup, conditional_data, conditional_loss, checkpoint, select, torch
from train import HORIZONS, METRICS

BASE = ['gru','lstm','tcn','transformer']
CANDIDATES = ['c1','c2']


@torch.no_grad()
def member(name, data, ids):
    model, ck = checkpoint(name)
    cfg = ck['config']
    directory = OUT/name
    epochs = pd.read_json(directory/'epochs.jsonl',lines=True)
    assert len(epochs)==24 and ck['epoch']==epochs.loc[epochs.validation_score.idxmin(),'epoch']
    last = torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    assert last['epoch']==24 and last['updates']==2976
    for state in (ck,last):
        assert all(int(s['step'])==state['updates'] for s in state['optimizer']['state'].values())
    units=torch.tensor(cfg['units'],device='cuda')
    parts, masks, scores = [], [], []
    for j in range(0,len(ids),cfg['batch']):
        batch=select(data,slice(j,j+cfg['batch']))
        _,score,sq,mask=conditional_loss(model,batch,units)
        parts.append(sq.cpu().numpy());masks.append(mask.cpu().numpy());scores.append(score.cpu().numpy())
    sq,mask=np.concatenate(parts),np.concatenate(masks)
    flight=ids.flight_id.to_numpy()
    score=pd.Series(np.concatenate(scores)).groupby(flight).mean().mean()
    np.testing.assert_allclose(score,ck['validation_score'],rtol=1e-5,atol=1e-6)
    rows=[]
    for k,h in enumerate(HORIZONS):
        for i,metric in enumerate(METRICS):
            valid=mask[:,k,i]
            agg=pd.Series(sq[valid,k,i].astype(np.float64)).groupby(flight[valid]).agg(['mean','size'])
            rows.append(agg.rename(columns={'mean':'mse','size':'windows'}).rename_axis('flight_id').reset_index()
                       .assign(model=name,horizon_s=h,metric=metric))
    return pd.concat(rows),dict(epoch=int(ck['epoch']),updates=int(ck['updates']),score=float(score),
                                run_seconds=json.loads((directory/'completed.json').read_text())['seconds'])


def main():
    setup()
    data,ids=conditional_data(HISTORY,'validation',TAPES)
    assert ids.window_id.is_unique and len(ids)==44524 and ids.flight_id.nunique()==39
    historical=pd.read_csv(SUITE/'per_flight.csv').query("control=='logged'").drop(columns='control')
    frames,scores=[historical],{}
    for name in CANDIDATES:
        df,scores[name]=member(name,data,ids)
        frames.append(df)
    records=pd.concat(frames,ignore_index=True).query('horizon_s >= .5')
    records.to_csv(OUT/'paired_flights.csv',index=False)
    idx=['horizon_s','metric','flight_id']
    mse=records.pivot(index=idx,columns='model',values='mse')
    support=records.pivot(index=idx,columns='model',values='windows')
    assert len(mse)==39*12 and np.isfinite(mse).all().all() and support.eq(support['gru'],axis=0).all().all()
    rmse=mse.groupby(level=idx[:2]).mean().pow(.5)
    ref=pd.read_csv(SUITE/'comparison.csv').query("control=='logged' and horizon_s >= .5").set_index(idx[:2])
    np.testing.assert_allclose(rmse[BASE+['slowfast']],ref.loc[rmse.index,BASE+['slowfast']],rtol=1e-7)
    rows=[]
    for (h,metric),r in rmse.iterrows():
        factor=180/np.pi if metric=='attitude_rad' else 1
        unit={'attitude_rad':'deg','position_m':'m','rate_rad_s':'rad/s','velocity_m_s':'m/s'}[metric]
        best=r[BASE].idxmin()
        byflight=mse.loc[(h,metric)]
        v=(r*factor).to_dict()
        row=dict(horizon_s=h,metric=metric.removesuffix('_rad') if metric=='attitude_rad' else metric,
                 unit=unit,**v,best_baseline=best,best_baseline_rmse=v[best],
                 original_gap=v['slowfast']-v[best],windows=int(support.loc[(h,metric),'gru'].sum()),flights=len(byflight))
        for c in CANDIDATES:
            row[c+'_gap']=v[c]-v[best]
            closed=v['slowfast']-v[c]
            row[c+'_recovered']=closed
            row[c+'_gap_closed_pct']=100*closed/row['original_gap'] if row['original_gap']>0 else np.nan
            row[c+'_wins_vs_baseline']=int((byflight[c]<byflight[best]).sum())
            row[c+'_wins_vs_original']=int((byflight[c]<byflight.slowfast).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    (OUT/'scores.json').write_text(json.dumps(scores,indent=2))
    print(scores)


if __name__=='__main__':
    main()
