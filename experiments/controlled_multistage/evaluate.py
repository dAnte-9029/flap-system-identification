import argparse
import json
import numpy as np
import pandas as pd
import torch
from workspace import *
from train_three import HORIZONS3, data_three, evaluate, model_factory

H2=HORIZONS3[:-1]


def restore(name,three=False):
    if not three and (name in BASELINES or name=='c2'):
        return checkpoint(name)
    directory=(RUN/'trained3' if three else RUN)/name
    cfg=json.loads((directory/'config.json').read_text())
    ck=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    params=ck['model']
    m=model_factory(name,params['mean'],params['scale']).cuda().eval()
    m.load_state_dict(params)
    assert sum(p.numel() for p in m.parameters())==cfg['params']
    epochs=pd.read_json(directory/'epochs.jsonl',lines=True)
    assert ck['epoch']==int(epochs.loc[epochs.validation_score.idxmin(),'epoch'])
    assert json.loads((directory/'completed.json').read_text())['updates']==2976
    return m,ck


def tables(parts,settings,representative):
    flights=pd.concat(parts,ignore_index=True)
    flights['setting']=settings
    flights.to_csv(RUN/f'{settings}_paired.csv',index=False)
    idx=['horizon_s','metric','flight_id']
    mse=flights.pivot(index=idx,columns='model',values='mse')
    counts=flights.pivot(index=idx,columns='model',values='windows')
    assert np.isfinite(mse).all().all() and counts.eq(counts['gru'],axis=0).all().all()
    rows=[]
    for (h,metric),block in mse.groupby(level=idx[:2]):
        if h<.5: continue
        vals=block.mean().pow(.5)
        best=vals[list(BASELINES)].idxmin()
        factor=180/np.pi if metric=='attitude_rad' else 1.
        scaled=vals*factor
        win=int((block[representative]<block[best]).sum())
        rows.append(dict(setting=settings,horizon_s=h,metric=metric.replace('attitude_rad','attitude_deg'),
                         **scaled.to_dict(),representative=representative,best_baseline=best,
                         gap=scaled[representative]-scaled[best],wins_vs_best=win,flights=len(block),
                         windows=int(counts.xs((h,metric),level=idx[:2]).gru.sum())))
    frame=pd.DataFrame(rows)
    frame.to_csv(RUN/f'{settings}_comparison.csv',index=False)
    return frame


@torch.no_grad()
def main(setting,names):
    setup()
    if setting=='2s':
        data,ids=conditional_data(HISTORY,'validation',TAPES)
        rows=pd.read_csv(STAGE1/'paired_flights.csv')
        parts=[rows[rows.model.isin([*BASELINES,'slowfast','c1','c2'])]]
        scores={'c2':json.loads((STAGE1/'scores.json').read_text())['c2']['score']}
    else:
        lock=json.loads((RUN/'freeze.json').read_text())
        names=[lock['representative'],*BASELINES]
        data,ids=data_three('validation')
        parts=[];scores={}
    for name in names:
        model,ck=restore(name,setting=='trained3')
        score,rows=evaluate(model,data,ids,torch.tensor(ck['config']['units'],device='cuda'),
                            H2 if setting=='2s' else HORIZONS3)
        if setting!='extrapolated3':
            np.testing.assert_allclose(score,ck['validation_score'],atol=1e-6,rtol=1e-5)
        if setting=='2s':
            rows=rows.query('horizon_s>=.5')
        parts.append(rows.assign(model=name))
        scores[name]=score
    rep=min(scores,key=scores.get) if setting=='2s' else lock['representative']
    frame=tables(parts,setting,rep)
    (RUN/f'{setting}_scores.json').write_text(json.dumps(dict(representative=rep,shared_scores=scores),indent=2))
    print(rep, 'wins', int((frame.gap<0).sum()),'of',len(frame))
    print(frame[['horizon_s','metric',rep,'best_baseline','gap','wins_vs_best']].round(5).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('setting',choices=['2s','extrapolated3','trained3'])
    p.add_argument('names',nargs='*')
    args=p.parse_args()
    main(args.setting,args.names)
