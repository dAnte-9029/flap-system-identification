"""Final cross-check of model selection, member identity, optimizer budgets and delivered metrics."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from workspace import RUN,BASELINES,HERE
from freeze import names


def read(path):
    return json.loads(path.read_text())


def main():
    lock=read(RUN/'freeze.json')
    rep=lock['representative']
    assert rep==min(names,key=lock['shared_scores'].get)
    all_files=[RUN/n for n in names[1:]]+[(RUN/'trained3')/n for n in (*BASELINES,rep)]
    assert len(all_files)==11
    rng={};train_units={}
    for directory in all_files:
        last=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        stage=int(directory.parent.name=='trained3')
        epochs=pd.read_json(directory/'epochs.jsonl',lines=True)
        assert len(epochs)==24 and int(last['updates'])==2976
        assert best['epoch']==epochs.loc[epochs.validation_score.idxmin(),'epoch']
        assert best['updates']==124*best['epoch'] and best['config']['batch']==1024 and best['config']['seed']==17
        for checkpoint in (best,last):
            assert all(int(v['step'])==checkpoint['updates'] for v in checkpoint['optimizer']['state'].values())
            assert all(t.isfinite().all() for t in checkpoint['model'].values())
        if stage not in rng:
            rng[stage]=last['generator'];train_units[stage]=np.array(last['config']['units'])
        assert torch.equal(rng[stage],last['generator'])
        np.testing.assert_array_equal(train_units[stage],last['config']['units'])
    df=pd.read_csv(RUN/'comparison.csv'); paired=pd.read_csv(RUN/'paired_flights.csv')
    for (setting,h,metric),group in df.groupby(['setting','horizon_s','metric']):
        core=group[group.model.isin(BASELINES)].nsmallest(1,'rmse').iloc[0]
        assert group.best_baseline.eq(core.model).all() and group.flights.eq(39).all()
        native=metric.replace('attitude_deg','attitude_rad')
        sub=paired[(paired.setting==setting)&(paired.horizon_s==h)&(paired.metric==native)]
        mean=sub.groupby('model').mse.mean().pow(.5)*(180/np.pi if native=='attitude_rad' else 1.)
        np.testing.assert_allclose(group.rmse,mean.loc[group.model],rtol=1e-12)
        assert sub.groupby('model').flight_id.nunique().eq(39).all()
        assert group.windows.eq(44524 if setting=='2s' else 43987).all()
    for setting,expected in [('2s',12),('matched_original2',12),('extrapolated3',16),('trained3',16)]:
        assert len(df[(df.setting==setting)&(df.model==rep)])==expected
    for setting in ('2s','extrapolated3','trained3'):
        rec=read(RUN/f'{setting}_pipeline.json')
        assert rec['windows']==39 and min(rec['removed_raw_rows'].values())>0
        for c in rec['checks'].values():
            assert c['observation_removal']==0 and c['control_prefix']<=1e-5 and c['post_change']>1e-4 and c['cut_to_2s']<=1e-5
    for a,b in [('preupdate_padding_attempts/paired_before.csv','extrapolated3_paired.csv'),
                ('preupdate_padding_attempts/extrapolated_before.csv','extrapolated3_comparison.csv')]:
        assert (RUN/a).read_bytes()==(RUN/b).read_bytes()
    assert all((RUN/p).is_file() and (RUN/p).stat().st_size>3000 for p in ['network.pdf','rmse_horizons.pdf','network.png','rmse_horizons.png'])
    for ref in re.findall(r'\]\(([^)]+)\)', (RUN/'report_zh.md').read_text()):
        assert (RUN/ref).exists(),ref
    print('11 completed training records; selected epochs, optimizer and sampler states; four paired tables and all report links verified.')


if __name__=='__main__': main()
