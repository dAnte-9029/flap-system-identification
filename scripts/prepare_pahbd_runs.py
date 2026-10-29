#!/usr/bin/env python3
"""Freeze the authorized six-run budget and shared validation membership."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.data.pahbd import NativeWindows
from system_identification.models.pahbd import Config
from system_identification.training.pahbd import FlightBalancedSampler,require_training_admission
from run_pahbd import validation_indices


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--device',required=True)
    a=p.parse_args();root=a.run_root.resolve();root.mkdir(parents=True,exist_ok=False);(root/'configs').mkdir()
    cfg=json.loads(Path('configs/pahbd/default.json').read_text())
    train=NativeWindows(**cfg['data'],split='train');val=NativeWindows(**cfg['data'],split='validation')
    require_training_admission(cfg,Config(**cfg['model']),train.windows,val.windows)
    index=train.windows.loc[train.windows.shared_periodic_eligible].reset_index(drop=True)
    vf=val.windows.loc[val.windows.shared_periodic_eligible]
    sizes=index.groupby(['flight_id','horizon_us']).size()
    assert index.flight_id.nunique()==114 and len(sizes)==456
    assert index.groupby('flight_id').horizon_us.nunique().eq(4).all()
    vi=validation_indices(vf,1701,cfg['training']['validation_windows_per_group'])
    identity=dict(manifest_sha256=train.manifest_sha256,phase_contract=train.phase_contract,
                  train_windows=len(index),train_flights=114,validation_windows=len(vi),
                  validation_window_ids=val.windows.loc[vi].window_id.tolist(),validation_seed=1701,
                  train_by_horizon=index.groupby('horizon_us').size().to_dict(),
                  validation_by_horizon=vf.groupby('horizon_us').size().to_dict())
    (root/'data_identity.json').write_text(json.dumps(identity,indent=2))
    tasks=[]
    for seed in (17,29,43):
        sampler=FlightBalancedSampler(index,seed)
        # Preserve the full paired draw sequence without materializing native windows.
        draws=[sampler.draw() for _ in range(10944)]
        assert all(g['draws']==24 for g in sampler.coverage()['groups'])
        (root/f'sampling_seed{seed}.json').write_text(json.dumps(index.iloc[draws].window_id.tolist()))
        for mode in ('A','B'):
            c=copy.deepcopy(cfg);c['seed']=seed;c['model']['mode']=mode
            c['validation_window_ids']=identity['validation_window_ids']
            c['training'].update(validation_seed=1701,validation_interval=912,fixed_budget=True,total_steps=10944,rounds=24)
            c['execution_contract']='Authorized formal A/B; float64; fixed budget; sealed test excluded'
            name=f'{mode}_seed{seed}'
            (root/'configs'/f'{name}.json').write_text(json.dumps(c,indent=2))
            tasks.append(dict(name=name,seed=seed,variant=mode,steps=10944,state='pending',device=a.device))
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (root/'queue.json').write_text(json.dumps(dict(state='prepared',code_revision=sha,device=a.device,created=time.time(),tasks=tasks),indent=2))
    print(json.dumps(dict(run_root=str(root),code_revision=sha,**{k:v for k,v in identity.items() if k not in ('validation_window_ids','phase_contract')}),indent=2))


if __name__=='__main__':main()
