"""Reference-hold ablation on frozen candidates, not measured causal responses."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT,load
from train_reference_unit_gravity import model_from
from train_reference_bandwidth_v5 import metrics
from control_windows_v3 import predict
from evaluate_control_windows_v2 import METRICS

def main():
    torch.set_num_threads(1);x,y,_,w=load();status=json.loads((OUT/'formal_launch/status.json').read_text());assert status['status']=='complete'
    meta=w[w.partition=='validation'];ix=meta.array_index.to_numpy();frames=[]
    for run in status['completed']:
        arm=run['arm'];seed=run['seed'];path=OUT/'formal'/f'{arm}_{seed}'/'final.pt';model=model_from(path)
        pred=predict(model,x,ix,hold=True)
        f,_=metrics(pred,{k:v[ix] for k,v in y.items()},x['history_state'][ix,:,3:6],meta,arm,seed);f['intervention']='held';frames.append(f)
        factual=pd.read_csv(path.parent/'per_flight.csv');factual=factual[factual.cohort=='validation'].copy();factual['intervention']='observed';frames.append(factual)
    a=pd.concat(frames);a.to_csv(OUT/'reference_hold_per_flight.csv',index=False)
    a.groupby(['arm','readout','intervention','subset','horizon_s'])[METRICS].mean().to_csv(OUT/'reference_hold_summary.csv')
    base.write(OUT/'reference_hold_scope.json',dict(script_sha256=base.sha(__file__),partitions=['validation'],frozen_models=True,
        purpose='model reliance on planned reference sequence; output difference is NOT a measured counterfactual or causal-effect validation',
        no_optimizer=True,sealed_test_accessed=False))
if __name__=='__main__':main()
