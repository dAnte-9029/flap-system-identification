"""Shared explicit contracts for the raw100 versus anti-aliased50 experiment."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from system_identification.models.reference_antialias_readout import antialias_rate_readout
OUT=base.ROOT/'docs/analysis/results/control_windows_v5'
DATA=base.ROOT/'dataset/control_windows_v5'

def load():
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    for name,digest in manifest['pins'].items():assert base.sha(base.ROOT/name)==digest,name
    with np.load(DATA/'windows_100hz.npz') as z:
        x={k[6:]:torch.tensor(z[k],dtype=torch.float32) for k in z.files if k.startswith('input_')}
        y={k[6:]:torch.tensor(z[k],dtype=torch.float32) for k in z.files if k.startswith('label_')}
    with np.load(base.ROOT/'dataset/control_windows_v2/normalization.npz') as z:stats={k:z[k] for k in z.files}
    w=pd.read_csv(OUT/'windows.csv');assert set(w.partition)=={'train','validation'}
    return x,y,stats,w


def forecast_loss(pred,target,model,history,arm):
    if arm=='raw100':pr,yr=pred['rates_b'],target['rates_b']
    elif arm=='aa50':pr,yr=antialias_rate_readout(history,pred['rates_b']),antialias_rate_readout(history,target['rates_b'])
    else:raise ValueError(arm)
    rate=((pr-yr)/model.rate_scale).square().mean()
    # Pose/velocity labels retain original50Hz publication contract and loss times.
    velocity=((pred['velocity_n'][:,1::2]-target['velocity_n'][:,1::2])/model.velocity_scale).square().mean()
    dot=(pred['quaternion'][:,1::2]*target['quaternion'][:,1::2]).sum(-1).clamp(-1,1)
    return rate+velocity+.2*(4*(1-dot.square()).clamp_min(0)/.01).mean()


def as_50hz(pred,truth,history,readout):
    pp={k:v[:,1::2] for k,v in pred.items()};yy={k:v[:,1::2] for k,v in truth.items()}
    if readout=='aa50':
        pp['rates_b']=antialias_rate_readout(history,pred['rates_b']);yy['rates_b']=antialias_rate_readout(history,truth['rates_b'])
    elif readout!='raw50':raise ValueError(readout)
    return pp,yy


def causal_expand_50hz(pred,initial):
    result={}
    for k,v in pred.items():
        result[k]=torch.stack([torch.cat([initial[k][:,None],v[:,:-1]],1),v],2).flatten(1,2)
    return result
