"""Paired seed-17 training and validation of logged-control conditional trajectories."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dynamics import ControlledGRU, query
from backbones import BACKBONES
from model import trajectory_losses
from train import load, run


def make_model(name, mean, scale):
    if name not in ('slowfast','gru',*BACKBONES):
        raise ValueError(f'unknown model {name}')
    return ControlledGRU(mean,scale,name=='slowfast',name if name in BACKBONES else None)


def conditional_data(cache, split, tapes):
    batch,ids = load(cache,split)
    selected = pd.read_parquet(tapes/f'{split}_ids.parquet').window_id
    ix = pd.Index(ids.window_id).get_indexer(selected)
    if (ix<0).any() or not selected.is_unique:
        raise ValueError('control window identity mismatch')
    if not np.array_equal(ix,np.arange(len(ids))):
        it = torch.tensor(ix,device='cuda')
        batch = [x[it] for x in batch]
        ids = ids.iloc[ix].reset_index(drop=True)
    commands = torch.from_numpy(np.load(tapes/f'{split}_commands.npy')).cuda()
    if len(commands)!=len(ids):
        raise ValueError('command pack cardinality mismatch')
    return [*batch,commands],ids


def conditional_loss(model,batch,units):
    history,initial,y,t,valid,beat,endpoint,tape = batch
    roll = model(history,initial,tape)
    pv,q,w = query(roll,t[:,:,0],'pv'),query(roll,t[:,:,1],'q'),query(roll,t[:,:,2],'w')
    sv = query(roll,t[:,:,0],'slow_v') if model.structured else None
    sw = query(roll,t[:,:,2],'slow_w') if model.structured else None
    return trajectory_losses(pv,q,w,y,valid,beat,endpoint,units,sv,sw)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('cache','tapes','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--model',choices=['slowfast','gru',*BACKBONES],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--batch',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=17)
    a=parser.parse_args()
    run(a,load_data=lambda cache,split:conditional_data(cache,split,a.tapes),
        build_model=lambda mean,scale:make_model(a.model,mean,scale),
        loss_fn=conditional_loss,extra_config=dict(macro_dt=.1 if a.model=='slowfast' else .02,control_tapes=str(a.tapes.resolve()),
            task='logged ZOH commands; autonomous physical state and phase',
            frequency_coefficients=[3.6354511983566393,23.393570068972757,-5.126725395181777],
            frequency_source='frozen PACLD v1 train-fit linear motor-frequency ODE'))
