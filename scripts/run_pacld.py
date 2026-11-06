#!/usr/bin/env python
"""PACLD smoke/train/validate/calibrate. Formal training requires explicit steps."""
import argparse
import json
from pathlib import Path
import sys
import time
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pacld import PACLD, Config
from system_identification.data.pacld import open_windows, subset, fixed_validation_indices, move_window
from system_identification.training.pacld import (seed_all,calibrate,calibration_positions,calibrated_weight,
    update,fit,load_checkpoint,save_checkpoint,FlightBalancedSampler)
from system_identification.evaluation.pacld import evaluate


def datasets(config):
    train=open_windows(config,'train'); validation=open_windows(config,'validation')
    indices=fixed_validation_indices(validation.index,config['training']['validation_seed'],config['training']['validation_windows_per_group'])
    selected=subset(validation,indices)
    identity=dict(train=train.identity(),validation=validation.identity(),
                  validation_window_ids=selected.index.window_id.tolist())
    return train,selected,identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/pacld/default.json')
    p.add_argument('--mode',choices=['smoke','train','validate','calibrate'],default='smoke')
    p.add_argument('--source',choices=['synthetic','native'],default='native')
    p.add_argument('--steps',type=int,help='train: total step target including resume; smoke: 1-3')
    p.add_argument('--device',default='cpu')
    p.add_argument('--checkpoint')
    p.add_argument('--coverage',help='train-only pair census parquet for calibration')
    p.add_argument('--calibration',help='frozen calibration JSON; defaults to configuration')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.mode in ('train','smoke') and (args.steps is None or args.steps<1): p.error('explicit positive --steps required')
    if args.mode=='smoke' and args.steps>3: p.error('smoke is bounded to 1-3 optimizer updates')
    if args.mode=='validate' and not args.checkpoint: p.error('validate requires --checkpoint')
    if args.mode!='smoke' and args.source!='native': p.error('formal modes require native data')
    if args.source=='synthetic' and args.checkpoint: p.error('synthetic smoke does not resume native checkpoints')
    if args.mode=='calibrate' and (not args.coverage or args.checkpoint): p.error('calibrate needs --coverage and an initial model')
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    config=json.loads(Path(args.config).read_text())
    if config['training']['selection']!='FULL state loss: windows mean within flight/horizon, horizons equal within flight, flights equal':
        raise ValueError('unsupported checkpoint selection rule')
    torch.set_num_threads(config['training']['cpu_threads']); seed_all(config['seed'])
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; not silently falling back')
        torch.cuda.set_device(device)
    model=PACLD(Config(**config['model'])).to(device=device,dtype=getattr(torch,config['training']['dtype']))
    optimizer=torch.optim.Adam(model.parameters(),**config['optimizer']); bundle=None
    if args.source=='synthetic':
        from system_identification.data.pahbd_synthetic import controlled_periodic_window
        from system_identification.models.pahbd import Config as SyntheticConfig
        w=move_window(controlled_periodic_window(SyntheticConfig(),.5)[0],device)
        out.mkdir(parents=True)
        records=[update(model,w,optimizer,.1) for _ in range(args.steps)]
        (out/'smoke.json').write_text(json.dumps(dict(source='synthetic',lambda_macro=.1,
            weight_scope='engineering smoke only; native training uses frozen train calibration',records=records),indent=2))
        return
    if args.mode=='calibrate':
        train=open_windows(config,'train')
        positions=calibration_positions(train,pd.read_parquet(args.coverage),
                                        config['loss']['calibration_seed'],config['loss']['calibration_windows'])
        out.mkdir(parents=True)
        result=calibrate(model,train,positions,config,out/'calibration.json')
        print(json.dumps({k:result[k] for k in ('windows','lambda_macro','median_full','median_macro')})); return
    train,validation,identity=datasets(config)
    weight=calibrated_weight(args.calibration or config['training']['calibration'],config,train)
    if args.checkpoint:
        model,optimizer,bundle=load_checkpoint(args.checkpoint,config,identity,device,restore_rng=args.mode=='train')
        if bundle['lambda_macro']!=weight: raise ValueError('checkpoint and calibrated weight differ')
    if args.mode=='train':
        fit(model,train,validation,optimizer,config,identity,weight,args.steps,out,bundle)
    elif args.mode=='validate':
        out.mkdir(parents=True)
        result=evaluate(model,validation,config,diagnostics=True,include_trace=True)
        result.update(checkpoint=args.checkpoint,data_identity=identity)
        (out/'validation.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'selection_score':result['selection_score'],'coverage':result['coverage']}))
    else:
        if args.checkpoint: p.error('use train --checkpoint for resume; smoke starts fresh')
        out.mkdir(parents=True)
        sampler=FlightBalancedSampler(train.index,config['seed'])
        records=[]
        for _ in range(args.steps):
            records.append(update(model,move_window(train[sampler.draw()],device),optimizer,weight))
        # Fixed, small smoke cohort: first member of each horizon/group in fixed validation selection.
        vi=validation.index.groupby(['horizon_us','sampling_group'],sort=True).head(1).index.tolist()
        result=evaluate(model,subset(validation,vi),config,diagnostics=True)
        schedule=dict(step=args.steps,best_score=result['selection_score'],best_checkpoint=None)
        # Smoke checkpoint cannot masquerade as a resumable formal training checkpoint.
        smoke_identity=dict(identity,scope='smoke_only')
        save_checkpoint(out/'smoke.pt',model,optimizer,config,smoke_identity,sampler,schedule,weight)
        (out/'smoke.json').write_text(json.dumps(dict(records=records,validation=result,data_identity=smoke_identity),indent=2))
        print(json.dumps({'updates':len(records),'validation_score':result['selection_score']}))


if __name__=='__main__': main()
