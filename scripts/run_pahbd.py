#!/usr/bin/env python3
"""Explicit PA-HBD smoke, formal fitting, validation and frozen response evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import train_steps,save_checkpoint,load_checkpoint,validate,formal_fit,require_training_admission


def validation_indices(frame,seed,count):
    if count<1:raise ValueError('positive fixed validation cohort size required')
    rng=np.random.default_rng(seed)
    return [int(i) for _,g in frame.groupby(['flight_id','horizon_us','phase_mode'],sort=True)
            for i in rng.choice(g.index,size=min(count,len(g)),replace=False)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/pahbd/default.json')
    p.add_argument('--mode',choices=['smoke','train','validate','response'],default='smoke')
    p.add_argument('--source',choices=['synthetic','native'],default='synthetic')
    p.add_argument('--variant',choices=['A','B'])
    p.add_argument('--steps',type=int)
    p.add_argument('--windows',type=int,default=2,help='smoke only; formal fitting always uses the full eligible index')
    p.add_argument('--device',default='cpu')
    p.add_argument('--checkpoint')
    p.add_argument('--response-source')
    p.add_argument('--response-split',choices=['train','validation'],default='validation')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.mode=='train' and (args.source!='native' or args.steps is None or args.steps<1):
        p.error('formal train requires native source and explicit positive --steps')
    if args.mode=='smoke' and (args.windows<1 or args.device!='cpu' or (args.steps is not None and not 1<=args.steps<=3)):
        p.error('smoke requires CPU, positive windows and 1-3 optimizer steps')
    if args.mode in ('validate','response') and (not args.checkpoint or args.source!='native'):
        p.error('evaluation requires native source and explicit checkpoint')
    output=Path(args.output)
    if output.exists():p.error('output exists; choose a new directory')
    cfg=json.loads(Path(args.config).read_text());torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    model_cfg=Config(**cfg['model'])
    if args.variant:model_cfg.mode=args.variant
    bundle=None
    if args.checkpoint:
        model,optimizer,bundle=load_checkpoint(args.checkpoint,args.device,optimizer=args.mode=='train')
        model_cfg=model.config
        if args.variant and args.variant!=model_cfg.mode:p.error('variant disagrees with checkpoint')
    else:model=PAHBD(model_cfg).to(device=args.device,dtype=torch.float64);optimizer=None
    if optimizer is None:optimizer=torch.optim.Adam(model.parameters(),**cfg['optimizer'])
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    metadata=dict(config=cfg,model_config=asdict(model_cfg),seed=cfg['seed'],code_revision=revision,
                  code_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                  mode=args.mode,source=args.source,device=args.device,formal_performance_claim=False)
    if args.source=='synthetic':windows=[controlled_periodic_window(model_cfg)[0]]
    else:
        split=args.response_split if args.mode=='response' else ('validation' if args.mode=='validate' else 'train')
        dataset=NativeWindows(**cfg['data'],split=split,device=args.device)
        if args.mode=='smoke':indices=dataset.windows.groupby('flight_id',sort=True).head(1).index[:args.windows]
        elif args.mode=='validate':indices=validation_indices(dataset.windows,cfg['seed'],cfg['training']['validation_windows_per_group'])
        else:indices=range(len(dataset))
        windows=ConfiguredWindows(dataset,indices,model_cfg)
        metadata['data']=dict(split=split,windows=len(windows),manifest_sha256=dataset.manifest_sha256,
                              phase_policy=dataset.phase_policy,selection='full eligible training index; seeded per-flight/horizon validation cohort')
    if not windows:raise RuntimeError('no eligible windows')
    if args.mode=='train':
        require_training_admission(cfg,model_cfg,dataset.windows)
        validation=NativeWindows(**cfg['data'],split='validation',device=args.device)
        require_training_admission(cfg,model_cfg,dataset.windows,validation.windows)
        vi=validation_indices(validation.windows,cfg['seed'],cfg['training']['validation_windows_per_group'])
        val=ConfiguredWindows(validation,vi,model_cfg)
        metadata['validation_window_ids']=validation.windows.iloc[vi].window_id.tolist()
        result=formal_fit(model,windows,val,optimizer,args.steps,output,metadata,seed=cfg['seed'],
                          validation_interval=cfg['training']['validation_interval'],patience=cfg['training']['patience'],
                          loss_config=cfg['loss'],resume=bundle)
    elif args.mode=='response':
        if not args.response_source:p.error('response requires --response-source frozen experiment directory')
        from system_identification.evaluation.pahbd import FrozenRollResponse,factual_predictor
        model.eval();adapter=FrozenRollResponse(args.response_source)
        result=adapter.evaluate(factual_predictor(model,windows),args.response_split)
        output.mkdir(parents=True);result.pop('rows').to_csv(output/'response_rows.csv',index=False)
    else:
        output.mkdir(parents=True)
        if args.mode=='validate':result=validate(model,windows)
        else:
            from system_identification.training.pahbd import window_loss
            result=dict(optimizer_steps=train_steps(model,windows,optimizer,args.steps or 1,cfg['loss'],seed=cfg['seed']))
            checks=[]
            for w in windows:
                optimizer.zero_grad(set_to_none=True);loss,*_=window_loss(model,w,**cfg['loss']);loss.backward()
                if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('smoke backward failed')
                checks.append(dict(flight_id=w.metadata['flight_id'],finite_loss=True,finite_gradients=True))
            result['per_window_checks']=checks
            save_checkpoint(output/'checkpoint.pt',model,optimizer,metadata,args.steps or 1)
    (output/'result.json').write_text(json.dumps(dict(metadata=metadata,result=result),indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
