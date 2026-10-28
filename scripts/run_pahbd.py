#!/usr/bin/env python3
"""Explicit CPU smoke / train / validation entry; no implicit formal training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd import NativeWindows,configure_latent,ConfiguredWindows
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import train_steps,save_checkpoint,load_checkpoint,validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/pahbd/default.json')
    p.add_argument('--mode',choices=['smoke','train','validate'],default='smoke')
    p.add_argument('--source',choices=['synthetic','native'],default='synthetic')
    p.add_argument('--variant',choices=['A','B'])
    p.add_argument('--steps',type=int)
    p.add_argument('--windows',type=int,default=2,help='number of frozen windows; 0 means all in train/validate')
    p.add_argument('--device',default='cpu')
    p.add_argument('--checkpoint')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.windows<0 or (args.mode=='smoke' and args.windows==0):p.error('smoke requires a positive window count')
    if args.mode=='train' and (args.steps is None or args.steps<1):p.error('train requires explicit positive --steps')
    if args.mode=='smoke' and (args.device!='cpu' or (args.steps is not None and not 1<=args.steps<=3)):
        p.error('smoke is limited to CPU and 1-3 optimizer steps')
    if args.mode=='validate' and (not args.checkpoint or args.source!='native'):
        p.error('validate requires --checkpoint and --source native')
    output=Path(args.output)
    if output.exists():p.error('output exists; choose a new directory')
    cfg=json.loads(Path(args.config).read_text());torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    model_cfg=Config(**cfg['model'])
    if args.variant:model_cfg.mode=args.variant
    if args.checkpoint:
        model,optimizer,bundle=load_checkpoint(args.checkpoint,args.device,optimizer=args.mode!='validate')
        model_cfg=model.config
        if args.variant and args.variant!=model_cfg.mode:p.error('variant disagrees with checkpoint')
    else:model=PAHBD(model_cfg).to(device=args.device,dtype=torch.float64);optimizer=None
    if optimizer is None:optimizer=torch.optim.Adam(model.parameters(),**cfg['optimizer'])
    if args.source=='synthetic':
        if args.device!='cpu':p.error('synthetic engineering path is CPU only')
        windows=[controlled_periodic_window(model_cfg)[0]]
    else:
        split='validation' if args.mode=='validate' else 'train'
        dataset=NativeWindows(**cfg['data'],split=split,device=args.device)
        # Smoke samples distinct flights; formal mode uses the declared prefix of frozen windows.
        indices=(dataset.windows.groupby('flight_id',sort=False).head(1).index[:args.windows]
                 if args.mode=='smoke' else range(min(args.windows or len(dataset),len(dataset))))
        windows=ConfiguredWindows(dataset,indices,model_cfg)
    if not windows:raise RuntimeError('no eligible windows')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    metadata=dict(config=cfg,model_config=asdict(model_cfg),seed=cfg['seed'],code_revision=revision,
                  code_dirty=bool(status),mode=args.mode,source=args.source,device=args.device,
                  windows=([w.metadata for w in windows] if args.mode=='smoke' or args.source=='synthetic' else
                           dict(split=split,count=len(windows),manifest_sha256=dataset.manifest_sha256,
                                selection='frozen window prefix',indices=windows.indices)),formal_performance_claim=False)
    if args.mode=='validate':result=validate(model,windows)
    else:
        steps=args.steps or 1
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        result=dict(optimizer_steps=train_steps(model,windows,optimizer,steps,cfg['loss']),
                    changed_parameter_tensors=sum(not torch.equal(before[k],v) for k,v in model.named_parameters()),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    events=([dict(history=len(w.history),targets=len(w.targets),controls=len(w.controls.times)) for w in windows]
                            if args.mode=='smoke' else None))
        # A native smoke must forward/backward every selected flight, even when steps=1.
        if args.mode=='smoke':
            from system_identification.training.pahbd import window_loss
            checks=[]
            for w in windows:
                optimizer.zero_grad(set_to_none=True);loss,*_=window_loss(model,w,**cfg['loss']);loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('smoke backward failed')
                checks.append(dict(flight_id=w.metadata['flight_id'],finite_loss=bool(torch.isfinite(loss)),
                                   finite_gradients=True))
            result['per_window_checks']=checks
    output.mkdir(parents=True)
    (output/'result.json').write_text(json.dumps(dict(metadata=metadata,result=result),indent=2))
    if args.mode!='validate':
        save_checkpoint(output/'checkpoint.pt',model,optimizer,metadata,steps)
        restored,_,_=load_checkpoint(output/'checkpoint.pt',args.device)
        if not all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()):
            raise RuntimeError('checkpoint roundtrip differs')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
