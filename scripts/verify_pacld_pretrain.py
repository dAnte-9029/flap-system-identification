#!/usr/bin/env python
"""Bounded real-data cache/equivalence and sequential end-to-end benchmarks.

Each benchmark arm is an independent seed17 temporary model. Baseline source
comes from the declared Git commit, never from an approximate reimplementation
of the loss. This script does not call formal training or validation selection.
"""
import argparse
import copy
from dataclasses import fields, is_dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import psutil
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.data.pacld import open_windows, PACLDWindows, move_window
from system_identification.models.pacld import PACLD, Config
from system_identification.training.pacld import (seed_all,update,FlightBalancedSampler,diagnostics_due,
    save_checkpoint,load_checkpoint,calibrated_weight)

BASELINE='371765df7b94d89864d415fd19b6c31f68518705'


class Memory:
    def __enter__(self):
        self.process=psutil.Process();self.peak=self.process.memory_info().rss;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.005):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()


def old_module(out,kind):
    path=out/'baseline_snapshot'/'src'/'system_identification'/kind/'baseline.py'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(subprocess.check_output(['git','show',f'{BASELINE}:src/system_identification/{kind}/pahbd.py' if kind=='data'
                                              else f'{BASELINE}:src/system_identification/training/pacld.py']))
    spec=importlib.util.spec_from_file_location(f'system_identification.{kind}._pretrain_baseline',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif is_dataclass(a):
        for field in fields(a):same(getattr(a,field.name),getattr(b,field.name))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
    return time.perf_counter()


def baseline_update(old,model,w,optimizer):
    """Instrument the exact prior objective/backward/update ordering, without clipping."""
    device=next(model.parameters()).device;timing={};start=sync(device)
    encode=model.encode;rollout=model.flow.rollout
    def timed_encode(window):
        t=sync(device);result=encode(window);timing['encode_seconds']=sync(device)-t;return result
    def timed_rollout(*args,**kwargs):
        t=sync(device);result=rollout(*args,**kwargs)
        timing[kwargs.get('mode','FULL').lower()+'_forward_seconds']=sync(device)-t;return result
    model.train();optimizer.zero_grad(set_to_none=True)
    model.encode=timed_encode;model.flow.rollout=timed_rollout
    t=sync(device)
    try:r=old.window_objective(model,w,1.)
    finally:model.encode=encode;model.flow.rollout=rollout
    timing['loss_pair_overhead_seconds']=sync(device)-t-sum(timing.values())
    t=sync(device);norms=old.gradient_norms(model,r);timing['gradient_diagnostic_seconds']=sync(device)-t
    t=sync(device);r['total'].backward();timing['backward_seconds']=sync(device)-t
    t=sync(device)
    if not torch.isfinite(r['total']) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError('baseline objective/gradient nonfinite')
    norm=float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])))
    timing['gradient_check_clip_seconds']=sync(device)-t
    t=sync(device);optimizer.step();timing['optimizer_seconds']=sync(device)-t
    timing['update_seconds']=sync(device)-start
    return dict(full=float(r['full'].detach()),macro=None if r['macro'] is None else float(r['macro'].detach()),
                total=float(r['total'].detach()),pairs=len(r['pairs']),gradient_norms=norms,
                pre_clip_grad_norm=norm,gradient_clipped=False,max_grad_norm=None,timing=timing)


def equivalence(old,config):
    from system_identification.data.pahbd_synthetic import controlled_periodic_window
    from system_identification.models.pahbd import Config as SyntheticConfig
    seed_all(17);m=PACLD(Config(width=16)).double();other=copy.deepcopy(m)
    w=controlled_periodic_window(SyntheticConfig(),.45)[0]
    a=torch.optim.Adam(m.parameters(),**config['optimizer']);b=torch.optim.Adam(other.parameters(),**config['optimizer'])
    baseline_update(old,m,w,a);update(other,w,b,1.,diagnostics=True,max_grad_norm=None)
    same(m.state_dict(),other.state_dict());same(a.state_dict(),b.state_dict())
    for x,y in zip(m.parameters(),other.parameters()):same(x.grad,y.grad)
    return 'bitwise CPU equality: baseline vs new with clipping disabled'


def cache_check(config,out):
    old=old_module(out,'data');plain=old.NativeWindows(**config['data'],split='train')
    small=copy.deepcopy(config);small['data_cache']['max_flights']=2
    cached=open_windows(small,'train');frame=cached.index
    a=int(frame.index[(frame.sampling_group=='main_50hz')&(frame.horizon_us==500000)][0])
    b=int(frame.index[(frame.sampling_group=='sparse_10hz')&(frame.horizon_us==500000)][0])
    c=int(frame.index[(frame.sampling_group=='main_50hz')&(frame.horizon_us==500000)&(frame.flight_id!=frame.loc[a,'flight_id'])][0])
    records=[]
    for label,pos in [('cold_main',a),('hit_main',a),('cold_sparse',b),('switch_hit_main',a),('evict_with_third',c),('reload_sparse',b),('reload_main',a)]:
        before=cached.dataset.cache_stats();counts={'baseline':0,'cached':0};original=pd.read_parquet
        def read_old(*args,**kwargs):counts['baseline']+=1;return original(*args,**kwargs)
        def read_new(*args,**kwargs):counts['cached']+=1;return original(*args,**kwargs)
        with Memory() as memory:
            t=time.perf_counter()
            with patch('pandas.read_parquet',side_effect=read_old):expected=plain[cached.indices[pos]]
            old_seconds=time.perf_counter()-t;t=time.perf_counter()
            with patch('pandas.read_parquet',side_effect=read_new):actual=cached[pos]
            new_seconds=time.perf_counter()-t
        same(expected,actual)
        records.append(dict(case=label,window_id=actual.metadata['window_id'],sampling_group=actual.metadata['sampling_group'],
            baseline_seconds=old_seconds,cached_seconds=new_seconds,parquet_reads=counts,
            cache_before=before,cache_after=cached.dataset.cache_stats(),rss_peak_mib=memory.peak/2**20,all_fields_equal=True))
    # Verify a different native window on the same retained flight, not only an identical window.
    next_pos=int(frame.index[(frame.flight_id==frame.loc[a,'flight_id'])&(frame.horizon_us==1000000)][0])
    same(plain[cached.indices[next_pos]],cached[next_pos])
    assert all(key[0]=='train' for key in cached.dataset._flights)
    assert cached.dataset.cache_stats()['retained_bytes']<=small['data_cache']['max_bytes']
    (out/'cache_checks.json').write_text(json.dumps(dict(records=records,second_window_equal=True,
        scope='all dataclass tensors, clocks, controls, masks, phase and metadata recursively bitwise/equal; no window pre-materialization'),indent=2))


def benchmark(config,out,variant,device,steps):
    old=old_module(out,'training');equivalent=equivalence(old,config)
    seed_all(17)
    with Memory() as memory:
        startup=time.perf_counter()
        if variant=='baseline':
            reader=old_module(out,'data').NativeWindows(**config['data'],split='train')
            windows=PACLDWindows(reader)
        else:windows=open_windows(config,'train')
        weight=calibrated_weight(config['training']['calibration'],config,windows);assert weight==1.
        model=PACLD(Config(**config['model'])).double().to(device)
        optimizer=torch.optim.Adam(model.parameters(),**config['optimizer'])
        sampler=FlightBalancedSampler(windows.index,17)
        identity={'train':windows.identity(),'scope':'pretrain_optimization_benchmark'}
        startup=time.perf_counter()-startup
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        records=[];global_start=sync(device)
        for i in range(steps):
            t=sync(device);pos=sampler.draw();draw_seconds=sync(device)-t
            t=sync(device);w=windows[pos];read_seconds=sync(device)-t
            t=sync(device);w=move_window(w,device);move_seconds=sync(device)-t
            t=sync(device)
            if variant=='baseline':record=baseline_update(old,model,w,optimizer)
            else:record=update(model,w,optimizer,weight,diagnostics=diagnostics_due(i,config['training']['gradient_diagnostics_interval']),
                               max_grad_norm=config['training']['max_grad_norm'])
            compute_seconds=sync(device)-t
            record.update(step=i+1,window_id=w.metadata['window_id'],flight_id=w.metadata['flight_id'],
                          horizon_us=w.metadata['horizon_us'],sampling_group=w.metadata['sampling_group'],
                          sampler_seconds=draw_seconds,read_seconds=read_seconds,move_seconds=move_seconds,
                          compute_seconds=compute_seconds,e2e_seconds=draw_seconds+read_seconds+move_seconds+compute_seconds,
                          cache=windows.dataset.cache_stats() if hasattr(windows.dataset,'cache_stats') else None)
            records.append(record)
            with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({k:record[k] for k in ('step','horizon_us','sampling_group','e2e_seconds','total','pre_clip_grad_norm')}),flush=True)
            if variant=='optimized' and i+1==steps//2:
                # Real checkpoint restores RNG, Adam, global step and the next sampler draw.
                save_checkpoint(out/'midpoint.pt',model,optimizer,config,identity,sampler,{'step':i+1,'best_score':None},weight)
                next_sampler=copy.deepcopy(sampler);expected_next=next_sampler.draw()
                model,optimizer,bundle=load_checkpoint(out/'midpoint.pt',config,identity,device,sampler_index=windows.index)
                sampler.load_state_dict(bundle['sampler']);probe=copy.deepcopy(sampler)
                assert probe.draw()==expected_next
        wall=sync(device)-global_start
        summary=dict(baseline_revision=BASELINE,variant=variant,device=str(device),steps=steps,startup_seconds=startup,
            wall_seconds_including_checkpoint=wall,mean_step_seconds=float(np.mean([r['e2e_seconds'] for r in records])),
            windows_per_second=steps/sum(r['e2e_seconds'] for r in records),rss_peak_mib=memory.peak/2**20,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
            flights=len({r['flight_id'] for r in records}),all_finite=all(np.isfinite(r['total']) for r in records),
            clipped_steps=sum(r['gradient_clipped'] for r in records),
            diagnostic_steps=[r['step'] for r in records if r['gradient_norms'] is not None],
            cache=windows.dataset.cache_stats() if hasattr(windows.dataset,'cache_stats') else None,
            equivalence_probe=equivalent,midpoint_resume_verified=variant=='optimized',
            effective_training_settings=dict(gradient_diagnostics_interval=1 if variant=='baseline' else config['training']['gradient_diagnostics_interval'],
                max_grad_norm=None if variant=='baseline' else config['training']['max_grad_norm'],
                cache='legacy current-flight only; encoder every window' if variant=='baseline' else config['data_cache']),
            data_identity=windows.identity(),config=config,records=records)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','records')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['cache','benchmark'],required=True)
    p.add_argument('--variant',choices=['baseline','optimized'],default='optimized');p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=16);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/pacld/default.json');args=p.parse_args()
    if not 16<=args.steps<=32:p.error('bounded benchmark requires 16-32 steps per arm')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text());torch.set_num_threads(1)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
        torch.cuda.set_device(device)
    if args.mode=='cache':cache_check(config,out)
    else:benchmark(config,out,args.variant,device,args.steps)


if __name__=='__main__':main()
