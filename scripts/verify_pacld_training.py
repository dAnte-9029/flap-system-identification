#!/usr/bin/env python
"""Bounded readiness: synthetic backward, native horizon timings, <=3 updates, validation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import sys
import threading
import time
import psutil
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pacld import PACLD,Config
from system_identification.data.pacld import open_windows,subset,fixed_validation_indices,move_window
from system_identification.training.pacld import (seed_all,calibrated_weight,window_objective,gradient_norms,
    update,save_checkpoint,FlightBalancedSampler)
from system_identification.evaluation.pacld import evaluate


class Memory:
    def __enter__(self):
        self.process=psutil.Process();self.baseline=self.process.memory_info().rss
        self.peak=self.baseline;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.005): self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()


def timed_backward(model,window,weight):
    device=next(model.parameters()).device
    if device.type=='cuda':
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    with Memory() as memory:
        start=time.perf_counter();r=window_objective(model,window,weight)
        if device.type=='cuda':torch.cuda.synchronize(device)
        forward=time.perf_counter()-start;start=time.perf_counter()
        norms=gradient_norms(model,r)
        if device.type=='cuda':torch.cuda.synchronize(device)
        diagnostic=time.perf_counter()-start;start=time.perf_counter()
        r['total'].backward()
        if device.type=='cuda':torch.cuda.synchronize(device)
        backward=time.perf_counter()-start
    if not torch.isfinite(r['total']) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError('nonfinite readiness loss/gradient')
    return dict(window_id=window.metadata['window_id'],flight_id=window.metadata['flight_id'],
                horizon_us=window.metadata.get('horizon_us'),sampling_group=window.metadata.get('sampling_group','synthetic'),
                actual_horizon_s=r['times'][-1],native_queries=len(r['times']),pairs=len(r['pairs']),
                device=str(device),dtype=str(next(model.parameters()).dtype),forward_seconds=forward,
                backward_seconds=backward,gradient_diagnostic_seconds=diagnostic,
                windows_per_second=1/(forward+backward+diagnostic),
                rss_peak_mib=memory.peak/2**20,rss_increase_mib=(memory.peak-memory.baseline)/2**20,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
                full=float(r['full'].detach()),macro=None if r['macro'] is None else float(r['macro'].detach()),
                gradient_norms=norms)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/pacld/default.json')
    p.add_argument('--cuda-device',default=None,help='explicit idle device, e.g. cuda:0')
    p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text());torch.set_num_threads(1);seed_all(config['seed'])
    train=open_windows(config,'train');validation=open_windows(config,'validation')
    weight=calibrated_weight(config['training']['calibration'],config,train)
    initial=PACLD(Config(**config['model'])).double()
    from system_identification.data.pahbd_synthetic import controlled_periodic_window
    from system_identification.models.pahbd import Config as SyntheticConfig
    synth=controlled_periodic_window(SyntheticConfig(),.5)[0]
    records=[timed_backward(initial,synth,weight)]
    materialized=[]
    for horizon in config['data']['horizon_us']:
        pos=int(train.index.index[(train.index.sampling_group=='main_50hz') & (train.index.horizon_us==horizon)][0])
        materialized.append(train[pos])
    sparse_pos=int(train.index.index[(train.index.sampling_group=='sparse_10hz') & (train.index.horizon_us==500000)][0])
    sparse=train[sparse_pos]
    sparse_short_pos=int(train.index.index[(train.index.sampling_group=='sparse_10hz') & (train.index.horizon_us==200000)][0])
    sparse_short=train[sparse_short_pos]
    for window in materialized+[sparse_short,sparse]:
        r=timed_backward(initial,window,weight);records.append(r)
        print(json.dumps({k:r[k] for k in ('device','horizon_us','sampling_group','forward_seconds','backward_seconds','gradient_diagnostic_seconds','rss_peak_mib')}),flush=True)
    # Two real CPU updates, followed by the same model/Adam state on CUDA if available.
    model=PACLD(Config(**config['model'])).double();model.load_state_dict(initial.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),**config['optimizer'])
    updates=[]
    for window in (materialized[1],sparse):
        start=time.perf_counter();record=update(model,window,optimizer,weight)
        record.update(device='cpu',seconds=time.perf_counter()-start);updates.append(record)
    hardware=dict(cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count(),cpu_threads=1,
                  cpu_model=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].lstrip('\t :'),
                  cuda_requested=args.cuda_device)
    if args.cuda_device:
        if not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
        device=torch.device(args.cuda_device);torch.cuda.set_device(device)
        hardware['cuda_name']=torch.cuda.get_device_name(device)
        gpu=PACLD(Config(**config['model'])).double().to(device);gpu.load_state_dict(initial.state_dict())
        # CUDA initialization/warmup, no optimizer update.
        warm=move_window(synth,device);timed_backward(gpu,warm,weight)
        for window in materialized:
            r=timed_backward(gpu,move_window(window,device),weight);records.append(r)
            print(json.dumps({k:r[k] for k in ('device','horizon_us','forward_seconds','backward_seconds','gradient_diagnostic_seconds','cuda_peak_allocated_mib')}),flush=True)
        gpu.load_state_dict(model.state_dict())
        gpuopt=torch.optim.Adam(gpu.parameters(),**config['optimizer']);gpuopt.load_state_dict(optimizer.state_dict())
        torch.cuda.synchronize(device);start=time.perf_counter()
        record=update(gpu,move_window(sparse,device),gpuopt,weight);torch.cuda.synchronize(device)
        record.update(device=str(device),seconds=time.perf_counter()-start);updates.append(record)
        model.load_state_dict(gpu.state_dict());optimizer.load_state_dict(gpuopt.state_dict())
    else:
        start=time.perf_counter();record=update(model,materialized[0],optimizer,weight)
        record.update(device='cpu',seconds=time.perf_counter()-start);updates.append(record)
    fixed=subset(validation,fixed_validation_indices(validation.index,config['training']['validation_seed'],1))
    positions=fixed.index.groupby(['horizon_us','sampling_group'],sort=True).head(1).index.tolist()
    small=subset(fixed,positions)
    start=time.perf_counter();result=evaluate(model,small,config,diagnostics=True,include_trace=True)
    validation_seconds=time.perf_counter()-start
    identity=dict(train=train.identity(),validation=validation.identity(),validation_window_ids=small.index.window_id.tolist(),
                  scope='readiness_only_not_formal_resume')
    sampler=FlightBalancedSampler(train.index,config['seed'])
    save_checkpoint(out/'readiness.pt',model,optimizer,config,identity,sampler,
                    dict(step=len(updates),best_score=result['selection_score']),weight)
    (out/'performance.json').write_text(json.dumps(dict(hardware=hardware,records=records,
        timing_scope='materialized window; forward + separate gradient norms + total backward; no optimizer in timing rows',
        rss_scope='sampled process RSS incl loaded dataset, not allocator-only; single-process peaks may retain prior allocations'),indent=2))
    (out/'updates.json').write_text(json.dumps(dict(lambda_macro=weight,updates=updates,optimizer_steps=len(updates)),indent=2))
    (out/'validation.json').write_text(json.dumps(dict(**result,seconds=validation_seconds,data_identity=identity),indent=2))
    print(json.dumps({'optimizer_steps':len(updates),'validation_score':result['selection_score'],'validation_seconds':validation_seconds}),flush=True)


if __name__=='__main__':main()
