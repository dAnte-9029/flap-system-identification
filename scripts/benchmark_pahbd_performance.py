"""Fixed real Hall windows; isolated profiling/consistency snapshots, never formal training."""
import argparse,json,time,sys,os
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');p.add_argument('--compile',action='store_true');p.add_argument('--cpu',type=int,default=12);p.add_argument('--fresh-window',action='store_true');p.add_argument('--horizons',nargs='+',type=int,default=[200000,500000,1000000,2000000]);p.add_argument('--modes',nargs='+',default=['A','B']);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
sys.path.insert(0,str(Path(a.source)/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.training.pahbd import trajectory_loss
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
# Fix host scheduling on this hybrid CPU for matched throughput measurements.
if a.cpu>=0:os.sched_setaffinity(0,{a.cpu})
torch.set_num_threads(1);torch.cuda.set_device(0)
c=json.loads(Path('configs/pahbd/default.json').read_text());d=NativeWindows(**c['data'],split='train',device='cuda:0')
def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time();e1=torch.cuda.Event(enable_timing=True);e2=torch.cuda.Event(enable_timing=True);e1.record();v=fn();e2.record();torch.cuda.synchronize()
    return v,dict(wall=time.perf_counter()-start,cpu=time.process_time()-cpu,cuda_elapsed=e1.elapsed_time(e2)/1000)
# Annotate the legacy loss without changing its arithmetic for stage profiling.
import inspect
src=inspect.getsource(trajectory_loss)
if "record_function('state_loss')" not in src:
    start=src.index('    for e in targets:');end=src.index('    cycle_terms={}',start)
    src=src[:start]+"    with torch.profiler.record_function('state_loss'):\n"+''.join('    '+line+'\n' for line in src[start:end].splitlines())+src[end:]
    start=src.index('    cycle_terms={}');end=src.index('    total=state+',start)
    src=src[:start]+"    with torch.profiler.record_function('cycle_loss'):\n"+''.join('    '+line+'\n' for line in src[start:end].splitlines())+src[end:]
    ns=dict(trajectory_loss.__globals__);exec(src,ns);trajectory_loss=ns['trajectory_loss']
(root/"metadata.json").write_text(json.dumps(dict(argv=sys.argv,torch_version=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),device=torch.cuda.get_device_name(0),cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))))
records=[]
for mode in a.modes:
 for horizon in a.horizons:
    torch.manual_seed(17);cfg=Config(**dict(c['model'],mode=mode));m=PAHBD(cfg).double().cuda();opt=torch.optim.Adam(m.parameters(),**c['optimizer'])
    if a.compile:m.flow.derivative=torch.compile(m.flow.derivative,fullgraph=True,dynamic=False)
    i=int(d.windows[(d.windows.horizon_us==horizon)&d.windows.shared_periodic_eligible].index[0]);w,load=timed(lambda:ConfiguredWindows(d,[i],cfg)[0]);times=sorted({e.measurement for e in w.targets})
    last={}
    def compute(save=False):
        opt.zero_grad(set_to_none=True)
        with torch.profiler.record_function('history_belief'):b,hist=timed(lambda:m.encode(w))
        # Optimized model may provide forward-local preparation separately.
        with torch.profiler.record_function('future_lie'):pred,future=timed(lambda:m.flow.rollout(b,times,w.controls))
        with torch.profiler.record_function('trajectory_loss'):res,lt=timed(lambda:trajectory_loss(pred,w.targets,times,cycles=w.metadata.get('cycle_pairs',()),**c['loss']))
        loss,parts=res
        from dataclasses import replace
        last.update(pred=[replace(s,x=s.x.detach(),variance=s.variance.detach()) for s in pred])
        with torch.profiler.record_function('backward'):_,back=timed(loss.backward)
        if save:
            torch.save(dict(pred={k:torch.stack([getattr(s,k) for s in pred]).detach().cpu() for k in ('x','f','phi','h','variance','phase_covariance')},loss={k:v.detach().cpu() for k,v in parts.items()},grads={k:p.grad.detach().cpu() for k,p in m.named_parameters() if p.grad is not None},window_id=w.metadata['window_id']),root/f'{mode}_{horizon}.pt')
        return dict(history=hist,future=future,loss=lt,backward=back)
    if a.compile:
        explain=torch._dynamo.explain(m.flow.derivative)(w.initial)
        (root/f'{mode}_{horizon}_compile.json').write_text(json.dumps(dict(graph_count=explain.graph_count,graph_break_count=explain.graph_break_count,break_reasons=[str(x) for x in explain.break_reasons])))
    # Warm-up does not change parameters; fixed input repeated in each implementation.
    warm=time.perf_counter();compute();warm=time.perf_counter()-warm;torch.cuda.reset_peak_memory_stats()
    samples=[]
    for j in range(a.repeats):
        if a.fresh_window:
            d._cached_id=None
            w,load=timed(lambda:ConfiguredWindows(d,[i],cfg)[0])
            if hasattr(m.flow,'_noise_cache'):m.flow._noise_cache.clear()
            if hasattr(m.flow.phase,'_transition_cache'):m.flow.phase._transition_cache.clear()
        samples.append(compute(save=j==0))
    if a.profile:
        # Bounded CUDA trace from real integration, avoiding millions of trace records.
        prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
        original=m.flow.step;calls=[0]
        def sampled(*args,**kwargs):
            calls[0]+=1
            if calls[0]==1:prof.start()
            result=original(*args,**kwargs)
            if calls[0]==3:prof.stop()
            return result
        m.flow.step=sampled
        compute()
        m.flow.step=original
        with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as lp:
            trajectory_loss(last['pred'],w.targets,times,cycles=w.metadata.get('cycle_pairs',()),**c['loss'])
        (root/f'{mode}_{horizon}_loss_profile.txt').write_text(lp.key_averages().table(sort_by='cpu_time_total',row_limit=12))
        events=prof.key_averages();(root/f'{mode}_{horizon}_profile.txt').write_text(events.table(sort_by='self_cpu_time_total',row_limit=35))
        summary=[dict(key=e.key,count=e.count,cpu_us=e.self_cpu_time_total,cuda_us=e.self_device_time_total) for e in events]
        (root/f'{mode}_{horizon}_operators.json').write_text(json.dumps(summary))
    def validation():
        with torch.no_grad():
            b=m.encode(w);pred=m.flow.rollout(b,times,w.controls)
            return trajectory_loss(pred,w.targets,times,cycles=w.metadata.get('cycle_pairs',()),**c['loss'])
    _,validation_time=timed(validation)
    _,optimizer_time=timed(lambda:(torch.nn.utils.clip_grad_norm_(m.parameters(),10.),opt.step()))
    r=dict(warmup_seconds=warm,validation=validation_time,optimizer=optimizer_time,mode=mode,horizon=horizon,load=load,samples=samples,peak_bytes=torch.cuda.max_memory_allocated(),window_id=w.metadata['window_id']);records.append(r);print(json.dumps(r),flush=True);(root/'timings.json').write_text(json.dumps(records,indent=2))
