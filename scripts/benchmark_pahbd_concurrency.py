#!/usr/bin/env python3
"""Measure short serial and two-GPU updates, with bounded CPU prefetch comparison."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
children=[];logs=[]
def launch(name,gpu,cpu,prefetch=False,barrier=None):
    command=[sys.executable,str(Path(__file__).with_name('benchmark_pahbd_workers.py')),'--output',str(root/f'{name}.json'),'--cpu',str(cpu)]
    if prefetch:command.extend(['--prefetch','--prefetch-cpu',str(16+gpu)])
    if barrier:command.extend(['--barrier',str(barrier),'--worker',name])
    log=(root/f'{name}.log').open('w');logs.append(log)
    child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
    children.append(child);return child

def wait(child,name):
    code=child.wait()
    if code:raise RuntimeError(f'{name} failed: {code}')
    result=json.loads((root/f'{name}.json').read_text());print(name,result['wall_seconds'],flush=True);return result
try:
    off=wait(launch('single0',0,12),'single0')
    on=wait(launch('prefetch0',0,12,True),'prefetch0')
    use_prefetch=on['wall_seconds']<off['wall_seconds']*.98
    one=on if use_prefetch else off
    two=wait(launch('single1',1,14,use_prefetch),'single1')
    barrier=root/'barrier';barrier.mkdir()
    jobs=[launch(f'dual{i}',i,12+2*i,use_prefetch,barrier) for i in (0,1)]
    deadline=time.monotonic()+240
    while not all((barrier/f'dual{i}').exists() for i in (0,1)):
        if any(c.poll() is not None for c in jobs) or time.monotonic()>deadline:raise RuntimeError('dual warmup failed/timed out')
        time.sleep(.2)
    start=time.perf_counter();(barrier/'GO').touch()
    results=[wait(c,f'dual{i}') for i,c in enumerate(jobs)]
    wall=max(r['wall_seconds'] for r in results)
    summary=dict(prefetch_speedup=off['wall_seconds']/on['wall_seconds'],selected_prefetch=use_prefetch,
                 serial_seconds=one['wall_seconds']+two['wall_seconds'],dual_seconds=wall,
                 dual_speedup=(one['wall_seconds']+two['wall_seconds'])/wall,
                 dual_steps_per_second=16/wall,dual_process_elapsed=time.perf_counter()-start)
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
finally:
    for child in children:
        if child.poll() is None:child.terminate();child.wait()
    for log in logs:log.close()
