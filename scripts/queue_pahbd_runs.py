#!/usr/bin/env python3
"""Persistent per-device sequential workers with global pause/fail-stop semantics."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.training.pahbd_status import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    root=a.run_root.resolve();q=json.loads((root/'queue.json').read_text())
    if q['state']!='prepared':raise RuntimeError('queue already started; refusing duplicate run')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if revision!=q['code_revision'] or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('source revision changed or dirty')
    workers=q.get('workers') or [dict(id='default',device=q['device'],physical_gpu=q.get('physical_gpu'),cpu_affinity=q.get('cpu_affinity'))]
    ids=[w['id'] for w in workers]
    if len(set(ids))!=len(ids):raise ValueError('duplicate worker id')
    gpus=[str(w['physical_gpu']) for w in workers if w.get('physical_gpu') is not None]
    if len(gpus)!=len(set(gpus)):raise ValueError('workers must use distinct physical GPUs')
    claimed={}
    for w in workers:
        cpus=list(w.get('cpu_affinity') or [])
        if q.get('prefetch') and w.get('prefetch_cpu') is not None:cpus.append(w['prefetch_cpu'])
        for cpu in cpus:
            topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            core=((topology/'physical_package_id').read_text().strip(),(topology/'core_id').read_text().strip())
            if core in claimed and claimed[core]!=w['id']:raise ValueError('workers share a physical CPU core')
            claimed[core]=w['id']
    for i,t in enumerate(q['tasks']):
        t.setdefault('worker',ids[i%len(ids)])
        if t['worker'] not in ids:raise ValueError('unknown task worker')
    q.update(state='running',pid=os.getpid(),started=time.time(),workers=workers)
    active={};stop=[None]
    def persist():
        q['heartbeat']=time.time();atomic_json(root/'queue.json',q)
    def interrupted(signum,frame):stop[0]=stop[0] or 'paused'
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    persist()
    try:
        while True:
            if (root/'PAUSE_REQUEST').exists():stop[0]=stop[0] or 'paused'
            # A task-local pause stops the whole queue before another task starts.
            if any((root/t['name']/'PAUSE_REQUEST').exists() for t,_,_ in active.values()):
                stop[0]=stop[0] or 'paused'
            for wid,(task,child,log) in list(active.items()):
                code=child.poll()
                if code is None:continue
                log.close();del active[wid]
                task.update(exit_code=code,ended=time.time(),state='completed' if code==0 else ('paused' if code==75 else 'failed'))
                atomic_json(root/f"{task['name']}.exit.json",dict(exit_code=code,ended=task['ended']))
                if code==75:stop[0]=stop[0] or 'paused'
                elif code:stop[0]='failed'
                persist()
            if stop[0]:
                for task,child,log in active.values():
                    # File request is safe even while the child is loading data;
                    # never send SIGTERM before its safe signal handler exists.
                    output=root/task['name']
                    if output.is_dir():(output/'PAUSE_REQUEST').touch()
                if not active:break
            else:
                for w in workers:
                    if w['id'] in active:continue
                    task=next((t for t in q['tasks'] if t['worker']==w['id'] and t['state']=='pending'),None)
                    if task is None:continue
                    device='cuda:0' if w.get('physical_gpu') is not None else w.get('device',q['device'])
                    command=[sys.executable,str(Path(__file__).with_name('run_pahbd.py')),'--mode','train','--source','native',
                             '--config',str(root/'configs'/f"{task['name']}.json"),'--device',device,
                             '--steps',str(task['steps']),'--output',str(root/task['name'])]
                    if task.get('checkpoint'):command.extend(['--checkpoint',task['checkpoint']])
                    if q.get('prefetch',False):
                        command.append('--prefetch')
                        if w.get('prefetch_cpu') is not None:command.extend(['--prefetch-cpu',str(w['prefetch_cpu'])])
                    if w.get('cpu_affinity'):command=['taskset','-c',','.join(map(str,w['cpu_affinity'])),*command]
                    env=dict(os.environ,PYTHONUNBUFFERED='1')
                    if w.get('physical_gpu') is not None:env['CUDA_VISIBLE_DEVICES']=str(w['physical_gpu'])
                    log=(root/f"{task['name']}.log").open('x')
                    try:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    except BaseException:log.close();raise
                    task.update(state='running',started=time.time(),pid=child.pid,device=device,
                                physical_gpu=w.get('physical_gpu'),cpu_affinity=w.get('cpu_affinity'))
                    active[w['id']]=(task,child,log);persist()
                if not active:break
            persist();time.sleep(1)
        state=stop[0] or 'completed';code={'completed':0,'paused':75,'failed':1}[state]
        q.update(state=state,exit_code=code,ended=time.time());persist()
        atomic_json(root/f'{state.upper()}.json',dict(exit_code=code,ended=q['ended']))
        if state=='failed':raise SystemExit(1)
    except SystemExit:raise
    except BaseException as exc:
        # Exceptional scheduler failure: preserve error and safely stop children.
        q.update(state='failed',error=str(exc),exit_code=1);persist()
        for task,child,log in active.values():
            while child.poll() is None:
                output=root/task['name']
                if output.is_dir():(output/'PAUSE_REQUEST').touch()
                time.sleep(1)
            task.update(state='completed' if child.returncode==0 else ('paused' if child.returncode==75 else 'failed'),exit_code=child.returncode,ended=time.time())
            atomic_json(root/f"{task['name']}.exit.json",dict(exit_code=child.returncode,ended=task['ended']))
            log.close()
        q['ended']=time.time();persist();atomic_json(root/'FAILED.json',dict(error=str(exc),exit_code=1))
        raise


if __name__=='__main__':main()
