#!/usr/bin/env python3
"""Sequential fail-stop queue, launched inside a detached tmux session."""
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
    q.update(state='running',pid=os.getpid(),started=time.time())
    child=None
    def persist():
        q['heartbeat']=time.time();atomic_json(root/'queue.json',q)
    def interrupted(signum,frame):
        raise RuntimeError(f'queue received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    persist()
    try:
        for task in q['tasks']:
            task.update(state='running',started=time.time());persist()
            command=[sys.executable,str(Path(__file__).with_name('run_pahbd.py')),'--mode','train','--source','native',
                     '--config',str(root/'configs'/f"{task['name']}.json"),'--device',q['device'],
                     '--steps',str(task['steps']),'--output',str(root/task['name'])]
            with (root/f"{task['name']}.log").open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
                task['pid']=child.pid;persist()
                while True:
                    try:code=child.wait(timeout=10);break
                    except subprocess.TimeoutExpired:persist()
            task.update(exit_code=code,ended=time.time(),state='completed' if code==0 else 'failed');persist()
            atomic_json(root/f"{task['name']}.exit.json",dict(exit_code=code,ended=task['ended']))
            if code:raise RuntimeError(f"{task['name']} exited {code}; remaining tasks not started")
        q.update(state='completed',exit_code=0,ended=time.time());persist()
        atomic_json(root/'COMPLETED.json',dict(exit_code=0,ended=q['ended']))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate();child.wait()
        for task in q['tasks']:
            if task['state']=='running':task.update(state='failed',exit_code=child.returncode if child else None)
        q.update(state='failed',error=str(exc),exit_code=1,ended=time.time());persist()
        atomic_json(root/'FAILED.json',dict(error=str(exc),exit_code=1,ended=q['ended']))
        raise


if __name__=='__main__':main()
