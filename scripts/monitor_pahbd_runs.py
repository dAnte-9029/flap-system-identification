#!/usr/bin/env python3
"""Read persistent queue and training status from any SSH terminal."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return {}


def alive(pid):
    if not pid:return False
    try:os.kill(pid,0);return True
    except (ProcessLookupError,PermissionError):return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    q=read(a.run_root/'queue.json');now=time.time();remaining=0.;known=True
    print('Run:',a.run_root,'\nQueue:',q.get('state','not started'),'PID:',q.get('pid'),'alive:',alive(q.get('pid')))
    baseline=read(a.run_root/'benchmark.json').get('task_eta_seconds')
    for task in q.get('tasks',[]):
        s=read(a.run_root/task['name']/'status.json');state=task['state']
        if state=='running' and not alive(task.get('pid')):state='ABNORMAL STOP'
        eta=s.get('eta_seconds')
        if state in ('pending','running'):
            if eta is None and baseline is not None:eta=baseline*(1-s.get('step',0)/task['steps'])
            if eta is None:known=False
            else:remaining+=eta
        age=now-s['heartbeat'] if 'heartbeat' in s else None
        print(f"{task['name']}: {state} {s.get('phase','queued')} {s.get('step',0)}/{task['steps']} "
              f"loss={s.get('loss','-')} val={s.get('validation_done','-')}/{s.get('validation_total','-')} "
              f"ETA={str(round(eta/3600,2))+'h' if eta is not None else '数据不足'} "
              f"heartbeat={str(round(age,1))+'s ago' if age is not None else '-'} exit={task.get('exit_code','-')}")
    print('Queue remaining:',f'{remaining/3600:.2f}h; finish ~{dt.datetime.fromtimestamp(now+remaining).isoformat(timespec="minutes")}' if known else '数据不足')
    subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv'],check=False)
    pids=[str(t['pid']) for t in q.get('tasks',[]) if t.get('state')=='running' and t.get('pid')]
    if pids:subprocess.run(['ps','-p',','.join(pids),'-o','pid,etime,pcpu,pmem,stat,args'],check=False)


if __name__=='__main__':main()
