"""Finite independent CPU jobs; launch state retained for review."""
import concurrent.futures
import os
import subprocess
import sys
import train_control_windows_v2 as base
from prepare_reference_frequency_weight_v13 import OUT


def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    jobs=[(0,17,True)]+[(f,s,False) for f in range(5) for s in [17,23,42]]
    launch=OUT/'launch';launch.mkdir(exist_ok=False);done=[];failed=[]
    def run(job):
        fold,seed,identity=job;cmd=[sys.executable,str(base.ROOT/'scripts/train_reference_frequency_weight_v13.py'),'--fold',str(fold),'--seed',str(seed)]
        if identity:cmd+=['--identity']
        with open(launch/f'fold{fold}_seed{seed}_identity{identity}.log','w') as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        return dict(fold=fold,seed=seed,identity=identity,exit_code=p.returncode)
    base.write(launch/'status.json',dict(status='running',completed=[],failures=[],total=len(jobs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for result in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            row=result.result();(failed if row['exit_code'] else done).append(row)
            base.write(launch/'status.json',dict(status='running',completed=done,failures=failed,total=len(jobs)));print(row,flush=True)
    base.write(launch/'status.json',dict(status='failed' if failed else 'complete',completed=done,failures=failed,total=len(jobs)))
    if failed:raise RuntimeError(failed)

if __name__=='__main__':main()
