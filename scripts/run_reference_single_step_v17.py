"""Run independent deterministic CPU training jobs, retaining launch evidence."""
import concurrent.futures
import os
import subprocess
import sys
import train_control_windows_v2 as base
from reference_single_step_v17 import OUT


def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    launch=OUT/'launch';launch.mkdir(exist_ok=False);done=[];failed=[]
    def run(job):
        fold,seed=job
        command=[sys.executable,str(base.ROOT/'scripts/train_reference_single_step_v17.py'),'--fold',str(fold),'--seed',str(seed)]
        with open(launch/f'fold{fold}_seed{seed}.log','w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        return dict(fold=fold,seed=seed,exit_code=result.returncode)
    base.write(launch/'status.json',dict(status='running',completed=[],failures=[],total=15))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run,(f,s)) for f in range(5) for s in [17,23,42]]
        for future in concurrent.futures.as_completed(jobs):
            row=future.result();(failed if row['exit_code'] else done).append(row)
            base.write(launch/'status.json',dict(status='running',completed=done,failures=failed,total=15));print(row,flush=True)
    base.write(launch/'status.json',dict(status='failed' if failed else 'complete',completed=done,failures=failed,total=15))
    if failed:raise RuntimeError(failed)


if __name__=='__main__':main()
