"""Thirty prespecified CPU training trajectories; independent single-thread jobs."""
import concurrent.futures
import json
import os
import subprocess
import sys
import train_control_windows_v2 as base
from reference_groupcv_v8 import OUT


def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    protocol=json.loads((OUT/'protocol.json').read_text());jobs=[(fold,arm,seed) for fold in range(5) for arm in protocol['arms'] for seed in protocol['seeds']]
    out=OUT/'launch';out.mkdir(exist_ok=False);completed=[];failures=[]
    base.write(out/'status.json',dict(status='running',completed=completed,failures=failures,total=len(jobs)))
    def run(job):
        fold,arm,seed=job
        with open(out/f'fold{fold}_{arm}_{seed}.log','w') as log:
            result=subprocess.run([sys.executable,str(base.ROOT/'scripts/train_reference_groupcv_v8.py'),'--fold',str(fold),'--arm',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
        return dict(fold=fold,arm=arm,seed=seed,exit_code=result.returncode)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            row=future.result();(failures if row['exit_code'] else completed).append(row)
            base.write(out/'status.json',dict(status='running',completed=completed,failures=failures,total=len(jobs)))
    base.write(out/'status.json',dict(status='failed' if failures else 'complete',completed=completed,failures=failures,total=len(jobs)))
    if failures:raise RuntimeError(failures)

if __name__=='__main__':main()
