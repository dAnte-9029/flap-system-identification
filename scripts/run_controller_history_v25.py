"""Run the frozen pair for every fold and seed; persistent logs and status."""
import concurrent.futures
import json
import os
import subprocess
import sys
import train_control_windows_v2 as base
from controller_history_training_v25 import OUT,ARMS


def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    assert json.loads((OUT/'preflight.json').read_text())['status']=='passed'
    dest=OUT/'launch';dest.mkdir(exist_ok=False);results=[]
    jobs=[(f,s,a) for f in range(5) for s in [17,23,42] for a in ARMS]
    base.write(dest/'status.json',dict(status='running',total=30,completed=results))
    def run(job):
        fold,seed,arm=job
        with open(dest/f'fold{fold}_{arm}_seed{seed}.log','w') as log:
            r=subprocess.run([sys.executable,str(base.ROOT/'scripts/train_controller_history_v25.py'),
                '--fold',str(fold),'--seed',str(seed),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        return dict(fold=fold,seed=seed,arm=arm,exit_code=r.returncode)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            r=f.result();results.append(r);print(r,flush=True)
            base.write(dest/'status.json',dict(status='running',total=30,completed=results))
    failed=any(r['exit_code'] for r in results)
    base.write(dest/'status.json',dict(status='failed' if failed else 'complete',total=30,completed=results))
    if failed:raise RuntimeError('See per-job failure logs')


if __name__=='__main__':main()
