"""Independent fixed-budget trajectories, at most six single-thread CPU jobs."""
import concurrent.futures
import json
import os
import subprocess
import sys
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT

def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    gate=json.loads((OUT/'tiny_decision.json').read_text());jobs=[(arm,seed) for arm in gate['formal_arms'] for seed in gate['seeds']]
    out=OUT/'formal_launch';out.mkdir(exist_ok=False);workers=min(6,len(jobs))
    base.write(out/'protocol.json',dict(script_sha256=base.sha(__file__),gate_sha256=base.sha(OUT/'tiny_decision.json'),jobs=jobs,workers=workers,cpu_threads_per_job=1))
    if not jobs:base.write(out/'status.json',dict(status='not_launched',reason='no train-only gate passed'));return
    completed=[];failures=[];base.write(out/'status.json',dict(status='running',completed=completed,failures=failures))
    def run(item):
        arm,seed=item
        with open(out/f'{arm}_{seed}.log','w') as log:
            result=subprocess.run([sys.executable,str(base.ROOT/'scripts/train_reference_capacity_v6.py'),'--arm',arm,'--seed',str(seed),'--formal'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
        return dict(arm=arm,seed=seed,exit_code=result.returncode)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            row=future.result();(failures if row['exit_code'] else completed).append(row)
            base.write(out/'status.json',dict(status='running',completed=completed,failures=failures))
    base.write(out/'status.json',dict(status='failed' if failures else 'complete',completed=completed,failures=failures))
    if failures:raise RuntimeError(failures)
if __name__=='__main__':main()
