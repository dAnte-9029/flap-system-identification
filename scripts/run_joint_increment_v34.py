"""Future launch: verify one full zero-weight control before15 candidates."""
import concurrent.futures
import os
import subprocess
import sys
import train_control_windows_v2 as base
from train_joint_increment_v34 import PREP,OUT,read,verify_frozen


def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    verify_frozen();assert read(PREP/'verification.json')['status']=='passed'
    OUT.mkdir(exist_ok=False);logs=OUT/'launch';logs.mkdir()
    base.write(OUT/'protocol.json',dict(stage='v35 execute v34 frozen joint supervision',
        preparation_delivery_sha256=base.sha(PREP/'delivery_manifest.json'),preparation_protocol_sha256=base.sha(PREP/'protocol.json'),
        candidate_runs=15,zero_weight_reproduction_runs=1,coefficient=1.0,steps=1600,batch=32,max_workers=6,
        validation_accessed=False,sealed_test_accessed=False))
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    def run(fold,seed,zero=False):
        name='zero_control' if zero else f'fold{fold}_seed{seed}'
        with (logs/(name+'.log')).open('w') as output:
            args=[sys.executable,str(base.ROOT/'scripts/train_joint_increment_v34.py'),'--fold',str(fold),'--seed',str(seed)]
            if zero:args.append('--zero-control')
            result=subprocess.run(args,stdout=output,stderr=subprocess.STDOUT,env=env)
        return dict(fold=fold,seed=seed,zero_control=zero,exit_code=result.returncode)
    base.write(logs/'status.json',dict(status='zero_control_running',completed=[],total_candidates=15))
    first=run(0,17,True)
    if first['exit_code']:
        base.write(logs/'status.json',dict(status='failed',completed=[first],total_candidates=15))
        raise RuntimeError('Zero-weight reproduction failed; candidate launch blocked')
    assert read(OUT/'zero_weight_reproduction.json')['status']=='passed'
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(run,f,s) for f in range(5) for s in [17,23,42]]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result());print(completed[-1],flush=True)
            base.write(logs/'status.json',dict(status='running',completed=completed,total_candidates=15,zero_control=first))
    failed=any(r['exit_code'] for r in completed)
    base.write(logs/'status.json',dict(status='failed' if failed else 'training_complete_evaluation_pending',completed=completed,total_candidates=15,zero_control=first))
    if failed:raise RuntimeError('Candidate failed; inspect job logs')


if __name__=='__main__':main()
