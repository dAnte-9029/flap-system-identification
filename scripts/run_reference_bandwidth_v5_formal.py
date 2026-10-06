"""Run only arms admitted by the train-only tiny gate, paired CPU jobs per seed."""
import json
import os
import subprocess
import sys
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT

def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    gate=json.loads((OUT/'tiny_decision.json').read_text());arms=gate['formal_arms'];dest=OUT/'formal_launch';dest.mkdir(exist_ok=False)
    base.write(dest/'protocol.json',dict(script_sha256=base.sha(__file__),gate_sha256=base.sha(OUT/'tiny_decision.json'),arms=arms,seeds=[17,23,42],workers=len(arms),sealed_test_accessed=False))
    completed=[]
    if not arms:
        base.write(dest/'status.json',dict(status='not_launched',reason='training-only tiny gate did not admit any arm'));return
    for seed in [17,23,42]:
        base.write(dest/'status.json',dict(status='running',seed=seed,completed=completed));jobs=[]
        for arm in arms:
            log=open(dest/f'{arm}_{seed}.log','w');cmd=[sys.executable,str(base.ROOT/'scripts/train_reference_bandwidth_v5.py'),'--arm',arm,'--seed',str(seed),'--formal']
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
            jobs.append((arm,process,log))
        failures=[]
        for arm,process,log in jobs:
            rc=process.wait();log.close()
            if rc:failures.append(dict(arm=arm,seed=seed,exit_code=rc))
            else:completed.append(dict(arm=arm,seed=seed))
        if failures:
            base.write(dest/'status.json',dict(status='failed',failures=failures,completed=completed));raise RuntimeError(failures)
    base.write(dest/'status.json',dict(status='complete',completed=completed))
if __name__=='__main__':main()
