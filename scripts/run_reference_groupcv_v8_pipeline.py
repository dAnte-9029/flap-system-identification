"""Wait for training, freeze OOF decision, then execute its conditional followups."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import train_control_windows_v2 as base
from reference_groupcv_v8 import OUT,SEEDS


def run(name,extra=()):
    print('Running',name,*extra,flush=True)
    subprocess.run([sys.executable,str(base.ROOT/'scripts'/name),*extra],check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/mpl-reference-v8'})


def main():
    base.write(OUT/'pipeline_status.json',dict(status='running',stage='waiting for training'))
    while True:
        try:status=json.loads((OUT/'launch/status.json').read_text())
        except json.JSONDecodeError:time.sleep(1);continue
        if status['status']=='failed':raise RuntimeError(status['failures'])
        if status['status']=='complete':break
        time.sleep(10)
    base.write(OUT/'pipeline_status.json',dict(status='running',stage='OOF and matched-model development evaluation'))
    run('evaluate_reference_groupcv_v8.py')
    decision=json.loads((OUT/'cv_decision.json').read_text())
    if decision['cv_gate']:
        base.write(OUT/'pipeline_status.json',dict(status='running',stage='conditional full training'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda seed:run('train_reference_groupcv_v8_full.py',('--seed',str(seed))),SEEDS))
        base.write(OUT/'full_training_status.json',dict(status='complete',seeds=SEEDS))
    else:
        base.write(OUT/'full_training_status.json',dict(status='not_launched',reason='prespecified train-flight OOF gate failed'))
    base.write(OUT/'pipeline_status.json',dict(status='running',stage='report and provenance verification'))
    run('report_reference_groupcv_v8.py')
    base.write(OUT/'pipeline_status.json',dict(status='complete'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        base.write(OUT/'pipeline_status.json',dict(status='failed',error=str(exc)));raise
