#!/usr/bin/env python3
"""Request a safe pause, or prepare (never launch) a fresh resume queue."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.training.pahbd_status import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['pause','prepare-resume'])
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();root=a.run_root.resolve();q=json.loads((root/'queue.json').read_text())
    if a.action=='pause':
        if q['state']!='running':p.error('queue is not running')
        (root/'PAUSE_REQUEST').touch()
        print('Queue pause requested; all active workers will save and pause, pending tasks stay queued.');return
    if q['state'] not in ('paused','stopped_by_user','failed'):p.error('source queue must be stopped')
    if not a.output or not a.checkpoint:p.error('resume requires --output and explicit --checkpoint')
    from system_identification.training.pahbd import load_checkpoint
    checkpoint=a.checkpoint.resolve();_,_,bundle=load_checkpoint(checkpoint)
    if not bundle.get('training_state'):p.error('checkpoint is not resumable')
    candidates=[t for t in q['tasks'] if checkpoint.parent==root/t['name']]
    if len(candidates)!=1:p.error('checkpoint must belong to a source task')
    task=candidates[0]
    selected_workers=[w for w in (q.get('workers') or []) if w['id']==task.get('worker')]
    cfg=root/'configs'/f"{task['name']}.json"
    if json.loads(cfg.read_text())!=bundle['metadata']['config']:p.error('checkpoint config differs')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);(output/'configs').mkdir()
    shutil.copy2(cfg,output/'configs'/cfg.name)
    for name in ('data_identity.json',f"sampling_seed{task['seed']}.json"):
        if (root/name).exists():shutil.copy2(root/name,output/name)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=Path(__file__).resolve().parents[1]).strip()
    task={k:task[k] for k in ('name','seed','variant','steps')}
    task.update(state='pending',checkpoint=str(checkpoint))
    prepared=dict(state='prepared',created=time.time(),code_revision=revision,device=q['device'],
                  physical_gpu=q.get('physical_gpu'),cpu_affinity=q.get('cpu_affinity'),
                  resumed_from=str(root),tasks=[task],workers=selected_workers or None,prefetch=q.get('prefetch',False))
    atomic_json(output/'queue.json',prepared)
    print(f'Prepared only: {output}; total budget {task["steps"]}, saved step {bundle["step"]}.')


if __name__=='__main__':main()
