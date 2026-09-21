"""Finish paper artifacts after a detached pilot exits, without training polling.

The Linux pidfd wait blocks until process exit; no epochs/metrics are monitored.
The training runner already performs the unified evaluation on completion.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/analysis/results/paper_baseline_comparison_v1'
ART=ROOT/'artifacts/paper_baseline_comparison_v1'
MODELS=['B0_ConstantVelocity','B1_MLP','B2_StandardGRU','B3_ActuatorAwareGRU']


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def open_pidfd(pid):
    if hasattr(os,'pidfd_open'):
        return os.pidfd_open(pid)
    # Some Conda Python/glibc builds omit the wrapper; use the Linux syscall.
    libc=ctypes.CDLL(None,use_errno=True)
    if os.uname().machine not in ('x86_64','aarch64'):
        raise RuntimeError('pidfd syscall fallback unsupported on this architecture')
    libc.syscall.restype=ctypes.c_long
    fd=int(libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0)))
    if fd<0:
        code=ctypes.get_errno()
        raise OSError(code,os.strerror(code))
    return fd


def finalize():
    status=json.loads((ART/'status.json').read_text())
    if status['stage']!='pilot_complete':
        raise RuntimeError(f"pilot did not complete: {status}")
    summary=pd.read_csv(OUT/'summary.csv')
    per=pd.read_csv(OUT/'per_flight.csv')
    checks=json.loads((OUT/'sanity_checks.json').read_text())
    if (set(summary.model)!=set(MODELS) or len(summary)!=48 or len(per)!=272
            or not np.isfinite(summary.select_dtypes('number')).all().all()):
        raise ValueError('incomplete/nonfinite four-model report')
    for c,n,flights in [('ALL',2582,17),('Sep7',1481,9),('Sep17',1101,8)]:
        g=summary[summary.cohort==c]
        if len(g)!=16 or not (g.n_windows==n).all() or not (g.n_flights==flights).all():
            raise ValueError(f'cohort counts differ: {c}')
    if not checks['checkpoint_unchanged'] or checks['sealed_test_opened_this_run']:
        raise ValueError('frozen-checkpoint/test contract failure')
    if not checks['same_origins_all_models_horizons'] or not checks['normalization_train_only_reproduced']:
        raise ValueError('fairness sanity failure')
    histories={}
    for model in MODELS[1:3]:
        for stage,count in [('base',40),('continuation',25)]:
            path=ART/f'{model}_{stage}_history.csv'
            history=pd.read_csv(path)
            if (len(history)!=count or history.epoch.tolist()!=list(range(1,count+1))
                    or not np.isfinite(history.select_dtypes('number')).all().all()
                    or int(history.iloc[-1].optimizer_steps)!=111*count):
                raise ValueError(f'incomplete/nonfinite training history: {path}')
            histories[f'{model}/{stage}']=dict(epochs=count,windows_per_epoch=28293,
                optimizer_steps=int(history.iloc[-1].optimizer_steps),
                wall_time_s=float(history.iloc[-1].wall_time_s),last_epoch_loss=float(history.iloc[-1].loss))
    subprocess.run([sys.executable,str(ROOT/'scripts/report_paper_baseline_comparison.py')],cwd=ROOT,check=True)
    protocol=json.loads((OUT/'protocol.json').read_text())
    # Metadata only, no raw excluded flight or evaluation file is accessed.
    config_path=ROOT/'configs/data/trajectory_september_v2.yaml'
    config=yaml.safe_load(config_path.read_text())
    protocol['historical_excluded_flight_names']=list(config['excluded_logs'])
    protocol['historical_exclusion_metadata_source']=dict(path=str(config_path.relative_to(ROOT)),
        sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    protocol['model_phase_source_detail']='encoder-count-derived relative_flap_phase_rad, re-anchored at each origin; logged Hall/wing phase columns are not model inputs'
    protocol['completion_finalizer_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol['training_completion']=histories
    protocol['status']='seed17_pilot_complete_reported'
    write_json(OUT/'protocol.json',protocol)
    appendix='''\n## Completion verification and inherited exclusions\n\nAll four model/horizon/cohort grids, training epochs, optimizer-step counts and finite metrics were checked automatically after training exited. Full training histories remain under the artifact directory. No training-time monitoring or validation-based selection was performed by this finalizer.\n\nThree additional historical exclusions inherited from the September v2 configuration (names only, not newly opened):\n\n'''
    appendix+='\n'.join('- `'+name+'`' for name in config['excluded_logs'])+'\n'
    appendix+='\nThe physical phase feature is encoder-count-derived relative phase re-anchored at the prediction origin. Logged Hall/wing-phase metadata does not imply that this frozen network consumes absolute mechanical phase.\n'
    appendix+='\nFrozen Ours evaluator parity with 5,164 existing validation endpoints is recorded in `frozen_ours_evaluator_parity.json`; maximum differences are at CPU/GPU rounding scale. `unit_tests.json` records 29 passing focused tests.\n'
    appendix+='\nSuggested commit message: `feat: add frozen short-horizon paper baseline pilot`\n'
    with (OUT/'report.md').open('a') as stream:stream.write(appendix)
    git_status=subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True)
    (OUT/'git_status_at_completion.txt').write_text(git_status)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
               if p.is_file() and p.name!='completion.json'}
    write_json(OUT/'completion.json',dict(status='complete',completed_at_unix=time.time(),
        seed_replicates=1,models=MODELS,summary_rows=48,per_flight_rows=272,
        full_summary='summary.csv',report='report.md',sanity='sanity_checks.json',
        sealed_test_opened_this_run=False,additional_seeds_started=False,ablation_started=False,
        artifact_sha256=artifacts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args()
    try:
        fd=open_pidfd(args.wait_pid)
        write_json(ART/'finalizer_status.json',dict(stage='waiting_for_process_exit',pid=os.getpid(),
                   training_pid=args.wait_pid,method='blocking pidfd; no epoch polling'))
        write_json(OUT/'completion.json',dict(status='running',training_pid=args.wait_pid,
                   finalizer_pid=os.getpid(),training_monitoring=False,
                   additional_seeds_started=False,ablation_started=False))
        try:
            select.select([fd],[],[])
        finally:
            os.close(fd)
        finalize()
        write_json(ART/'finalizer_status.json',dict(stage='complete',pid=os.getpid()))
    except BaseException as exc:
        write_json(ART/'finalizer_status.json',dict(stage='failed',pid=os.getpid(),error=repr(exc)))
        write_json(OUT/'completion.json',dict(status='failed',error=repr(exc),
                   traceback=traceback.format_exc(),additional_seeds_started=False))
        raise


if __name__=='__main__':main()
