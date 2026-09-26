"""Read-only evidence audit; keep research completion distinct from control readiness."""
from pathlib import Path
import json
import subprocess
import numpy as np
import torch

import run_paper_rollout_horizon_ablation as source
from package_control_dynamics_candidate import EXPERIMENTS

m=source.m
ROOT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'


def verify():
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    assert branch=='control-dynamics-realdata-v1'
    subprocess.run(['git','merge-base','--is-ancestor','e64a1ba','HEAD'],check=True)
    counts={}
    for experiment in [*EXPERIMENTS,'delivery']:
        path=ROOT/experiment/'completion.json';c=m.read_json(path)
        assert c['status']=='complete' and c['heldout_accessed'] is False
        m.verify_pins(c['outputs']);counts[experiment]=len(c['outputs'])
        protocol=ROOT/experiment/'protocol.json'
        if protocol.exists():m.verify_pins(m.read_json(protocol)['pins'])
    sp,batches,_=source.inputs()
    assert set(batches)=={'train','validation'}
    assert not set(batches['train'].trajectory.log_ids)&set(batches['validation'].trajectory.log_ids)
    current=m.read_json(ROOT/'ensemble_candidate/completion.json')
    assert current['prediction_gate_passed'] and current['control_ready'] is False
    envelope=m.read_json(ROOT/'planning_error_margin/completion.json')
    assert envelope['passed']==0 and envelope['control_validated'] is False
    artifact=m.read_json(ROOT/'delivery/completion.json')
    assert artifact['standalone_smoke']['status']=='passed'
    assert artifact['standalone_smoke']['dataset_access'] is False
    receipt=dict(status='verified',branch=branch,base='e64a1ba',experiments=counts,
        observed_partitions=list(batches),train_flights=len(set(batches['train'].trajectory.log_ids)),
        validation_flights=len(set(batches['validation'].trajectory.log_ids)),
        archived_candidate_sha256=artifact['archive_sha256'],
        delivery_ready=True,prediction_improvement_verified=True,control_ready=False,
        unresolved=['causal action gain and physical delay','counterfactual error coverage','full flight-chain validation','RL transfer'],
        verifier_sha256=m.file_hash(Path(__file__)))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':verify()
