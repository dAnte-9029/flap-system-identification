"""Launch exactly15 frozen candidates only after complete v42 delivery."""
import concurrent.futures
import subprocess
import sys
import train_control_windows_v2 as base
from train_attitude_v42 import PREP,OUT,read
from audit_controller_support_v26 import verify_parent


def execute(run):
    name=f"fold{run['fold']}_seed{run['seed']}"
    with (OUT/'launch'/f'{name}.log').open('w') as f:
        result=subprocess.run([sys.executable,'scripts/train_attitude_v42.py','--fold',str(run['fold']),'--seed',str(run['seed'])],cwd=base.ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; see launch log')
    return name


def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    verify_parent(PREP);assert read(PREP/'verification.json')['status']=='passed'
    assert read(PREP/'zero_weight_reproduction.json')['status']=='passed'
    OUT.mkdir(exist_ok=False);(OUT/'launch').mkdir()
    base.write(OUT/'protocol.json',dict(stage='v43 fixed tilt-supervision training',preparation_delivery_sha256=base.sha(PREP/'delivery_manifest.json'),
        training_plan_sha256=base.sha(PREP/'training_plan.json'),candidate_runs=15,updates_each=1600,
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    completed=[];base.write(OUT/'launch/status.json',dict(status='running',completed=[]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
        for name in pool.map(execute,read(PREP/'training_plan.json')['runs']):
            completed.append(name);base.write(OUT/'launch/status.json',dict(status='running',completed=completed));print(name,flush=True)
    base.write(OUT/'launch/status.json',dict(status='training_complete_evaluation_pending',completed=completed))


if __name__=='__main__':main()
