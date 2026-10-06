"""Execute the 15 frozen jobs with durable logs; CPU settings match controls."""
import concurrent.futures
import os
import subprocess
import sys
import train_control_windows_v2 as base
from actuator_training_v30 import OUT, verify_protocol


def main():
    assert sys.executable == '/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    verify_protocol()
    dest = OUT / 'launch'; dest.mkdir(exist_ok=False)
    results = []; jobs = [(f, s) for f in range(5) for s in [17, 23, 42]]
    base.write(dest / 'status.json', dict(status='running', completed=results, total=15))
    def run(job):
        fold, seed = job
        with (dest / f'fold{fold}_seed{seed}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(base.ROOT / 'scripts/train_actuator_v30.py'),
                '--fold', str(fold), '--seed', str(seed)], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        return dict(fold=fold, seed=seed, exit_code=result.returncode)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, job) for job in jobs]):
            results.append(future.result()); print(results[-1], flush=True)
            base.write(dest / 'status.json', dict(status='running', completed=results, total=15))
    failed = any(r['exit_code'] for r in results)
    base.write(dest / 'status.json', dict(status='failed' if failed else 'complete', completed=results, total=15))
    if failed: raise RuntimeError('Training failed; inspect per-job logs')


if __name__ == '__main__': main()
