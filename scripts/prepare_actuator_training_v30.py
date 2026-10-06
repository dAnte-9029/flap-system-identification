"""Freeze execution metadata without changing the parent experiment recipe."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import train_control_windows_v2 as base
from actuator_training_v30 import OUT, DATA, CONTROL, load, verify_parent
from prepare_actuator_inputs_v29 import fingerprint
from system_identification.models.actuator_command_gru import ActuatorCommandGRU


def main():
    assert sys.executable == '/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    torch.set_num_threads(1)
    for root in [DATA, CONTROL]: verify_parent(root)
    p = json.loads((DATA / 'protocol.json').read_text())
    pins = dict(p['source_pins'])
    for name in ['actuator_training_v30.py', 'train_actuator_v30.py', 'run_actuator_v30.py',
                 'evaluate_actuator_v30.py', 'prepare_actuator_training_v30.py']:
        path = base.ROOT / 'scripts' / name; pins[str(path.relative_to(base.ROOT))] = base.sha(path)
    assert p['candidate_runs'] == 15 and p['steps'] == 1600 and p['batch'] == 32
    OUT.mkdir(exist_ok=False)
    base.write(OUT / 'protocol.json', dict(stage='v30 execute frozen v29 actuator-conditioned recipe',
        status='frozen_before_training', parent_protocol=p,
        parent_manifests={str((r/'delivery_manifest.json').relative_to(base.ROOT)): base.sha(r/'delivery_manifest.json') for r in [DATA, CONTROL]},
        source_pins=pins, device='cpu', threads_per_job=1, max_workers=6,
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__),
        new_training_runs=15, reused_control_runs=15, validation_accessed=False, sealed_test_accessed=False))
    x, y = load(); checks = []
    for run in json.loads((DATA / 'training_plan.json').read_text())['runs']:
        fold = run['fold']; dest = DATA / f'fold{fold}'
        info = json.loads((dest / 'manifest.json').read_text())
        assert base.sha(dest / 'normalization.npz') == run['normalization_sha256']
        assert base.sha(base.ROOT / run['control_checkpoint']) == run['control_checkpoint_sha256']
        assert base.sha(base.ROOT / run['draws_path']) == run['draws_sha256']
        with np.load(base.ROOT / run['draws_path']) as z: draws = z['array_index']
        assert draws.shape == (1600,32) and set(draws.ravel()) <= set(info['fit_indices'])
        with np.load(dest / 'normalization.npz') as z: stats = {k:z[k] for k in z.files}
        torch.manual_seed(run['seed']); model = ActuatorCommandGRU(stats)
        assert fingerprint(model) == run['initial_parameter_sha256']
        ids = draws[0][:2]
        pred = model(**{k:v[ids] for k,v in x.items()})
        loss = base.loss_fn(pred, {k:v[ids] for k,v in y.items()}, model)
        assert torch.isfinite(loss)
        checks.append(dict(fold=fold, seed=run['seed'], initial_parameters_and_draws_match=True))
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_actuator_inputs_v29.py',
                           'tests/test_rate_reference_evaluation_v23.py'], text=True, capture_output=True, cwd=base.ROOT)
    base.write(OUT / 'test_results.json', dict(returncode=test.returncode, stdout=test.stdout, stderr=test.stderr))
    assert test.returncode == 0, test.stdout + test.stderr
    base.write(OUT / 'preflight.json', dict(status='passed', checks=checks, tests_passed=4,
        original407_train_windows=True, parent_hashes_verified=True, formal_training_started=False))
    print('v30 frozen; all15 initializations/draws and4 tests passed', flush=True)


if __name__ == '__main__': main()
