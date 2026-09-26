"""Verify and run the self-contained research candidate package, without datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();root=args.bundle.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('invalid bundle path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f'bundle checksum mismatch: {name}')
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import torch
    from system_identification.integration.control_dynamics_candidate import load_candidate,JointSupport,support_vector
    from system_identification.integration.control_error_envelope import EmpiricalErrorEnvelope
    from system_identification.models.control_ensemble import PHYSICAL
    torch.set_num_threads(2)
    model=load_candidate(root/'model.pt',manifest['files']['model.pt'])
    with np.load(root/'example_inputs.npz',allow_pickle=False) as z:
        future=z['future_controls'].copy();dt=z['dt_s'].copy()
        initial={k:torch.as_tensor(z[k],dtype=torch.bool if k=='history_mask' else torch.float32) for k in [*PHYSICAL,'history_state_features','history_controls','history_mask']}
    with torch.inference_mode():
        out=model(**initial,future_controls=torch.tensor(future,dtype=torch.float32),dt_s=torch.tensor(dt,dtype=torch.float32))
    differences={}
    with np.load(root/'expected_predictions.npz',allow_pickle=False) as expected:
        for key in PHYSICAL:
            result=getattr(out,key).numpy()
            np.testing.assert_allclose(result,expected[key],rtol=1e-5,atol=1e-5)
            differences[key]=float(np.max(np.abs(result-expected[key])))
    with np.load(root/'support.npz',allow_pickle=False) as z:
        support=JointSupport(**{k:z[k] for k in ['reference','mean','scale','lower','upper','distance_limit']})
    observed=initial['history_state_features'][:,-1].numpy()
    command=initial['history_controls'][:,-1].numpy();previous=initial['history_controls'][:,-6].numpy()
    coverage=support.query(support_vector(observed,command,previous))
    with (root/'empirical_envelope.csv').open() as f:envelope=EmpiricalErrorEnvelope(list(csv.DictReader(f)))
    print(json.dumps(dict(status='passed',dataset_access=False,example='observed initial history; synthesized held current command; expected output is frozen model prediction, not flight truth',
        max_absolute_difference=differences,initial_support_accepted=coverage['accepted'].tolist(),
        q_100ms_empirical_error=envelope.bound(5,'q'),control_validated=False),indent=2))


if __name__=='__main__':main()
