"""Frozen v29 data and inference helpers for the v30 training execution."""
import json
import numpy as np
import torch
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from system_identification.models.actuator_command_gru import ActuatorCommandGRU

OUT = base.ROOT / 'docs/analysis/results/control_windows_v30'
DATA = OUT.parent / 'control_windows_v29'
CONTROL = OUT.parent / 'control_windows_v23'


def load():
    m = json.loads((DATA / 'dataset_manifest.json').read_text())
    assert base.sha(DATA / 'windows.npz') == m['arrays_sha256']
    with np.load(DATA / 'windows.npz') as z:
        x = {k[6:]: torch.tensor(z[k]) for k in z.files if k.startswith('input_')}
        y = {k[6:]: torch.tensor(z[k]) for k in z.files if k.startswith('label_')}
    assert len(x['history_state']) == 407 and x['future_command'].shape == (407, 50, 4)
    return x, y


def verify_protocol():
    p = json.loads((OUT / 'protocol.json').read_text())
    for name, digest in p['source_pins'].items(): assert base.sha(base.ROOT / name) == digest, name
    for name, digest in p['parent_manifests'].items(): assert base.sha(base.ROOT / name) == digest, name
    return p


def model_from(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['input_contract'] == 'post_allocation_motor0_servo012_v29'
    names = ['state_mean', 'state_scale', 'reference_mean', 'reference_scale',
             'actuator_mean', 'actuator_scale', 'rate_scale', 'velocity_scale']
    model = ActuatorCommandGRU({k: ck['state_dict'][k] for k in names})
    model.load_state_dict(ck['state_dict'])
    return model.eval()
