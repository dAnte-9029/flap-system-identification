"""Frozen data/model helpers for the paired historical-state experiment."""
import json
import hashlib
import numpy as np
import torch
import train_control_windows_v2 as base
from rate_reference_v23 import load as load_reference,V8
from system_identification.models.reference_gru_controller_history import ReferenceGRUControllerHistory

OUT=base.ROOT/'docs/analysis/results/control_windows_v25'
V23=OUT.parent/'control_windows_v23';V24=OUT.parent/'control_windows_v24'
ARMS=['zero_history_control','observed_history_control']


def fingerprint(model,common=False):
    values=[]
    for key,p in model.named_parameters():
        if common and key=='encoder.weight_ih_l0':p=p[:,:25]
        values.append(p.detach().cpu().numpy().tobytes())
    return hashlib.sha256(b''.join(values)).hexdigest()


def load():
    x,y=load_reference();m=json.loads((V24/'dataset_manifest.json').read_text())
    assert base.sha(V24/'history_controller.npz')==m['arrays_sha256']
    with np.load(V24/'history_controller.npz') as z:
        np.testing.assert_array_equal(z['array_index'],np.arange(407))
        assert (z['publication_us']<=z['query_us'][:,:,None]).all()
        x['history_controller']=torch.tensor(z['history_controller'])
    return x,y


def model(fold,arm):
    assert arm in ARMS
    mf=json.loads((V23/f'fold{fold}/manifest.json').read_text());cf=json.loads((V24/f'fold{fold}/manifest.json').read_text())
    assert mf['fit_indices']==cf['fit_indices'] and mf['query_indices']==cf['query_indices']
    assert base.sha(V23/f'fold{fold}/normalization.npz')==mf['normalization_sha256']
    assert base.sha(V24/f'fold{fold}/controller_normalization.npz')==cf['normalization_sha256']
    with np.load(V23/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
    with np.load(V24/f'fold{fold}/controller_normalization.npz') as z:
        return ReferenceGRUControllerHistory(stats,z['mean'],z['scale'],arm=='observed_history_control')
