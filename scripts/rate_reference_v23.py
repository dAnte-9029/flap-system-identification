"""Explicit body-rate reference contract and frozen-budget train helpers."""
import json
import numpy as np
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import ReferenceGRUPhaseSchedule

OUT=base.ROOT/'docs/analysis/results/control_windows_v23'
V8=OUT.parent/'control_windows_v8'
FIELDS=['roll','pitch','yaw','thrust_body[0]','reset_integral']


def reference_values(data,query):
    from reference_timing_v14 import effective_indices
    t=np.asarray(data['timestamp'],dtype=np.int64)
    sample=np.asarray(data.get('timestamp_sample',t),dtype=np.int64)
    ix=effective_indices(t,sample,query)
    value=np.column_stack([data[k][ix] for k in FIELDS]).astype(np.float32)
    if not np.isfinite(value).all():raise ValueError('nonfinite rate reference')
    if not np.isin(value[:,4],[0,1]).all():raise ValueError('invalid reset flag')
    return value,ix


def load():
    m=json.loads((OUT/'dataset_manifest.json').read_text())
    assert base.sha(OUT/'windows.npz')==m['arrays_sha256']
    with np.load(OUT/'windows.npz') as z:
        x={k[6:]:torch.tensor(z[k]) for k in z.files if k.startswith('input_')}
        y={k[6:]:torch.tensor(z[k]) for k in z.files if k.startswith('label_')}
    return x,y


def reference_stats(values):
    mean=values.astype(float).mean(0);scale=values.astype(float).std(0).clip(1e-4)
    # A binary reset command is not a tiny-variance continuous reference.
    mean[4]=0.;scale[4]=1.
    return mean,scale
