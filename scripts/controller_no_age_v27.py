"""Frozen data and fold statistics, without refitting or changing any input tensor."""
import json
import numpy as np
import train_control_windows_v2 as base
from controller_history_training_v25 import V23,V24,V8,load,fingerprint
from system_identification.models.reference_gru_controller_no_age import ReferenceGRUControllerNoAge

OUT=base.ROOT/'docs/analysis/results/control_windows_v27'
V25=OUT.parent/'control_windows_v25';V26=OUT.parent/'control_windows_v26'
ARMS=['no_age_history_control']


def model(fold,arm='no_age_history_control'):
    assert arm in ARMS
    a=json.loads((V23/f'fold{fold}/manifest.json').read_text());b=json.loads((V24/f'fold{fold}/manifest.json').read_text())
    assert a['fit_indices']==b['fit_indices'] and a['query_indices']==b['query_indices']
    assert base.sha(V23/f'fold{fold}/normalization.npz')==a['normalization_sha256']
    assert base.sha(V24/f'fold{fold}/controller_normalization.npz')==b['normalization_sha256']
    with np.load(V23/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
    with np.load(V24/f'fold{fold}/controller_normalization.npz') as z:
        return ReferenceGRUControllerNoAge(stats,z['mean'],z['scale'])
