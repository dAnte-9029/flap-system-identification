"""Train-flight normalization and cross-fitted phase contracts for v8."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_generalization_v7 import phase_prediction
from system_identification.models.reference_gru_phase_schedule import ReferenceGRUPhaseSchedule

OUT = base.ROOT/'docs/analysis/results/control_windows_v8'
PHASE = OUT.parent/'control_windows_v4/diagnostic_future_phase.npz'
SEEDS = [17,23,42]
STAT_KEYS = ['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale']


def fit_normalization(arrays, windows, targets):
    """Same unique-row recipe as v2, restricted to explicitly supplied fit flights."""
    if not (windows.partition=='train').all():
        raise ValueError('normalization requires original training flights')
    history=[];reference=[];actuator=[];rates=[]
    for log,g in windows.groupby('log_id'):
        a=arrays[log]
        hh=np.unique(np.concatenate([np.arange(i-25,i+1) for i in g.origin_index]))
        rr=np.unique(np.concatenate([np.arange(i-25,i+50) for i in g.origin_index]))
        history.append(a['state_features'][hh]);reference.append(a['references'][rr]);actuator.append(a['actuator_commands'][hh-1])
        rates.extend(a['rates_b'][i:i+51] for i in g.origin_index)
    stats={}
    for name,items in [('state',history),('reference',reference),('actuator',actuator)]:
        z=np.concatenate(items).astype(float)
        stats[name+'_mean']=z.mean(0);stats[name+'_scale']=z.std(0).clip(1e-4)
    stats['state_mean'][9:11]=0;stats['state_scale'][9:11]=1
    stats['rate_scale']=np.concatenate(rates).std(0).clip(.1)
    stats['velocity_scale']=targets[windows.array_index.to_numpy()].reshape(-1,3).std(0).clip(.5)
    return stats


def crossfit_phase(history, phase, logs, fit_indices, query_indices):
    """Each GRU fit sample's phase prediction excludes its entire source flight."""
    fit_indices=np.asarray(fit_indices);query_indices=np.asarray(query_indices)
    if set(logs[fit_indices])&set(logs[query_indices]):
        raise ValueError('fit/query flights overlap')
    fit_prediction=np.zeros((len(fit_indices),50));provenance=[];parameters=[]
    for log in sorted(set(logs[fit_indices])):
        train=fit_indices[logs[fit_indices]!=log];local=np.flatnonzero(logs[fit_indices]==log);query=fit_indices[local]
        pp,pars=phase_prediction('ridge10',history[query],history[train],phase[train],logs[train])
        fit_prediction[local]=pp
        provenance.append(dict(fit_indices=train.tolist(),query_indices=query.tolist()))
        parameters.append(pars)
    query_prediction,pars=phase_prediction('ridge10',history[query_indices],history[fit_indices],phase[fit_indices],logs[fit_indices])
    provenance.append(dict(fit_indices=fit_indices.tolist(),query_indices=query_indices.tolist()))
    parameters.append(pars)
    return fit_prediction,query_prediction,provenance,parameters


def phase_start(end):
    return np.column_stack([np.zeros(len(end)),end[:,:-1]]).astype(np.float32)


def load_train():
    # Verify immutable archive, but materialize only the authorized training windows.
    manifest=json.loads((base.DATA/'manifest.json').read_text())
    for name,digest in manifest['tables'].items():assert base.sha(base.DATA/name)==digest,name
    for name,digest in manifest['arrays'].items():assert base.sha(base.ROOT/name)==digest,name
    w=pd.read_csv(base.DATA/'windows_v2.csv');w=w[w.partition=='train'].copy();ids=w.array_index.to_numpy()
    assert np.array_equal(ids,np.arange(len(ids))), 'explicit index remapping required'
    with np.load(base.ROOT/'dataset/control_windows_v2/windows.npz') as z:
        x={k[6:]:torch.tensor(z[k][ids],dtype=torch.float32) for k in z.files if k.startswith('input_')}
        y={k[6:]:torch.tensor(z[k][ids],dtype=torch.float32) for k in z.files if k.startswith('label_')}
    return x,y,w


def predict(model,x,ids,schedule=None):
    result={k:[] for k in ['rates_b','velocity_n','quaternion']}
    with torch.no_grad():
        for start in range(0,len(ids),64):
            ix=ids[start:start+64];batch={k:v[ix] for k,v in x.items()}
            if schedule is not None:batch['phase_start']=schedule[ix]
            pred=model(**batch)
            for k,v in pred.items():
                assert torch.isfinite(v).all();result[k].append(v.numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def model_from(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=ReferenceGRUPhaseSchedule({k:ck['state_dict'][k] for k in STAT_KEYS})
    model.load_state_dict(ck['state_dict']);return model.eval()
