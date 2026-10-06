"""Repeat final data checks without fitting a model or reading protected logs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from system_identification.data.reference_windows import extract_window

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    m=json.loads((OUT/'manifest.json').read_text());w=pd.read_csv(ROOT/m['windows_path'])
    assert sha(ROOT/m['windows_path'])==m['windows_sha256']
    for p,h in m['cache_sha256'].items():assert sha(ROOT/p)==h
    assert not set(w[w.partition=='train'].log_id)&set(w[w.partition=='validation'].log_id)
    stats_rows={'state':[],'reference':[],'past_actuator':[]};count=0;caches=0
    for path,rows in w.groupby('cache'):
        with np.load(ROOT/path) as z:a={k:z[k] for k in z.files}
        for r in rows.itertuples():
            x,_=extract_window(a,int(r.origin_index));assert x['future_reference'].shape==(50,5);count+=1
        i=int(rows.iloc[0].origin_index);original,_=extract_window(a,i)
        mutated={k:v.copy() for k,v in a.items()}
        for k in ['state_features','phase_rad','quaternion','velocity_n','rates_b']:mutated[k][i+1:]=123
        mutated['actuator_commands'][i:]=-123
        changed,_=extract_window(mutated,i)
        for k in original:np.testing.assert_array_equal(original[k],changed[k])
        caches+=1
        if rows.partition.iloc[0]=='train':
            h=np.unique(np.concatenate([np.arange(i-25,i+1) for i in rows.origin_index]))
            ref=np.unique(np.concatenate([np.arange(i-25,i+50) for i in rows.origin_index]))
            stats_rows['state'].append(a['state_features'][h]);stats_rows['reference'].append(a['references'][ref]);stats_rows['past_actuator'].append(a['actuator_commands'][h-1])
    with np.load(ROOT/m['normalization_path']) as norm:
        for name,items in stats_rows.items():
            a=np.concatenate(items).astype(float);mean=a.mean(0);scale=a.std(0).clip(1e-6)
            if name=='state':mean[9:11]=0;scale[9:11]=1
            np.testing.assert_allclose(norm[name+'_mean'],mean);np.testing.assert_allclose(norm[name+'_scale'],scale)
    receipt=dict(status='passed',windows_checked=count,caches_checked=caches,normalization_recomputed_train_only=True,
        future_measurement_poisoning_passed=True,split_overlap=False,sealed_test_accessed=False,
        loader_sha256=sha(ROOT/'src/system_identification/data/reference_windows.py'),verifier_sha256=sha(__file__),
        manifest_sha256=sha(OUT/'manifest.json'),training_started=False)
    (OUT/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':run()
