"""Full prepared-window integrity and causal-input preflight, without training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from system_identification.data.reference_windows import extract_window

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/analysis/results/reference_dynamics_data_v1'
CACHE=ROOT/'dataset/reference_dynamics_data_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    manifest=json.loads((CACHE/'manifest.json').read_text())
    for p,h in {**manifest['cache_sha256'],**manifest['tables_sha256']}.items():assert sha(ROOT/p)==h,p
    w=pd.read_csv(OUT/'windows.csv');primary=pd.read_csv(OUT/'primary_windows.csv')
    seg=pd.read_csv(OUT/'segments.csv').set_index('segment_id')
    assert w.window_id.is_unique
    assert not set(w[w.partition=='train'].log_id)&set(w[w.partition=='validation'].log_id)
    count=0;normstate=[];normref=[];normact=[]
    examples=[]
    for path,rows in w.groupby('cache'):
        with np.load(ROOT/path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        for r in rows.itertuples():
            s=seg.loc[r.segment_id];i=int(r.origin_index)
            assert s.start_index<=i-26 and i+50<s.stop_index
            assert int(arrays['timestamp_us'][i])==r.origin_timestamp_us
            inputs,labels=extract_window(arrays,i)
            assert inputs['history_state'].shape==(26,14)
            assert inputs['history_actuator'].shape==(26,4)
            assert inputs['future_reference'].shape==(50,4)
            assert labels['rates_b'].shape==(50,3)
            count+=1
        # One real-input mutation test per cache; preserve arrays after check.
        i=int(rows.iloc[0].origin_index);before,_=extract_window(arrays,i)
        altered={k:v.copy() for k,v in arrays.items()}
        for name in ['state_features','phase_rad','quaternion','velocity_n','rates_b']:
            altered[name][i+1:]=123
        altered['actuator_commands'][i:]=-321
        after,_=extract_window(altered,i)
        for k in before:np.testing.assert_array_equal(before[k],after[k])
        tr=primary[(primary.cache==path)&(primary.partition=='train')]
        if len(tr):
            h=np.unique(np.concatenate([np.arange(i-25,i+1) for i in tr.origin_index]))
            rr=np.unique(np.concatenate([np.arange(i-25,i+50) for i in tr.origin_index]))
            normstate.append(arrays['state_features'][h]);normref.append(arrays['references'][rr]);normact.append(arrays['actuator_commands'][h-1])
        selected=primary[primary.cache==path]
        if len(selected) and len(examples)<4:
            r=selected.iloc[0];x,y=extract_window(arrays,int(r.origin_index))
            dest=CACHE/f'preflight_{len(examples)}.npz'
            np.savez_compressed(dest,**{'input_'+k:v for k,v in x.items()},**{'label_'+k:v for k,v in y.items()})
            examples.append(dict(window_id=r.window_id,partition=r.partition,path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
    with np.load(CACHE/'normalization.npz') as stats:
        for name,items in [('state',normstate),('reference',normref),('past_actuator',normact)]:
            a=np.concatenate(items).astype(float);mean=a.mean(0);scale=a.std(0).clip(1e-6)
            if name=='state':mean[9:11]=0;scale[9:11]=1
            np.testing.assert_allclose(stats[name+'_mean'],mean,atol=1e-10)
            np.testing.assert_allclose(stats[name+'_scale'],scale,atol=1e-10)
    refs=np.concatenate(normref)
    ranges=pd.DataFrame(dict(feature=['roll_reference_rad','pitch_reference_rad','yaw_rate_reference','thrust_body_x'],
        minimum=refs.min(0),q01=np.quantile(refs,.01,axis=0),q99=np.quantile(refs,.99,axis=0),maximum=refs.max(0)))
    ranges.to_csv(OUT/'primary_train_reference_ranges.csv',index=False)
    w.groupby(['partition','mode','kind','active_reference']).size().rename('windows').reset_index().to_csv(OUT/'window_summary.csv',index=False)
    primary.groupby(['partition','mode','kind','active_reference']).size().rename('windows').reset_index().to_csv(OUT/'primary_summary.csv',index=False)
    w.groupby(['group','partition']).agg(windows=('window_id','size'),flights=('log_id','nunique')).reset_index().to_csv(OUT/'group_summary.csv',index=False)
    receipt=dict(status='passed',windows_checked=count,log_caches_checked=w.cache.nunique(),split_overlap=False,
        train_only_normalization_recomputed=True,future_measurement_poisoning_passed=True,examples=examples,
        manifest_sha256=sha(CACHE/'manifest.json'),loader_sha256=sha(ROOT/'src/system_identification/data/reference_windows.py'),
        preflight_script_sha256=sha(__file__),model_training_started=False,sealed_test_accessed=False)
    (OUT/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':run()
