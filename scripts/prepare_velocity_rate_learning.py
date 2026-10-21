"""Read-only registry verification and vectorized train/validation packing."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from system_identification.data.velocity_rate_dataset import STATE, CONTROL, PHASE, TARGET, file_hash, write_json
from system_identification.models.control_events import (
    CANONICAL_CONTROL_EVENTS, LEGACY_CONTROL_EVENTS, canonicalize_control_events_numpy)

EXP = ROOT/'experiments/velocity_rate_gru64_round1'
CACHE = ROOT/'artifacts/velocity_rate_gru64_round1'


def resolve(config):
    registry = yaml.safe_load((ROOT/config['registry']).read_text())
    assert registry['default_dataset_id'] == config['dataset_id']
    entry = registry['datasets'][config['dataset_id']]
    manifest_path = ROOT/entry['manifest_path']
    assert file_hash(manifest_path) == entry['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    root = ROOT/entry['repository_relative_root']
    checked = {}
    # Do not even hash/open test artifacts. Only shared metadata and authorized partitions.
    for name, expected in manifest['artifact_sha256'].items():
        if name.endswith('_test.parquet'):
            continue
        if name.endswith('.parquet') and not name.endswith(('_train.parquet', '_validation.parquet')):
            continue
        assert file_hash(root/name) == expected, name
        checked[name] = expected
    split = json.loads((root/'split_manifest.json').read_text())
    assert split['dataset_id'] == config['dataset_id']
    return root, dict(registry=entry, manifest_sha256=file_hash(manifest_path),
                      checked_artifact_sha256=checked, test_artifacts_opened=False,
                      phase_contract=manifest['config']['phase'], split_policy=split['policy'])


def interval_events(motor, servo, starts, ends, maximum_age=160000,
                    *, event_policy=CANONICAL_CONTROL_EVENTS):
    """Half-open [start,end) ZOH, canonicalized without losing real jumps."""
    if event_policy not in (CANONICAL_CONTROL_EVENTS, LEGACY_CONTROL_EVENTS):
        raise ValueError(f"Unknown control event policy: {event_policy}")
    ts = np.unique(np.r_[motor.timestamp_us, servo.timestamp_us]).astype(np.int64)
    left = np.searchsorted(ts, starts, side='right')
    right = np.searchsorted(ts, ends, side='left')
    length = right-left+1
    width = int(length.max())
    j = np.arange(width)[None, :]
    times = ts[np.minimum(left[:, None]+np.maximum(j-1, 0), len(ts)-1)]
    times[:, 0] = starts
    times = np.where(j < length[:, None], times, ends[:, None])
    next_times = np.minimum(np.column_stack((times[:, 1:], ends)), ends[:, None])
    duration = (next_times-times)*1e-6
    result = []
    for source, fields in [(motor, ['control[0]']), (servo, ['control[0]','control[1]','control[2]'])]:
        native_t = source.timestamp_us.to_numpy()
        idx = np.searchsorted(native_t, times, side='right')-1
        used = j < length[:, None]
        assert (idx[used] >= 0).all()
        age = times-native_t[np.maximum(idx, 0)]
        assert ((age[used] >= 0)&(age[used] <= maximum_age)).all()
        values = source[fields].to_numpy()[np.maximum(idx, 0)]
        assert np.isfinite(values[used]).all()
        result.append(values)
    events = np.concatenate((*result, duration[..., None]), -1).astype('float32')
    events[j >= length[:, None]] = 0
    assert (duration >= 0).all()
    np.testing.assert_allclose(duration.sum(1), (ends-starts)*1e-6, atol=1e-10)
    if event_policy == CANONICAL_CONTROL_EVENTS:
        events, length = canonicalize_control_events_numpy(events, length, (ends-starts)*1e-6)
    return events, length


def pack(root, split, config):
    assert split in ('train', 'validation'), 'test is sealed'
    w = pd.read_parquet(root/f'windows_{split}.parquet')
    w = w.loc[(w.sampling_group==config['sampling_group']) & (w.history_us==config['history_us'])
              & (w.horizon_us==config['horizon_us'])].reset_index(drop=True)
    assert len(w)==config['expected_windows'][split]
    assert w.flight_id.nunique()==config['expected_flights'][split]
    samples = pd.read_parquet(root/f'samples_{split}.parquet')
    natives = {kind: dict(tuple(pd.read_parquet(root/f'native_actuator_{kind}_{split}.parquet').groupby('flight_id', sort=False))) for kind in ('motors','servos')}
    groups = dict(tuple(samples.groupby('flight_id', sort=False)))
    n, H, K = len(w), int(w.history_count.max()), int(w.future_count.max())
    data = dict(history=np.zeros((n,H,20),np.float32), history_mask=np.zeros((n,H,20),bool),
                history_dt=np.zeros((n,H),np.float32), history_length=w.history_count.to_numpy(np.int64),
                initial=np.zeros((n,13),np.float32), target=np.zeros((n,K,13),np.float32),
                dt=np.zeros((n,K),np.float32), future_mask=np.zeros((n,K),bool),
                event_length=np.ones((n,K),np.int64))
    event_parts=[]
    control_change=np.zeros(n)
    for fid, windows in w.groupby('flight_id', sort=False):
        ids=windows.index.to_numpy()
        s=groups[fid].reset_index(drop=True)
        a,o,e=(windows[c].to_numpy(int) for c in ('start_sample','origin_sample','end_sample'))
        ih=np.minimum(a[:,None]+np.arange(H), o[:,None])
        fi=np.minimum(o[:,None]+np.arange(1,K+1),e[:,None])
        hm=np.arange(H)[None,:]<windows.history_count.to_numpy()[:,None]
        fm=np.arange(K)[None,:]<windows.future_count.to_numpy()[:,None]
        assert (s.segment_id.to_numpy()[a]==s.segment_id.to_numpy()[e]).all()
        assert s.valid_core.to_numpy()[ih].all() and s.valid_core.to_numpy()[fi].all()
        values=s[STATE+CONTROL+PHASE].to_numpy(float)
        mask=np.isfinite(values)
        mask[:,13:17] &= s.valid_control.to_numpy()[:,None]
        mask[:,17:19] &= s.valid_encoder.to_numpy()[:,None]
        mask[:,19] &= s.valid_frequency.to_numpy()
        data['history'][ids]=np.where(mask,values,0)[ih]
        data['history_mask'][ids]=mask[ih]&hm[...,None]
        t=s.state_measurement_us.to_numpy(np.int64)
        data['history_dt'][ids,1:]=np.diff(t[ih],axis=1)*1e-6
        data['initial'][ids]=values[o,:13]
        # Translation invariance: position is exclusively an integration origin.
        data['history'][ids,:,:3] -= values[o,:3][:,None,:]
        data['initial'][ids,:3]=0
        data['target'][ids]=s[TARGET].to_numpy()[fi]
        data['target'][ids,:,:3] -= values[o,:3][:,None,:]
        previous=np.column_stack((o,fi[:,:-1]))
        data['dt'][ids]=(t[fi]-t[previous])*1e-6
        data['future_mask'][ids]=fm
        transitions=np.unique(previous[fm])
        events,lengths=interval_events(natives['motors'][fid],natives['servos'][fid],t[transitions],t[transitions+1])
        lookup=np.searchsorted(transitions,previous).clip(0,len(transitions)-1)
        ev=events[lookup].copy()
        ev[~fm]=0
        le=np.where(fm,lengths[lookup],1)
        data['event_length'][ids]=le
        event_parts.append((ids,ev))
        # Condition grouping includes history and the authorized future command tape; never a model feature.
        for local,global_id in enumerate(ids):
            commands=np.concatenate((values[ih[local,hm[local]],13:17],
                np.concatenate([ev[local,k,:le[local,k],:4] for k in np.flatnonzero(fm[local])])),axis=0)
            control_change[global_id]=np.linalg.norm(np.ptp(commands,axis=0))
    max_events=max(ev.shape[2] for _,ev in event_parts)
    data['events']=np.zeros((n,K,max_events,5),np.float32)
    for ids,ev in event_parts:
        data['events'][ids,:,:ev.shape[2]]=ev
    w['control_change']=control_change
    assert np.isfinite(data['history']).all() and np.isfinite(data['target']).all()
    for name in ('history','target','initial'):
        assert np.isfinite(data[name]).all()
    w.to_parquet(EXP/f'{split}_windows.parquet', index=False)
    np.savez(CACHE/f'{split}.npz',**data)
    return dict(control_event_policy=CANONICAL_CONTROL_EVENTS,
                windows=n,flights=w.flight_id.nunique(),history_counts=w.history_count.value_counts().to_dict(),
                future_counts=w.future_count.value_counts().to_dict(),maximum_events_per_step=max_events,
                dt_quantiles=np.quantile(data['dt'][data['future_mask']],[0,.01,.5,.99,1]).tolist(),
                cache_bytes=sum(v.nbytes for v in data.values()),
                cache_sha256=file_hash(CACHE/f'{split}.npz'))


def main():
    start=time.time()
    config=json.loads((EXP/'config.json').read_text())
    CACHE.mkdir(parents=True,exist_ok=True)
    root, audit=resolve(config)
    audit['partitions']={s:pack(root,s,config) for s in ('train','validation')}
    a=pd.read_parquet(EXP/'train_windows.parquet'); b=pd.read_parquet(EXP/'validation_windows.parquet')
    assert set(a.flight_id).isdisjoint(b.flight_id)
    audit['packing_seconds']=time.time()-start
    write_json(EXP/'data_audit.json',audit)
    print(json.dumps(audit['partitions'],indent=2),flush=True)

if __name__=='__main__': main()
