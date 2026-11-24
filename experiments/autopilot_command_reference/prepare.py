"""Cache fixed-time causal input integrals and native endpoint observations on the existing flight split."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path('/home/zn/flap-system-identification')
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'outputs/autopilot_command_reference_20261124'
HISTORY = ROOT / 'outputs/history_slow_fast_20261119_cache'
TAPES = ROOT / 'outputs/controlled_slow_fast_20261119_tapes'
HORIZONS = (.5, 1., 2.)
TAUS = (.2, 1.)


def digest(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pool_commands(tape, times):
    """Integrate dz/dt=(phi(u)-z)/tau exactly under piecewise-constant u up to each query."""
    commands = tape[..., :4].reshape(len(tape), -1, 4)
    dt = tape[..., 4].reshape(len(tape), -1).astype('float64') * 1e-6
    ends = dt.cumsum(1)
    starts = ends - dt
    phi = np.concatenate((commands, np.tanh(2 * commands)), axis=2)
    summary = []
    for time in times.T:
        age = np.maximum(time[:, None] - ends, 0.)
        elapsed = np.clip(time[:, None] - starts, 0., dt)
        pieces = []
        for tau in TAUS:
            weight = np.exp(-age / tau) * (-np.expm1(-elapsed / tau))
            pieces.append(np.einsum('bs,bsc->bc', weight, phi))
        summary.append(np.concatenate(pieces, axis=1))
    return np.stack(summary, axis=1).astype('float32')


def prepare(part):
    ids = pd.read_parquet(HISTORY / f'{part}_windows.parquet')
    selected = pd.read_parquet(TAPES / f'{part}_ids.parquet').window_id
    assert selected.is_unique and ids.window_id.is_unique
    take = pd.Index(ids.window_id).get_indexer(selected)
    assert (take >= 0).all()
    ids = ids.iloc[take].reset_index(drop=True)
    history,initial,t,y,valid = [np.load(HISTORY / f'{part}_{key}.npy', mmap_mode='r') for key in ('history','initial','t','y','valid')]
    commands = np.load(TAPES / f'{part}_commands.npy', mmap_mode='r')
    assert len(commands) == len(ids)
    accum = {}
    for offset in range(0,len(ids),384):
        group = ids.iloc[offset:offset+384]
        ix = take[offset:offset+len(group)]
        hist, init = history[ix], initial[ix]
        tape = commands[offset:offset+len(ix)]
        np.testing.assert_allclose(tape[:,0,0,:4],hist[:,-1,9:13],atol=2e-7,rtol=0)
        tolerance = np.where(group.sampling_group=='main_50hz',.035,.105)
        r = Rotation.from_quat(init[:,[7,8,9,6]])
        time, target, mask = [], [], []
        for horizon in HORIZONS:
            for ch in (0,2):
                distance = np.where(valid[ix,:,ch],np.abs(t[ix,:,ch]-horizon),np.inf)
                nearest = np.argmin(distance,axis=1)
                ok = distance[np.arange(len(ix)),nearest] <= tolerance
                qtime = np.where(ok,t[ix,nearest,ch],horizon)
                value = (r.apply(y[ix,nearest,3:6]-init[:,3:6],inverse=True) if ch==0
                         else y[ix,nearest,10:13]-init[:,10:13])
                time.append(qtime)
                target.append(np.where(ok[:,None],value,0))
                mask.append(ok)
        times = np.stack(time,axis=1)
        assert np.max(times) <= 2.1 + 1e-5
        row = dict(hist=hist[:,[0,5,10,15,20,25]].reshape(len(ix),-1),
                   u0=hist[:,-1,9:13],time=times,pooled=pool_commands(tape,times),
                   y=np.stack(target,axis=1),mask=np.stack(mask,axis=1))
        for k,v in row.items(): accum.setdefault(k,[]).append(v.astype('float32'))
    values={k:np.concatenate(v) for k,v in accum.items()}
    assert all(np.isfinite(v).all() for v in values.values())
    dest=DATA / f'{part}.npz'
    np.savez(dest,**values,flight=ids.flight_id.to_numpy(dtype=str),window=ids.window_id.to_numpy(dtype=str))
    return dict(file=str(dest),sha256=digest(dest),windows=len(ids),flights=ids.flight_id.nunique(),
                native_endpoint_counts=values['mask'].sum(0).astype(int).tolist())


if __name__=='__main__':
    reg=yaml.safe_load((HERE.parents[1]/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
    version=reg['default_dataset_id']
    assert version=='velocity_rate_multistep_v1'
    record=reg['datasets'][version]
    assert digest(ROOT/record['manifest_path'])==record['manifest_sha256']
    DATA.mkdir(exist_ok=False)
    split={p:prepare(p) for p in ('train','validation')}
    with np.load(split['train']['file']) as a, np.load(split['validation']['file']) as b:
        assert set(a['flight']).isdisjoint(b['flight'])
    (HERE/'data.json').write_text(json.dumps(dict(dataset=version,manifest_sha256=record['manifest_sha256'],split=split),indent=2)+'\n')
