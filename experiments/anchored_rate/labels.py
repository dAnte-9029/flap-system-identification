"""Strictly published historical anchors and native, time-weighted training beat labels."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path('/home/zn/flap-system-identification')
HOME = Path(__file__).resolve().parents[2]
# Caches are private to this round even when the experiment runner checks out the code.
CACHE = ROOT / '.worktrees/anchored-slow-rate-b/outputs/anchored_rate'
DIAG = ROOT / 'research/rounds/periodic-motion-20261125/diagnostics'
sys.path.insert(0, str(DIAG))
import prepare_reference as ref

def make(part):
    assert part in ('train','validation')
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = {m['flight_id']:m for m in json.loads((ref.DATA/'flight_provenance.json').read_text()) if m['split']==part}
    phase_fn = ref.load_module('anchored_phase',ref.SOURCE/'src/system_identification/data/pahbd_phase.py').prepare_phase
    contract = json.loads((ref.SOURCE/'configs/pahbd/phase_reference.json').read_text())
    win = pd.read_parquet(ref.HISTORY/f'{part}_windows.parquet').reset_index(names='history_idx').set_index('window_id')
    with np.load(ROOT/f'outputs/autopilot_flight_dynamics_20261124/{part}.npz') as old:
        win = win.loc[old['window']].reset_index()
        np.testing.assert_array_equal(win.flight_id,old['flight'])
    cols = ['flight_id','state_measurement_us','state_publication_us','segment_id','valid_core','firmware_verified','review','provenance_review','excluded']
    state = pd.read_parquet(ref.DATA/f'samples_{part}.parquet',columns=cols).groupby('flight_id')
    phase = pd.read_parquet(ref.DATA/f'native_wing_phase_{part}.parquet').groupby('flight_id')
    rates = pd.read_parquet(ref.DATA/f'native_vehicle_angular_velocity_{part}.parquet').groupby('flight_id')
    anchor = np.zeros((len(win),3),np.float32)
    eligible = np.zeros(len(win),bool)
    init = np.load(ref.HISTORY/f'{part}_initial.npy',mmap_mode='r')[win.history_idx,10:13]
    lead, weights, means, ids = [],[],[],[]
    rng = np.random.default_rng(17)
    for fid, windows in win.groupby('flight_id'):
        ph = phase_fn(phase.get_group(fid),meta[fid]['transmission_ratio'],contract)
        ss = state.get_group(fid)
        raw = rates.get_group(fid)
        cyc = ref.make_cycles(ph,meta[fid],ss,raw,False,Counter())
        if not cyc: continue
        st, pub, seg = [ss[k].to_numpy() for k in ['state_measurement_us','state_publication_us','segment_id']]
        assert np.all(np.diff(st)>=0)
        rt = raw.loc[raw.in_reviewed_flight].drop_duplicates('timestamp_sample_us').sort_values('timestamp_sample_us')
        rclock, rpub = rt.timestamp_sample_us.to_numpy(), rt.timestamp_us.to_numpy()
        pt, pav = ph.timestamp_us.to_numpy(), ph.availability_us.to_numpy()
        bad = pav[~ph.phase_usable.to_numpy()]
        resets = np.sort(meta[fid]['reset_times_us'])
        beats = list(cyc.values())
        left = np.array([c['left'] for c in beats])
        right = np.array([c['right'] for c in beats])
        segments = np.array([c['segment'] for c in beats])
        mean = np.array([c['weight']@c['data'] for c in beats])
        ready = []
        for c in beats:
            t = np.rint(c['t']*1e6).astype('int64')
            j = np.searchsorted(st,t).clip(0,len(st)-1)
            k = np.where(st[j]==t,j,(j-1).clip(0))
            ready.append(max(pav[np.searchsorted(pt,c['right'])],rpub[np.searchsorted(rclock,t)].max(),pub[j].max(),pub[k].max()))
        ready = np.array(ready)
        for wi, now in zip(windows.index,windows.decision_us.to_numpy()):
            past = np.arange(left.searchsorted(now-500000), right.searchsorted(now,'right'))
            if not len(past): continue
            si = st.searchsorted(now,side='right')-1
            while si>=0 and pub[si]>now: si-=1
            if si<0: continue
            past = past[(ready[past]<=now)&(segments[past]==seg[si])]
            if not len(past): continue
            k = past[-1]
            if (bad.searchsorted(now,'right')>bad.searchsorted(pav[np.searchsorted(pt,right[k])],'right') or
                resets.searchsorted(now,'right')>resets.searchsorted(left[k],'left')): continue
            eligible[wi]=True
            anchor[wi]=mean[k]
            if part!='train': continue
            future = np.arange(left.searchsorted(now),right.searchsorted(now+2000000,'right'))
            if not len(future): continue
            future = future[segments[future]==segments[k]]
            if not len(future): continue
            f = rng.choice(future)
            c=beats[f]
            lead.append(c['t']-now*1e-6)
            weights.append(c['weight'])
            means.append(mean[f])
            ids.append(wi)
    native = {}
    if ids:
        native = dict(wi=np.array(ids), lengths=np.array(list(map(len,lead))), t=np.concatenate(lead).astype('float32'),
                      weight=np.concatenate(weights), target=np.array(means))
        assert native['t'].min()>0 and native['t'].max()<=2.000002
    np.savez(CACHE/f'{part}.npz',anchor=anchor,ok=eligible,origin=init,window=win.window_id.to_numpy(dtype=str),**native)
    print(part,len(win),int(eligible.sum()),len(ids),flush=True)

if __name__=='__main__':
    make(sys.argv[1])
