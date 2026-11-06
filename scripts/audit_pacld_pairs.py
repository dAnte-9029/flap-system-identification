#!/usr/bin/env python
"""Exact native cycle-pair census without materializing model history per window.

Only encoder phase is interpolated, exactly as NativeWindows does; state targets
are never interpolated. Monotone per-flight count is checked, not assumed across
resets. Every source uses its own native timestamp/mask. Includes reader checks.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.data.pacld import open_windows
from system_identification.data.pahbd import FIELDS


def count_pairs(phase, masks):
    """Same nearest one-cycle successor and half-native-neighbor tolerance."""
    if len(phase) < 2:
        return 0, 0, 0
    if np.any(np.diff(phase) < 0):
        raise ValueError('nonmonotone encoder phase: requires original per-window analysis')
    i = np.arange(len(phase)-1)
    right = np.searchsorted(phase, phase[i]+2*math.pi).clip(i+1, len(phase)-1)
    left = (right-1).clip(i+1, len(phase)-1)
    # Original Python min chooses the earlier index on ties.
    j = np.where(abs(phase[left]-phase[i]-2*math.pi) <= abs(phase[right]-phase[i]-2*math.pi), left, right)
    spacing = np.minimum(np.r_[np.inf, np.diff(phase)], np.r_[np.diff(phase), np.inf])
    match = abs(phase[j]-phase[i]-2*math.pi) <= spacing[j]/2
    valid = (masks[i] & masks[j]).any(axis=1)
    return int((match & valid).sum()), int(match.sum()), len(i)


def source_arrays(dataset, fid, source, encoder):
    tab = pd.read_parquet(dataset.root/f'native_{source}_{dataset.split}.parquet', filters=[('flight_id', '==', fid)])
    mt = tab.timestamp_sample_us.where(tab.timestamp_sample_us > 0, tab.timestamp_us)
    av = tab.get('availability_us', tab.timestamp_us)
    tab = tab[(av >= mt) & tab.in_reviewed_flight].copy()
    tab['mt'] = mt.loc[tab.index]
    tab = tab.sort_values('mt', kind='stable')
    values = tab[FIELDS[source]].to_numpy(float)
    masks = np.isfinite(values)
    if source == 'vehicle_local_position':
        masks &= tab[['xy_valid','xy_valid','z_valid','v_xy_valid','v_xy_valid','v_z_valid']].to_numpy(bool)
    elif source == 'vehicle_attitude':
        masks = np.repeat((np.isfinite(values).all(1) & (np.linalg.norm(values, axis=1) >= .5))[:, None], 3, axis=1)
    ts = tab.mt.to_numpy()
    et = encoder.timestamp_us.to_numpy(); ep = encoder.total_count.to_numpy(dtype=float)
    ep = (ep-dataset.provenance[fid]['phase_origin_count'])*2*math.pi/(4096*dataset.provenance[fid]['transmission_ratio'])
    j = np.searchsorted(et, ts)
    safe = j.clip(1, len(et)-1)
    ok = (j > 0) & (j < len(et)) & (et[safe]-et[safe-1] <= 30000)
    phase = np.interp(ts, et, ep)
    return ts, phase, masks, ok, et[safe-1], et[safe]


def census(windows):
    dataset = windows.dataset
    rows = []
    before = time.perf_counter()
    for flight_num, (fid, group) in enumerate(windows.index.groupby('flight_id', sort=True), 1):
        enc = pd.read_parquet(dataset.root/f'native_encoder_count_{dataset.split}.parquet',
                              filters=[('flight_id', '==', fid)]).sort_values('timestamp_us')
        source_data = {s: source_arrays(dataset, fid, s, enc) for s in FIELDS}
        for w in group.itertuples():
            row = dict(window_id=w.window_id, flight_id=fid, split=dataset.split,
                       horizon_us=int(w.horizon_us), sampling_group=w.sampling_group)
            for source, (ts, p, masks, valid, elo, ehi) in source_data.items():
                lo, hi = np.searchsorted(ts, [w.decision_us, w.last_target_measurement_us], side='right')
                keep = valid[lo:hi] & (elo[lo:hi] >= w.support_start_us) & (ehi[lo:hi] <= w.support_end_us)
                count, matches, candidates = count_pairs(p[lo:hi][keep], masks[lo:hi][keep])
                row[source] = count
                row[source+'_matches'] = matches
                row[source+'_candidates'] = candidates
            rows.append(row)
        if flight_num % 10 == 0:
            print(f'{dataset.split}: {flight_num} flights, {len(rows)} windows, {time.perf_counter()-before:.1f}s', flush=True)
    return pd.DataFrame(rows)


def summarize(frame):
    rows = []
    for (split, horizon, sampling), g in frame.groupby(['split', 'horizon_us', 'sampling_group']):
        for source in FIELDS:
            good = g[g[source] > 0]
            rows.append(dict(split=split, horizon_us=horizon, sampling_group=sampling, source=source,
                             eligible_flights=g.flight_id.nunique(), paired_flights=good.flight_id.nunique(),
                             eligible_windows=len(g), paired_windows=len(good), window_ratio=len(good)/len(g),
                             valid_pairs=int(g[source].sum()), matched_pairs=int(g[source+'_matches'].sum()),
                             candidate_origins=int(g[source+'_candidates'].sum()),
                             valid_per_candidate=float(g[source].sum()/max(g[source+'_candidates'].sum(), 1))))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/pacld/default.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    all_rows = []; checks = []
    for split in ('train', 'validation'):
        windows = open_windows(config, split)
        frame = census(windows)
        # Verify accelerated census against unmodified reader in each horizon/group.
        for _, g in frame.groupby(['horizon_us', 'sampling_group']):
            row = g.iloc[0]
            pos = int(windows.index.index[windows.index.window_id == row.window_id][0])
            w = windows[pos]
            actual = {s: 0 for s in FIELDS}
            for i, j in w.metadata['cycle_pairs']:
                a, b = w.targets[i], w.targets[j]
                weight = a.mask[:12]*b.mask[:12]*a.quality[:12]*b.quality[:12]
                if b.measurement > a.measurement > 0 and bool((weight > 0).any()):
                    actual[a.source] += 1
            expected = {s: int(row[s]) for s in FIELDS}
            if actual != expected:
                raise RuntimeError(f'census disagrees with NativeWindows: {row.window_id}: {actual} != {expected}')
            checks.append(dict(window_id=row.window_id, counts=actual, split=split))
        frame.to_parquet(out/f'windows_{split}.parquet', index=False)
        all_rows.append(frame)
    summary = summarize(pd.concat(all_rows, ignore_index=True))
    summary.to_csv(out/'summary.csv', index=False)
    (out/'reader_checks.json').write_text(json.dumps(checks, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
