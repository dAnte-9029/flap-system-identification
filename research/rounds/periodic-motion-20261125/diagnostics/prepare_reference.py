"""Form independent leave-one-wingbeat-out references at native development observations."""
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
DATA = PROJECT / 'dataset/velocity_rate_multistep_v1'
HISTORY = PROJECT / 'outputs/history_slow_fast_20261119_cache'
SOURCE = PROJECT / '.worktrees/slow-fast-separation-validation-ab-v1'
OLD = PROJECT / 'outputs/autopilot_flight_dynamics_20261124/validation.npz'
TAU = 2 * np.pi
N_WINDOWS = 8


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cycles(phase, meta, samples, native, velocity, counts):
    time = phase.timestamp_us.to_numpy(dtype=float)
    angle = phase.phase_rad.to_numpy()
    usable = phase.phase_usable.to_numpy()
    cross = np.flatnonzero((np.diff(angle) < -np.pi) & usable[1:] & usable[:-1]) + 1
    boundaries = time[cross - 1] + (time[cross] - time[cross - 1]) * (TAU - angle[cross - 1]) / (angle[cross] + TAU - angle[cross - 1])
    raw = native.loc[native.in_reviewed_flight].drop_duplicates('timestamp_sample_us').sort_values('timestamp_sample_us')
    clock = raw.timestamp_sample_us.to_numpy()
    values = raw[['vx', 'vy', 'vz'] if velocity else [f'xyz[{i}]' for i in range(3)]].to_numpy(float)
    state_ok = np.isfinite(values).all(1)
    if velocity:
        state_ok &= raw.v_xy_valid & raw.v_z_valid
    state_ok = np.asarray(state_ok)
    registered_t = samples.state_measurement_us.to_numpy()
    segments = samples.segment_id.to_numpy()
    reg_ok = (samples.valid_core & samples.firmware_verified & ~samples.review &
              ~samples.provenance_review & ~samples.excluded & (segments >= 0)).to_numpy()
    j = np.searchsorted(registered_t, clock).clip(0, len(registered_t) - 1)
    k = (j - 1).clip(0)
    exact = registered_t[j] == clock
    k[exact] = j[exact]
    state_ok &= (clock >= registered_t[k]) & (clock <= registered_t[j]) & reg_ok[k] & reg_ok[j] & (segments[k] == segments[j])
    resets = np.asarray(meta['reset_times_us'])
    cycles = {}
    for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        counts['crossing_pair'] += 1
        a, b = cross[i] - 1, cross[i + 1] + 1
        if not np.all(usable[a:b]):
            counts['phase_qc'] += 1
            continue
        l, r = np.searchsorted(clock, [left, right])
        t = clock[l:r]
        if len(t) < 8 or np.max(np.diff(np.r_[left, t, right])) > 40000:
            counts['sampling'] += 1
            continue
        if not state_ok[l:r].all() or np.ptp(segments[j[l:r]]) or ((resets >= left) & (resets <= right)).any():
            counts['state_segment'] += 1
            continue
        p = np.interp(t, time[a:b], np.unwrap(angle[a:b])) - TAU
        if np.any(np.diff(p) <= 0) or p[0] < -1e-6 or p[-1] > TAU + 1e-6 or np.max(np.diff(np.r_[p, p[0] + TAU])) > np.pi / 2:
            counts['phase_coverage'] += 1
            continue
        w = np.diff(np.r_[left, (t[1:] + t[:-1]) / 2, right]) / (right - left)
        cycles[i] = dict(t=t * 1e-6, phase=p, weight=w, data=values[l:r], left=left, right=right,
                         segment=segments[j[l]], rate_hz=1e6 / np.median(np.diff(t)))
    counts['usable_beat'] += len(cycles)
    return cycles


def design(cycle, center, period):
    x = (cycle['t'] - center) / period
    co, si = np.cos(cycle['phase']), np.sin(cycle['phase'])
    return np.column_stack([np.ones_like(x), x, co, si, x * co, x * si])


def solve(target, others):
    center = (target['left'] + target['right']) / 2e6
    period = (target['right'] - target['left']) / 1e6
    xx = np.vstack([design(c, center, period) for c in others])
    w = np.sqrt(np.concatenate([c['weight'] for c in others]))[:, None]
    if np.linalg.cond(xx * w) > 1e3:
        return None
    z = design(target, center, period)
    yy = np.vstack([c['data'] for c in others])
    coef = np.linalg.lstsq(xx * w, yy * w, rcond=None)[0]
    only = np.linalg.lstsq(xx[:, :2] * w, yy * w, rcond=None)[0]
    return z[:, :2] @ coef[:2], z[:, 2:] @ coef[2:], z[:, :2] @ only


def synthetic(cycle, neighbors):
    center = (cycle['left'] + cycle['right']) / 2e6
    period = (cycle['right'] - cycle['left']) / 1e6
    def injected(c):
        u = c['t'] - center
        carrier = np.sin(c['phase'] + .37)
        step = np.clip(u / .5 + .5, 0, 1)
        smooth = step * step * (3 - 2 * step)
        linear = .6 + .19 * u
        linearamp = (1. + .3 * u) * carrier
        varamp = (1. + .5 * np.sin(2 * np.pi * u / (3.5 * period) + .4)) * carrier
        return np.stack([linear + linearamp, smooth, varamp], axis=1), linear, linearamp, smooth, varamp
    slow, fast, _ = solve(cycle, [{**c, 'data': injected(c)[0]} for c in neighbors])
    _, line, linearamp, step, varamp = injected(cycle)
    w = cycle['weight']
    def power(x):
        return float(w @ (x * x))
    return dict(exact_slow=power(slow[:, 0] - line), exact_fast=power(fast[:, 0] - linearamp),
                step_to_fast=power(fast[:, 1]), step_error=power(slow[:, 1] - step), step=power(step),
                oscillation_to_slow=power(slow[:, 2]), oscillation_error=power(fast[:, 2] - varamp), oscillation=power(varamp))


def main():
    contract = json.loads((SOURCE / 'configs/pahbd/phase_reference.json').read_text())
    prepare_phase = load_module('diagnostic_phase_qc', SOURCE / 'src/system_identification/data/pahbd_phase.py').prepare_phase
    history = pd.read_parquet(HISTORY / 'validation_windows.parquet').reset_index(names='history_idx').set_index('window_id')
    with np.load(OLD) as old:
        wins = history.loc[old['window']].reset_index()
        np.testing.assert_array_equal(wins.flight_id, old['flight'])
    assert len(wins) == 44524 and wins.flight_id.nunique() == 39 and wins.window_id.is_unique
    wins['cache_idx'] = np.arange(len(wins))
    init = np.load(HISTORY / 'validation_initial.npy', mmap_mode='r')
    cols = ['flight_id', 'state_measurement_us', 'segment_id', 'valid_core', 'firmware_verified', 'review', 'provenance_review', 'excluded']
    samples = pd.read_parquet(DATA / 'samples_validation.parquet', columns=cols).groupby('flight_id')
    meta = {m['flight_id']: m for m in json.loads((DATA / 'flight_provenance.json').read_text()) if m['split'] == 'validation'}
    phase_groups = pd.read_parquet(DATA / 'native_wing_phase_validation.parquet').groupby('flight_id')
    names = ['vehicle_local_position', 'vehicle_angular_velocity']
    streams = [pd.read_parquet(DATA / f'native_{n}_validation.parquet').groupby('flight_id') for n in names]
    output = defaultdict(list)
    syn = []
    counts = defaultdict(Counter)
    coverage = []
    block = 0
    for fid, windows in wins.groupby('flight_id'):
        windows = windows.sort_values('decision_us')
        picks = windows.iloc[np.unique(np.rint(np.linspace(0, len(windows) - 1, N_WINDOWS)).astype(int))]
        group = windows.sampling_group.iloc[0]
        phase = prepare_phase(phase_groups.get_group(fid), meta[fid]['transmission_ratio'], contract)
        for ch, stream in enumerate(streams):
            count = counts[f'{group}/{names[ch]}']
            cycles = make_cycles(phase, meta[fid], samples.get_group(fid), stream.get_group(fid), ch == 0, count)
            usable = sorted(i for i, c in cycles.items() if all(j in cycles and cycles[j]['segment'] == c['segment'] for j in [i - 2, i - 1, i + 1, i + 2]))
            starts = np.array([cycles[i]['left'] for i in usable])
            nrows, nw, nb = 0, 0, 0
            for row in picks.itertuples():
                lo, hi = starts.searchsorted([row.decision_us, row.decision_us + 2000000])
                selected = [i for i in usable[lo:hi] if cycles[i]['right'] <= row.decision_us + 2000000]
                any_obs = False
                initial = np.asarray(init[row.history_idx], float)
                rot = Rotation.from_quat(initial[[7, 8, 9, 6]])
                origin = rot.apply(initial[3:6], inverse=True) if ch == 0 else initial[10:13]
                for i in selected:
                    c = cycles[i]
                    adj = [cycles[i + a] for a in [-2, -1, 1, 2]]
                    fit = solve(c, adj)
                    if fit is None:
                        count['ill_conditioned'] += 1
                        continue
                    slow, fast, linear = fit
                    data = c['data']
                    if ch == 0:
                        slow, fast, linear, data = [rot.apply(a, inverse=True) for a in (slow, fast, linear, data)]
                    if not any_obs:
                        syn.append(synthetic(c, adj))
                    any_obs = True
                    length = len(data)
                    for key, v in dict(data=data, slow=slow, fast=fast, trend_only=linear,
                                       time=c['t'] - row.decision_us * 1e-6, phase=c['phase'], weight=c['weight'],
                                       origin=np.broadcast_to(origin, (length, 3))).items():
                        output[key].append(v)
                    for key, value in dict(flight=fid, window_idx=row.cache_idx, ch=ch, block=block).items():
                        output[key].append(np.full(length, value))
                    block += 1
                    nrows += length
                    nb += 1
                nw += int(any_obs)
            coverage.append(dict(flight=fid,group=group, ch=ch,selected_windows=len(picks),usable_windows=nw,beat_window_pairs=nb,
                                 native_points=nrows,usable_time_beats=len(usable),median_samples_per_beat=float(np.median([len(c['t']) for c in cycles.values()])) if cycles else None))
    save = {k: np.concatenate(v) for k, v in output.items()}
    assert len(save['time']) and sum(r['native_points'] for r in coverage) == len(save['time'])
    assert np.min(save['time']) > 0 and np.max(save['time']) <= 2.000001
    # Flight identifiers are stored once per flight.
    ids, ix = np.unique(save.pop('flight'), return_inverse=True)
    np.savez_compressed(HERE / 'reference.npz', **save, flight_index=ix.astype(np.int16), flight_ids=ids)
    synth = {k: np.mean([r[k] for r in syn]) for k in syn[0]}
    summary = dict(number_synthetic_evaluations=len(syn),exact_trend_rmse=np.sqrt(synth['exact_slow']),
                   exact_periodic_rmse=np.sqrt(synth['exact_fast']),step_to_periodic_rms_ratio=np.sqrt(synth['step_to_fast']/synth['step']),
                   step_recovery_relative_rmse=np.sqrt(synth['step_error']/synth['step']),periodic_to_trend_rms_ratio=np.sqrt(synth['oscillation_to_slow']/synth['oscillation']),
                   varying_envelope_relative_rmse=np.sqrt(synth['oscillation_error']/synth['oscillation']))
    (HERE / 'coverage.json').write_text(json.dumps(dict(qc={k:dict(v) for k,v in counts.items()}, coverage=coverage,
                      synthetic=summary), indent=2) + '\n')
    print(pd.DataFrame(coverage).groupby(['group','ch']).agg({'usable_windows':'sum','beat_window_pairs':'sum','native_points':'sum'}))
    print(summary)


if __name__ == '__main__':
    main()
