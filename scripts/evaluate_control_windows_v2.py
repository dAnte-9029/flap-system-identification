"""Common-flight evaluation of frozen A/B/C/D final checkpoints; no fitting."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from train_control_windows_v2 import DATA, ART, ROOT, load, write
from system_identification.models.reference_gru import ReferenceGRU

OUT = DATA / 'evaluation'
HORIZONS = [5, 10, 25, 50]
METRICS = ['p', 'q', 'r', 'velocity', 'attitude']


def error_rows(prediction, truth, meta, arm, seed):
    rates = np.square(prediction['rates_b'] - truth['rates_b'])
    vel = np.square(prediction['velocity_n'] - truth['velocity_n']).mean(-1)
    # Angular distance is sign-invariant and reported in radians.
    dot = np.abs((prediction['quaternion'] * truth['quaternion']).sum(-1)).clip(0, 1)
    angle = np.square(2 * np.arccos(dot))
    result = []
    for step in HORIZONS:
        frame = meta.copy()
        frame['arm'] = arm
        frame['seed'] = seed
        frame['horizon_s'] = step * .02
        for axis, name in enumerate(['p','q','r']):
            frame[name] = rates[:, step-1, axis]
        frame['velocity'] = vel[:, step-1]
        frame['attitude'] = angle[:, step-1]
        result.append(frame)
    return pd.concat(result, ignore_index=True)


def summarize(errors):
    frames = []
    selectors = dict(all=np.ones(len(errors), dtype=bool),
        roll_active=errors.roll_range_rad >= np.deg2rad(1),
        pitch_active=errors.pitch_range_rad >= np.deg2rad(1),
        quiet=errors.quality == 'Q2', hard=errors.quality == 'Q3',
        mission=errors['mode'] == 3, stabilized=errors['mode'] == 15)
    for subset, mask in selectors.items():
        grouped = errors.loc[mask].groupby(['arm','seed','log_id','horizon_s'])
        result = np.sqrt(grouped[METRICS].mean()).reset_index()
        result['windows'] = grouped.size().to_numpy()
        result['subset'] = subset
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def primary_cube(per_flight, flights):
    """Each flight remains the resampling unit across axes, horizons and seeds."""
    cube = np.full((4, 3, len(flights), 2, 3), np.nan)
    for ai, arm in enumerate('ABCD'):
        for si, seed in enumerate([17,23,42]):
            for fi, flight in enumerate(flights):
                for axis, (subset, metric) in enumerate([('roll_active','p'),('pitch_active','q')]):
                    for hi, horizon in enumerate([.1,.2,.5]):
                        row = per_flight[(per_flight.arm == arm) & (per_flight.seed == seed)
                            & (per_flight.log_id == flight) & (per_flight.subset == subset)
                            & np.isclose(per_flight.horizon_s, horizon)]
                        if len(row):
                            cube[ai,si,fi,axis,hi] = row[metric].iloc[0]
    return cube


def primary_mean(cube):
    # Equal axes/horizons after equal weight of available flights; same support for all arms.
    return np.nanmean(np.nanmean(cube, axis=2), axis=(2,3))


def main():
    status = json.loads((DATA/'training/status.json').read_text())
    assert status['status'] == 'complete' and len(status['completed']) == 12
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    x, y, stats = load()
    meta = pd.read_csv(DATA/'evaluation_windows.csv')
    indices = meta.array_index.to_numpy()
    truth = {key: value[indices].numpy() for key, value in y.items()}
    errors = []
    waveform = []
    for run in status['completed']:
        arm, seed = run['arm'], run['seed']
        destination = ART/f'{arm}_{seed}'
        p = destination/'validation_predictions.npz'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == run['predictions_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['array_index'], indices)
            pred = {key: z[key] for key in truth}
        errors.append(error_rows(pred, truth, meta, arm, seed))
        initial = x['initial_rates_b'][indices].numpy()[:,None,:]
        real_delta = truth['rates_b'] - initial
        pred_delta = pred['rates_b'] - initial
        # Supplemental 1s waveform diagnostics, NOT G2 full-event/causal-effect evidence.
        for axis, name in enumerate(['p','q','r']):
            frame = meta[['window_id','log_id','quality','mode']].copy()
            frame['arm'], frame['seed'], frame['axis'] = arm, seed, name
            frame['increment_mae'] = np.abs(pred_delta[:,:,axis]-real_delta[:,:,axis]).mean(1)
            frame['absolute_peak_amplitude_error'] = np.abs(np.max(np.abs(pred_delta[:,:,axis]),1)-np.max(np.abs(real_delta[:,:,axis]),1))
            frame['absolute_peak_time_difference_s'] = .02*np.abs(np.argmax(np.abs(pred_delta[:,:,axis]),1)-np.argmax(np.abs(real_delta[:,:,axis]),1))
            waveform.append(frame)
    # Reference-independent persistence baseline: hold v and omega, integrate quaternion.
    hold = ReferenceGRU(stats).eval()
    held = {key: [] for key in truth}
    with torch.no_grad():
        for start in range(0,len(indices),64):
            ix = indices[start:start+64]
            pred = hold(**{key:value[ix] for key,value in x.items()})
            for key in held:
                held[key].append(pred[key].numpy())
    errors.append(error_rows({key:np.concatenate(value) for key,value in held.items()}, truth, meta, 'hold', 0))
    per_window = pd.concat(errors, ignore_index=True)
    per_window.to_csv(OUT/'per_window_squared_errors.csv', index=False)
    pd.concat(waveform, ignore_index=True).to_csv(OUT/'one_second_waveform_diagnostics.csv', index=False)
    per_flight = summarize(per_window)
    per_flight.to_csv(OUT/'per_flight_rmse.csv', index=False)
    aggregate = per_flight.groupby(['arm','seed','subset','horizon_s'])[METRICS].mean().reset_index()
    aggregate.to_csv(OUT/'flight_equal_rmse.csv', index=False)
    worst = per_flight[per_flight.subset == 'all'].groupby(['arm','seed','horizon_s'])[METRICS].max().reset_index()
    worst.to_csv(OUT/'worst_flight_rmse.csv', index=False)
    flights = sorted(meta.log_id.unique())
    cube = primary_cube(per_flight, flights)
    scores = primary_mean(cube)
    rng = np.random.default_rng(1309)
    draws = rng.integers(0,len(flights),(2000,len(flights)))
    boot = np.stack([primary_mean(cube[:,:,draw,:,:]).mean(1) for draw in draws])
    mean = scores.mean(1)
    decisions = {}
    for ci, candidate in [(2,'C'),(3,'D')]:
        comparisons = {}
        for bi, baseline in [(0,'A'),(1,'B')]:
            gain = 1 - mean[ci] / mean[bi]
            interval = np.quantile(1-boot[:,ci]/boot[:,bi],[.025,.975])
            comparisons[baseline] = dict(improvement_fraction=float(gain),
                flight_bootstrap_95ci=interval.tolist(),
                paired_seed_improvements=(1-scores[ci]/scores[bi]).tolist())
        same_direction = int(((scores[ci]<scores[0]) & (scores[ci]<scores[1])).sum())
        safeguards = {}
        for subset in ['all','quiet','hard']:
            short = aggregate[(aggregate.subset == subset) & (aggregate.horizon_s < 1)]
            means = short.groupby('arm')[METRICS].mean()
            ratios = (means.loc[candidate]/means.loc['A'] - 1)
            limit = .03 if subset == 'all' else .05
            safeguards[subset] = dict(relative_degradation=ratios.to_dict(), maximum_allowed=limit,
                passed=bool((ratios<=limit).all()),
                flights=int(per_flight[per_flight.subset==subset].log_id.nunique()),
                windows=int(len(meta) if subset=='all' else (meta.quality=={'quiet':'Q2','hard':'Q3'}[subset]).sum()))
        point = all(value['improvement_fraction'] >= .05 for value in comparisons.values())
        ci_pass = all(value['flight_bootstrap_95ci'][0] > 0 for value in comparisons.values())
        passed = point and ci_pass and same_direction>=2 and all(v['passed'] for v in safeguards.values())
        decisions[candidate] = dict(primary_vs=comparisons, jointly_improving_seeds=same_direction,
            safeguards=safeguards, point_threshold_passed=point, confidence_passed=ci_pass, G1_passed=passed)
    result = dict(primary_metric='flight-equal event-axis rate RMSE averaged p/q and 0.1/0.2/0.5s, rad/s',
        seeds=[17,23,42], scores_by_arm={arm:scores[i].tolist() for i,arm in enumerate('ABCD')},
        mean_by_arm={arm:float(mean[i]) for i,arm in enumerate('ABCD')}, decisions=decisions,
        G0_data_integrity='passed; nonoverlap, hashes, common eval, train-only normalization, causal-input tests',
        G0_channel_coverage='roll/pitch/thrust contextual clusters count gate met; manual yaw and yaw_sp_move_rate insufficient; no causal identification claim',
        G2='not accepted: 1s waveform table is supplemental and cannot substitute full-context response/delay validation',
        G3='not started; requires G1 and G2', sealed_test_accessed=False,
        validation_status='previously inspected development validation, not a new blind test',
        source_training_protocol_sha256=hashlib.sha256((DATA/'training/protocol.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(OUT/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
