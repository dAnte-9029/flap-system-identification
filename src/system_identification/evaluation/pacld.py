"""Native-time PACLD diagnostics; observational prediction, not physical separation proof."""
import bisect
import math
import numpy as np
import torch
from system_identification.models.pacld.dynamics import phase_carrier
from system_identification.models.pahbd import geometry as so3
from system_identification.training.pacld import window_objective, macro_loss, transition_residual
from system_identification.data.pacld import move_window
from .pahbd import native_horizon_errors, aggregate_flights, FrozenRollResponse, factual_predictor


COMPONENTS = {'p': slice(0, 3), 'v': slice(3, 6), 'R': slice(6, 9), 'omega': slice(9, 12)}


def hierarchical(rows, key):
    """Window -> horizon within flight -> equal flights; omit missing, never zero-fill."""
    groups = {}
    for row in rows:
        if row.get(key) is not None:
            groups.setdefault((row['flight_id'], row['horizon_us']), []).append(row[key])
    per_group = [dict(flight_id=f, horizon_us=h, value=float(np.mean(v)), windows=len(v))
                 for (f, h), v in sorted(groups.items())]
    flights = {}
    for r in per_group: flights.setdefault(r['flight_id'], []).append(r['value'])
    per_flight = {f: float(np.mean(v)) for f, v in flights.items()}
    return dict(score=float(np.mean(list(per_flight.values()))) if per_flight else None,
                per_flight=per_flight, per_flight_horizon=per_group)


def paired_physical_errors(window, result, mode):
    lookup = dict(zip(result['times'], result['predictions'][mode])); values = {}
    for i, j in result['pairs']:
        a, b = window.targets[i], window.targets[j]
        r = transition_residual(lookup[a.measurement].x, lookup[b.measurement].x, a.value[:13], b.value[:13])
        for name, sl in COMPONENTS.items():
            if bool((a.mask[sl]*b.mask[sl]*torch.minimum(a.quality[sl], b.quality[sl]) > 0).all()):
                values.setdefault(name, []).append(float(r[sl].square().sum()))
    return {name: dict(mean_squared=float(np.mean(v)), pairs=len(v)) for name, v in values.items()}


def phase_correlation(phases, errors, harmonics=3):
    finite = np.isfinite(phases) & np.isfinite(errors)
    phase, error = np.asarray(phases)[finite], np.asarray(errors)[finite]
    count = len(phase)
    if count < 2*harmonics+2 or np.ptp(phase) < 2*math.pi:
        return dict(samples=count, r2=None, reason='insufficient native phase support')
    angle = phase[:, None]*np.arange(1, harmonics+1)
    A = np.column_stack((np.ones(count), np.cos(angle), np.sin(angle)))
    if np.linalg.matrix_rank(A) < A.shape[1] or np.var(error) < 1e-20:
        return dict(samples=count, r2=None, reason='rank deficient or constant residual')
    coef = np.linalg.lstsq(A, error, rcond=None)[0]
    return dict(samples=count, r2=float(1-np.sum((error-A@coef)**2)/np.sum((error-error.mean())**2)),
                amplitude=np.hypot(coef[1:harmonics+1], coef[harmonics+1:]).tolist(),
                phase_span_rad=float(np.ptp(phase)),
                median_native_phase_step_rad=float(np.median(np.diff(phase))),
                caution='descriptive in-sample harmonic association, not independent periodic truth')


def intracycle(window, result, config):
    cfg = config['diagnostics']; lookup = {m: dict(zip(result['times'], p)) for m, p in result['predictions'].items()}
    tape = window.controls.canonical()
    changes = [tape.times[i] for i in range(1, len(tape.times))
               if float((tape.values[i]-tape.values[i-1]).abs().max()) > cfg['control_change_threshold']]
    phase = window.metadata.get('target_phase_reference', [None]*len(window.targets))
    groups = {}
    for i, e in enumerate(window.targets):
        for name, source, value_index, mask_index in [('q','vehicle_angular_velocity',11,10), ('vD','vehicle_local_position',5,5)]:
            if e.source != source or float(e.mask[mask_index]*e.quality[mask_index]) <= 0: continue
            regime = 'control_change' if any(e.measurement-cfg['control_change_lookback_s'] <= t <= e.measurement for t in changes) else 'stable'
            for m in ('FULL', 'BASE'):
                group = groups.setdefault((name, regime, m), dict(error=[], phase=[], difference=[]))
                group['error'].append(float(lookup[m][e.measurement].x[value_index]-e.value[value_index]))
                group['phase'].append(float('nan') if phase[i] is None else phase[i])
                group['difference'].append(float(lookup['FULL'][e.measurement].x[value_index]-lookup['BASE'][e.measurement].x[value_index]))
    rows = []
    for (name, regime, mode), g in sorted(groups.items()):
        rows.append(dict(variable=name, regime=regime, mode=mode, samples=len(g['error']),
                         mean_squared=float(np.mean(np.square(g['error']))),
                         full_base_rms=float(np.sqrt(np.mean(np.square(g['difference'])))),
                         phase_residual=phase_correlation(np.array(g['phase']), np.array(g['error']), cfg['phase_harmonics'])))
    return rows


def latent_diagnostics(model, window, result, include_trace=False):
    states = [result['belief']]+result['predictions']['FULL']
    bases = [result['belief']]+result['predictions']['BASE']
    times = [0.]+result['times']; tape = window.controls.canonical()
    trace = []
    for t, s, base in zip(times, states, bases):
        u = tape.values[bisect.bisect_right(tape.times, t)-1]
        dz, dc = model.flow.dynamics.latent_rates(s, u)
        dfull = model.flow.dynamics.decode(s, u, 'FULL')
        explicit = dfull-model.flow.dynamics.decode(s, u, 'BASE')
        path_difference = dfull-model.flow.dynamics.decode(base, u, 'BASE')
        def array(tensor): return tensor.detach().cpu().tolist()
        trace.append(dict(time=t, z_s=array(s.z_s), c=array(s.c), z_f=array(phase_carrier(s.phi, s.c)),
                          phi=float(s.phi.item()), f=float(s.f.item()), slow_rate=array(dz), envelope_rate=array(dc),
                          explicit_decoder_difference=array(explicit), trajectory_decoder_difference=array(path_difference)))
    c = np.array([r['c'] for r in trace]).reshape(-1, 3, 2)
    phi = np.array([r['phi'] for r in trace]); z = np.array([r['z_s'] for r in trace])
    modes = []
    for k in range(3):
        amplitude = np.linalg.norm(c[:, k], axis=1)
        valid = amplitude > 1e-8  # numerical angle validity only, not a scientific collapse threshold
        angle = np.unwrap(np.arctan2(c[:, k, 1], c[:, k, 0]))
        slope = None
        if valid.sum() >= 3 and np.ptp(phi[valid]) > 1e-8:
            slope = float(np.polyfit(phi[valid], angle[valid], 1)[0])
        modes.append(dict(harmonic=k+1, envelope_amplitude_rms=float(np.sqrt(np.mean(amplitude**2))),
                          envelope_phase_slope_vs_phi=slope, exact_counter_rotation_slope=-(k+1),
                          angle_valid_samples=int(valid.sum())))
    explicit = np.array([r['explicit_decoder_difference'] for r in trace])
    summary = dict(slow_std=z.std(axis=0).tolist(), envelope_std=c.reshape(-1,6).std(axis=0).tolist(),
                   modes=modes, explicit_acceleration_rms=np.sqrt(np.mean(explicit[:,:3]**2,axis=0)).tolist(),
                   explicit_angular_acceleration_rms=np.sqrt(np.mean(explicit[:,3:]**2,axis=0)).tolist(),
                   interpretation='descriptive branch activity/counter-rotation probes; no physical latent truth or collapse verdict')
    if include_trace: summary['trace'] = trace
    return summary


@torch.no_grad()
def evaluate(model, windows, config, diagnostics=True, include_trace=False):
    model.eval(); rows = []; native = []; paired = []; within = []
    for w in windows:
        if w.metadata['split'] not in ('train','validation'): raise ValueError('sealed split forbidden')
        w = move_window(w, next(model.parameters()).device, next(model.parameters()).dtype)
        result = window_objective(model, w, 0., diagnostic_base=True)
        full_macro, _ = macro_loss(result['predictions']['FULL'], w.targets, result['times'], result['pairs'])
        row = dict(window_id=w.metadata['window_id'], flight_id=w.metadata['flight_id'],
                   horizon_us=w.metadata.get('horizon_us', round(result['times'][-1]*1e6)),
                   sampling_group=w.metadata.get('sampling_group','synthetic'),
                   full=float(result['full']), macro=None if result['macro'] is None else float(result['macro']),
                   full_macro=None if full_macro is None else float(full_macro), pairs=len(result['pairs']),
                   proposed_pairs=len(w.metadata.get('cycle_pairs',())), rejected_pairs=result['rejected_pairs'])
        rows.append(row)
        # Match PA-HBD's nearest native measurement within 30ms; no interpolation.
        errors = native_horizon_errors(result['predictions']['FULL'], result['times'], w.targets, w.metadata)
        for e in errors: e['window_horizon_us'] = row['horizon_us']
        native.extend(errors)
        for mode in ('BASE','FULL'):
            for component, values in paired_physical_errors(w, result, mode).items():
                paired.append(dict(**{k:row[k] for k in ('window_id','flight_id','horizon_us','sampling_group')},
                                   mode=mode, component=component, **values))
        if diagnostics:
            for r in intracycle(w, result, config):
                within.append(dict(**{k:row[k] for k in ('window_id','flight_id','horizon_us','sampling_group')}, **r))
            row['latent'] = latent_diagnostics(model, w, result, include_trace)
    reduction = {name: hierarchical(rows, name) for name in ('full','macro','full_macro')}
    # Preserve native metric definition, but first equalize window-horizon groups.
    grouped_native = {}
    for r in native:
        key=(r['flight_id'],r['window_horizon_us'],r['horizon_s'],r['component'])
        grouped_native.setdefault(key,[]).append(r['squared_error'])
    balanced_native = [dict(flight_id=f,horizon_s=t,component=c,squared_error=float(np.mean(v)))
                       for (f,h,t,c),v in grouped_native.items()]
    paired_summary = []
    for mode in ('BASE','FULL'):
        for component in COMPONENTS:
            for horizon in sorted({r['horizon_us'] for r in paired}):
                chosen=[r for r in paired if r['mode']==mode and r['component']==component and r['horizon_us']==horizon]
                agg=hierarchical(chosen,'mean_squared')
                if agg['score'] is not None:
                    paired_summary.append(dict(mode=mode,component=component,horizon_us=horizon,
                                               rmse=math.sqrt(agg['score']),flights=len(agg['per_flight'])))
    return dict(selection_metric=config['training']['selection'], selection_score=reduction['full']['score'],
                loss_reduction=reduction, windows=rows, native_flight_equal=aggregate_flights(balanced_native),
                native_rows=native, paired_rows=paired, paired_flight_equal=paired_summary,
                intracycle_rows=within, intracycle_flight_equal=summarize_intracycle(within),
                coverage=dict(windows=len(rows),paired_windows=sum(r['pairs']>0 for r in rows),
                              paired_flights=len({r['flight_id'] for r in rows if r['pairs']>0}),
                              valid_pairs=sum(r['pairs'] for r in rows),rejected_pairs=sum(r['rejected_pairs'] for r in rows)))



def summarize_intracycle(rows):
    summary = []
    keys = sorted({(r['sampling_group'], r['variable'], r['regime']) for r in rows})
    for group, variable, regime in keys:
        scores = {}
        for mode in ('BASE', 'FULL'):
            selected = [r for r in rows if (r['sampling_group'], r['variable'], r['regime'], r['mode']) == (group, variable, regime, mode)]
            scores[mode] = hierarchical(selected, 'mean_squared')
        if any(v['score'] is None for v in scores.values()):
            continue
        summary.append(dict(sampling_group=group, variable=variable, regime=regime,
                            full_rmse=math.sqrt(scores['FULL']['score']), base_rmse=math.sqrt(scores['BASE']['score']),
                            base_minus_full_mse=scores['BASE']['score']-scores['FULL']['score'],
                            flights=len(scores['FULL']['per_flight']),
                            interpretation='native sample prediction error; no interpolated high-frequency truth'))
    return summary


def frozen_response(model, windows, source, split='validation'):
    """Reuse frozen pairs/identities/metrics verbatim; explicit trained caller only."""
    if split not in ('train','validation'): raise ValueError('sealed split forbidden')
    adapter = FrozenRollResponse(source)
    cohort = adapter.pairs.loc[adapter.pairs.split == split]
    needed = set(cohort[['window_a', 'window_b']].to_numpy().ravel())
    missing = needed-set(windows.index.window_id)
    if missing:
        raise ValueError(f'{len(missing)} frozen response windows unavailable; refusing to reselect pairs')
    class DeviceWindows:
        index = windows.index
        def __getitem__(self, i):
            return move_window(windows[i], next(model.parameters()).device, next(model.parameters()).dtype)
    return adapter.evaluate(factual_predictor(model, DeviceWindows()), split)
