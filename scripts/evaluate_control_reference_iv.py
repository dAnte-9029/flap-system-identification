"""Frozen validation of the train-selected manual-reference 100ms diagnostic."""
from pathlib import Path
import numpy as np
import pandas as pd

from diagnose_control_reference_iv import BASE, source, design, fit, predict, iv_gain

m = source.m
OUT = BASE / 'reference_iv_validation'


def features(batch):
    dt = batch.trajectory.dt_s[:, :5]
    means = np.sum(batch.trajectory.controls[:, :5] * dt[:, :, None], axis=1) / dt.sum(1)[:, None]
    common = (means[:, 1] + means[:, 2]) / 2
    other = np.column_stack([means[:, 0], (means[:, 1] - means[:, 2]) / 2, means[:, 3]])
    return np.column_stack([design(batch, 5)[0], other]), common


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    m.configure()
    parent = BASE / 'reference_iv_diagnostic'
    completion = m.read_json(parent / 'completion.json')
    m.verify_pins(completion['outputs'])
    assert next(x for x in completion['diagnostic_gates'] if x['steps'] == 5 and x['variant'] == 'iv_manual')['gate']
    protocol = m.read_json(parent / 'protocol.json')
    m.verify_pins(protocol['pins'])
    ensemble = BASE / 'ensemble_candidate/predictions.npz'
    m.verify_pins(m.read_json(BASE / 'ensemble_candidate/completion.json')['outputs'])
    pins = dict(protocol['pins'])
    for path in [Path(__file__), parent / 'completion.json', ensemble]:
        pins[m.rel(path)] = m.file_hash(path)
    m.write_json(OUT / 'protocol.json', dict(
        pins=pins, selected='manual reference, 100ms only; fixed by training diagnostic gate',
        partitions=['train', 'validation'], sealed_test_accessed=False,
        fit='same 6477 train origins; five-fold train nuisance residuals for beta; final nuisance/direct ridge on all admitted train; ridge0.01 unchanged',
        inference='current/past states and controls plus known future commands only; no validation raw references',
        evaluation='all original validation origins, flight-equal q100 endpoint RMSE; population differs from reference-fresh Stabilized training subset',
        gate='must improve vs no-common and direct ridge and match or improve frozen ensemble mean q100 RMSE; diagnostic only, no causal certification',
        limitations='reference exclusion/exogeneity unproven; no control promotion even if prediction gate passes'))
    _, batches, _ = source.inputs()
    tr, val = batches['train'], batches['validation']
    with np.load(BASE / 'reference_semantics/references.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['window_ids'], tr.trajectory.window_ids)
        admitted = archive['admitted']
        reference = archive['past_current_reference'][admitted, :, 0]
    x, u = features(tr)
    x, u = x[admitted], u[admitted]
    ids = tr.trajectory.log_ids[admitted]
    y = (tr.trajectory.truth.angular_velocity_b[:, 5, 1] - tr.trajectory.truth.angular_velocity_b[:, 0, 1])[admitted]
    fold_map = m.read_json(BASE / 'local_identification/protocol.json')['folds']
    folds = np.array([fold_map[g] for g in ids])
    values = np.column_stack([u, y, reference])
    residual = np.full_like(values, np.nan)
    for fold in range(5):
        keep = folds != fold
        residual[~keep] = values[~keep] - predict(fit(x[keep], values[keep], ids[keep]), x[~keep])
    assert np.isfinite(residual).all()
    beta, denominator, signal = iv_gain(residual[:, 0], residual[:, 1], residual[:, 2:], ids)
    assert np.isfinite(beta)
    nuisance = fit(x, values[:, :2], ids)
    direct = fit(np.column_stack([x, u]), y[:, None], ids)
    # Freeze coefficients before any validation outcome is scored.
    m.write_json(OUT / 'fit.json', dict(beta=beta, denominator=denominator, instrument_rms=signal,
        train_origins=len(ids), train_flights=len(set(ids)), validation_used_for_fit=False))
    xv, uv = features(val)
    n = predict(nuisance, xv)
    q0 = val.trajectory.truth.angular_velocity_b[:, 0, 1]
    predictions = dict(no_common=q0 + n[:, 1], direct=q0 + predict(direct, np.column_stack([xv, uv]))[:, 0],
        iv_manual=q0 + n[:, 1] + beta * (uv - n[:, 0]))
    with np.load(ensemble, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['window_ids'], val.trajectory.window_ids)
        predictions['ensemble'] = archive['angular_velocity_b'][:, 5, 1]
    truth = val.trajectory.truth.angular_velocity_b[:, 5, 1]
    rows = []
    for name, pred in predictions.items():
        assert np.isfinite(pred).all()
        for log in sorted(set(val.trajectory.log_ids)):
            take = val.trajectory.log_ids == log
            rows.append(dict(variant=name, log_id=log, n=int(take.sum()), rmse=float(np.sqrt(np.mean((pred[take] - truth[take]) ** 2)))))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'per_flight.csv', index=False)
    summary = frame.groupby('variant').rmse.mean()
    summary.to_csv(OUT / 'summary.csv')
    np.savez_compressed(OUT / 'predictions.npz', **predictions, truth=truth, window_ids=val.trajectory.window_ids.astype(str))
    m.verify_pins(pins)
    m.write_json(OUT / 'completion.json', dict(status='complete', sealed_test_accessed=False,
        prediction_gate=bool(summary.iv_manual < min(summary.no_common, summary.direct) and summary.iv_manual <= summary.ensemble),
        instrument_validated=False, plant_promoted=False,
        outputs={m.rel(p): m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    run()
