"""Secondary rolling diagnostics, never substituted for the full rollout gates."""
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_joint_increment_v34 import OUT, DATA, LOCAL, read, verify_frozen, load_inputs
from actuator_training_v30 import model_from
from rollout_trace_v32 import accumulated_rates


def rate_metrics(predicted, derivative, previous, actual, dt):
    """Return parent-level errors; summed local increments are not a rollout."""
    predicted, derivative, previous, actual, dt = [np.asarray(v, dtype=float)
        for v in (predicted, derivative, previous, actual, dt)]
    assert predicted.shape == previous.shape == actual.shape
    assert dt.shape == actual.shape[:2] and (dt > 0).all()
    accum = accumulated_rates(previous[:, 0], derivative, dt) - actual
    return dict(local_mse=((predicted-actual)**2).mean(1),
        accumulated_mse=(accum**2).mean(1), accumulated_end_mse=accum[:, -1]**2,
        derivative_bias=(derivative[:, :, 3:]-(actual-previous)/dt[:, :, None]).mean(1),
        persistence_mse=((previous-actual)**2).mean(1))


@torch.no_grad()
def rolling_predict(model, inputs, ids):
    keys = set(inputs)-{'history_reference', 'future_reference'}
    steps = {k: [] for k in ['rates_b', 'derivative', 'raw_head']}
    for offset in range(50):
        batches = {k: [] for k in steps}
        for start in range(0, len(ids), 64):
            batch = {k: inputs[k][ids[start:start+64], offset] for k in keys}
            capture = {}
            def hook(module, args, output):
                capture['raw'] = output
            handle = model.head.register_forward_hook(hook)
            try:
                pred = model(**batch)
            finally:
                handle.remove()
            raw = capture['raw']
            deriv = raw.clamp(-6., 6.)*model.derivative_scale
            expected = batch['initial_rates_b']+deriv[:, 3:]*batch['dt_s'][:, 0, None]
            torch.testing.assert_close(expected, pred['rates_b'][:, 0], rtol=0, atol=0)
            for key, value in [('rates_b', pred['rates_b'][:, 0]), ('derivative', deriv), ('raw_head', raw)]:
                assert torch.isfinite(value).all()
                batches[key].append(value.numpy())
        for key, values in batches.items():
            steps[key].append(np.concatenate(values))
    return {k: np.stack(v, 1) for k, v in steps.items()}


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); verify_frozen()
    assert read(OUT/'launch/status.json')['status'] == 'training_complete_evaluation_pending'
    _, _, inputs, labels = load_inputs()
    windows = pd.read_csv(DATA/'windows.csv')
    rows, clips, checks = [], [], []
    for fold in range(5):
        ids = read(DATA/f'fold{fold}/manifest.json')['query_indices']
        actual = labels['rates_b'][ids].numpy()
        previous = inputs['initial_rates_b'][ids].numpy()
        dt = inputs['dt_s'][ids, :, 0].numpy()
        for seed in [17, 23, 42]:
            root = OUT/f'fold{fold}/joint_increment/seed{seed}'
            digest = base.sha(root/'final.pt')
            model = model_from(root/'final.pt')
            candidate = rolling_predict(model, inputs, ids)
            np.savez_compressed(root/'rolling_predictions.npz', array_index=ids, **candidate)
            with np.load(root/'oof_predictions.npz') as z:
                np.testing.assert_array_equal(candidate['rates_b'][:, 0], z['rates_b'][:, 0])
            with np.load(LOCAL/f'fold{fold}/actuator/seed{seed}/trace.npz') as z:
                np.testing.assert_array_equal(z['array_index'], ids)
                baseline = {k: z['rolling_current_'+k] for k in candidate}
            for arm, values in [('baseline', baseline), ('joint', candidate)]:
                metrics = rate_metrics(values['rates_b'], values['derivative'], previous, actual, dt)
                for j, axis in enumerate(['p', 'q', 'r']):
                    frame = windows.iloc[ids][['array_index', 'log_id', 'mode']].copy()
                    frame = frame.assign(fold=fold, seed=seed, arm=arm, axis=axis)
                    for key, val in metrics.items(): frame[key] = val[:, j]
                    rows.append(frame)
                clips.append(dict(fold=fold, seed=seed, arm=arm,
                    clipped_components=int((abs(values['raw_head']) > 6).sum()),
                    max_raw_abs=float(abs(values['raw_head']).max())))
            assert base.sha(root/'final.pt') == digest
            checks.append(dict(fold=fold, seed=seed, checkpoint_unchanged=True,
                derivative_reconstructs_native_step_exact=True, original_origin_exact=True))
        print(f'fold{fold}: all 50 rolling anchors completed', flush=True)
    metrics = pd.concat(rows, ignore_index=True)
    for _, group in metrics.groupby(['seed', 'arm', 'axis']):
        np.testing.assert_array_equal(np.sort(group.array_index), np.arange(407))
    cols = ['local_mse', 'accumulated_mse', 'accumulated_end_mse', 'persistence_mse']
    keys = ['array_index', 'log_id', 'mode', 'fold', 'seed', 'axis']
    old = pd.read_csv(LOCAL/'per_window.csv')
    old = old[(old.arm == 'actuator') & (old.variant == 'rolling_current')].set_index(keys)
    new = metrics[metrics.arm == 'baseline'].set_index(keys)
    np.testing.assert_allclose(new[cols+['derivative_bias']], old.loc[new.index, cols+['derivative_bias']],
        rtol=1e-11, atol=1e-11)
    metrics.to_csv(OUT/'rolling_per_window.csv', index=False)
    metrics[metrics.array_index == 97].to_csv(OUT/'rolling_window97.csv', index=False)
    flight = metrics.groupby(['arm', 'axis', 'seed', 'log_id'])[cols].mean().pow(.5)
    flight.to_csv(OUT/'rolling_per_flight.csv')
    flight.groupby(['arm', 'axis']).mean().to_csv(OUT/'rolling_summary.csv')
    metrics.groupby(['arm', 'axis', 'seed', 'log_id']).derivative_bias.mean().groupby(['arm', 'axis']).mean().to_csv(OUT/'rolling_derivative_bias.csv')
    pd.DataFrame(clips).to_csv(OUT/'rolling_clipping.csv', index=False)
    verify_frozen()
    base.write(OUT/'rolling_verification.json', dict(status='passed', checks=checks,
        previous_v33_control_metrics_reproduced=True, unique407_parents_per_arm_seed=True,
        anchors_per_parent=50, new_independent_windows=0, forecast_from_parent_origin=False,
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False))


if __name__ == '__main__': main()
