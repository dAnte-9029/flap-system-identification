"""Frozen-budget reference-model data-strategy comparison (train/validation only)."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from system_identification.models.reference_gru import ReferenceGRU

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/analysis/results/control_windows_v2'
OUT = DATA / 'training'
ART = ROOT / 'artifacts/control_windows_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for p, digest in manifest['tables'].items():
        assert sha(DATA / p) == digest, p
    for p, digest in manifest['arrays'].items():
        assert sha(ROOT / p) == digest, p
    assert not set(manifest['train_flights']) & set(manifest['validation_flights'])
    with np.load(ROOT / 'dataset/control_windows_v2/windows.npz') as z:
        x = {k[6:]: torch.tensor(z[k], dtype=torch.float32) for k in z.files if k.startswith('input_')}
        y = {k[6:]: torch.tensor(z[k], dtype=torch.float32) for k in z.files if k.startswith('label_')}
    with np.load(ROOT / 'dataset/control_windows_v2/normalization.npz') as z:
        stats = {k: z[k] for k in z.files}
    return x, y, stats


def loss_fn(pred, target, model):
    rate = ((pred['rates_b'] - target['rates_b']) / model.rate_scale).square().mean()
    velocity = ((pred['velocity_n'] - target['velocity_n']) / model.velocity_scale).square().mean()
    # Sign-invariant quaternion chordal loss, approximately angle squared / .1^2.
    dot = (pred['quaternion'] * target['quaternion']).sum(-1).clamp(-1, 1)
    attitude = (4 * (1 - dot.square()).clamp_min(0) / .01).mean()
    return rate + velocity + .2 * attitude


def train_one(arm, seed, steps, x, y, stats, save=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ReferenceGRU(stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    cohort = pd.read_csv(DATA / f'arm_{arm}.csv')
    assert (cohort.partition == 'train').all()
    indices = cohort.array_index.to_numpy()
    probability = cohort.sampling_probability.to_numpy()
    probability /= probability.sum()
    exposure = np.zeros(len(indices), dtype=int)
    started = time.monotonic()
    curve = []
    for step in range(1, steps + 1):
        local = rng.choice(len(indices), 32, replace=True, p=probability)
        np.add.at(exposure, local, 1)
        ix = indices[local]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**{k: v[ix] for k, v in x.items()})
        loss = loss_fn(prediction, {k: v[ix] for k, v in y.items()}, model)
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss {arm}/{seed}/{step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            item = dict(arm=arm, seed=seed, step=step, loss=float(loss.detach()),
                        gradient_norm=float(norm), elapsed_s=time.monotonic()-started)
            curve.append(item)
            print(json.dumps(item), flush=True)
    result = dict(arm=arm, seed=seed, steps=steps, elapsed_s=time.monotonic()-started,
                  final_batch_loss=curve[-1]['loss'], exposure_min=int(exposure.min()),
                  exposure_median=float(np.median(exposure)), exposure_max=int(exposure.max()))
    if save:
        destination = ART / f'{arm}_{seed}'
        destination.mkdir(parents=True, exist_ok=False)
        checkpoint = destination / 'final.pt'
        torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=seed, steps=steps,
                        architecture='ReferenceGRU64', protocol_sha256=sha(OUT/'protocol.json'),
                        data_manifest_sha256=sha(DATA/'manifest.json')), checkpoint)
        pd.DataFrame(curve).to_csv(destination/'training_curve.csv', index=False)
        cohort.assign(draws=exposure).to_csv(destination/'exposure.csv', index=False)
        # Fixed common validation set; no validation-dependent checkpoint selection.
        evaluation = pd.read_csv(DATA/'evaluation_windows.csv')
        assert (evaluation.partition == 'validation').all()
        ev = evaluation.array_index.to_numpy()
        model.eval()
        predictions = {k: [] for k in y}
        with torch.no_grad():
            for offset in range(0, len(ev), 64):
                ix = ev[offset:offset+64]
                pred = model(**{k: v[ix] for k, v in x.items()})
                for key in predictions:
                    value = pred[key].cpu().numpy()
                    assert np.isfinite(value).all(), key
                    predictions[key].append(value)
        np.savez_compressed(destination/'validation_predictions.npz', array_index=ev,
                            **{k: np.concatenate(v) for k, v in predictions.items()})
        result.update(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                      predictions_sha256=sha(destination/'validation_predictions.npz'))
        write(destination/'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--steps', type=int, default=800)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y, stats = load()
    if args.preflight:
        path = DATA/'training_preflight.json'
        assert not path.exists()
        result = train_one('A', 0, 20, x, y, stats, save=False)
        result.update(purpose='train-only finite-gradient/runtime check; not a model candidate',
                      validation_scored=False, torch_version=torch.__version__, cpu_threads=1)
        write(path, result)
        return
    assert args.steps > 0
    OUT.mkdir(exist_ok=False)
    protocol = dict(arms=['A','B','C','D'], seeds=[17,23,42], updates=args.steps,
        architecture='ReferenceGRU64; causal observed history + planned references; held current frequency/airspeed',
        batch_size=32, learning_rate=3e-4, optimizer='AdamW', weight_decay=1e-4,
        gradient_norm_clip=1., checkpoint='final fixed update; no validation selection',
        loss='mean(rate_error/rate_scale)^2 + mean(velocity_error/velocity_scale)^2 + 0.2*mean(4*(1-dot(q,qtruth)^2)/0.01)',
        cpu_threads=1, device='cpu', deterministic_algorithms=True,
        normalization='shared A-train unique rows; frozen dataset normalization.npz',
        evaluation='739 common validation windows from 9 whole flights; previously seen development validation, not blind test',
        horizons_steps=[5,10,25,50], primary_horizons_steps=[5,10,25],
        primary_metric='flight-equal RMSE, equal mean over roll-active p and pitch-active q and 0.1/0.2/0.5s; active if future reference range >=1degree',
        G1='C/D improve >=5% vs BOTH A/B; >=2/3 paired seeds improve vs both; flight-bootstrap95% improvement CI >0 vs both; full short-horizon velocity/attitude/p/q/r <=3% degradation vs A; Q2/Q3 <=5%',
        G2='complete input-event context response: amplitude/increment >=10% better vs A/B; median timing <=max(40ms,2*source input interval); coverage gates per sign10clusters/3flights',
        G3='only after G1 and G2; otherwise do not run model-based reference optimizer',
        bootstrap='2000 paired resamples of whole validation flights, seed1309; seeds averaged, not counted as flights',
        weak_contrast='C395/A407; B and C share384 windows; do not tune selection after outcomes',
        manual_yaw='insufficient independent events; no control-response pass claim',
        yaw_sp_move_rate='constant zero; unidentified nonzero effect',
        data_manifest_sha256=sha(DATA/'manifest.json'),
        script_sha256=sha(__file__), model_sha256=sha(ROOT/'src/system_identification/models/reference_gru.py'),
        sealed_test_accessed=False, torch_version=torch.__version__)
    write(OUT/'protocol.json', protocol)
    completed = []
    try:
        for arm in protocol['arms']:
            for seed in protocol['seeds']:
                write(OUT/'status.json', dict(status='running', current_arm=arm, current_seed=seed, completed=completed))
                completed.append(train_one(arm, seed, args.steps, x, y, stats))
                write(OUT/'status.json', dict(status='running', completed=completed))
        write(OUT/'status.json', dict(status='complete', completed=completed))
    except Exception as exc:
        write(OUT/'status.json', dict(status='failed', error=repr(exc), completed=completed))
        raise


if __name__ == '__main__':
    main()
