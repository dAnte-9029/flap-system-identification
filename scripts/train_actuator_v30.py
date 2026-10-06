"""One v29-prespecified 1600-step actuator-conditioned training run."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from actuator_training_v30 import OUT, DATA, load, verify_protocol, model_from, ActuatorCommandGRU
from prepare_actuator_inputs_v29 import fingerprint
from reference_groupcv_v8 import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    parser.add_argument('--seed', type=int, required=True, choices=[17, 23, 42])
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    verify_protocol()
    plan = json.loads((DATA / 'training_plan.json').read_text())
    run = next(r for r in plan['runs'] if (r['fold'], r['seed']) == (args.fold, args.seed))
    fold = DATA / f'fold{args.fold}'; info = json.loads((fold / 'manifest.json').read_text())
    assert base.sha(fold / 'normalization.npz') == run['normalization_sha256'] == info['normalization_sha256']
    assert base.sha(base.ROOT / run['draws_path']) == run['draws_sha256']
    with np.load(fold / 'normalization.npz') as z: stats = {k: z[k] for k in z.files}
    with np.load(base.ROOT / run['draws_path']) as z: draws = z['array_index']
    assert draws.shape == (1600, 32) and set(draws.ravel()) <= set(info['fit_indices'])
    x, y = load()
    torch.manual_seed(args.seed); model = ActuatorCommandGRU(stats)
    assert fingerprint(model) == run['initial_parameter_sha256']
    assert sum(p.numel() for p in model.parameters()) == 40070
    out = OUT / f'fold{args.fold}/seed{args.seed}'; out.mkdir(parents=True, exist_ok=False)
    base.write(out / 'run_protocol.json', {**run, 'protocol_sha256': base.sha(OUT / 'protocol.json')})
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    curve = []; start = time.monotonic()
    for step, ids in enumerate(draws, 1):
        opt.zero_grad(set_to_none=True)
        pred = model(**{k: v[ids] for k, v in x.items()})
        loss = base.loss_fn(pred, {k: v[ids] for k, v in y.items()}, model)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 200 == 0:
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), elapsed_s=time.monotonic()-start)
            curve.append(row); pd.DataFrame(curve).to_csv(out / 'training_curve.csv', index=False)
            base.write(out / 'status.json', dict(status='running', **row))
            print(row, flush=True)
    checkpoint = out / 'final.pt'
    torch.save(dict(state_dict=model.state_dict(), steps=1600, fold=args.fold, seed=args.seed,
        input_contract='post_allocation_motor0_servo012_v29', protocol_sha256=base.sha(OUT / 'protocol.json')), checkpoint)
    for name, ids in [('fit', info['fit_indices']), ('oof', info['query_indices'])]:
        pred = predict(model.eval(), x, ids)
        np.savez_compressed(out / f'{name}_predictions.npz', array_index=ids, **pred)
    held = dict(x); held['future_command'] = x['future_command'][:, :1].expand(-1, 50, -1)
    pred = predict(model, held, info['query_indices'])
    np.savez_compressed(out / 'oof_held.npz', array_index=info['query_indices'], **pred)
    restored = model_from(checkpoint); ids = info['query_indices'][:2]
    factual = predict(model, x, ids); replay = predict(restored, x, ids)
    altered = dict(x); altered['future_command'] = x['future_command'].clone()
    altered['future_command'][:, 10:] += .1
    changed = predict(restored, altered, ids)
    for k in factual:
        np.testing.assert_array_equal(factual[k], replay[k])
        np.testing.assert_array_equal(factual[k][:, :10], changed[k][:, :10])
    base.write(out / 'status.json', dict(status='complete', step=1600, elapsed_s=time.monotonic()-start,
        checkpoint_sha256=base.sha(checkpoint), exact_init_and_draws=True, reload_and_prefix_causality=True))


if __name__ == '__main__': main()
