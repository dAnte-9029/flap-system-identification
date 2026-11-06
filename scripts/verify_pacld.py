#!/usr/bin/env python
"""Bounded CPU verification only: one synthetic pair and optional two train windows."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import torch
from system_identification.models.pacld import PACLD, Config
from system_identification.models.pahbd import Config as SyntheticConfig
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import trajectory_loss


def check(model, window):
    model.zero_grad(set_to_none=True)
    before = time.perf_counter()
    belief, pair = model.predict_pair(window)
    forward_seconds = time.perf_counter()-before
    times = sorted({e.measurement for e in window.targets})
    losses = {mode: trajectory_loss(states, window.targets, times)[0] for mode, states in pair.items()}
    before = time.perf_counter()
    sum(losses.values()).backward()
    backward_seconds = time.perf_counter()-before
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError(f'missing/nonfinite gradient: {name}')
    gradient_l1 = {name: float(sum(p.grad.abs().sum() for p in module.parameters()))
                   for name, module in [('shared', model.belief.shared), ('slow_head', model.belief.slow_head),
                                        ('fast_head', model.belief.fast_head), ('slow', model.flow.dynamics.slow),
                                        ('equilibrium', model.flow.dynamics.equilibrium),
                                        ('relaxation', model.flow.dynamics.relaxation),
                                        ('coupling', model.flow.dynamics.coupling), ('decoder', model.flow.dynamics.decoder)]}
    if min(gradient_l1.values()) <= 0:
        raise RuntimeError('zero module gradient')
    for states in pair.values():
        for s in states:
            for field in ('x', 'z_s', 'c', 'f', 'phi', 'variance', 'phase_covariance'):
                if not torch.isfinite(getattr(s, field)).all():
                    raise RuntimeError(f'nonfinite state: {field}')
            if abs(float(s.x[6:10].norm())-1) > 1e-10:
                raise RuntimeError('quaternion norm drift')
    meta = window.metadata
    return dict(window_id=meta['window_id'], flight_id=meta['flight_id'], split=meta['split'],
                sampling_group=meta.get('sampling_group', 'synthetic_native'),
                segment_id=meta.get('segment_id'), decision_us=meta.get('decision_us'),
                manifest_sha256=meta.get('manifest_sha256'),
                history_events=len(window.history), native_queries=len(times),
                history_start=window.start, horizon_s=times[-1],
                phase_quality=float(belief.phase_quality.item()),
                forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                loss={k: float(v.detach()) for k, v in losses.items()}, gradient_l1=gradient_l1,
                interpretation='untrained computation/gradient smoke; losses are not performance evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pacld/default.json')
    parser.add_argument('--real-train', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    model = PACLD(Config(**cfg['model'])).double().cpu()
    records = [check(model, controlled_periodic_window(SyntheticConfig(), .2)[0])]
    load_seconds = None
    if args.real_train:
        from system_identification.data.pahbd import NativeWindows
        before = time.perf_counter()
        data_cfg = dict(cfg['data']); data_cfg['horizon_us'] = [200000]
        dataset = NativeWindows(**data_cfg, split='train', device='cpu', dtype=torch.float64)
        load_seconds = time.perf_counter()-before
        # One deterministic first eligible train window per existing sampling group.
        for group in ('main_50hz', 'sparse_10hz'):
            eligible = dataset.windows.index[(dataset.windows.sampling_group == group)
                                              & dataset.windows.shared_periodic_eligible]
            if len(eligible) == 0:
                raise RuntimeError(f'no eligible train window in {group}')
            record = check(model, dataset[int(eligible[0])])
            if record['phase_quality'] <= 0:
                raise RuntimeError('train window has no causal Hall reference')
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    result = dict(schema='pacld_cpu_smoke_v1', seed=cfg['seed'], config=asdict(model.config),
                  code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  revision_note='base revision before the PACLD source commit; source and this report are committed together',
                  device='cpu', dtype='float64', threads=1, torch_version=torch.__version__,
                  dataset_load_seconds=load_seconds, optimizer_steps=0, records=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'output': str(output), 'windows': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
