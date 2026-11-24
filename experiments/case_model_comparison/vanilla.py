"""Memoryless direct next-motion MLP. Native observations supervise a single transition."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'controlled_long_horizon_stage1'))
from common import HISTORY, TAPES, conditional_data, setup
from dynamics import inverse, prefix_rotations, prefix_start, qexp, qmul, query, rotate, unit
from model import trajectory_losses
from train import atomic_save

OUT = HISTORY.parent / 'case_model_comparison_20261124'
BATCH = 1024
EPOCHS = 24
SEED = 17


def current_features(v, q, w):
    down = torch.zeros_like(v)
    down[..., 2] = 1.
    iq = inverse(q)
    return torch.cat([rotate(iq, v), w, rotate(iq, down)], -1)


class VanillaMLP(nn.Module):
    def __init__(self, feature_mean, feature_scale, motion_mean, motion_scale):
        super().__init__()
        for name in ('feature_mean','feature_scale','motion_mean','motion_scale'):
            self.register_buffer(name, locals()[name])
        self.net = nn.Sequential(nn.Linear(14, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 6))

    def direct(self, features, commands, duration):
        # Each broadcasted entry is a single held-command next-state prediction.
        shape = torch.broadcast_shapes(features.shape[:-1], commands.shape[:-1], duration.shape[:-1])
        raw = torch.cat([features.expand(*shape, 9), commands.expand(*shape, 4)], -1)
        inputs = torch.cat([(raw-self.feature_mean)/self.feature_scale,
                            ((duration-.02)/.08).expand(*shape, 1)], -1)
        return self.net(inputs)*self.motion_scale + self.motion_mean

    def forward(self, history, initial, tape):
        p, v, q, w = initial.split([3, 3, 4, 3], -1)
        starts, slopes = [], []
        for block in tape.unbind(1):
            dt = block[..., 4:5]*1e-6
            duration = dt.sum(1, keepdim=True)
            features = current_features(v, q, w)
            motion = torch.cat([v, w], -1)
            predicted = self.direct(features[:, None], block[..., :4], duration)
            next_motion = torch.cat([rotate(q[:, None], predicted[..., :3]), predicted[..., 3:]], -1)
            # Interpolate direct endpoints over each command's actual duration.
            acc = (next_motion-motion[:, None])/duration
            dv = acc*dt
            segment_motion = motion[:, None]+prefix_start(dv)
            dp = dt*(segment_motion[..., :3]+.5*dv[..., :3])
            dq = qexp(dt*(segment_motion[..., 3:]+.5*dv[..., 3:]))
            quats = unit(qmul(q[:, None], prefix_rotations(dq)))
            starts.append(torch.cat([p[:, None]+prefix_start(dp), segment_motion[..., :3],
                                     torch.cat([q[:, None], quats[:, :-1]], 1), segment_motion[..., 3:]], -1))
            slopes.append(acc)
            p, q = p+dp.sum(1), quats[:, -1]
            v, w = (motion+dv.sum(1)).split(3, -1)
        return tape[..., 4].flatten(1).cumsum(1), torch.cat(starts,1), torch.cat(slopes,1), None


@torch.no_grad()
def examples(data):
    # Two native clocks may observe the next velocity and angular rate at different times.
    initial, labels, times, valid = data[1:5]
    tau, idx = times.masked_fill(~valid, float('inf')).min(1)
    tau = tau[:, [0, 2]]
    assert torch.isfinite(tau).all() and (tau > 0).all()
    rows = torch.arange(len(tau), device=tau.device)
    q = initial[:, 6:10]
    x = current_features(initial[:, 3:6], q, initial[:, 10:])
    target = torch.cat([rotate(inverse(q), labels[rows, idx[:, 0], 3:6]),
                        labels[rows, idx[:, 2], 10:]], -1)
    count = math.ceil(tau.max().item() / .02)
    tape = data[-1][:, :count].flatten(1, 2)
    us = tape[..., 4]
    left = (us.cumsum(1)-us)[:, None]
    overlap = ((tau*1e6).round()[:, :, None]-left).clamp_min(0).minimum(us[:, None])
    weights = overlap / overlap.sum(2, keepdim=True)
    # Collapse adjacent repeated ZOH values, including repetitions across 20 ms boundaries.
    commands = tape[..., :4]
    active = overlap.any(1)
    slots = torch.arange(us.shape[1], device=us.device).expand_as(us)
    last = torch.where(active, slots, -1).cummax(1).values
    prev = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], 1)
    old = commands.gather(1, prev.clamp_min(0)[..., None].expand_as(commands))
    changed = active & ((prev < 0) | (commands != old).any(-1))
    group = (changed.long().cumsum(1)-1).clamp_min(0)
    size = int(group.max())+1
    packed = commands.new_zeros(len(tau), size, 4)
    packed.scatter_add_(1, group[..., None].expand_as(commands), commands*changed[..., None])
    total = weights.new_zeros(len(tau), 2, size)
    total.scatter_add_(2, group[:, None].expand_as(weights), weights)
    torch.testing.assert_close(total.sum(2), torch.ones_like(tau))
    return x, packed, tau, total, target


def one_step(model, batch):
    x, commands, tau, weights, target = batch
    predicted = model.direct(x[:, None, None], commands[:, None], tau[:, :, None, None])
    endpoints = (predicted*weights[..., None]).sum(2)
    motion = torch.cat([endpoints[:, 0, :3], endpoints[:, 1, 3:]], -1)
    return ((motion-target)/model.motion_scale).square().mean()


@torch.no_grad()
def selection_scores(model, data, units):
    _, state, y, t, valid, beat, endpoints, tape = data
    roll = model(None, state, tape)
    pv, q, w = [query(roll, t[..., i], field) for i, field in enumerate(('pv','q','w'))]
    return trajectory_losses(pv, q, w, y, valid, beat, endpoints, units)[1]


def validation_graph(model, data, units):
    static = [x[:BATCH].clone() for x in data]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        selection_scores(model, static, units)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        score = selection_scores(model, static, units)

    # Captured kernels keep raw pointers, so retain ownership of the normalizers and weights.
    def evaluate(flight, flight_counts, _capture_inputs=(units, model)):
        parts = []
        for start in range(0, len(flight), BATCH):
            size = min(BATCH, len(flight)-start)
            for dst, src in zip(static[1:], data[1:]):
                dst[:size].copy_(src[start:start+size])
            graph.replay()
            parts.append(score[:size].clone())
        value = torch.cat(parts).double()
        return float(torch.zeros_like(flight_counts).scatter_add_(0, flight, value).div(flight_counts).mean())
    return evaluate


def train():
    import pandas as pd
    setup()
    torch.manual_seed(SEED)
    folder = OUT / 'mlp'
    folder.mkdir(exist_ok=False)
    train_data, ids = conditional_data(HISTORY, 'train', TAPES)
    training = examples(train_data)
    n = len(ids)
    assert n == 126237 and ids.flight_id.nunique() == 121
    del train_data
    val_data, vid = conditional_data(HISTORY, 'validation', TAPES)
    assert (len(vid), vid.flight_id.nunique()) == (44524, 39)
    x, cmd, tau, _, targets = training
    features = torch.cat([x, cmd[:, 0]], -1)
    model = VanillaMLP(features.mean(0), features.std(0).clamp_min(.05),
                       targets.mean(0), targets.std(0).clamp_min(.05)).cuda()
    # Same whole-flight rollout selection as the archived reference checkpoints.
    ref = json.loads((HISTORY.parent/'controlled_baselines_20261119_run1/tcn/config.json').read_text())
    units = x.new_tensor(ref['units'])
    assert (ref['seed'], ref['epochs'], ref['batch'], ref['cache'], ref['control_tapes']) == (
        SEED, EPOCHS, BATCH, str(HISTORY), str(TAPES))
    per_epoch = math.ceil(n/BATCH)
    config = dict(model='Vanilla MLP', units=ref['units'], hidden=[64,64], params=sum(p.numel() for p in model.parameters()),
                  dataset=ref['dataset'], cache=str(HISTORY), control_tapes=str(TAPES), seed=SEED, epochs=EPOCHS,
                  batch=BATCH, train_windows=n, train_flights=121, validation_windows=len(vid), validation_flights=39,
                  updates_per_epoch=per_epoch, gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), rollout_dt=.02,
                  code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
                  training='Single next native velocity and rate, absolute state, train-axis-standardized MSE',
                  features='present body velocity, FRD rate, body down vector, 4 ZOH commands, transition duration',
                  native_target_seconds_min_max=[tau.min(0).values.tolist(), tau.max(0).values.tolist()],
                  selection='original shared flight-equal autonomous validation score, identical to frozen baselines',
                  optimizer='AdamW lr .001 cosine to .0001, weight decay .0001, clip 5')
    (folder/'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, fused=True)
    gen = torch.Generator(device='cuda').manual_seed(SEED)
    flight = torch.as_tensor(pd.factorize(vid.flight_id)[0], device='cuda')
    flight_counts = torch.bincount(flight).double()
    evaluate = validation_graph(model, val_data, units)
    best, start = math.inf, time.monotonic()
    before = model.net[0].weight.detach().clone()
    for epoch in range(1, EPOCHS+1):
        perm = torch.randperm(n, device='cuda', generator=gen)
        perm = torch.cat([perm, perm[:per_epoch*BATCH-n]])
        losses = []
        for j, ix in enumerate(perm.split(BATCH)):
            step = (epoch-1)*per_epoch+j+1
            opt.zero_grad(set_to_none=True)
            loss = one_step(model, [v[ix] for v in training])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.param_groups[0]['lr'] = .001*(.55+.45*math.cos(math.pi*(step-1)/(per_epoch*EPOCHS)))
            opt.step()
            losses.append(loss.detach())
        score = evaluate(flight, flight_counts)
        loss = float(torch.stack(losses).mean())
        change = float((model.net[0].weight-before).abs().max())
        if not (math.isfinite(score) and math.isfinite(loss) and change > 0):
            raise ValueError(f'Invalid epoch {epoch} metrics')
        state = dict(model=model.state_dict(), epoch=epoch, updates=step, config=config, validation_score=score)
        atomic_save(state, folder/'last.pt')
        if score < best:
            best = score
            atomic_save(state, folder/'best.pt')
        row = dict(epoch=epoch,updates=step,one_step_train_loss=loss,validation_score=score,
                   best=best,parameter_delta=change,elapsed_s=round(time.monotonic()-start, 1))
        with (folder/'epochs.jsonl').open('a') as f:
            print(json.dumps(row), file=f)
        print(json.dumps(row), flush=True)
    (folder/'completed.json').write_text(json.dumps(dict(updates=step,best=best,epochs=EPOCHS)))


def restore_mlp():
    ck = torch.load(OUT/'mlp/best.pt', map_location='cpu', weights_only=False)
    sd = ck['model']
    model = VanillaMLP(*(sd[k] for k in ('feature_mean','feature_scale','motion_mean','motion_scale'))).cuda().eval()
    model.load_state_dict(sd)
    return model, ck


if __name__ == '__main__':
    train()
