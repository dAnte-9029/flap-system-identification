"""Two-step, true-origin supervision of the unchanged Main V2 transition."""
import time
import numpy as np
import pandas as pd
import torch
from system_identification.training.trajectory_main_v1 import _model_call, trajectory_rollout_loss


def increment_terms(prediction, truth, scales, lag=2):
    """Paired simulator transitions, not an independent future-state head.

    Prediction and truth include the same true t0. The forward already integrates
    the two distinct native dt values. Scales are train RMS vector increments.
    """
    if lag < 1 or any(float(s) <= 0 for s in scales):
        raise ValueError('positive lag and increment scales required')
    terms = []
    for key, scale in zip(('velocity_n', 'angular_velocity_b'), scales, strict=True):
        p, t = getattr(prediction, key), getattr(truth, key)
        if p.shape[1] <= lag or t.shape[1] <= lag:
            raise ValueError('increment exceeds available transitions')
        error = (p[:, lag] - p[:, 0]) - (t[:, lag] - t[:, 0])
        terms.append((error / scale).square().sum(-1).mean())
    return tuple(terms)


def train_increment_stage(model, batch, *, scales, weights, device, epochs, seed,
                          learning_rate, actuator=False, steps=50, batch_size=256,
                          callback=None):
    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    count = len(batch.trajectory.window_ids)
    history = []; started = time.monotonic()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        permutation = torch.randperm(count, generator=generator).numpy()
        sums = np.zeros(4); maxnorm = 0.
        model.train()
        for start in range(0, count, batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            p, t = _model_call(model, batch, indices, use_history=True,
                              rollout_steps=steps, device=torch.device(device))
            original = trajectory_rollout_loss(p, t, objective_steps=steps)
            if actuator:
                drive = (p.flap_frequency_hz[:, 1:steps+1] - t.flap_frequency_hz[:, 1:steps+1]).square().mean()
                original = original + .2 * drive + model.control_regularization_loss(residual_l2=1e-3, tail_gate_l1=1e-2)
            iv, iw = increment_terms(p, t, scales)
            loss = original
            if any(weights):
                loss = loss + weights[0] * iv + weights[1] * iw
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite gradient')
            optimizer.step()
            sums += np.array([float(x.detach()) for x in (loss, original, iv, iw)]) * len(indices)
            maxnorm = max(maxnorm, float(norm))
        elapsed = time.monotonic() - started
        row = dict(epoch=epoch+1, loss=sums[0]/count, original_loss=sums[1]/count,
                   increment_v_normalized=sums[2]/count, increment_omega_normalized=sums[3]/count,
                   increment_weighted_fraction=float((weights[0]*sums[2]+weights[1]*sums[3])/sums[0]),
                   gradient_norm_max=maxnorm, wall_time_s=elapsed,
                   windows_per_second=(epoch+1)*count/elapsed,
                   optimizer_steps=(epoch+1)*int(np.ceil(count/batch_size)),
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0)
        history.append(row)
        if callback:
            callback(model, pd.DataFrame(history))
    return model.cpu().eval(), pd.DataFrame(history)


def gradient_probe(model, batch, indices, *, scales, weights, device, actuator=False):
    """Read-only gradient balance on fixed train rows; no optimizer update."""
    model.to(device).eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    p, t = _model_call(model, batch, indices, use_history=True, rollout_steps=50, device=torch.device(device))
    main = trajectory_rollout_loss(p, t, objective_steps=50)
    if actuator:
        main = main + .2*(p.flap_frequency_hz[:,1:51]-t.flap_frequency_hz[:,1:51]).square().mean() + model.control_regularization_loss(residual_l2=1e-3, tail_gate_l1=1e-2)
    iv, iw = increment_terms(p, t, scales)
    inc = weights[0]*iv + weights[1]*iw
    def norm(loss):
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((v.square().sum() for v in values if v is not None), loss.new_zeros(()))))
    result = dict(original_loss=float(main), increment_v=float(iv), increment_omega=float(iw),
                  weighted_increment=float(inc), original_gradient_norm=norm(main), increment_gradient_norm=norm(inc))
    result['gradient_ratio'] = result['increment_gradient_norm']/max(result['original_gradient_norm'],1e-12)
    model.cpu().eval()
    return result
