"""PACLD-only objectives, fixed train calibration, resumable bounded training."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from system_identification.models.pacld import PACLD, Config
from system_identification.models.pahbd import geometry as so3
from system_identification.training.pahbd import FlightBalancedSampler

SCHEMA = 'pacld_training_v2'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, config, data_identity, sampler, schedule, lambda_macro):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    from .pacld_compatibility import current_contract, optimization_rule
    contract = current_contract()
    state = np.random.get_state()
    rng = dict(torch=torch.get_rng_state(), python=random.getstate(),
               numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
               cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    bundle = dict(schema=SCHEMA, source_contract=contract, optimization_rule=optimization_rule(config),
                  sampler_identity=sampler.identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                  config=config, model_config=asdict(model.config), data_identity=data_identity,
                  seed=config['seed'], rng=rng, sampler=sampler.state_dict(), schedule=schedule.copy(),
                  step=schedule['step'], lambda_macro=float(lambda_macro),
                  dtype=str(next(model.parameters()).dtype).split('.')[-1],
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_checkpoint(path, config, data_identity, device='cpu', restore_rng=True, sampler_index=None):
    b = torch.load(path, map_location='cpu', weights_only=True)
    if b.get('schema') != SCHEMA:
        raise ValueError('PACLD v2 source-verified checkpoint required; legacy/v1 has no verified source/clipping contract')
    from .pacld_compatibility import check_source_contract, scientific_config, optimization_rule
    current = check_source_contract(b.get('source_contract',{}))
    if b.get('optimization_rule') != optimization_rule(config):
        raise ValueError('checkpoint optimizer/clipping rule mismatch')
    if b.get('sampler_identity') != b['sampler']['identity'] or b['sampler']['seed'] != config['seed']:
        raise ValueError('checkpoint sampler identity mismatch')
    if sampler_index is not None:
        expected_sampler = FlightBalancedSampler(sampler_index,config['seed'])
        expected_sampler.load_state_dict(b['sampler'])
    if scientific_config(b['config']) != scientific_config(config) or b['data_identity'] != data_identity:
        raise ValueError('checkpoint configuration/data identity mismatch')
    if (json.loads(json.dumps(asdict(Config(**b['model_config'])))) !=
            json.loads(json.dumps(asdict(Config(**config['model'])))) or b['dtype'] != config['training']['dtype']):
        raise ValueError('checkpoint embedded model/dtype configuration mismatch')
    if any(group.get(key) != value for group in b['optimizer']['param_groups']
           for key,value in config['optimizer'].items()):
        raise ValueError('checkpoint embedded Adam configuration mismatch')
    if not math.isfinite(b['lambda_macro']) or b['lambda_macro'] < 0:
        raise ValueError('invalid checkpoint lambda_macro')
    model = PACLD(Config(**b['model_config'])).to(device=device, dtype=getattr(torch, b['dtype']))
    model.load_state_dict(b['model'])
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    optimizer.load_state_dict(b['optimizer'])
    if restore_rng:
        torch.set_rng_state(b['rng']['torch'])
        random.setstate(b['rng']['python'])
        ns = b['rng']['numpy']; np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
        if b['rng']['cuda'] is not None and torch.cuda.is_available():
            if len(b['rng']['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA device topology changed for exact RNG restore')
            torch.cuda.set_rng_state_all(b['rng']['cuda'])
    b['resume_compatibility'] = dict(saved_git=b['git_revision'],
        current_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_contract=current, saved_operational_config={k:b['config'].get(k) for k in ('training','data_cache')},
        current_operational_config={k:config.get(k) for k in ('training','data_cache')})
    return model, optimizer, b


SOURCES = ('vehicle_local_position', 'vehicle_attitude', 'vehicle_angular_velocity')


def valid_pairs(window):
    """Validate existing pair membership; never synthesize pairs or relax phase QC."""
    targets = window.targets
    pairs = []; rejected = 0; seen = set()
    phases = window.metadata.get('target_phase_reference')
    for pair in window.metadata.get('cycle_pairs', ()):
        if len(pair) != 2:
            rejected += 1; continue
        i, j = pair
        if not (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)) and 0 <= i < j < len(targets)):
            rejected += 1; continue
        a, b = targets[i], targets[j]
        ok = (a.source == b.source and a.source in SOURCES and a.frame == b.frame == 'NED/FRD/Hamilton-wxyz'
              and math.isfinite(a.measurement) and math.isfinite(b.measurement)
              and 0 < a.measurement < b.measurement and a.measurement <= a.availability
              and b.measurement <= b.availability and (i, j) not in seen)
        if ok:
            weight = a.mask[:12]*b.mask[:12]*torch.minimum(a.quality[:12], b.quality[:12])
            ok = bool(torch.isfinite(weight).all() and (weight > 0).any())
        if ok and phases is not None:
            ok = len(phases) == len(targets) and phases[i] is not None and phases[j] is not None
            if ok:
                eligible = [(k, abs(phases[k]-phases[i]-2*math.pi)) for k in range(i+1, len(targets))
                            if targets[k].source == a.source and phases[k] is not None and math.isfinite(phases[k])]
                neighbors = [abs(phases[k]-phases[j]) for k in range(len(targets))
                             if k != j and targets[k].source == a.source and phases[k] is not None and math.isfinite(phases[k])]
                ok = (math.isfinite(phases[i]) and bool(eligible) and bool(neighbors)
                      and min(eligible, key=lambda x: x[1])[0] == j
                      and abs(phases[j]-phases[i]-2*math.pi) <= min(neighbors)/2)
        if ok:
            pairs.append((i, j)); seen.add((i, j))
        else:
            rejected += 1
    return pairs, rejected


def state_loss(predictions, targets, times):
    """Pure PA-HBD state reduction; no predictive variance or default cycle branch."""
    lookup = dict(zip(times, predictions)); families = {}
    for e in targets:
        if not math.isfinite(e.measurement) or e.measurement <= 0 or e.source not in SOURCES:
            raise ValueError('future native physical-state targets required')
        weights = e.mask[:12]*e.quality[:12]
        if not bool((weights > 0).any()):
            continue
        r = so3.state_residual(lookup[e.measurement].x, e.value[:13])
        score = (r.square()/e.variance[:12].clamp_min(1e-4)*weights).sum()/weights.sum().clamp_min(1)
        family = 'gyro' if e.source == 'vehicle_angular_velocity' else 'ekf'
        families.setdefault(family, []).append(score)
    if not families:
        raise ValueError('no valid physical target')
    return torch.stack([torch.stack(v).mean() for v in families.values()]).mean()


def transition_residual(pa, pb, ya, yb):
    pred_rotation = so3.multiply(so3.conjugate(pa[6:10]), pb[6:10])
    true_rotation = so3.multiply(so3.conjugate(ya[6:10]), yb[6:10])
    return torch.cat(((pb[:6]-pa[:6])-(yb[:6]-ya[:6]),
                      so3.residual(true_rotation, pred_rotation),
                      (pb[10:]-pa[10:])-(yb[10:]-ya[10:])))


def macro_loss(predictions, targets, times, pairs):
    """Equal sources, equal valid pairs within source; None means no supervision."""
    if not pairs:
        return None, {}
    lookup = dict(zip(times, predictions)); sources = {}
    for i, j in pairs:
        a, b = targets[i], targets[j]
        r = transition_residual(lookup[a.measurement].x, lookup[b.measurement].x, a.value[:13], b.value[:13])
        weight = a.mask[:12]*b.mask[:12]*torch.minimum(a.quality[:12], b.quality[:12])
        noise = (a.variance[:12]+b.variance[:12]).clamp_min(1e-4)
        sources.setdefault(a.source, []).append((r.square()/noise*weight).sum()/weight.sum().clamp_min(1))
    reduced = {source: torch.stack(values).mean() for source, values in sources.items()}
    return torch.stack(list(reduced.values())).mean(), reduced


def window_objective(model, window, lambda_macro, diagnostic_base=False, timings=None):
    if not math.isfinite(lambda_macro) or lambda_macro < 0:
        raise ValueError('nonnegative finite fixed lambda required')
    pairs, rejected = valid_pairs(window)
    times = sorted({e.measurement for e in window.targets})
    mark = synchronized_time(model) if timings is not None else None
    belief = model.encode(window)  # Exactly once; no detach between the branches.
    if timings is not None:
        now = synchronized_time(model); timings['encode_seconds'] = now-mark; mark = now
    full = model.flow.rollout(belief, times, window.controls, mode='FULL')
    if timings is not None:
        now = synchronized_time(model); timings['full_forward_seconds'] = now-mark; mark = now
    base = model.flow.rollout(belief, times, window.controls, mode='BASE') if pairs or diagnostic_base else None
    if timings is not None:
        now = synchronized_time(model); timings['base_forward_seconds'] = now-mark; mark = now
    full_loss = state_loss(full, window.targets, times)
    macro, sources = macro_loss(base, window.targets, times, pairs)
    total = full_loss if macro is None else full_loss+lambda_macro*macro
    if timings is not None: timings['loss_seconds'] = synchronized_time(model)-mark
    return dict(total=total, full=full_loss, macro=macro, macro_sources=sources,
                pairs=pairs, rejected_pairs=rejected, belief=belief,
                predictions={'FULL': full, 'BASE': base}, times=times)


def gradient_norms(model, terms):
    """Separate norms on the SAME graph; diagnostics never set weights or gradients."""
    named = list(model.named_parameters())
    result = {}
    for label in ('full', 'macro'):
        loss = terms[label]
        if loss is None:
            result[label] = None
            continue
        grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        groups = {}
        for (name, _), grad in zip(named, grads):
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f'nonfinite {label} gradient: {name}')
            group = '.'.join(name.split('.')[:3]) if name.startswith('flow.dynamics') else '.'.join(name.split('.')[:2])
            groups[group] = groups.get(group, 0)+grad.square().sum()
        result[label] = dict(total=float(torch.stack(list(groups.values())).sum().sqrt()),
                             modules={k: float(v.sqrt()) for k, v in groups.items()})
    return result


def synchronized_time(model):
    device = next(model.parameters()).device
    if device.type == 'cuda': torch.cuda.synchronize(device)
    return time.perf_counter()


def diagnostics_due(completed_steps, interval):
    """Global steps 1, 101, 201 for interval=100; resume never restarts the phase."""
    if interval < 1 or completed_steps < 0: raise ValueError('invalid diagnostic schedule')
    return completed_steps % interval == 0


def check_states(result):
    states = [result['belief']]
    for predictions in result['predictions'].values():
        if predictions is not None: states.extend(predictions)
    fields = ('x','z_s','c','f','phi','actuator','variance','phase_covariance','phase_quality')
    values = [getattr(s, field).detach().reshape(-1) for s in states for field in fields
              if getattr(s, field) is not None]
    if not torch.isfinite(torch.cat(values)).all():
        raise FloatingPointError('nonfinite autonomous state; update aborted')
    q = torch.stack([s.x[6:10].detach() for s in states])
    if (q.norm(dim=-1)-1).abs().max() > 1e-6:
        raise FloatingPointError('nonunit attitude; update aborted')


def update(model, window, optimizer, lambda_macro, diagnostics=True, max_grad_norm=10.0):
    if window.metadata['split'] != 'train':
        raise ValueError('optimizer updates require train split')
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError('max_grad_norm must be positive finite or None (disabled)')
    model.train(); optimizer.zero_grad(set_to_none=True)
    timing = {}; start = synchronized_time(model)
    result = window_objective(model, window, lambda_macro, timings=timing)
    mark = synchronized_time(model)
    if not torch.isfinite(result['total']):
        raise FloatingPointError('nonfinite objective; update aborted')
    check_states(result)
    now = synchronized_time(model); timing['state_check_seconds'] = now-mark; mark = now
    norms = gradient_norms(model, result) if diagnostics else None
    now = synchronized_time(model); timing['gradient_diagnostic_seconds'] = now-mark; mark = now
    result['total'].backward()
    now = synchronized_time(model); timing['backward_seconds'] = now-mark; mark = now
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters or not torch.isfinite(torch.cat([p.grad.reshape(-1) for p in parameters])).all():
        raise FloatingPointError('nonfinite/missing gradient; update aborted')
    if max_grad_norm is None:
        pre_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]),2)
        coefficient = 1.
    else:
        pre_norm = torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,norm_type=2.,error_if_nonfinite=True)
        coefficient = min(1.,max_grad_norm/(float(pre_norm)+1e-6))
    if not torch.isfinite(pre_norm): raise FloatingPointError('nonfinite gradient norm; update aborted')
    now = synchronized_time(model); timing['gradient_check_clip_seconds'] = now-mark; mark = now
    optimizer.step()
    now = synchronized_time(model); timing['optimizer_seconds'] = now-mark; mark = now
    if not torch.isfinite(torch.cat([p.detach().reshape(-1) for p in model.parameters()])).all():
        raise FloatingPointError('nonfinite parameters after optimizer; stop without checkpointing')
    timing['parameter_check_seconds'] = synchronized_time(model)-mark
    timing['update_seconds'] = synchronized_time(model)-start
    return dict(full=float(result['full'].detach()), macro=None if result['macro'] is None else float(result['macro'].detach()),
                total=float(result['total'].detach()), pairs=len(result['pairs']), rejected_pairs=result['rejected_pairs'],
                gradient_norms=norms, lambda_macro=lambda_macro, max_grad_norm=max_grad_norm,
                pre_clip_grad_norm=float(pre_norm), gradient_clipped=coefficient < 1., clip_coefficient=coefficient,
                timing=timing, window_id=window.metadata['window_id'], flight_id=window.metadata['flight_id'],
                horizon_us=window.metadata.get('horizon_us'), sampling_group=window.metadata.get('sampling_group'))


def calibration_positions(windows, coverage, seed=17, count=64):
    """Round-robin horizon/group, shuffled flights within strata, unique windows."""
    if set(windows.index.split) != {'train'} or set(coverage.split) != {'train'}:
        raise ValueError('calibration is train-only')
    qualified = coverage.loc[coverage[list(SOURCES)].sum(axis=1) > 0, 'window_id']
    frame = windows.index[windows.index.window_id.isin(qualified)]
    if len(frame) == 0:
        raise ValueError('blocked: no valid macro calibration windows')
    rng = np.random.default_rng(seed)
    strata = {}
    for key, g in frame.groupby(['horizon_us', 'sampling_group'], sort=True):
        flights = list(rng.permutation(sorted(g.flight_id.unique())))
        queues = {f: list(rng.permutation(g.index[g.flight_id == f])) for f in flights}
        strata[key] = (flights, queues)
    chosen = []
    while len(chosen) < min(count, len(frame)):
        for key in sorted(strata):
            flights, queues = strata[key]
            if not flights:
                continue
            f = flights.pop(0); chosen.append(int(queues[f].pop()))
            if queues[f]: flights.append(f)
            if len(chosen) == min(count, len(frame)): break
    return chosen


@torch.no_grad()
def calibrate(model, windows, positions, config, output):
    from system_identification.data.pacld import move_window
    records = []; model.eval()
    for i in positions:
        w = move_window(windows[i], next(model.parameters()).device, next(model.parameters()).dtype)
        if w.metadata['split'] != 'train': raise ValueError('train-only calibration')
        result = window_objective(model, w, 0.)
        if result['macro'] is None: raise ValueError('census selected a window with no valid pairs')
        records.append(dict(window_id=w.metadata['window_id'], flight_id=w.metadata['flight_id'],
                            horizon_us=w.metadata['horizon_us'], sampling_group=w.metadata['sampling_group'],
                            full=float(result['full']), macro=float(result['macro']), pairs=len(result['pairs'])))
        if len(records) % 8 == 0: print(f'calibration {len(records)}/{len(positions)}', flush=True)
    if not records or len({r['flight_id'] for r in records}) < 2 or len({r['horizon_us'] for r in records}) < 2:
        raise ValueError('blocked: insufficient flight/horizon coverage for calibration')
    med_full = float(np.median([r['full'] for r in records])); med_macro = float(np.median([r['macro'] for r in records]))
    if not all(math.isfinite(v) and v >= 0 for v in (med_full, med_macro)):
        raise FloatingPointError('invalid calibration distribution')
    weight = float(np.clip(.1*med_full/max(med_macro, 1e-8), .01, 1.))
    result = dict(schema='pacld_lambda_v1', seed=config['seed'], model_config=asdict(model.config),
                  loss_config=config['loss'], data_identity=windows.identity(), lambda_macro=weight,
                  median_full=med_full, median_macro=med_macro, windows=len(records),
                  distributions={k: dict(zip(['min','q25','median','q75','max'],
                                             np.quantile([r[k] for r in records], [0,.25,.5,.75,1]).tolist())) for k in ('full','macro')},
                  records=records, git_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(result, f, indent=2)
    return result


def calibrated_weight(path, config, train):
    result = json.loads(Path(path).read_text())
    if (result.get('schema') != 'pacld_lambda_v1' or result['seed'] != config['seed']
            or result['model_config'] != json.loads(json.dumps(asdict(Config(**config['model']))))
            or result['loss_config'] != config['loss'] or result['data_identity'] != train.identity()):
        raise ValueError('calibration model/loss/train identity mismatch')
    value = result['lambda_macro']
    if not math.isfinite(value) or value < 0: raise ValueError('invalid lambda')
    return value


def fit(model, train, validation, optimizer, config, data_identity, lambda_macro, steps, output, resume=None):
    """Explicit total-step budget; only FULL validation score selects checkpoints."""
    from system_identification.data.pacld import move_window
    from system_identification.evaluation.pacld import evaluate
    if steps < 1: raise ValueError('positive explicit total steps required')
    if set(train.index.split) != {'train'} or set(validation.index.split) != {'validation'}:
        raise ValueError('train/validation roles required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    from .pacld_compatibility import current_contract
    contract=current_contract()
    (output/'run_contract.json').write_text(json.dumps(dict(config=config,data_identity=data_identity,
        source_contract=contract,lambda_macro=lambda_macro,
        resume_compatibility=resume.get('resume_compatibility') if resume else None),indent=2))
    sampler=FlightBalancedSampler(train.index,config['seed'])
    schedule=dict(step=0,best_score=float('inf'),best_checkpoint=None)
    if resume:
        if resume['lambda_macro'] != lambda_macro: raise ValueError('fixed lambda changed')
        sampler.load_state_dict(resume['sampler']); schedule=resume['schedule'].copy()
    if steps <= schedule['step']: raise ValueError('steps must exceed resumed step')
    interval=config['training']['validation_interval']
    diagnostic_interval=config['training']['gradient_diagnostics_interval']
    if min(interval,diagnostic_interval)<1: raise ValueError('positive intervals required')
    device=next(model.parameters()).device; dtype=next(model.parameters()).dtype
    while schedule['step'] < steps:
        i=sampler.draw(); begin=time.perf_counter()
        w=move_window(train[i],device,dtype)
        if device.type=='cuda': torch.cuda.synchronize(device)
        load_seconds=time.perf_counter()-begin; begin=time.perf_counter()
        try:
            draw=update(model,w,optimizer,lambda_macro,
                        diagnostics=diagnostics_due(schedule['step'],diagnostic_interval),
                        max_grad_norm=config['training']['max_grad_norm'])
        except (FloatingPointError, ValueError, RuntimeError) as error:
            (output/'failure.json').write_text(json.dumps(dict(completed_steps=schedule['step'],
                attempted_window=w.metadata['window_id'],error=str(error)),indent=2))
            raise
        if device.type=='cuda': torch.cuda.synchronize(device)
        schedule['step']+=1
        draw.update(step=schedule['step'],load_seconds=load_seconds,compute_seconds=time.perf_counter()-begin)
        if hasattr(getattr(train,'dataset',None),'cache_stats'): draw['data_cache']=train.dataset.cache_stats()
        with (output/'training.jsonl').open('a') as f: f.write(json.dumps(draw)+'\n')
        if schedule['step']%interval==0 or schedule['step']==steps:
            validation_result=evaluate(model,validation,config,diagnostics=False)
            score=validation_result['selection_score']
            if score is None or not math.isfinite(score): raise FloatingPointError('nonfinite validation score')
            name=f"step_{schedule['step']:08d}.pt"
            if score<schedule['best_score']:
                schedule.update(best_score=score,best_checkpoint=str((output/name).resolve()))
            save_checkpoint(output/name,model,optimizer,config,data_identity,sampler,schedule,lambda_macro)
            (output/f"validation_{schedule['step']:08d}.json").write_text(json.dumps(validation_result,indent=2))
            (output/'best.json').write_text(json.dumps(schedule,indent=2))
        if schedule['step']==1 or schedule['step']%config['training'].get('terminal_log_interval',100)==0 or schedule['step']==steps:
            print(json.dumps({k:draw[k] for k in ('step','full','macro','total','pre_clip_grad_norm','compute_seconds')}),flush=True)
    return schedule
