"""Run pinned NeuralProphet model math on CPU, avoiding its optional Lightning and plotting frontend."""
import ast
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np
import torch
from torch import nn


ROOT = Path('neural_prophet/neuralprophet')
PERIODS = {'monthly': 30., 'yearly': 365.}
ORDER = 5
N_TRAIN = 4800
N_ALL = 6000
KNOTS = np.arange(.1, .9, .1)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def original_definitions(file, names, scope):
    path = ROOT / file
    tree = ast.parse(path.read_text())
    selected = {x.name: x for x in ast.walk(tree) if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names}
    assert set(selected) == set(names)
    module = ast.Module(body=[*ast.parse('from __future__ import annotations').body,
                              *(selected[k] for k in names)], type_ignores=[])
    exec(compile(module, str(path), 'exec'), scope)


def load_author():
    scope = dict(torch=torch, nn=nn, np=np, abstractmethod=abstractmethod, OrderedDict=OrderedDict)
    for file, names in [
        ('utils_torch.py', ['init_parameter']),
        ('utils.py', ['config_seasonality_to_model_dims']),
        ('components/base.py', ['BaseComponent']),
        ('components/trend/base.py', ['Trend']),
        ('components/seasonality/base.py', ['Seasonality']),
        ('components/trend/piecewise_linear.py', ['PiecewiseLinearTrend','GlobalPiecewiseLinearTrend']),
        ('components/seasonality/fourier.py', ['FourierSeasonality','GlobalFourierSeasonality']),
        ('time_dataset.py', ['compute_fourier_features']),
    ]:
        original_definitions(file, names, scope)
    scope['utils'] = SimpleNamespace(config_seasonality_to_model_dims=scope['config_seasonality_to_model_dims'])
    return scope


AUTHOR = load_author()


def heads():
    common = dict(id_list=['s'], quantiles=[.5], n_forecasts=1, device='cpu')
    trend_config = SimpleNamespace(n_changepoints=len(KNOTS), changepoints=KNOTS, growth='linear', trend_reg=0)
    trend = AUTHOR['GlobalPiecewiseLinearTrend'](trend_config, num_trends_modelled=1, **common).double()
    season_config = SimpleNamespace(computation='fourier', periods={k: SimpleNamespace(resolution=ORDER) for k in PERIODS})
    season = AUTHOR['GlobalFourierSeasonality'](season_config, num_seasonalities_modelled=1,
                                                num_seasonalities_modelled_dict={k: 1 for k in PERIODS}, **common).double()
    return trend, season


def harmonic_numpy(days, period):
    # Independent reference uses complex powers of a unit phasor.
    z = np.exp(2j * np.pi * days / period)
    powers = np.stack([z ** k for k in range(1, ORDER + 1)], axis=1)
    return np.concatenate([powers.imag, powers.real], axis=1)


def features(days):
    return {k: AUTHOR['compute_fourier_features'](torch.from_numpy(days), p, ORDER).double().unsqueeze(1)
            for k,p in PERIODS.items()}


def branches(trend, season, time, x):
    slow = trend(time, None).flatten()
    periodic = {k: season.compute_fourier(v, k).flatten() for k,v in x.items()}
    return slow, periodic


def measure():
    stage = os.environ['REPRO_STAGE']
    days = np.arange(N_ALL, dtype=np.float64)
    normalized = days / (N_TRAIN - 1.)
    time = torch.from_numpy(normalized)[:,None]
    x = features(days)
    exact = {k: harmonic_numpy(days,v) for k,v in PERIODS.items()}
    if stage == 'preflight':
        return {'rows': (float(len(days)), 'days'), 'rank': (float(np.linalg.matrix_rank(np.column_stack([np.ones(N_TRAIN),normalized[:N_TRAIN],*[(normalized[:N_TRAIN]-k).clip(min=0) for k in KNOTS], *[v[:N_TRAIN] for v in exact.values()]]))), 'columns')}
    if stage == 'modules':
        torch.manual_seed(771)
        trend, season = heads()
        slow, periodic = branches(trend, season, time, x)
        with torch.no_grad():
            k0=trend.trend_k0.item()
            slopes=trend.trend_deltas.numpy().reshape(-1)+k0
            boundaries=trend.trend_changepoints_t.numpy()[1:]
            piece=trend.bias.item()+slopes[0]*normalized
            for cp,delta in zip(boundaries,np.diff(slopes)):
                piece += delta*np.maximum(normalized-cp,0)
            harm=max(float(np.max(np.abs(periodic[k].numpy()-exact[k]@season.season_params[k].numpy().reshape(-1)))) for k in x)
            # Evaluate actual monthly and annual time-offset invariance separately.
            shift=max(float((season.compute_fourier(features(days+p)[k],k).flatten()-periodic[k]).abs().max()) for k,p in PERIODS.items())
            return {'trend_formula':(float(np.max(np.abs(piece-slow.numpy()))),'normalized_y'),
                    'season_formula':(harm,'normalized_y'),'period_invariance':(shift,'normalized_y')}
    fits=[]
    for seed in range(5):
        rng=np.random.default_rng(700+seed)
        cp=rng.integers(1450,3201)
        slow_true=np.minimum(days/cp,(N_ALL-days)/(N_ALL-cp))
        comps=[slow_true, *[v@rng.random(2*ORDER) for v in exact.values()]]
        comps=np.stack([(s-s[:N_TRAIN].min())/np.ptp(s[:N_TRAIN]) for s in comps])
        clean=comps.sum(0)
        noisy=clean+rng.normal(0,.02,N_ALL)
        scale=np.ptp(noisy[:N_TRAIN])
        target=torch.from_numpy((noisy-noisy[:N_TRAIN].min())/scale)
        truth=comps/scale
        torch.manual_seed(700+seed)
        trend,season=heads()
        optimizer=torch.optim.LBFGS([*trend.parameters(),*season.parameters()],lr=1,max_iter=450,
            tolerance_grad=1e-9,tolerance_change=1e-15,line_search_fn='strong_wolfe')
        xs={k:v[:N_TRAIN] for k,v in x.items()}
        def objective():
            optimizer.zero_grad()
            s,p=branches(trend,season,time[:N_TRAIN],xs)
            loss=(s+sum(p.values())-target[:N_TRAIN]).square().mean()
            loss.backward()
            return loss
        optimizer.step(objective)
        with torch.no_grad():
            s,p=branches(trend,season,time,x)
            pred=np.stack([s.numpy(),*(v.numpy() for v in p.values())])
        held=slice(N_TRAIN,None)
        e=(pred-truth)[:,held]
        rmse=np.sqrt(np.mean((e-e.mean(1,keepdims=True))**2,axis=1))
        # Independent least-squares optimum on same noisy training samples.
        bounds=trend.trend_changepoints_t.numpy()[1:]
        design=np.column_stack([np.ones(N_ALL),normalized,*[np.maximum(normalized-c,0) for c in bounds],*exact.values()])
        coeff=np.linalg.lstsq(design[:N_TRAIN],target.numpy()[:N_TRAIN],rcond=None)[0]
        fit_delta=np.max(np.abs((pred.sum(0)-design@coeff)[held]))
        fits.append(dict(seed=700+seed,changepoint=int(cp),centered_component_rmse=rmse.tolist(),
                         max_held_disagreement_ols=float(fit_delta), iterations=int(optimizer.state[trend.bias]['n_iter']),
                         uncentered_total_held_rmse=float(np.sqrt(np.mean((pred.sum(0)[held]-target.numpy()[held])**2)))))
    Path('synthetic_runs.json').write_text(json.dumps(fits,indent=2))
    return {'slow':(max(row['centered_component_rmse'][0] for row in fits),'normalized_y'),
            'season':(max(max(row['centered_component_rmse'][1:]) for row in fits),'normalized_y'),
            'solved':(max(row['max_held_disagreement_ols'] for row in fits),'normalized_y')}


if __name__=='__main__':
    values=measure()
    Path(os.environ['REPRO_METRICS']).write_text(json.dumps({k:{'value':v,'unit':u} for k,(v,u) in values.items()}))
    print(values,flush=True)
