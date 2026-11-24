"""Flight-equal endpoint scores on common native targets for the CASE model table."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'transformer_slowfast_v1'))
from latent import LatentTransformerHarmonics
from common import HISTORY, TAPES, conditional_data, frozen, setup
from model import geodesic, trajectory_losses
from stage_query import query

OUT = HISTORY.parent / 'case_model_comparison_20261124'
HORIZONS = (.5, 1., 1.5)
MODELS = ('mlp', 'gru', 'lstm', 'tcn', 'transformer', 'v2')
METRICS = ('position_m', 'velocity_m_s', 'attitude_deg', 'rate_rad_s')


def native_endpoints(data, ids):
    t, valid = data[3:5]
    tolerance = t.new_tensor(np.where(ids.sampling_group == 'main_50hz', .035, .105))
    indices = []
    for h in HORIZONS:
        diff = (t-h).abs().masked_fill(~valid, float('inf'))
        distance, index = diff.min(1)
        if not (distance <= tolerance[:, None]).all():
            raise ValueError('requested endpoints do not all have native support')
        indices.append(index)
    index = torch.stack(indices, 1)
    times = t.gather(1, index)
    truth = [data[2][..., channel].gather(1, index[..., i, None].expand(-1, -1, size))
             for i, channel, size in [(0, slice(0, 6), 6), (1, slice(6, 10), 4), (2, slice(10, 13), 3)]]
    return times, truth


def restore(name):
    if name in ('gru', 'lstm', 'tcn', 'transformer'):
        return frozen(name)
    if name == 'mlp':
        from vanilla import restore_mlp
        return restore_mlp()
    folder = HISTORY.parent / 'transformer_slowfast_latent_v2_20261124/model'
    ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    state = ck['model']
    model = LatentTransformerHarmonics(state['mean'], state['scale']).cuda().eval()
    model.load_state_dict(state)
    return model, ck


@torch.no_grad()
def evaluate(names):
    setup()
    data, ids = conditional_data(HISTORY, 'validation', TAPES)
    assert ids.window_id.is_unique and (len(ids), ids.flight_id.nunique()) == (44524, 39)
    times, truth = native_endpoints(data, ids)
    steps = math.ceil(times.max().item() / .02)
    for name in names:
        model, ck = restore(name)
        errors, selection = [], []
        if name == 'mlp':
            units = data[1].new_tensor(ck['config']['units'])
        for start in range(0, len(ids), 1024):
            size = min(1024, len(ids)-start)
            sl = slice(start, start+size)
            if name == 'mlp' and size < 1024:
                # Match the training evaluator's fixed CUDA-graph batch shape and padding.
                sl = torch.cat((torch.arange(start, len(ids), device=times.device),
                                torch.arange(start-1024+size, start, device=times.device)))
            tape = data[-1][sl] if name == 'mlp' else data[-1][sl, :steps]
            roll = model(data[0][sl], data[1][sl], tape)
            if name == 'mlp':
                native = [query(roll, data[3][sl, :, i], kind) for i, kind in enumerate(('pv','q','w'))]
                scores = trajectory_losses(*native, data[2][sl], data[4][sl], data[5][sl], data[6][sl], units)[1]
                selection.append(scores[:size].cpu().numpy())
            pv, q, w = [query(roll, times[sl, :, i], kind) for i, kind in enumerate(('pv','q','w'))]
            diff = pv-truth[0][sl]
            err = torch.stack([diff[..., :3].square().sum(-1), diff[..., 3:].square().sum(-1),
                               geodesic(q, truth[1][sl]).square(),
                               (w-truth[2][sl]).square().sum(-1)], -1)
            errors.append(err[:size].cpu().numpy())
        errors = np.concatenate(errors).astype(np.float64)
        assert np.isfinite(errors).all()
        if selection:
            replayed = pd.Series(np.concatenate(selection).astype('float64')).groupby(ids.flight_id.to_numpy()).mean().mean()
            print('checkpoint score, padded eager replay', ck['validation_score'], replayed, flush=True)
            np.testing.assert_allclose(replayed, ck['validation_score'], rtol=2e-5)
        records = []
        for i, horizon in enumerate(HORIZONS):
            for j, metric in enumerate(METRICS):
                values = errors[:, i, j]
                if metric == 'attitude_deg':
                    values = values * (180 / np.pi)**2
                groups = pd.Series(values).groupby(ids.flight_id.to_numpy()).agg(['mean','size'])
                groups.columns = ['mse','windows']
                records.append(groups.rename_axis('flight_id').reset_index().assign(
                    model=name, horizon_s=horizon, metric=metric))
        table = pd.concat(records, ignore_index=True)
        table.to_csv(OUT / f'{name}_flights.csv', index=False)
        print(f'{name} evaluated, epoch {ck["epoch"]}, {len(ids)} windows', flush=True)


def write_latex(summary):
    names = ['Vanilla MLP', 'GRU', 'LSTM', 'TCN', 'Transformer', 'Ours (V2)']
    units = [r'$p$ (m)', r'$v$ (m/s)', r'$R$ ($^\circ$)', r'$\omega$ (rad/s)']
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{Flight-equal endpoint RMSE on 39 validation flights (44,524 windows). '
             r'Bold indicates the lowest error in each column.}',
             r'\label{tab:flight-endpoints}', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lrrrrrrrrrrrr}', r'\hline',
             r' & \multicolumn{4}{c}{0.5 s} & \multicolumn{4}{c}{1.0 s} & \multicolumn{4}{c}{1.5 s} \\',
             r'\cline{2-5}\cline{6-9}\cline{10-13}', 'Model & '+' & '.join(units*3)+r' \\', r'\hline']
    best = summary.min()
    for name, (_, row) in zip(names, summary.iterrows()):
        cells = []
        for col, val in row.items():
            fmt = '.2f' if 'attitude' in col else '.3f'
            number = format(val, fmt)
            cells.append(r'\textbf{'+number+'}' if val == best[col] else number)
        lines.append(name+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\hline', r'\end{tabular}', r'\end{table*}'])
    (OUT / 'model_comparison.tex').write_text('\n'.join(lines)+'\n')


def summarize():
    detail = pd.concat([pd.read_csv(OUT / f'{name}_flights.csv') for name in MODELS])
    keys = ['horizon_s','metric','flight_id']
    counts = detail.pivot(index=keys, columns='model', values='windows')
    assert len(counts) == 39*4*len(HORIZONS) and counts.eq(counts.mlp, axis=0).all().all()
    assert counts.mlp.groupby(level=keys[:2]).sum().eq(44524).all()
    result = detail.groupby(['model','horizon_s','metric']).mse.mean().pow(.5).unstack()
    archived = pd.read_csv(HISTORY.parent / 'transformer_slowfast_direct_v3_20261124/all_models_comparison.csv')
    old = archived[archived.model.isin(MODELS) & archived.horizon_s.isin((.5, 1.))].set_index(['model','horizon_s'])
    assert len(old) == 10
    np.testing.assert_allclose(result.loc[old.index, list(METRICS)], old[list(METRICS)], rtol=2e-5, atol=2e-5)
    summary = result.unstack('horizon_s').swaplevel(axis=1)
    summary = summary.loc[list(MODELS), pd.MultiIndex.from_product([HORIZONS,METRICS])]
    summary.columns = [f'{h:.1f}s_{metric}' for h, metric in summary.columns]
    summary.to_csv(OUT / 'model_comparison.csv')
    write_latex(summary)
    print(summary.round(4).to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('members', choices=['existing','mlp','table'])
    option = p.parse_args().members
    if option == 'table':
        summarize()
    else:
        evaluate(MODELS[1:] if option == 'existing' else ['mlp'])
