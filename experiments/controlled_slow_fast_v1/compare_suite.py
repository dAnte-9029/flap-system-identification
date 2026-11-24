"""Score the five completed models with identical data, native targets and weighting."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from compare import evaluate
from fit import conditional_data


MODELS = ('gru', 'lstm', 'tcn', 'transformer', 'slowfast')
MATCH = ('dataset', 'task', 'cache', 'control_tapes', 'frequency_coefficients', 'seed',
         'epochs', 'batch', 'updates_per_epoch', 'history_s', 'rollout_dt', 'training_s',
         'units', 'optimizer', 'lr', 'weight_decay', 'schedule', 'gradient_clip',
         'tf32', 'sample_policy', 'train_windows', 'train_flights', 'validation_windows',
         'validation_flights')


def compare_suite(reference, output, gpu):
    locations = {m: reference if m in ('gru','slowfast') else output for m in MODELS}
    configs = {m: json.loads((root/m/'config.json').read_text()) for m,root in locations.items()}
    for name,cfg in configs.items():
        mismatched = [key for key in MATCH if cfg[key] != configs['gru'][key]]
        directory = locations[name]/name
        finished = json.loads((directory/'completed.json').read_text())
        expected = cfg['epochs']*cfg['updates_per_epoch']
        if mismatched or cfg['model'] != name or finished['updates'] != expected:
            raise ValueError(f'{name}: unmatched contract {mismatched} or incomplete training')
        if name != 'slowfast' and (cfg['aux_weight'] != 0 or cfg['macro_dt'] != .02):
            raise ValueError(f'{name}: unmatched baseline clock or auxiliary loss')
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    data,ids = conditional_data(Path(configs['gru']['cache']), 'validation', Path(configs['gru']['control_tapes']))
    parts,summary = [],{}
    for name in MODELS:
        frame,summary[name] = evaluate(name,locations[name],data,ids)
        parts.append(frame)
    rows = pd.concat(parts,ignore_index=True)
    keys = ['control','horizon_s','metric']
    support = rows.pivot(index=keys+['flight_id'],columns='model',values='windows')[list(MODELS)]
    if support.isna().any().any() or not support.eq(support.gru,axis='index').all().all():
        raise ValueError('unequal per-flight native target coverage')
    errors = rows.pivot(index=keys+['flight_id'],columns='model',values='mse')[list(MODELS)]
    metrics = errors.groupby(level=keys).mean().pow(.5)
    metrics['windows'] = support.gru.groupby(level=keys).sum()
    metrics['flights'] = support.groupby(level=keys).size()
    paired = errors.iloc[:,:-1].gt(errors.slowfast,axis='index').groupby(level=keys).sum()
    rows.to_csv(output/'per_flight.csv',index=False)
    metrics.to_csv(output/'comparison.csv')
    paired.to_csv(output/'slowfast_winning_flight_counts.csv')
    (output/'scores.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(metrics.loc['logged'].round(5).to_string())
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('reference',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    compare_suite(a.reference,a.output,a.gpu)
