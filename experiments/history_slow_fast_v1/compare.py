"""Compare the validation-selected checkpoints from three matched, completed runs."""
import argparse
import json
from pathlib import Path

import pandas as pd


def compare(reference, baseline, output):
    models = {'slowfast': reference/'slowfast', 'single': reference/'single', 'gru': baseline}
    shared = ['seed','epochs','batch','updates_per_epoch','units','history_s','rollout_dt',
              'training_s','optimizer','lr','weight_decay','schedule','gradient_clip',
              'sample_policy','cache','train_windows','validation_windows','validation_flights']
    rows = []
    for name, path in models.items():
        config = json.loads((path/'config.json').read_text())
        end = json.loads((path/'completed.json').read_text())
        if config['model'] != name or end['updates'] != config['epochs'] * config['updates_per_epoch']:
            raise ValueError(f'incomplete or unexpected model in {path}')
        contract = {k: config[k] for k in shared}
        if rows and contract != reference_contract:
            raise ValueError(f'unmatched evaluation or training contract in {path}')
        reference_contract = contract
        metrics = pd.read_csv(path/'best_metrics.csv').set_index(['horizon_s','metric']).sort_index()
        if rows and not rows[0][['windows','flights']].equals(metrics[['windows','flights']]):
            raise ValueError(f'different native-target coverage in {path}')
        rows.append(metrics.rename(columns={'flight_equal_rmse':name}))
    table = rows[0].join([rows[1][['single']],rows[2][['gru']]])
    table['slowfast_rmse_reduction_vs_gru_pct'] = 100 * (1 - table.slowfast / table.gru)
    table.to_csv(output)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for key in ('reference','baseline','output'):
        parser.add_argument(key,type=Path)
    args = parser.parse_args()
    compare(args.reference,args.baseline,args.output)
