"""Report paired flight bootstrap differences, averaging same-seed RMSE contrasts."""
import json
import argparse
import numpy as np
import pandas as pd
from prepare_reference import HERE

SEEDS=(17,29,43)
FIELDS={
    'reference_beats_local_trend':('trend_only','trend_wave'),
    'wave_beats_zero':('zero_periodic','periodic_error'),
    'centered_wave_beats_zero':('centered_zero','centered_fast_error'),
    'centered_wave_beats_swapped_slow':('swapped_fast_error','centered_fast_error'),
    'slow_beats_persistence':('zero_slow_change','slow_change_error'),
    'slow_beats_swapped_wave':('swapped_slow_error','slow_change_error'),
    'measured_beat_mean_sensitivity':('unchanged_observed_mean','predicted_observed_change'),
    'future_wave_beats_drop':('drop_future_wave','full_observation'),
    'future_wave_beats_flip':('flip_future_wave','full_observation'),
}
INTERVALS=[(0,2.,'all'),(0,.5,'0-0.5'),(.5,1.,'0.5-1'),(1.,2.,'1-2')]


def paired(a,b):
    """a,b shape: paired training seeds by flight, values are per-flight MSE."""
    ar,br=np.sqrt(a.mean(1)),np.sqrt(b.mean(1))
    ix=np.random.default_rng(17).integers(a.shape[1],size=(10000,a.shape[1]))
    u=np.sqrt(a[:,ix].mean(2)); v=np.sqrt(b[:,ix].mean(2))
    gain=ar-br
    return dict(mean_reference_rmse=ar.mean(),mean_model_rmse=br.mean(),mean_gain=gain.mean(),
                interval95=np.quantile((u-v).mean(0),[.025,.975]).tolist(),
                gains_by_seed=dict(zip(map(str,SEEDS),gain)),flights=a.shape[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',default='full')
    parser.add_argument('--control',default='control')
    parser.add_argument('--output',default='three_seed_scores.json')
    args=parser.parse_args()
    labels={'full':args.full,'control':args.control}
    tables={(arm,seed):pd.read_parquet(HERE/f'{name}{seed}_beats.parquet') for arm,name in labels.items() for seed in SEEDS}
    for table in tables.values():
        pd.testing.assert_frame_equal(table[['block','ch','flight','window','lead']],tables['full',17][['block','ch','flight','window','lead']],check_exact=True)
        measured=table[[f'd_observation{i}' for i in range(3)]].to_numpy()
        predicted=table[[f'd_slow_pred{i}' for i in range(3)]].to_numpy()
        table['unchanged_observed_mean']=(measured**2).sum(1)
        table['predicted_observed_change']=((measured-predicted)**2).sum(1)
    out={}
    for ch,stream in enumerate(('velocity','angular_rate')):
        rr={}
        for l,r,tag in INTERVALS:
            tabs={}
            for (name,seed),df in tables.items():
                base=df[(df.ch==ch)&(df.lead>l)&(df.lead<=r)]
                for after in (False,True):
                    v=base[base.is_later] if after else base
                    tabs[name,seed,after]=v.groupby(['flight','window']).mean(numeric_only=True).groupby('flight').mean()
            def get(arm,col):
                later=col in {'zero_slow_change','slow_change_error','swapped_slow_error','unchanged_observed_mean','predicted_observed_change'}
                return np.stack([tabs[arm,s,later][col] for s in SEEDS])
            stats={arm:{metric:paired(get(arm,base),get(arm,full)) for metric,(base,full) in FIELDS.items()} for arm in ('full','control')}
            stats['full_vs_control']={metric:paired(get('control',col),get('full',col)) for metric,col in (
                ('wave','periodic_error'),('centered_wave','centered_fast_error'),('slow_change','slow_change_error'),('measured_motion','full_observation'))}
            rr[tag]=stats
        out[stream]=rr
    (HERE/args.output).write_text(json.dumps(out,indent=2)+'\n')
    for stream,vals in out.items():
        print(stream)
        for name,metric in vals['all']['full'].items():
            print(name,metric['mean_gain'],metric['interval95'],metric['gains_by_seed'])


if __name__=='__main__':
    main()
