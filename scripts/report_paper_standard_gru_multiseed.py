"""Descriptive three-seed summaries; no selection or window-level inference."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ['position_rmse_m','velocity_rmse_m_s','attitude_error_deg','body_rate_rmse_rad_s']
LABELS = ['Position RMSE [m]','Velocity RMSE [m/s]','Attitude geodesic RMS [deg]','Body-rate RMSE [rad/s]']


def across_seeds(summary):
    keys = ['seed','cohort','horizon_s']
    if summary.duplicated(keys).any():
        raise ValueError('duplicate seed/cohort/horizon')
    if len(summary) != 36 or set(summary.cohort) != {'ALL','Sep7','Sep17'} or set(summary.horizon_s) != {.1,.2,.5,1.}:
        raise ValueError('incomplete seed grid')
    for _,group in summary.groupby(['cohort','horizon_s']):
        if set(group.seed) != {17,23,42}:
            raise ValueError('exactly three declared seeds required')
    if not np.isfinite(summary[METRICS].to_numpy()).all():
        raise ValueError('nonfinite summary')
    long = summary.melt(id_vars=keys,value_vars=METRICS,var_name='metric',value_name='value')
    return long.groupby(['cohort','horizon_s','metric']).value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n_seeds'})


def interpret(summary, multiseed, screen):
    stats = multiseed.copy()
    stats['cv'] = stats['std']/stats['mean']
    primary = stats[stats.horizon_s == .5]
    extended = stats[stats.horizon_s == 1.]
    worst = primary.loc[primary.cv.idxmax()]
    gaps = {}
    for metric in METRICS:
        p = summary[summary.horizon_s == .5].pivot(index='seed',columns='cohort',values=metric)
        delta = p.Sep17-p.Sep7
        gaps[metric] = dict(sep17_minus_sep7={str(k):float(v) for k,v in delta.items()},
                           consistent_sign=bool((delta>0).all() or (delta<0).all() or (delta==0).all()))
    growth = []
    for (seed,cohort),g in summary.groupby(['seed','cohort']):
        g = g.set_index('horizon_s')
        for metric in METRICS:
            growth.append(dict(seed=int(seed),cohort=cohort,metric=metric,
                ratio_1s_to_100ms=float(g.loc[1.,metric]/g.loc[.1,metric]),
                ratio_1s_to_500ms=float(g.loc[1.,metric]/g.loc[.5,metric])))
    growth_consistent = all(r['ratio_1s_to_100ms']>1 for r in growth)
    gap_consistent = all(v['consistent_sign'] for v in gaps.values())
    passed = bool(primary.cv.max()<=screen['max_primary_cv'] and extended.cv.max()<=screen['max_1s_cv']
                  and growth_consistent and gap_consistent)
    return dict(decision='Multi-seed robustness passed' if passed else 'Robustness screen not passed; inspect seed/cohort variation before ablation',
        screen_passed=passed,screen=screen,primary_max_cv=float(primary.cv.max()),extended_max_cv=float(extended.cv.max()),
        greatest_primary_relative_seed_sensitivity=dict(metric=worst.metric,cohort=worst.cohort,cv=float(worst.cv)),
        cohort_gaps=gaps,error_growth=growth,growth_1s_above_100ms_all_seeds_cohorts=growth_consistent,
        failure='No failed or nonfinite seeds in completed results. Large finite variation is reported, not rerun.',
        statistical_claim='Descriptive n=3 screen; no significance claim. Thresholds frozen before training.',
        next_candidate='History Length Ablation' if passed else 'Inspect training instability; do not replace poor seeds',
        next_task_started=False)


def table(frame):
    cols=list(frame.columns)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def generate(out,protocol):
    out=Path(out)
    summary=pd.read_csv(out/'per_seed_summary.csv')
    multi=across_seeds(summary)
    multi.to_csv(out/'multiseed_summary.csv',index=False)
    result=interpret(summary,multi,protocol['robustness_screen'])
    (out/'interpretation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(result['error_growth']).to_csv(out/'error_growth.csv',index=False)
    for metric,label in zip(METRICS,LABELS):
        fig,ax=plt.subplots(figsize=(6,4))
        for seed,g in summary[summary.cohort=='ALL'].groupby('seed'):
            g=g.sort_values('horizon_s')
            ax.plot(g.horizon_s,g[metric],'o-',alpha=.45,label=f'Seed {seed}')
        g=multi[(multi.cohort=='ALL')&(multi.metric==metric)].sort_values('horizon_s')
        x=g.horizon_s.to_numpy(); mean=g['mean'].to_numpy(); sd=g['std'].to_numpy()
        ax.plot(x,mean,'k-',lw=2,label='Mean, n=3')
        ax.fill_between(x,mean-sd,mean+sd,color='gray',alpha=.18,label='±1 sample SD')
        ax.set(xlabel='Nominal horizon [s]',ylabel=label,xticks=[.1,.2,.5,1.])
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(out/f'error_vs_horizon_{metric}.{suffix}',dpi=180)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4))
        for cohort in ['Sep7','Sep17']:
            g=summary[(summary.cohort==cohort)&(summary.horizon_s==.5)].sort_values('seed')
            ax.plot(range(3),g[metric],'o-',label=cohort)
        ax.set(xticks=range(3),xticklabels=['17','23','42'],xlabel='Replicate seed',ylabel=label,
               title='Nominal 500 ms — equal-flight macro mean')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(out/f'seed_robustness_500ms_{metric}.{suffix}',dpi=180)
        plt.close(fig)
    primary=multi[multi.horizon_s==.5].copy()
    primary['mean ± std']=primary.apply(lambda r:f'{r["mean"]:.6f} ± {r["std"]:.6f}',axis=1)
    report=[
        '# Paper Step 3 — Standard GRU multi-seed robustness',
        '## 1. Goal',
        'Standard GRU (Step1 B2) is the current paper main model. Test initialization/minibatch robustness on open validation data only. Actuator-aware remains a historical negative ablation. No other models or ablations trained.',
        '## 2. Frozen model definition',
        protocol['architecture']+'. Trainable parameters: 21,383. History26; same state/control targets, same frozen buffers and unchanged integrator. Exact source hashes in protocol.json.',
        '40 epochs AdamW lr0.0003, then25 epochs lr0.0005 with a new AdamW optimizer. Batch256; all28,293 windows per epoch (111 updates). Weight decay1e-5, gradient clip5, full50-step loss. Same frozen two-step increment scales/weights; continuation adds0.2 frequency MSE through the existing zero-regularizer adapter. No validation checkpoint selection: last epoch65 for every seed.',
        '## 3. Data/split contract',
        '41 training flights /28,293 windows;17 validation flights /2,582 origins: Sep7 nine flights/1,481 origins; Sep17 eight flights/1,101 origins. Same windows at every horizon/seed. Freshly assembled arrays exactly match the Step1 cached histories, targets, controls, dt and identity order. No normalization fitting. Full flight names/exclusions and hashes are in protocol.json.',
        f'Manifest SHA256: `{protocol["manifest_sha256"]}`; train-origin CSV: `{protocol["train_window_sha256"]}`; validation-origin CSV: `{protocol["validation_origin_sha256"]}`; normalization canonical SHA256: `{protocol["normalization_sha256"]}`.',
        'Frames: position/velocity NED; body rates FRD; quaternion wxyz body-to-NED. Control order motor/left/right/rudder, normalized allocation commands before PWM. Relative encoder phase is reanchored at each origin. Nominal horizons are native steps5/10/25/50, integrating original dt, not resampling.',
        '## 4. Seeds',
        'Replicates17/23/42 use initialization and base-stage seed s, continuation seed s+12: (17,29), (23,35), (42,54). The fixed offset preserves the actual historical pilot definition. Seed17 checkpoint/predictions reused after hash and metric parity checks; no repeat training.',
        table(pd.read_csv(out/'training_runs.csv')),
        '## 5. Training reproducibility',
        protocol['seed17_rng_provenance'],
        'New workers explicitly seed Python, NumPy and PyTorch CPU/CUDA at each stage. The existing CPU torch.Generator controls minibatch permutations; no DataLoader or worker seeds. Runtime versions/device/determinism/TF32 settings are saved in each seed runtime.json. Deterministic algorithms enabled;4torchthreads;cudnn benchmark disabled. These controls do not prove bitwise reproducibility across software versions/devices.',
        '## 6. Per-seed results',
        'Each flight first computes endpoint vector RMSE (no division by3); attitude uses RMS quaternion geodesic degrees. Then flights are averaged equally. Increments share the measured origin and therefore Δv/Δω error equals the endpoint velocity/rate error. Results below are never pooled-origin RMSE.',
        table(summary[['seed','cohort','horizon_s']+METRICS]),
        '## 7. Across-seed mean/std',
        'Sample SD(ddof=1), min/max across exactly three seed-specific macro means. Full all-horizon statistics: multiseed_summary.csv. Primary500ms:',
        table(primary[['cohort','metric','mean ± std','min','max']]),
        '## 8. Sep7/Sep17 robustness',
        table(pd.DataFrame([dict(metric=k,consistent_gap_sign=v['consistent_sign'],**{f'seed{s}':x for s,x in v['sep17_minus_sep7'].items()}) for k,v in result['cohort_gaps'].items()])),
        'Gap above is Sep17 minus Sep7. Inspect absolute cohort errors and seed SD together; only two sessions are represented.',
        '## 9. Failure/anomaly check',
        'All65epochs/7,215 updates per seed, finite logged losses and gradients, strict checkpoint loading, bitwise fixed normalization, identical origins/native timing, finite predictions and unit quaternions are checked. Future-label poisoning and500ms prefix parity pass. No automatic retry is permitted. final_val_loss is evaluated after final checkpoint solely for reporting.',
        '## 10. Interpretation',
        result['decision']+'.',
        f'Max primary CV across metrics/cohorts: {result["primary_max_cv"]:.2%}; max1s CV: {result["extended_max_cv"]:.2%}. Greatest relative primary sensitivity: {result["greatest_primary_relative_seed_sensitivity"]}.',
        'Operational screen, frozen before new results: primary and1s CV≤20% in each cohort/metric; consistent500ms cohort-gap sign;1s error above100ms in all seeds/cohorts. This is a descriptive stability screen, not a significance test. Detailed1s/500ms and1s/100ms ratios: error_growth.csv.',
        '## 11. Limitations',
        'Only three seeds and17 correlated validation flights; no window-level confidence intervals. Baseline comparators remain seed17 pilots, so this does not establish statistically significant superiority over multi-seed B1/B3. B2 was designated main model after observing this same validation set; no unbiased test-generalization claim. Future logged commands are feedback-conditioned, not interventions. No long-rollout or simulation-readiness claim.',
        '## 12. Sealed-test status',
        'Sealed Sep8 and reserved Sep19 were not opened. Only train/validation Parquet allowlist and existing validation artifacts were accessed. No new seed selection, ablation, or test evaluation follows automatically. Next candidate only if robust: History Length Ablation; not executed.',
        'Suggested commit: `feat: freeze Standard GRU three-seed robustness experiment`',
    ]
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
