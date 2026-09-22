"""History ablation reporting: paired seeds, equal-flight means, no window tests."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_paper_standard_gru_multiseed import across_seeds,table,METRICS,LABELS

HS=(1,5,13,26)
DYNAMICS=METRICS[1:]
PAIRS=((1,5),(1,13),(1,26),(13,26),(5,13),(5,26))


def summarize(frame):
    if set(frame.history_steps)!=set(HS) or len(frame)!=144:
        raise ValueError('incomplete history/seed/cohort/horizon grid')
    return pd.concat([across_seeds(g).assign(history_steps=int(h))
                      for h,g in frame.groupby('history_steps')],ignore_index=True)


def relative_gains(multiseed):
    rows=[]
    for (cohort,horizon,metric),g in multiseed.groupby(['cohort','horizon_s','metric']):
        means=g.set_index('history_steps')['mean']
        for reference,h in PAIRS:
            rows.append(dict(reference_history_steps=reference,history_steps=h,cohort=cohort,
                horizon_s=horizon,metric=metric,reference_error=float(means[reference]),error=float(means[h]),
                relative_improvement_pct=float((means[reference]-means[h])/means[reference]*100)))
    return pd.DataFrame(rows)


def paired_directions(summary,perflight):
    seed_rows=[];flight_rows=[]
    for ref,h in PAIRS:
        for metric in METRICS:
            for (cohort,horizon),g in summary.groupby(['cohort','horizon_s']):
                p=g.pivot(index='seed',columns='history_steps',values=metric)
                delta=p[h]-p[ref]
                seed_rows.append(dict(reference_history_steps=ref,history_steps=h,cohort=cohort,horizon_s=horizon,
                    metric=metric,n_seeds=3,seeds_improved=int((delta<0).sum()),mean_paired_difference=float(delta.mean()),
                    min_paired_difference=float(delta.min()),max_paired_difference=float(delta.max())))
            # Average the three seed-specific per-flight RMSEs, then pair identical flights.
            p=perflight.groupby(['history_steps','cohort','log_id','horizon_s'])[metric].mean().unstack('history_steps')
            for (cohort,flight,horizon),row in p.iterrows():
                flight_rows.append(dict(reference_history_steps=ref,history_steps=h,cohort=cohort,flight_id=flight,
                    horizon_s=horizon,metric=metric,reference_error=float(row[ref]),error=float(row[h]),delta=float(row[h]-row[ref])))
    seeds=pd.DataFrame(seed_rows);flights=pd.DataFrame(flight_rows)
    both=pd.concat([flights,flights.assign(cohort='ALL')],ignore_index=True)
    counts=both.groupby(['reference_history_steps','history_steps','cohort','horizon_s','metric']).delta.agg(
        n_flights='count',flights_improved=lambda v:int((v<0).sum()),flights_worse=lambda v:int((v>0).sum()),
        mean_paired_difference='mean').reset_index()
    return seeds,flights,counts


def interpret(multi,gains,seeds):
    primary=multi[(multi.cohort=='ALL')&(multi.horizon_s==.5)]
    marginal=gains[(gains.cohort=='ALL')&(gains.horizon_s==.5)&(gains.reference_history_steps==13)&(gains.history_steps==26)&gains.metric.isin(DYNAMICS)]
    improvements=marginal.set_index('metric').relative_improvement_pct
    seed_marginal=seeds[(seeds.cohort=='ALL')&(seeds.horizon_s==.5)&(seeds.reference_history_steps==13)&(seeds.history_steps==26)&seeds.metric.isin(DYNAMICS)]
    best={m:int(g.loc[g['mean'].idxmin(),'history_steps']) for m,g in primary.groupby('metric')}
    # Fixed descriptive 2% band is NOT an equivalence test. Retain mixed outcomes.
    if all(best[m]==26 for m in DYNAMICS) and (improvements>2).all() and (seed_marginal.seeds_improved==3).all():
        outcome='A';text='H26 leads all three primary dynamics metrics, with >2% H13->H26 gains and matching direction across all seeds.'
    elif (improvements.abs()<=2).all():
        outcome='B';text='H13 and H26 primary dynamics means are within a descriptive 2% band; approximate saturation, not proven statistical equivalence.'
    else:
        shorter=[]
        for h in (5,13):
            p=primary.pivot(index='metric',columns='history_steps',values='mean').loc[DYNAMICS]
            gain=(p[26]-p[h])/p[26]*100
            if (gain>2).sum()>=2 and (gain>=-2).all():shorter.append(h)
        if shorter:
            outcome='C';text=f'Shorter context H{shorter} improves at least two primary dynamics means by >2%, without >2% regression on the third; inspect paired seed/flight evidence before attribution.'
        else:
            outcome='Mixed';text='Metric-dependent differences do not cleanly support a single A/B/C outcome; preserve the full results and keep H26 main model unchanged.'
    cv=multi.assign(cv=multi['std']/multi['mean'])
    sensitivity=cv[cv.horizon_s==.5].sort_values('cv',ascending=False).iloc[0]
    ranks=[]
    for (c,h,m),g in multi.groupby(['cohort','horizon_s','metric']):
        ranks.append(dict(cohort=c,horizon_s=h,metric=m,best_to_worst=list(map(int,g.sort_values('mean').history_steps))))
    return dict(outcome=outcome,interpretation=text,best_primary_history_by_metric=best,
        marginal_H13_to_H26_primary_pct=improvements.to_dict(),
        largest_primary_seed_cv=dict(history_steps=int(sensitivity.history_steps),cohort=sensitivity.cohort,metric=sensitivity.metric,cv=float(sensitivity.cv)),
        rankings=ranks,main_model_changed=False,next_ablation_started=False,
        classification_rule='Descriptive predeclared2% band, not significance/equivalence; A needs all3seeds consistent; mixed findings preserved.')


def formatted_results(multi):
    t=multi.copy()
    t['mean ± std']=t.apply(lambda r:f'{r["mean"]:.5f} ± {r["std"]:.5f}',axis=1)
    return t.pivot(index=['cohort','history_steps','horizon_s'],columns='metric',values='mean ± std').reset_index()


def plot_representative(out):
    with np.load(out/'representative_prediction.npz') as f:
        def euler(q):
            q=q/np.linalg.norm(q,axis=-1,keepdims=True)
            w,x,y,z=q.T
            angles=np.stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))],axis=-1)
            return np.degrees(np.unwrap(angles,axis=0))
        fig,axes=plt.subplots(3,3,figsize=(12,8),sharex=True)
        for i,(field,names,unit) in enumerate([
            ('velocity_n',['vN','vE','vD'],'m/s'),('quaternion_nb',['roll','pitch','yaw'],'deg'),
            ('angular_velocity_b',['p','q','r'],'rad/s')]):
            for label in ['truth','H1','H13','H26']:
                values=f[f'{label}_{field}'];values=euler(values) if field=='quaternion_nb' else values
                for j in range(3):
                    axes[i,j].plot(f['time_s'],values[:,j],label=label,lw=2 if label=='truth' else 1.2)
                    axes[i,j].set_ylabel(f'{names[j]} [{unit}]');axes[i,j].grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Native time from origin [s]')
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Frozen Step1 representative origin; seed17; Euler curves illustrative')
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(out/f'representative_prediction.{suffix}',dpi=170)
        plt.close(fig)


def generate(out,protocol):
    out=Path(out)
    summary=pd.read_csv(out/'per_seed_summary.csv');per=pd.read_csv(out/'per_flight_wide.csv')
    multi=summarize(summary);multi.to_csv(out/'multiseed_summary.csv',index=False)
    gains=relative_gains(multi);gains.to_csv(out/'relative_improvement.csv',index=False)
    seeds,paired,counts=paired_directions(summary,per)
    seeds.to_csv(out/'paired_seed_directions.csv',index=False)
    paired.to_csv(out/'paired_flight_differences.csv',index=False)
    counts.to_csv(out/'paired_flight_directions.csv',index=False)
    cv=multi.assign(coefficient_of_variation=multi['std']/multi['mean'])
    cv.to_csv(out/'seed_sensitivity.csv',index=False)
    interpretation=interpret(multi,gains,seeds)
    (out/'interpretation.json').write_text(json.dumps(interpretation,indent=2,allow_nan=False)+'\n')
    duration=pd.read_csv(out/'history_duration_summary.csv')
    median=duration[(duration.partition=='validation')&(duration.cohort=='ALL')].set_index('history_steps').median_s
    ticks=[f'H{h}\n{median[h]*1000:.0f} ms' for h in HS]
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=170)
        plt.close(fig)
    for metric,label in zip(METRICS,LABELS):
        g=multi[(multi.metric==metric)&(multi.cohort=='ALL')&(multi.horizon_s==.5)].sort_values('history_steps')
        fig,ax=plt.subplots(figsize=(6,4))
        ax.errorbar(range(4),g['mean'],yerr=g['std'],fmt='o-',capsize=4)
        ax.set(xticks=range(4),xticklabels=ticks,ylabel=label,xlabel='History context (median native span)',title='500 ms, 3-seed mean ± sample SD')
        ax.grid(alpha=.2);save(fig,f'error_vs_history_{metric}')
        if metric not in DYNAMICS:continue
        fig,ax=plt.subplots(figsize=(6,4))
        for h in HS:
            g=multi[(multi.metric==metric)&(multi.cohort=='ALL')&(multi.history_steps==h)].sort_values('horizon_s')
            ax.errorbar(g.horizon_s,g['mean'],yerr=g['std'],fmt='o-',capsize=3,label=f'H{h}')
        ax.set(xticks=[.1,.2,.5,1.],xlabel='Nominal prediction horizon [s]',ylabel=label)
        ax.legend();ax.grid(alpha=.2);save(fig,f'history_by_horizon_{metric}')
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,c in zip(axes,['Sep7','Sep17']):
            g=multi[(multi.metric==metric)&(multi.cohort==c)&(multi.horizon_s==.5)].sort_values('history_steps')
            ax.errorbar(range(4),g['mean'],yerr=g['std'],fmt='o-',capsize=4)
            ax.set(xticks=range(4),xticklabels=ticks,title=c,ylabel=label);ax.grid(alpha=.2)
        save(fig,f'cohort_consistency_{metric}')
    plot_representative(out)
    formatted=formatted_results(multi)
    primary=formatted[formatted.horizon_s==.5]
    primary_gains=gains[(gains.cohort=='ALL')&(gains.horizon_s==.5)&gains.metric.isin(DYNAMICS)]
    primary_seeds=seeds[(seeds.cohort=='ALL')&(seeds.horizon_s==.5)&seeds.metric.isin(DYNAMICS)]
    primary_counts=counts[(counts.cohort=='ALL')&(counts.horizon_s==.5)&counts.metric.isin(DYNAMICS)]
    # Range gains and ranks are explicit evidence, rather than a forced monotonic story.
    ranges=primary_gains[primary_gains[['reference_history_steps','history_steps']].apply(tuple,axis=1).isin([(1,5),(5,13),(13,26)])]
    biggest=ranges.loc[ranges.groupby('metric').relative_improvement_pct.idxmax()]
    src=protocol['source_frozen_contract']
    report=[
        '# Paper Step4 — History length ablation',
        '## 1. Goal',
        'Is past state/control context useful for short-horizon real-flight dynamics prediction, and where does its benefit saturate? Only H changes. H26 remains the paper main model regardless of this diagnostic.',
        '## 2. History semantics',
        'State tensor [N,H,12], control tensor [N,H,4], mask [N,H]. Samples range from origin-(H-1) through origin, inclusive. H26=current+25past; H1=current-only single-sample context. State/control use identical sample indices. Hidden state starts at zero per window, consumes allH history tokens including t; first rollout head then receives current state/control plus that hidden state, preserving the original architecture. H1 still has recurrent predicted-state transitions during rollout. Phase stays reanchored at origin; no re-anchoring at shortened-history start.',
        '## 3. Fair comparison protocol',
        'Every H uses exactly28,293 train and2,582 validation origins, with unchanged future targets/control tape/native dt. Short histories are suffix views of frozen H26 arrays; the trajectory object is shared. Full histories have no padding. Independent original-loader reconstruction of fixed origins checks suffix/phase/timestamp semantics. No extra short-history windows. Frozen buffers and increment scales/weights reused without fitting.',
        f'Normalization SHA256 `{protocol["normalization_sha256"]}`. Train origins `{protocol["train_window_sha256"]}`. Validation origins `{protocol["validation_origin_sha256"]}`. Source model commit `{protocol["source_main_model_commit"]}`. All12 models have21,383 trainable parameters.',
        '## 4. Actual history durations',
        'Span=(timestamp(origin)-timestamp(origin-H+1))*1e-6; H1=0. Native timestamps, no assumption of exact50Hz. Origin-weighted durations below; mean/median/sampleSD/p05/p95/min/max in seconds.',
        table(duration),
        '## 5. Training protocol',
        'Frozen Step3:40epochs AdamW lr3e-4, then25epochs lr5e-4 with optimizer reset; batch256, weight_decay1e-5, gradient_clip5; full50-step rollout plus frozen two-step increment objective; continuation adds0.2frequencyMSE. Last epoch65, no validation selection. Seeds17/23/42 use stage pairs17/29,23/35,42/54. New runs initialize exactly as H26 for the same seed. Runtime settings/versions in each artifact runtime.json. No retries based on unfavorable errors. H26 checkpoint/predictions/evaluation are reused, not trained or copied.',
        table(pd.read_csv(out/'training_runs.csv')),
        '## 6. Main results',
        'Per flight, compute endpoint vector RMSE (not divided by3), or RMS quaternion geodesic degrees. Average flights equally, then compute mean and sampleSD across three seeds. Delta-v/delta-omega endpoint increment errors equal velocity/body-rate endpoint errors because the same measured origin is subtracted. Primary500ms:',
        table(primary[['cohort','history_steps']+METRICS]),
        '## 7. Error vs history length',
        'Four error_vs_history figures use500ms and ±1seedSD. Positive relative gain means shorter-reference error decreased. Largest adjacent-range mean improvement for each dynamics metric (negative values would mean regression):',
        table(biggest[['metric','reference_history_steps','history_steps','relative_improvement_pct']]),
        '## 8. Error vs prediction horizon',
        'Complete nominal0.1/0.2/0.5/1.0s mean±SD below. Native5/10/25/50steps, same measured dt as Step3. History-by-horizon figures separately plot velocity/attitude/body rate.',
        table(formatted[['cohort','history_steps','horizon_s']+METRICS]),
        '## 9. Cohort consistency',
        'Cohort panels and ALL/Sep7/Sep17 tables preserve session differences; rankings for each metric/horizon/cohort are in interpretation.json. Per-flight paired differences first average the three seed-specific per-flight RMSEs; no windows treated as independent samples. Primary ALL flight directions:',
        table(primary_counts),
        '## 10. Seed robustness',
        'Full coefficient-of-variation and min/max: seed_sensitivity.csv. Paired seed directions below count how often the larger-H error decreases; each seed uses the same initialization/minibatch seed policy across H.',
        table(primary_seeds),
        f'Largest500ms relative seed variation: {interpretation["largest_primary_seed_cv"]}. No claim that smaller mean automatically implies stable improvement.',
        '## 11. Relative gains',
        'Formula100*(reference_error-error_H)/reference_error on three-seed macro means, not mean of per-window ratios. Includes H1 comparisons, adjacent-range gains, and H26 relative to H13; all horizons/cohorts in relative_improvement.csv.',
        table(primary_gains[['reference_history_steps','history_steps','metric','relative_improvement_pct']]),
        '## 12. Interpretation',
        f'Outcome {interpretation["outcome"]}: {interpretation["interpretation"]}',
        f'Best primary ALL history per metric: {interpretation["best_primary_history_by_metric"]}. H13->H26 gains(%): {interpretation["marginal_H13_to_H26_primary_pct"]}.',
        'Classification is only a descriptive aid: a predeclared2% mean-error band describes near-equal performance, not statistical equivalence. OutcomeA also requires all3seeds to favorH26 on all3dynamics metrics. Mixed tradeoffs remain mixed. Review cohort and flight directions rather than claiming a uniformly optimal H. Do not switch the main model automatically.',
        '## 13. Limitations',
        'Only3seeds,17flights and2open validation sessions; no window-level significance tests. History effects can mix physical memory with estimator/filter state and closed-loop correlations. Logged future controls are feedback-conditioned; this is prediction ablation, not causal actuator identification. Shorter history does not remove recurrence from future rollout. Phase pose provenance remains unchanged/unconfirmed. Reused H26 runtime provenance limitations from Step3 still apply. Representative seed17 window is frozen from Step1, never chosen by error; Euler curves illustrate orientation, while all primary attitude metrics are geodesic.',
        '## 14. Sealed-test status',
        'Sealed Sep8 and reserved Sep19 remain unopened. Only explicit train/validation files and previously opened validation artifacts are used. No phase/frequency/loss ablation or new model selection follows. H26 remains frozen pending user decision.',
        'Suggested commit: `feat: add frozen three-seed history length ablation`',
    ]
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
