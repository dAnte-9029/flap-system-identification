"""Report the frozen baseline pilot; plots never select models or checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper-baselines")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from system_identification.evaluation.paper_baselines import METRIC_MAP

OUT=ROOT/'docs/analysis/results/paper_baseline_comparison_v1'
ART=ROOT/'artifacts/paper_baseline_comparison_v1'
METRICS=['position_rmse_m','velocity_rmse_m_s','attitude_error_deg','body_rate_rmse_rad_s']
LABELS=['Position RMSE [m]','Velocity RMSE [m/s]','Attitude geodesic RMS [deg]','Body-rate RMSE [rad/s]']


def table(frame):
    def fmt(v):
        return f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def audit_report(protocol):
    coverage=pd.read_csv(OUT/'flight_coverage.csv')
    counts=pd.DataFrame([dict(model=k,**v) for k,v in protocol['parameter_counts'].items()]).fillna('')
    text='''# Phase A — Baseline comparison audit

The research target is short-horizon real-flight dynamics prediction, with 0.5 s primary. This is not an RL, uncertainty, OOD or long-rollout project.

## Reuse and required additions

- B0: reuse `models/trajectory.py::ConstantTwistPredictor` directly; no fitting.
- B1: existing `MLPRegressor(16,7,(64,64))` is suitable in size. Its historical trainer used one-step derivatives and different data/statistics, so historical checkpoints are not comparable. Reuse the network through a memoryless Main V1 rollout adapter and train on the same 28,293 windows.
- B2: reuse `CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True)` directly. Old September GRU checkpoints used the smaller v2 dataset and cannot be reused as matched baselines.
- B3: reuse the completed expanded September actuator-aware checkpoint, unchanged. The old August Main V2 and later joint-control experiments are not the proposed-method checkpoint for this pilot.
- Reuse window assembly, train-only normalization fitting, rollout integration, rollout loss, increment training loop and geodesic evaluator. Add only benchmark orchestration, flight aggregation, provenance and plots.

## Frozen method and training mismatch

Ours is **not** an end-to-end control-conditioned GRU: its history-only GRU64 is trained for 40 epochs (seed 17, LR 0.0003), then frozen while actuator residuals train for 25 epochs (seed 29, LR 0.0005). Last epoch is used in both stages. The second stage adds frequency MSE and actuator-specific regularization. Historical records report 1827.4 s total on GPU 1. B1/B2 are expected to require roughly 15–30 min each; the pilot trains these two only.

A standard controlled GRU cannot simultaneously match that frozen parameter set and differ only in the control path. Minimal proposed solution: use the same 40+25 epoch budget, optimizer resets, sampling seeds, learning rates and physical prediction loss; add the same second-stage frequency loss, omit inapplicable actuator regularizers. Disclose that this is a matched-budget comparison, not a strict single-factor ablation. A 40-epoch backbone-budget alternative is supported explicitly. Do not change/retrain Ours to conceal the mismatch.

The replicate label 17 must retain actual Ours seeds 17/29 in all metadata. A second and third proposed-method replicate would require a separately agreed stage-seed mapping and authorization to reproduce frozen Ours training; this run does not do so.

## Data and causality

Authority: registered `trajectory_v3_september_expanded`, same manifest and four train/validation artifacts as frozen Ours. 41 training flights / 28,293 windows; 17 validation flights / 2,582 windows (Sep7: 9 flights, Sep17: 8). Sep17 is validation, not training. Training days: Sep6, 11–16, 18. No other dataset, raw ULog, test metric or reserved prediction is loaded.

Every origin includes 26 samples with nominal 0.5 s history and 50 future native transitions. Nominal dt is 0.02 s; actual logged dt is integrated unchanged. Validation history spans 0.4889–0.5093 s, and the 25-step endpoint spans 0.4887–0.5189 s. Horizon labels refer to 5/10/25/50 native steps, not exactly resampled wall-clock intervals; full duration distributions are in the timing CSVs. Training stride is 10 samples, validation stride 50. No origin or flight is dropped by model/error. State: position/velocity NED, body rate FRD, wxyz FRD-to-NED quaternion, relative phase and flap frequency. Dynamics features: body velocity, body rate, body gravity, origin-relative phase sin/cos, frequency. Seven derivative targets: body-expressed inertial acceleration, angular acceleration and frequency derivative. Controls: motor, left, right, rudder.

Historical extraction uses causal publication-time zero-order hold. This pilot preserves that preprocessing, checks available source timestamps, verifies contiguous identities and dt, and refits statistics on train only to verify bitwise float32 equality with frozen buffers. Future logged commands are known-input replay; future measured states/frequency/phase are labels only. Feedback-generated future commands do not constitute independent causal interventions.

## Evaluation and statistical units

Nominal endpoint horizons 0.1/0.2/0.5/1.0 s; 0.5 s primary, 1 s extended. Per-flight vector RMSE is sqrt(mean(sum(error_xyz²))), not component-averaged RMSE. Attitude is RMS quaternion geodesic angle, not quaternion-component RMSE. Macro values average per-flight RMSE equally; standard deviation uses flights (ddof=1), never thousands of windows as independent trials. Single replicate has no across-seed SD or significance claim.

The requested Δv/Δω errors algebraically equal endpoint velocity/rate errors when both start at the same true t0. Report them transparently, not as independent evidence for dynamics learning. Representative figure uses the middle lexicographically sorted validation-origin identity, chosen without viewing predictions.

## Parameter counts

'''+table(counts)+'''

For B3, `trainable` denotes actuator-stage parameters, `base_stage_trainable` the earlier backbone stage; `total` is deployment size. In evaluation no parameters are optimized.

## Complete admitted flight list

'''+table(coverage)+'''

## Excluded/sealed assignments (metadata only)

'''
    for group,flights in protocol['excluded_sealed_flights'].items():
        text+=f'\n### {group}\n\n'+'\n'.join('- `'+x+'`' for x in flights)+'\n'
    text+='''
Six Sep8 sealed flights and 23 Sep19 reserved flights remain excluded. Reading their names from split metadata is the only access in this run. Existing raw-inventory metadata in the repository predates this task; the pilot does not assert that no historical metadata inventory ever occurred. No sealed/reserved raw samples, evaluations or predictions are opened here.

Registry, instructions and many unrelated artifacts were already dirty at startup. This task does not modify them. `protocol.json` records the current commit plus source/data/checkpoint hashes because commit identity alone cannot reproduce a dirty workspace.
'''
    (OUT/'audit.md').write_text(text)


def figures(summary):
    for metric,label in zip(METRICS,LABELS):
        fig,axes=plt.subplots(1,3,figsize=(14,4),sharex=True,layout='constrained')
        for ax,cohort in zip(axes,['ALL','Sep7','Sep17']):
            for model,g in summary[summary.cohort==cohort].groupby('model'):
                g=g.sort_values('horizon_s')
                ax.plot(g.horizon_s,g[metric],marker='o',label=model)
            ax.set(xlabel='Nominal prediction horizon [s]',ylabel=label,title=cohort,xticks=[.1,.2,.5,1.])
            ax.axvline(.5,alpha=.15,color='black');ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.savefig(OUT/f'error_vs_horizon_{metric}.png',dpi=180)
        fig.savefig(OUT/f'error_vs_horizon_{metric}.pdf')
        plt.close(fig)
    data=np.load(OUT/'representative_prediction.npz')
    def euler(q):
        w,x,y,z=np.moveaxis(q,-1,0)
        return np.rad2deg(np.stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            np.arcsin(np.clip(2*(w*y-z*x),-1,1)),
            np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))],axis=-1))
    fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,layout='constrained')
    for prefix,label in [('truth','Ground truth'),('B2_StandardGRU','Standard GRU'),('B3_ActuatorAwareGRU','Ours')]:
        arrays=[data[f'{prefix}_velocity_n'],euler(data[f'{prefix}_quaternion_nb']),data[f'{prefix}_angular_velocity_b']]
        for row,(array,names,unit) in enumerate(zip(arrays,[['vx','vy','vz'],['roll','pitch','yaw'],['p','q','r']],['m/s','deg','rad/s'])):
            for j in range(3):
                axes[row,j].plot(data['time_s'],array[:,j],label=label,linestyle='-' if prefix=='truth' else '--')
                axes[row,j].set_ylabel(f'{names[j]} [{unit}]');axes[row,j].grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Time from observed origin [s]')
    axes[0,0].legend(fontsize=8)
    fig.savefig(OUT/'representative_prediction.png',dpi=180)
    fig.savefig(OUT/'representative_prediction.pdf');plt.close(fig)


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    audit_report(protocol)
    if not (OUT/'summary.csv').exists():
        print('Audit report written; training/results still pending.');return
    summary=pd.read_csv(OUT/'summary.csv');per=pd.read_csv(OUT/'per_flight.csv')
    aggregate=pd.read_csv(OUT/'aggregate.csv')
    figures(summary)
    keys=['seed','cohort','log_id','horizon_s']
    ours=per[per.model=='B3_ActuatorAwareGRU'].set_index(keys)
    gru=per[per.model=='B2_StandardGRU'].set_index(keys)
    differences=pd.DataFrame(index=ours.index)
    for metric in METRICS:
        differences[metric+'_ours_minus_gru']=ours[metric]-gru[metric]
        differences[metric+'_ours_reduction_pct']=100*(1-ours[metric]/gru[metric])
    differences.reset_index().to_csv(OUT/'ours_vs_gru_per_flight.csv',index=False)
    rows=pd.read_csv(ART/'per_origin.csv')
    failures=[]
    for (model,cohort),g in rows[rows.horizon_s==.5].groupby(['model','cohort']):
        for metric in ['velocity_error_m_s','attitude_error_deg','body_rate_error_rad_s']:
            worst=g.sort_values([metric,'window_id'],ascending=[False,True]).head(3).copy()
            worst['ranking_metric']=metric;failures.append(worst)
    pd.concat(failures).to_csv(OUT/'failure_cases.csv',index=False)
    columns=['model','horizon_s',*METRICS,'delta_v_rmse_m_s','delta_omega_rmse_rad_s']
    text='''# Paper Experiment Step 1 — Baseline Comparison (single-replicate pilot)

## 1. Goal and status

Short-horizon dynamics modeling from real flight data. Primary horizon 0.5 s, supporting 0.1/0.2 s, extended 1.0 s. This is a single-replicate pilot, not final three-seed paper statistics. Ours was not modified, tuned or retrained. See [audit.md](audit.md) for the full implementation/data audit and [protocol.json](protocol.json) for exact provenance.

## 2. Dataset split

41 train flights / 28,293 windows; 17 validation flights / 2,582 identical origins across all models and all horizons. Sep7 has 9 validation flights; Sep17 has 8. Full flight identities and counts: [flight_coverage.csv](flight_coverage.csv). Complete sealed/reserved metadata lists are in the audit and protocol; no test data/results were read.

## 3. Models and formulas

B0: v(t+h)=v0, ω(t+h)=ω0, p(t+h)=p0+h v0, q(t+h)=normalize(q0 ⊗ [cos(|ω0|h/2), sin(|ω0|h/2) ω0/|ω0|]), with the continuous zero-rate limit. Existing constant-twist implementation integrates the actual dt values; flap frequency stays fixed. No fitted parameters.

B1: current normalized 12-state features plus four controls → existing 64×64 ReLU MLP → seven derivatives. No history or recurrent hidden state. The physical integrator, derivative scale/clipping and phase/frequency update are inherited from the existing recurrent model.

B2: existing GRU64 with the same 26-sample history, states/normalization, derivative head and physical integration. Raw normalized commands enter history encoding, current derivative prediction and the recurrent transition directly.

B3: frozen expanded Main V2, history-only GRU64 plus causal drive/tail filtered residuals (τ=0.10/0.04 s). Previous actuator state affects the current derivative; the current command updates the proxy for the next step. Base weights, gates, signs, τ, normalization and losses are unchanged.

'''
    text+=table(pd.DataFrame([dict(model=k,**v) for k,v in protocol['parameter_counts'].items()]).fillna(''))
    text+='\n\nFor Ours, total is deployment parameter count; trainable is its actuator-stage subset. Base-stage trainable count is reported separately.\n'
    text+='''
## 4. Fairness controls and checkpoint selection

'''+f"Selected schedule: **{protocol.get('schedule')}**. "+'''Use every registered training window, train-only statistics bitwise matched to Ours, identical batch size and native dt, and the existing multi-step rollout/increment objective. Last epoch only, no validation early stopping, checkpoint ranking, per-metric selection or baseline-driven Ours tuning. All normalization uses training only.

This is a matched-budget model-family comparison, **not a strict single-factor actuator ablation**: Ours freezes its backbone during the actuator stage; generic models update all parameters and have no actuator-specific regularizer. MLP also differs in capacity and activation, so an MLP–GRU difference is consistent with temporal value but does not isolate memory alone. The actual random seeds are recorded by stage; Ours uses 17/29, and label 17 denotes the pilot replicate. A second-stage sampling seed 29 matches the historical Ours permutation protocol where that stage is used.

## 5. Primary and supporting results

Horizon labels are nominal 5/10/25/50 native steps. Actual elapsed time is integrated and recorded per origin; validation 25-step durations range 0.4887–0.5189 s. See [validation_horizon_timing.csv](validation_horizon_timing.csv). No resampling or time stretching was applied. All values below are equal-flight means of per-flight endpoint RMSE. Attitude is RMS geodesic angle in degrees. Vector errors sum all three squared axes before taking the per-flight RMS. Flight SD (ddof=1) is in [aggregate.csv](aggregate.csv); axes are in [summary.csv](summary.csv). No standard error based on window count is reported.

'''
    for c in ['ALL','Sep7','Sep17']:
        text+=f'\n### {c}\n\n'+table(summary.loc[summary.cohort==c,columns])+'\n'
    text+='''
Δv and Δω are computed using the requested common true initial state. Their errors equal endpoint v and ω errors algebraically; these duplicate columns are included for transparency and do not provide independent evidence. The frozen training increment loss is a different, 40 ms supervision term; it is not retuned to the 500 ms reporting horizon.

## 6. Per-flight results

Full four-horizon results are in [per_flight.csv](per_flight.csv). Primary 500 ms values follow:

'''+table(per.loc[per.horizon_s==.5,['cohort','log_id','model',*METRICS]])+'\n'
    text+='''
## 7. Failure cases and representative rollout

[failure_cases.csv](failure_cases.csv) retains the three largest velocity, attitude and body-rate endpoint errors for every model/cohort at 500 ms. No outlier is removed from any aggregate. These are descriptive failure cases, not exclusion gates.

The representative 500 ms rollout is selected by fixed middle index after lexicographic origin sorting, independently of model errors. Identity: [representative_selection.json](representative_selection.json). Euler angles are used only for display; metrics remain quaternion geodesic errors.

![Representative rollout](representative_prediction.png)

'''
    for metric in METRICS:
        text+=f'![{metric}](error_vs_horizon_{metric}.png)\n\n'
    text+='''
## 8. Interpretation

Positive reductions below mean Ours has lower macro RMSE than Standard GRU; negative values favor Standard GRU. No criterion is changed to favor Ours.

'''
    comparisons=[]
    for c in ['ALL','Sep7','Sep17']:
        s=summary[(summary.cohort==c)&(summary.horizon_s==.5)].set_index('model')
        comparisons.append(dict(cohort=c,**{m:100*(1-s.loc['B3_ActuatorAwareGRU',m]/s.loc['B2_StandardGRU',m]) for m in METRICS}))
    text+=table(pd.DataFrame(comparisons))+'\n\n'
    for c in ['ALL','Sep7','Sep17']:
        s=summary[(summary.cohort==c)&(summary.horizon_s==.5)].set_index('model')
        for better,reference in [('B2_StandardGRU','B0_ConstantVelocity'),('B2_StandardGRU','B1_MLP'),('B3_ActuatorAwareGRU','B2_StandardGRU')]:
            wins=[m for m in METRICS if s.loc[better,m]<s.loc[reference,m]]
            text+=f'- {c}, 500 ms: {better} improves {len(wins)}/4 metrics over {reference}: '+(', '.join(wins) if wins else 'none')+'.\n'
    text+='''

These are observed differences, not statistically established significance. Cross-flight patterns are supported descriptively by the paired per-flight table, not by treating correlated windows as repeated trials. Logged control replay alone cannot establish action-dependent causal effectiveness.

## 9. Limitations and next step

Single replicate; two validation days and correlated flights; already-used development cohorts; family/training differences noted above; observed logged feedback controls; uncertain absolute mechanical phase pose. No OOD, uncertainty, long-horizon, controller or RL claims. Endpoint increments are redundant with endpoint state errors. Full history favors temporal models by design and is intentionally absent from memoryless B1.

After all sanity checks pass, the pipeline can support additional seeds technically. Final-paper execution still requires explicitly freezing the matched-budget interpretation and a two-stage seed mapping for Ours (the existing 17/29 checkpoint is only one replicate). Do not describe this pilot as final mean±SD across seeds, and do not open sealed test yet. No ablation or additional seeds are started here.

## 10. Sanity and sealed-test status

See [sanity_checks.json](sanity_checks.json): finite arrays, normalized quaternions, full 51-state rollout length, native timestamps/integrated-dt alignment (nominal rather than exact wall-clock horizons), full causal history, unchanged checkpoint, identical origins, train-only normalization reproduction, future-label poisoning and 500 ms prefix parity. Automated synthetic tests additionally cover constant-velocity/quaternion integration, history-free MLP, increment identities, unknown cohort rejection and equal-flight aggregation under unequal window counts.

Only explicit train/validation Parquet files were opened. Six Sep8 sealed flights and 23 Sep19 reserved flights were not evaluated, used for training/selection, or opened for exploratory analysis. Their names were read solely from split metadata. Historical metadata inventory predates this run; no claim of universally unopened historical metadata is made.

## Reproduction

Use `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`, without dependency changes:

```bash
python scripts/run_paper_baseline_comparison.py --phase audit
python scripts/run_paper_baseline_comparison.py --phase run --schedule '''+protocol.get('schedule','PENDING')+'''
python scripts/report_paper_baseline_comparison.py
```

The runner refuses to overwrite completed summary results. Full predictions, per-origin metrics, reusable prepared batches, training histories and learned checkpoints are under `artifacts/paper_baseline_comparison_v1/`. Source/data/model hashes accompany the protocol because the starting workspace was already dirty.
'''
    (OUT/'report.md').write_text(text)
    print('Report, paired comparisons, failure cases and figures written.')


if __name__=='__main__':main()
