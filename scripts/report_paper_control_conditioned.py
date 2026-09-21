"""Tables and figures for frozen control-conditioned prediction comparisons."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-paper-control-conditioned')
from pathlib import Path
import json
import hashlib
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=ROOT/'docs/analysis/results/paper_control_conditioned_eval_v1'
CORE=['velocity_rmse_m_s','attitude_error_deg','body_rate_rmse_rad_s']
LABELS=['Velocity RMSE [m/s]','Attitude geodesic RMS [deg]','Body-rate RMSE [rad/s]']
PRIMARY=['position_rmse_m',*CORE,'delta_v_rmse_m_s','delta_omega_rmse_rad_s']
CHANNEL_METRICS={
    'drive':['velocity_rmse_m_s','velocity_z_rmse_m_s'],
    'common':['body_rate_q_rmse_rad_s','attitude_pitch_component_rmse_deg'],
    'differential':['body_rate_p_rmse_rad_s','attitude_roll_component_rmse_deg'],
    'rudder':['body_rate_r_rmse_rad_s','attitude_yaw_component_rmse_deg'],
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(df):
    def fmt(x):
        return f'{x:.5f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |',
        *['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)]])


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)


def figures(summary,paired):
    colors={'B2':'#2679b3','B3':'#de7824'}
    for metric,ylabel in zip(CORE,LABELS):
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
        for ax,cohort in zip(axes,['ALL','Sep7','Sep17']):
            sub=summary[(summary.channel=='overall')&(summary.horizon_s==.5)&(summary.cohort==cohort)]
            for model in ['B2','B3']:
                y=sub[sub.model==model].set_index('activity_bin').loc[['Low','Medium','High'],metric]
                ax.plot(np.arange(3),y,marker='o',color=colors[model],label=model)
            ax.set(xticks=range(3),xticklabels=['Low','Medium','High'],xlabel='Control-change activity',ylabel=ylabel,title=f'{cohort} | nominal 0.5 s')
            ax.grid(alpha=.2);ax.legend()
        save(fig,'figure1_'+metric)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,ylabel in zip(axes,CORE,['Velocity [m/s]','Attitude [deg]','Body rate [rad/s]']):
        s=paired[(paired.channel=='overall')&(paired.horizon_s==.5)&(paired.cohort=='ALL')&(paired.metric==metric)].set_index('activity_bin').loc[['Low','Medium','High','Top20']]
        y=s.flight_equal_mean_paired_difference.to_numpy()
        ax.errorbar(range(4),y,yerr=np.stack([y-s.mean_paired_ci95_low,s.mean_paired_ci95_high-y]),fmt='o-',capsize=4)
        ax.axhline(0,color='black',linewidth=1)
        ax.set(xticks=range(4),xticklabels=['Low','Medium','High','Top20'],xlabel='Control-change activity',ylabel='Paired B3 - B2: '+ylabel,
               title='Equal-flight mean; flight bootstrap 95% CI')
        ax.grid(alpha=.2)
    save(fig,'figure2_paired_differences')
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for col,(channel,metrics) in enumerate(CHANNEL_METRICS.items()):
        for row,metric in enumerate(metrics):
            ax=axes[row,col]
            sub=summary[(summary.channel==channel)&(summary.horizon_s==.5)&(summary.cohort=='ALL')]
            for i,model in enumerate(['B2','B3']):
                y=sub[sub.model==model].set_index('activity_bin').loc[['High','Top20'],metric]
                ax.bar(np.arange(2)+(i-.5)*.34,y,width=.34,color=colors[model],label=model)
            label=metric.replace('_rmse_m_s',' [m/s]').replace('_rmse_rad_s',' [rad/s]').replace('_rmse_deg',' [deg]').replace('_',' ')
            ax.set(title=channel,ylabel=label,xticks=range(2),xticklabels=['High','Top20'])
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    save(fig,'figure3_channel_excitation')


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    summary=pd.read_csv(OUT/'conditional_summary.csv')
    paired=pd.read_csv(OUT/'paired_summary.csv')
    coverage=pd.read_csv(OUT/'bin_coverage.csv')
    flights=pd.read_csv(OUT/'per_flight_high_excitation.csv')
    figures(summary,paired)
    high=paired[(paired.cohort=='ALL')&(paired.channel=='overall')&(paired.horizon_s==.5)&paired.activity_bin.isin(['High','Top20'])&paired.metric.isin(CORE)]
    if not (len(high)==6 and (high.RMSE_difference_ci95_low>0).all() and (high.B2_flight_wins>high.B3_flight_wins).all()):
        raise ValueError('results no longer support the reviewed Outcome C; review without silently changing conclusion')
    text='''# Paper Experiment Step 2 — Control-Conditioned Evaluation

## Conclusion: Outcome C for the frozen pilot

There is no evidence that actuator-aware B3 gains a stable prediction advantage under stronger control changes. At nominal 500 ms, B2 is better on velocity, attitude and aggregate body rate in Low, Medium, High and Top20. For High and Top20, both equal-flight mean paired-origin error and paired per-flight RMSE differences favor B2, and their flight-bootstrap 95% intervals are above zero. Most flights agree. The channel-associated 500 ms point estimates also favor B2; differential Top20 p-rate has a CI crossing zero and should be treated as unresolved, not as a proven B2 gain.

This is not a claim that B2 wins every cell: Low-activity 500 ms position slightly favors B3; Sep7 Top20 velocity slightly favors B3 (−0.003842 m/s, 95% flight CI [−0.069219, +0.052717]), while Sep7 High velocity is effectively tied (+0.000375 m/s, CI [−0.041897, +0.041576]); some individual flights and shorter-horizon/channel cells favor B3. Those exceptions do not establish the proposed high-excitation advantage. Do not retune tau, normalization, signs, B2 or B3 to rescue a preferred narrative.

For paper planning, the supported next decision is to designate Standard GRU as the main method, with actuator-aware results as negative ablation/discussion/supplementary. This run does **not** implement that next step, launch training/seeds or open sealed test. The finding concerns these frozen single-replicate models, not a universal theorem about actuator-aware architectures.

## 1. Scope, frozen origins and prediction reuse

Source: `docs/analysis/results/paper_baseline_comparison_v1/`, committed in `cdf7625`. Exactly 2,582 origins from 17 validation flights, Sep7 (9 flights / 1,481 origins) and Sep17 (8 / 1,101). No new origins, filtering by errors, model inference or training. The two saved prediction arrays are joined to the SAME frozen window IDs and future truth and passed through the existing `endpoint_metrics` implementation. Per-origin metric parity and all unconditioned macro results reproduce Step 1. Checkpoint, baseline-output, origin and data hashes are checked before and after the run.

The 5/10/25/50 native steps are nominal 0.1/0.2/0.5/1.0 s; integrate the original timestamps, no resampling. Primary 500 ms actual spans are 0.488664–0.518876 s. 1 s is supplementary only. Only `samples_validation.parquet` and `windows_validation.parquet` are read from the registered expanded September dataset. Training data are not reread: existing train-only normalization buffers supply the activity scales.

## 2. Actual control semantics

Raw logged and dataset/model tape order:

| index | logged source | dataset meaning |
|---|---|---|
| 0 | actuator_motors.control[0] | normalized flapping-drive motor setpoint; **not a Hz command** |
| 1 | actuator_servos.control[0] | left elevon normalized command |
| 2 | actuator_servos.control[1] | right elevon normalized command |
| 3 | actuator_servos.control[2] | rudder normalized command |

These are post-allocation normalized setpoints, not measured servo angles and not `actuator_outputs` PWM. The extractor uses causal publication-time zero-order hold and copies these values without PWM conversion or a second reversal. The September configuration maps Servo1/Servo2/Motor1/Servo3 to MAIN1/MAIN2/MAIN3/MAIN5. `PWM_MAIN_REV=17` reverses MAIN1 and MAIN5 in the downstream output driver. Thus normalization into the actuator topic's dimensionless range has already occurred upstream, while PWM reversal is downstream of the logged coordinates. Do not multiply the logged left/rudder signs again. Measured flap frequency is a state/target, not a future command used for this grouping.

B2 standardizes `[motor,left,right,rudder]` with its frozen train means/stds. B3 uses the history-only state backbone, plus a normalized motor drive proxy and normalized tail coordinates. Conditional statistics use those exact pre-PWM command coordinates, before lag filters:

`u_drive = motor`

`u_common = (left + right) / 2`

`u_differential = (left - right) / 2`

`u_rudder = rudder`

The independent-coordinate terminology refers to an invertible command decomposition, not statistically independent inputs. Channels co-vary in closed-loop flight. Surface-angle calibration in degrees is unavailable, and common/q, differential/p, rudder/r associations do not isolate aerodynamic causality.

Local source evidence: `data/trajectory_dataset.py::CONTROL_COLUMNS` and `extract_trajectory_samples`; `models/trajectory_main_v2.py::transform_tail_controls`; `training/trajectory_main_v2.py::fit_main_v2_stats`; `configs/data/trajectory_september_v2.yaml`. The original September firmware `c9af571575af33d5403fbb479256848797c801e2` and all Sep17 validation flights' firmware `a644ca7868f07004e45658d4ae9a7a8c880c6bb5` were inspected locally (hashes/excerpts in `control_semantics_evidence.json`): `msg/versioned/ActuatorServos.msg`, `src/lib/mixer_module/functions/FunctionServos.hpp`, and `mixer_module.cpp::output_limit_calc_single` put reversal after actuator-topic consumption. No PX4 source or controller was edited.

## 3. Control-only activity protocol

For horizon K, only the applied tape `u[0:K]` is used. The command at the endpoint `u[K]` cannot influence that endpoint and is excluded. For each transformed channel j:

- Magnitude: `sqrt(mean_k((u_j[k]-mu_train,j)^2))`.
- Change: `D_j = max_k |u_j[k]-u_j[0]|`.
- Total variation: `TV_j = sum_k |u_j[k+1]-u_j[k]|` (saved as a descriptive diagnostic).
- Normalized channel score: `D_j / sigma_train,j`.
- Overall score: `sqrt(mean_j((D_j/sigma_train,j)^2))`.

The reference/scale are frozen B3 train statistics, originally fitted on training history and future-control rows, including the original overlapping-window weighting. Motor uses backbone control statistics; common/differential/rudder use tail statistics. No statistics are refitted, and neither model nor its normalization is changed. Constant large commands can have high magnitude but exactly zero change/TV; they cannot qualify as strong excitation through magnitude alone.

'''
    text+=table(pd.DataFrame(dict(channel=['drive','common','differential','rudder'],reference=protocol['controls']['reference'],scale=protocol['controls']['scale'])))
    text+='''

Quantiles are defined independently for each horizon from all validation **controls only**, and reused unchanged for ALL, Sep7 and Sep17. Low: score ≤ q33; Medium: q33 < score ≤ q67; High: score > q67. Top20: score ≥ q80 and strictly positive. Linear quantiles; ties retained, so realized proportions can differ in degenerate data. Thresholds and the entire `origin_activity.csv` were written before loading model errors. No alternative score was selected after looking at results. “High excitation” here means relatively large command changes, not a persistently exciting open-loop identification input.

Thresholds:

'''
    thresholds=[]
    for h,channels in protocol['bins']['thresholds'].items():
        for c,t in channels.items():thresholds.append(dict(horizon_s=float(h),channel=c,**t))
    threshold_frame=pd.DataFrame(thresholds);threshold_frame.to_csv(OUT/'activity_thresholds.csv',index=False)
    text+=table(threshold_frame[threshold_frame.horizon_s<=.5])
    text+='''

## 4. Coverage and flight/cohort balance

Overall bins at each primary horizon:

'''+table(coverage[(coverage.channel=='overall')&(coverage.cohort=='ALL')&(coverage.horizon_s<=.5)])
    text+='''

All overall bins retain all 17 flights. At 500 ms High has 323 Sep7 + 529 Sep17 origins; Top20 has 202 + 315. High/Top20 are relatively enriched in Sep17, so separate cohort results are required and provided. Their largest individual-flight shares are about 10.3%/12.0%, not one-flight dominance. Drive High/Top20 include 16 flights; all other 500 ms channel High/Top20 subsets include 17. Missing-flight coverage is explicit, not filled with zero error. Full coverage is in `bin_coverage.csv`; flight rows in `per_flight_all_bins.csv` preserve sparse contributions.

Channel coverage at 500 ms:

'''+table(coverage[(coverage.channel!='overall')&(coverage.cohort=='ALL')&(coverage.horizon_s==.5)&coverage.activity_bin.isin(['High','Top20'])])
    text+='''

## 5. Overall conditional results, all primary horizons and cohorts

RMSE convention matches Step 1: vector error norm per origin, RMS within each flight, then equal-flight macro mean. Attitude is sign-safe quaternion geodesic RMS in degrees. No window-count weighting of flights. Δv and Δω errors equal endpoint v and ω errors under the common observed initial state; repeated columns are not additional independent evidence.

'''
    for c in ['ALL','Sep7','Sep17']:
        text+=f'\n### {c}\n\n'+table(summary[(summary.channel=='overall')&(summary.cohort==c)&(summary.horizon_s<=.5)][['horizon_s','activity_bin','model',*PRIMARY]])+'\n'
    text+='''
## 6. Paired errors and flight bootstrap

For every matched origin, `delta_e = error_B3 - error_B2`; negative favors B3. The pooled origin mean/median/win fraction below are descriptive, not independent-trial statistics. Flight-level estimands and CIs are separate:

1. Mean paired-origin error: compute mean delta_e inside each flight, then average flights equally.
2. Paired flight RMSE difference: compute each model's RMSE inside each flight, subtract B2 from B3, then average flights equally. This equals the difference between conditional summary macro values.

10,000 bootstrap draws resample whole flights within Sep7/Sep17, retaining each represented stratum's flight count. Percentile 95% intervals; fixed control bins; no window bootstrap. The bootstrap RNG seed 17023 is a statistics seed, not an added model seed. Only two days are represented; within-day dependence can make intervals optimistic. Threshold and model-training uncertainty are not included. Channel/horizon comparisons are descriptive and are not multiplicity-adjusted significance tests.

Primary paired results (all bins, 500 ms):

'''+table(paired[(paired.channel=='overall')&(paired.cohort=='ALL')&(paired.horizon_s==.5)&paired.metric.isin(CORE)][[
        'activity_bin','metric','origin_pooled_mean_difference','origin_pooled_median_difference','origin_pooled_B3_win_fraction',
        'flight_equal_mean_paired_difference','mean_paired_ci95_low','mean_paired_ci95_high']])
    text+='\n\nHigh and Top20 flight-RMSE differences and robustness:\n\n'+table(high[[
        'activity_bin','metric','flight_equal_mean_RMSE_difference','RMSE_difference_ci95_low','RMSE_difference_ci95_high','B3_flight_wins','B2_flight_wins','flight_ties']])
    text+='\n\nIndividual-flight High/Top20 results, overall score at 500 ms:\n\n'+table(flights[(flights.channel=='overall')&(flights.horizon_s==.5)&flights.metric.isin(['velocity_rmse_m_s','attitude_error_deg','body_rate_rmse_rad_s','delta_v_rmse_m_s','delta_omega_rmse_rad_s'])][[
        'activity_bin','log_id','cohort','n_origins','metric','B2_error','B3_error','delta']])
    text+='''

## 7. Channel-associated response comparisons

Drive: velocity and NED vertical velocity (positive down), plus all rate/attitude metrics in the CSV. Common elevon: q and pitch-associated orientation error; differential: p and roll-associated error; rudder: r and yaw-associated error. Orientation components use the principal rotation vector `Log(q_truth^-1*q_pred)` in truth-body axes; they are not independent Euler-angle errors. Their norm is the geodesic attitude error. Association with a channel is not an intervention or a physical transfer-function identification.

'''
    channel_rows=[]
    for channel,metrics in CHANNEL_METRICS.items():
        text+=f'\n### {channel}\n\n'
        for c in ['ALL','Sep7','Sep17']:
            text+=f'\n{c}:\n\n'+table(summary[(summary.channel==channel)&(summary.cohort==c)&(summary.horizon_s==.5)&summary.activity_bin.isin(['High','Top20'])][['activity_bin','model','n_origins','n_flights',*metrics,*CORE]])+'\n'
        sub=paired[(paired.channel==channel)&(paired.cohort=='ALL')&(paired.horizon_s==.5)&paired.activity_bin.isin(['High','Top20'])&paired.metric.isin(metrics)]
        channel_rows.append(sub)
        text+='\nPaired flight-RMSE differences:\n\n'+table(sub[['activity_bin','metric','flight_equal_mean_RMSE_difference','RMSE_difference_ci95_low','RMSE_difference_ci95_high','B3_flight_wins','B2_flight_wins']])+'\n'
    pd.concat(channel_rows).to_csv(OUT/'channel_primary_comparison.csv',index=False)
    text+='''
## 8. Figures

Figure 1: error vs control-change bin at nominal 500 ms, including separate cohorts.

'''
    for metric in CORE:text+=f'![Error vs activity](figure1_{metric}.png)\n\n'
    text+='''Figure 2: equal-flight mean of origin-paired B3−B2 error, flight-bootstrap 95% CI; zero line marked. Positive is worse for B3.

![Paired difference](figure2_paired_differences.png)

Figure 3: channel-specific High and Top20 response-associated metrics, 500 ms.

![Channel comparison](figure3_channel_excitation.png)

## 9. Interpretation and limitations

The overall high-excitation advantage hypothesized for B3 is not observed. At 500 ms its velocity/body-rate differences do not cross into a stable favorable region as activity increases, and attitude disadvantage increases. High and Top20 core differences favor B2 in the combined aggregate and in Sep17. Sep7 velocity is effectively tied: High has B3/B2 flight wins 5/4 but near-zero macro difference, while Top20 macro velocity slightly favors B3 with a broad CI crossing zero and B3/B2 wins 4/5. Sep7 attitude/body-rate results still favor B2. Thus the combined velocity disadvantage is substantially driven by Sep17; no stable cross-cohort velocity advantage is claimed for either method on these subsets. Channel point estimates show no 500 ms reversal of this finding. Differential Top20 p-rate is a near-tie with CI spanning zero, so it is not described as a statistically resolved B2 improvement.

An isolated gain is insufficient to reframe the paper around actuator-aware superiority: Low-bin position at 500 ms favors B3 slightly, and individual flights sometimes do, but the requested velocity/attitude/body-rate high-excitation evidence is unfavorable. Tables retain all cases, including negative findings. No score, reference, threshold, checkpoint, loss or model was changed to obtain a win.

This is a conditional prediction diagnostic on previously used validation logs. Feedback commands respond to flight state/disturbances, and multiple channels move together. High change does not imply an independent control intervention or persistent excitation. Max-departure score ignores the pre-origin-to-origin jump and does not separately identify amplitude, bandwidth and duration; TV and magnitude are saved but not searched for favorable bins. Native horizons are approximate wall-clock durations. Step 1 also has training-family differences (frozen B3 backbone with actuator-only second stage versus fully trainable B2 continuation, actuator-specific regularization, mixed 17/29 stage seeds). No strict architectural causality is inferred from this comparison.

## 10. Sealed status, validation and reproducibility

No sealed/reserved raw flight, sample, prediction or evaluation was opened. Six Sep8 sealed and 23 Sep19 reserved identities are only exclusion metadata inherited from Step 1. No train data were loaded or statistics refitted; no model inference, optimization, checkpoint write, new model seed, RL or OOD analysis occurred. Checkpoint and baseline result hashes remain unchanged.

`sanity_checks.json` records parity/immutability; `unit_tests.json` records focused tests. `protocol.json` records exact checkpoints, origin hash, control representation/scales, all quantiles, metric definitions and bootstrap estimands. Supplementary 1 s values and all axis/Δ metrics remain in the CSVs. `paired_summary.csv` contains all origin-descriptive and flight-bootstrap results; `per_flight_high_excitation.csv` contains the requested individual-flight comparison.

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_paper_control_conditioned.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/report_paper_control_conditioned.py
```

The evaluator refuses to overwrite completed outputs. An incomplete run can resume only with identical control thresholds/activity hash. Report generation uses only the frozen conditional CSVs. The report is a decision deliverable; no subsequent experiment is started.

Suggested commit message: `feat: add frozen control-conditioned baseline evaluation`
'''
    (OUT/'report.md').write_text(text)
    decision=dict(outcome='C',scope='frozen single-replicate nominal 500 ms control-conditioned prediction comparison',
        recommendation='Standard GRU as paper main model; actuator-aware as negative ablation/discussion/supplementary',
        exceptions=['Low 500ms position slightly favors B3','Differential Top20 p-rate CI crosses zero','Sep7 High/Top20 velocity approximately tied; Top20 mean slightly favors B3 with CI crossing zero','Some individual flights/shorter-horizon cells favor B3'],
        next_step_started=False,strict_architectural_causality_claim=False)
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    protocol['report_source_sha256']=sha(__file__)
    protocol['state']='completed'
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    (OUT/'git_status_at_completion.txt').write_text(subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True))
    frozen=protocol['frozen_input_sha256']
    for p,h in frozen.items():
        if sha(ROOT/p)!=h:raise ValueError(f'frozen input changed: {p}')
    artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',outcome='C',
        origin_count=2582,training_run=False,inference_run=False,new_model_seeds=0,
        sealed_test_opened=False,reserved_test_opened=False,frozen_inputs_unchanged=True,
        artifact_sha256=artifacts),indent=2)+'\n')
    print('Completed control-conditioned report and figures; Outcome C; no next step started.')


if __name__=='__main__':main()
