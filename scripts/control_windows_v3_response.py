"""Complete-context observational waveform diagnostics; no controller tuning."""
import json
import numpy as np
import pandas as pd
import torch
from control_windows_v3 import OUT,OLD,ROOT,ART,SEEDS,base,model_from
from system_identification.data.reference_windows import extract_window


def causal_mean(values,width=10):
    c=np.concatenate([np.zeros((1,values.shape[1])),np.cumsum(values,axis=0)],axis=0)
    return (c[width:]-c[:-width])/width


def native_interval(a,left,right):
    names=list(a['age_topic_names']);col=names.index('vehicle_attitude_setpoint')
    source=np.rint(a['timestamp_us'][left:right]-a['age_s'][left:right,col]*1e6).astype(np.int64)
    intervals=np.diff(np.unique(source))/1e6
    return float(np.median(intervals[intervals>0]))


def waveform_summary(pred,truth,past,precontext,dt=.02):
    """Ten-sample causal averages; peak alignment is not physical actuator delay."""
    baseline=past[-10:].mean(0)
    observed=causal_mean(np.concatenate([past[-9:],truth]))-baseline
    forecast=causal_mean(np.concatenate([past[-9:],pred]))-baseline
    background=causal_mean(precontext)
    scale=1.4826*np.median(np.abs(background-np.median(background,axis=0)),axis=0)
    rows=[]
    for axis in range(3):
        t=observed[:,axis];p=forecast[:,axis]
        amplitude=float(np.max(np.abs(t)))
        corr=float(np.corrcoef(t,p)[0,1]) if t.std()>1e-8 and p.std()>1e-8 else None
        rows.append(dict(axis=['p','q','r'][axis],observed_peak=amplitude,
            predicted_peak=float(np.max(np.abs(p))),
            amplitude_error=float(abs(np.max(np.abs(p))-amplitude)),
            waveform_rmse=float(np.sqrt(np.mean((p-t)**2))),
            waveform_correlation=corr,
            peak_time_error_s=float(abs(np.argmax(np.abs(t))-np.argmax(np.abs(p)))*dt),
            terminal_mean_error=float(abs(p[-50:].mean()-t[-50:].mean())),
            measurable=bool(amplitude>max(3*float(scale[axis]),.05)),background_scale=float(scale[axis])))
    return rows,observed,forecast


def main(baseline_folder=None):
    out=OUT/'response';out.mkdir(exist_ok=False)
    base.write(out/'protocol.json',dict(models=['A800','A_long1600'],seeds=SEEDS,
        selection='all19 complete validation input-event contexts fixed by v2; no response-based removal',
        free_run='one observed initialization then full remaining context; 3.7-37.7s is extrapolation beyond1s training and not a nominal1s score',
        waveform='0.2s causal mean rate, baseline last0.2s observed history; all p/q/r coupling retained',
        measurable='observed peak magnitude > max(3*preinput background MAD scale,0.05rad/s); unmeasurable cases retained',
        timing='time of absolute waveform peak, NOT isolated input-response/physical actuator delay',
        nominal='same disjoint1s validation predictions restricted to support fully inside each complete context, separate from full-run stress',
        model_choice='A_long was predeclared before training; not selected from validation outcomes',sealed_test_accessed=False))
    clusters=pd.read_csv(OLD/'event_clusters.csv');clusters=clusters[clusters.complete_context]
    w=pd.read_csv(OLD/'windows_v2.csv');v=w[w.partition=='validation'];cache={}
    for log in clusters.log_id.unique():
        with np.load(ROOT/w[w.log_id==log].cache.iloc[0]) as z:cache[log]={k:z[k] for k in z.files}
    rows=[];nominal=[];traces={}
    for arm,folder in [('A',baseline_folder or ROOT/'artifacts/control_windows_v2'),('A_long',ART)]:
        for seed in SEEDS:
            model=model_from(folder/f'{arm}_{seed}/final.pt')
            with np.load(folder/f'{arm}_{seed}/validation_predictions.npz') as z:
                array_to_row={int(i):j for j,i in enumerate(z['array_index'])}
                one_second=z['rates_b']
            for c in clusters.itertuples():
                a=cache[c.log_id];origin=int(c.start_index)+26;horizon=int(c.stop_index)-origin-1
                x,y=extract_window(a,origin,horizon=horizon)
                with torch.no_grad():
                    p=model(**{k:torch.tensor(value[None],dtype=torch.float32) for k,value in x.items()})
                forecast=p['rates_b'][0].numpy();truth=y['rates_b'];assert np.isfinite(forecast).all()
                # Complete contexts start exactly one second before the first event.
                pre=a['rates_b'][c.start_index:c.start_index+50]
                statistics,observed,smoothed=waveform_summary(forecast,truth,x['history_state'][:,3:6],pre)
                interval=native_interval(a,c.start_index,c.stop_index)
                for row in statistics:
                    rows.append(dict(arm=arm,seed=seed,event_cluster_id=c.event_cluster_id,log_id=c.log_id,
                        horizon_s=horizon*.02,reference_interval_s=interval,
                        timing_target_s=max(.04,2*interval),**row))
                key=f'{arm}_{seed}_{c.event_cluster_id}'
                traces[key+'_predicted']=smoothed;traces[key+'_observed']=observed
                inside=v[(v.log_id==c.log_id)&(v.origin_index-26>=c.start_index)&(v.origin_index+50<c.stop_index)]
                for sample in inside.itertuples():
                    real=a['rates_b'][sample.origin_index+1:sample.origin_index+51]
                    pred=one_second[array_to_row[sample.array_index]]
                    for axis,name in enumerate(['p','q','r']):
                        nominal.append(dict(arm=arm,seed=seed,event_cluster_id=c.event_cluster_id,log_id=c.log_id,
                            window_id=sample.window_id,axis=name,mse=float(np.mean((real[:,axis]-pred[:,axis])**2))))
    pd.DataFrame(rows).to_csv(out/'full_context_waveforms.csv',index=False)
    pd.DataFrame(nominal).to_csv(out/'nominal_one_second_windows.csv',index=False)
    np.savez_compressed(out/'waveform_traces.npz',**traces)
    r=pd.DataFrame(rows);n=pd.DataFrame(nominal)
    means=r.groupby(['arm','axis'])[['waveform_rmse','amplitude_error','peak_time_error_s','terminal_mean_error']].mean()
    means.to_csv(out/'full_context_summary.csv')
    nominal_means=np.sqrt(n.groupby(['arm','seed','log_id','axis']).mse.mean()).groupby(['arm','axis']).mean()
    nominal_means.to_csv(out/'nominal_summary.csv')
    events=pd.read_csv(OLD/'events.csv');gates={}
    for axis,number in [('p',0),('q',1)]:
        scored=r[(r.axis==axis)&r.measurable]
        amp=scored.groupby(['arm','seed','log_id']).amplitude_error.mean().groupby('arm').mean()
        candidate=scored[scored.arm=='A_long']
        gain=float(1-amp['A_long']/amp['A']) if len(amp)==2 and amp['A']>0 else None
        timing=float(candidate.peak_time_error_s.median()) if len(candidate) else None
        limit=float(candidate.timing_target_s.median()) if len(candidate) else None
        eligible=set(candidate.event_cluster_id);ev=events[(events.axis==number)&events.event_cluster_id.isin(eligible)]
        coverage={}
        for sign in [-1,1]:
            subset=ev[ev.sign==sign]
            coverage[str(sign)]=dict(clusters=int(subset.event_cluster_id.nunique()),flights=int(subset.log_id.nunique()))
        coverage_pass=all(v['clusters']>=10 and v['flights']>=3 for v in coverage.values())
        passed=gain is not None and gain>=.1 and timing<=limit and coverage_pass
        gates[axis]=dict(amplitude_error_improvement=gain,median_peak_time_error_s=timing,
            timing_target_s=limit,measurable_coverage=coverage,passed=bool(passed))
    base.write(out/'status.json',dict(status='complete',contexts=len(clusters),flights=int(clusters.log_id.nunique()),
        waveform_rows=len(r),measurable_fraction=float(r[r.arm=='A'].measurable.mean()),
        G2_observational_thresholds_passed=all(v['passed'] for v in gates.values()),axis_gates=gates,
        scope='mixed-reference natural-record waveform alignment, not isolated physical delay; full-context free-run beyond nominal1s must remain separate from nominal error',
        sealed_test_accessed=False))
    print(means);print(nominal_means)


if __name__=='__main__':
    torch.set_num_threads(1);main()
