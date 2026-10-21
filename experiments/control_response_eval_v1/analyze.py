"""Response contrasts, seed-specific metrics and flight-dependence-aware summaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate import HERE,OUT,increments,contrast,response_error,flight_weights,write_json,file_hash
METRICS=['velocity','rate','position','attitude']

def weighted_quantile(x,w,q):
    order=np.argsort(x);x=np.asarray(x)[order];w=np.asarray(w)[order]
    return float(x[np.searchsorted(np.cumsum(w)/np.sum(w),q).clip(0,len(x)-1)])

def collect():
    cfg=json.loads((HERE/'config.json').read_text());pairs=pd.read_parquet(OUT/'cohorts.parquet');rows=[];brows=[]
    for h in cfg['horizons_ms']:
        z=np.load(OUT/f'predictions_{h}ms.npz');a,b=z['i'],z['j'];P=len(a);points=z['point_ms']
        truth_inc=increments(z['truth'],z['truth_initial'])
        ta={k:v[a] for k,v in truth_inc.items()};tb={k:v[b] for k,v in truth_inc.items()}
        timesA=z['actual_time_s'][a];timesB=z['actual_time_s'][b]
        for model in ['GRU-NoControl','GRU-Control']:
            for seed in cfg['seeds']:
                key=f'{model}_{seed}';pi=increments(z[key],z['initial']);pa={k:v[a] for k,v in pi.items()};pb={k:v[b] for k,v in pi.items()}
                si=increments(z[key+'_swapped'],z['initial'][z['recipient']]);ab={k:v[:P] for k,v in si.items()};ba={k:v[P:] for k,v in si.items()}
                ri=increments(z[key+'_refined'],z['initial'][z['recipient']]);ra={k:v[:P] for k,v in ri.items()};rb={k:v[P:] for k,v in ri.items()}
                ci=increments(z[key+'_common'],z['initial'][z['recipient']]);csi=increments(z[key+'_common_swapped'],z['initial'][z['recipient']])
                ca={k:v[:P] for k,v in ci.items()};cb={k:v[P:] for k,v in ci.items()}
                cab={k:v[:P] for k,v in csi.items()};cba={k:v[P:] for k,v in csi.items()}
                for metric in METRICS:
                    true=contrast(ta,tb,metric);pred=contrast(pa,pb,metric);error=response_error(pred,true,metric)
                    norm=np.linalg.norm(true,axis=-1);pnorm=np.linalg.norm(pred,axis=-1)
                    cosine=(true*pred).sum(-1)/np.maximum(norm*pnorm,1e-15)
                    cosine=np.where((norm>1e-8)&(pnorm>1e-8),cosine,np.nan)
                    uA=contrast(pa,ab,metric);uB=contrast(ba,pb,metric)
                    model_control=.5*(uA+uB);history=pred-model_control
                    if metric!='attitude':
                        # Exact additive decomposition of the model's A contrast, not a causal estimate.
                        alternative=.5*((pa[metric]-ba[metric])+(ab[metric]-pb[metric]))
                        np.testing.assert_allclose(history,alternative,atol=1e-12)
                    na,nb=np.linalg.norm(uA,axis=-1),np.linalg.norm(uB,axis=-1)
                    consistency=(uA*uB).sum(-1)/np.maximum(na*nb,1e-15)
                    consistency=np.where((na>1e-6)&(nb>1e-6),consistency,np.nan)
                    sham_a=np.linalg.norm(contrast(pa,ra,metric),axis=-1)
                    sham_b=np.linalg.norm(contrast(pb,rb,metric),axis=-1)
                    commonA=contrast(ca,cab,metric);commonB=contrast(cba,cb,metric)
                    common_a=np.linalg.norm(commonA,axis=-1);common_b=np.linalg.norm(commonB,axis=-1)
                    native_vector=.5*(uA+uB);common_vector=.5*(commonA+commonB)
                    gridcos=(native_vector*common_vector).sum(-1)/np.maximum(np.linalg.norm(native_vector,axis=-1)*np.linalg.norm(common_vector,axis=-1),1e-15)
                    errA=response_error(contrast(pa,ta,metric),np.zeros_like(pred),metric)
                    errB=response_error(contrast(pb,tb,metric),np.zeros_like(pred),metric)
                    for i,pidx in enumerate(z['pair_index']):
                        for j,t in enumerate(points):
                            common=dict(pair_uid=f'{h}:{pidx}',horizon_ms=h,point_ms=int(t),model=model,seed=seed,metric=metric)
                            rows.append(dict(**common,error=float(error[i,j]),squared_error=float(error[i,j]**2),
                                true_amplitude=float(norm[i,j]),pred_amplitude=float(pnorm[i,j]),
                                amplitude_error=float(abs(norm[i,j]-pnorm[i,j])),cosine=float(cosine[i,j]),
                                predicted_zero=pnorm[i,j]<=1e-6,per_trajectory_increment_mse=float(.5*(errA[i,j]**2+errB[i,j]**2)),
                                time_a_s=float(timesA[i,j]),time_b_s=float(timesB[i,j]),
                                **{f'true_{axis}':float(true[i,j,k]) for k,axis in enumerate('xyz')},
                                **{f'pred_{axis}':float(pred[i,j,k]) for k,axis in enumerate('xyz')}))
                            brows.append(dict(**common,swap_norm_a=float(na[i,j]),swap_norm_b=float(nb[i,j]),
                                swap_cosine=float(consistency[i,j]),timing_refinement_norm=float(.5*(sham_a[i,j]+sham_b[i,j])),
                                timing_refinement_squared_norm=float(.5*(sham_a[i,j]**2+sham_b[i,j]**2)),
                                control_contribution_norm=float(np.linalg.norm(model_control[i,j])),
                                history_contribution_norm=float(np.linalg.norm(history[i,j])),
                                common_grid_norm=float(.5*(common_a[i,j]+common_b[i,j])),
                                common_grid_squared_norm=float(.5*(common_a[i,j]**2+common_b[i,j]**2)),native_common_cosine=float(gridcos[i,j]),
                                **{f'swap_a_{axis}':float(uA[i,j,k]) for k,axis in enumerate('xyz')},
                                **{f'swap_b_{axis}':float(uB[i,j,k]) for k,axis in enumerate('xyz')}))
    meta=pairs.drop(columns=['horizon_ms']).copy()
    response=pd.DataFrame(rows).merge(meta,on='pair_uid',validate='many_to_many')
    sensitivity=pd.DataFrame(brows).merge(meta,on='pair_uid',validate='many_to_many')
    response.to_parquet(OUT/'response_metrics.parquet',index=False);sensitivity.to_parquet(OUT/'swap_metrics.parquet',index=False)
    # Direction eligibility from training references only; no pairs are dropped from RMSE/MAE.
    thresholds=[]
    for (h,t,metric),g in response.query('cohort=="small_control" and split=="train"').groupby(['horizon_ms','point_ms','metric']):
        g=g.drop_duplicates('pair_uid');floor=cfg['direction_floors'][metric]
        q=weighted_quantile(g.true_amplitude.to_numpy(),g.flight_weight.to_numpy(),.9)
        thresholds.append(dict(horizon_ms=h,point_ms=t,metric=metric,threshold=max(floor,q),reference_p90=q,
            reference_pairs=len(g),reference_flights=len(set(g.flight_a)|set(g.flight_b)),reference_components=g.component.nunique(),
            reference_adequate=len(g)>=10 and g.component.nunique()>=5,source='train small_control only'))
    th=pd.DataFrame(thresholds);th.to_csv(OUT/'direction_thresholds.csv',index=False)
    return response,sensitivity,th

def summarize(response,swap,th):
    cfg=json.loads((HERE/'config.json').read_text());r=response.merge(th,on=['horizon_ms','point_ms','metric'],validate='many_to_one')
    r['effective']=r.true_amplitude>r.threshold
    r['correct']=(r.cosine>0)&(~r.predicted_zero)
    basekeys=['horizon_ms','point_ms','cohort','split','model','metric']
    output=[];directions=[]
    for keys,g in r.groupby(basekeys):
        for seed in [17,29,43,'all']:
            d=g if seed=='all' else g[g.seed==seed]
            w=d.flight_weight.to_numpy();w=w/w.sum();w0=np.full(len(d),1/len(d))
            mse=float(np.sum(w*d.squared_error));amp2=float(np.sum(w*d.true_amplitude**2));effective=d.effective.to_numpy()
            row=dict(zip(basekeys,keys));row.update(seed=str(seed),pairs=d.pair_uid.nunique(),flights=len(set(d.flight_a)|set(d.flight_b)),components=d.component.nunique(),
                response_rmse=np.sqrt(mse),response_mae=np.sum(w*d.error),amplitude_mae=np.sum(w*d.amplitude_error),
                true_response_rms=np.sqrt(amp2),pred_response_rms=np.sqrt(np.sum(w*d.pred_amplitude**2)),
                zero_response_skill=1-mse/amp2 if amp2>0 else np.nan,
                error_to_response_ratio=np.sqrt(mse/amp2) if amp2>0 else np.nan,
                equal_pair_rmse=np.sqrt(np.sum(w0*d.squared_error)),trajectory_increment_rmse=np.sqrt(np.sum(w*d.per_trajectory_increment_mse)),
                effective_pairs=d.loc[d.effective,'pair_uid'].nunique(),direction_accuracy=np.sum(w[effective]*d.correct.to_numpy()[effective])/w[effective].sum() if effective.any() else np.nan,
                direction_threshold=d.threshold.iloc[0],direction_reference_adequate=bool(d.reference_adequate.iloc[0]))
            output.append(row)
            if seed=='all':
                for mult in cfg['direction_sensitivity_multipliers']:
                    mask=d.true_amplitude.to_numpy()>mult*d.threshold.to_numpy()
                    directions.append(dict(zip(basekeys,keys))|dict(multiplier=mult,effective_pairs=d.loc[mask,'pair_uid'].nunique(),
                        accuracy=float(np.average(d.correct.to_numpy()[mask],weights=w[mask])) if mask.any() else np.nan))
    summary=pd.DataFrame(output);summary.to_csv(OUT/'response_summary.csv',index=False)
    pd.DataFrame(directions).to_csv(OUT/'direction_sensitivity.csv',index=False)
    # Seed-averaged squared error contrasts: model seeds are not independent flight replicates.
    contrast_rows=[];group_rows=[];component_rows=[];rng=np.random.default_rng(cfg['random_seed'])
    for keys,g in r.groupby(['horizon_ms','point_ms','cohort','split','metric']):
        per=g.groupby(['pair_uid','model']).squared_error.mean().unstack()
        meta=g.drop_duplicates('pair_uid').set_index('pair_uid').loc[per.index]
        w=meta.flight_weight.to_numpy();w/=w.sum();a=per['GRU-Control'].to_numpy();b=per['GRU-NoControl'].to_numpy()
        row=dict(zip(['horizon_ms','point_ms','cohort','split','metric'],keys))
        rmsec=np.sqrt(np.sum(w*a));rmsen=np.sqrt(np.sum(w*b));comp=meta.component.to_numpy();unique=np.unique(comp)
        low=high=np.nan
        if len(unique)>=cfg['minimum_components_for_interval']:
            # Resample disconnected components as whole units, retain flight-equal weights inside units.
            mass=np.array([w[comp==c].sum() for c in unique]);ca=np.array([np.sum(w[comp==c]*a[comp==c]) for c in unique]);cb=np.array([np.sum(w[comp==c]*b[comp==c]) for c in unique])
            ix=rng.integers(len(unique),size=(cfg['bootstrap_replicates'],len(unique)))
            delta=np.sqrt(ca[ix].sum(1)/mass[ix].sum(1))-np.sqrt(cb[ix].sum(1)/mass[ix].sum(1))
            low,high=np.quantile(delta,[.025,.975])
        fd=meta.flight_disjoint.to_numpy(bool);fddelta=np.sqrt(a[fd].mean())-np.sqrt(b[fd].mean()) if fd.any() else np.nan
        contrast_rows.append(row|dict(control_rmse=rmsec,nocontrol_rmse=rmsen,improvement_pct=100*(1-rmsec/rmsen),
            rmse_delta=rmsec-rmsen,component_bootstrap_low=low,component_bootstrap_high=high,components=len(unique),
            interval_available=len(unique)>=cfg['minimum_components_for_interval'],flight_disjoint_pairs=int(fd.sum()),flight_disjoint_rmse_delta=fddelta,
            better_pairs=int((a<b).sum()),pairs=len(a)))
        for c in unique:
            for mode,mask in [('component_only',comp==c),('leave_component_out',comp!=c)]:
                if not mask.any():continue
                ww=w[mask]/w[mask].sum()
                component_rows.append(row|dict(component=int(c),mode=mode,pairs=int(mask.sum()),
                    control_rmse=np.sqrt(np.sum(ww*a[mask])),nocontrol_rmse=np.sqrt(np.sum(ww*b[mask]))))
        # Predetermined descriptive bins; no response-based sample selection.
        bins=dict(match_score=np.where(meta.score<=.5,'<=0.5','>0.5'),
                  control_strength=np.where(meta[['future_gap_'+c for c in ['motor','left_elevon','right_elevon','rudder']]].to_numpy().max(1)>=.2,'max RMS >=0.2','max RMS <0.2'),
                  same_firmware=meta.same_firmware.astype(str).to_numpy(),
                  any_review=meta.any_review.astype(str).to_numpy(),
                  high_dynamics=meta.high_dynamics.astype(str).to_numpy())
        for field,labels in bins.items():
            for label in np.unique(labels):
                mask=labels==label;ww=w[mask]/w[mask].sum()
                group_rows.append(row|dict(grouping=field,label=label,pairs=int(mask.sum()),control_rmse=np.sqrt(np.sum(ww*a[mask])),nocontrol_rmse=np.sqrt(np.sum(ww*b[mask]))))
    pd.DataFrame(contrast_rows).to_csv(OUT/'paired_comparisons.csv',index=False);pd.DataFrame(group_rows).to_csv(OUT/'condition_comparisons.csv',index=False)
    pd.DataFrame(component_rows).to_csv(OUT/'component_sensitivity.csv',index=False)
    brows=[]
    for keys,g in swap.groupby(basekeys):
        for seed in [17,29,43,'all']:
            d=g if seed=='all' else g[g.seed==seed];w=d.flight_weight.to_numpy();w/=w.sum()
            norm=.5*(d.swap_norm_a.to_numpy()+d.swap_norm_b.to_numpy());valid=np.isfinite(d.swap_cosine)
            c=d.control_contribution_norm.to_numpy();hist=d.history_contribution_norm.to_numpy()
            brows.append(dict(zip(basekeys,keys))|dict(seed=str(seed),pairs=d.pair_uid.nunique(),
                swap_rms=np.sqrt(np.sum(w*.5*(d.swap_norm_a**2+d.swap_norm_b**2))),swap_amplitude_mean=np.sum(w*norm),
                history_direction_agreement=np.average((d.swap_cosine[valid]>0),weights=w[valid]) if valid.any() else np.nan,
                history_cosine_mean=np.average(d.swap_cosine[valid],weights=w[valid]) if valid.any() else np.nan,
                insensitive_fraction=np.sum(w*(norm<1e-3)),timing_refinement_rms=np.sqrt(np.sum(w*d.timing_refinement_squared_norm)),
                common_grid_response_rms=np.sqrt(np.sum(w*d.common_grid_squared_norm)),
                native_common_direction_agreement=np.sum(w*(d.native_common_cosine>0)),
                model_control_share=np.sum(w*np.divide(c,c+hist,out=np.zeros_like(c),where=c+hist>1e-8))))
    pd.DataFrame(brows).to_csv(OUT/'swap_summary.csv',index=False)
    return summary

def main():
    r,b,t=collect();summarize(r,b,t)
    write_json(OUT/'analysis_manifest.json',dict(code_sha256=file_hash(Path(__file__)),
        aggregation='equal flight: each flight averages incident pairs; then equal flights; seeds average squared errors, not trajectories',
        direction='positive dot product (>0 cosine) only above train small-control amplitude p90/floor; zero predictions incorrect',
        uncertainty='connected components of the shared-flight pair graph; fewer than five components => no interval',
        noise_reference='observational small-command-difference pairs, not pure sensor noise or a causal lower bound'))
    print('Response metrics and dependency-aware summaries complete')

if __name__=='__main__':main()
