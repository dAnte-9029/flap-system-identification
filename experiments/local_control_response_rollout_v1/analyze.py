"""Outcome analysis on frozen memberships, with no extrapolation or reselection."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
VARIABLES=['vx','vy','vz','p','q','r']
CHANNELS=[3,4,5,10,11,12]


def interpolate(t,y,at):
    """Interpolation is an evaluation operation, never a model input."""
    t=np.asarray(t,float);at=np.asarray(at,float)
    assert np.all(np.diff(t)>0)
    good=(at>=t[0]-1e-10)&(at<=t[-1]+1e-10)
    result=np.full((len(at),y.shape[1]),np.nan)
    for j in range(y.shape[1]):result[good,j]=np.interp(at[good],t,y[:,j])
    return result,good


def labels(pairs):
    parent={x:x for x in sorted(set(pairs.flight_a)|set(pairs.flight_b))}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for r in pairs.itertuples():parent[find(r.flight_a)]=find(r.flight_b)
    keys={x:i for i,x in enumerate(sorted({find(x) for x in parent}))}
    return {r.pair_uid:keys[find(r.flight_a)] for r in pairs.itertuples()}


def weights(pairs,kind):
    if kind=='pair':return dict.fromkeys(pairs.pair_uid,1/len(pairs))
    if kind=='flight':
        counts=pd.Series(np.r_[pairs.flight_a,pairs.flight_b]).value_counts()
        w=np.array([1/counts[a]+1/counts[b] for a,b in zip(pairs.flight_a,pairs.flight_b)])
    elif kind=='component':
        comp=labels(pairs);counts=pd.Series(list(comp.values())).value_counts()
        w=np.array([1/counts[comp[x]] for x in pairs.pair_uid])
    else:raise ValueError(kind)
    return dict(zip(pairs.pair_uid,w/w.sum()))


def metrics(d,w,floor):
    ok=d.supported.to_numpy(bool)
    d=d[ok];w=np.asarray(w)[ok]
    if not len(d):return dict(n=0)
    w=w/w.sum();real=d.real.to_numpy();pred=d.pred.to_numpy();err=pred-real
    avg=lambda x:float(np.dot(w,np.asarray(x)))
    active=np.abs(real)>floor
    aw=w[active];aw=aw/aw.sum() if aw.sum() else aw
    direction=np.where(np.abs(pred)<=1e-8,0,np.sign(pred))
    ea=d.increment_error_a.to_numpy();eb=d.increment_error_b.to_numpy()
    denom=avg(real**2)
    return dict(n=len(d),pairs=d.pair_uid.nunique(),rmse=np.sqrt(avg(err**2)),mae=avg(np.abs(err)),bias=avg(err),
        real_rms=np.sqrt(denom),pred_rms=np.sqrt(avg(pred**2)),
        signed_gain=avg(pred*real)/denom if denom>1e-15 else np.nan,
        scale_ratio=np.sqrt(avg(pred**2)/denom) if denom>1e-15 else np.nan,
        near_zero_fraction=avg(~active),active_pairs=d.loc[active,'pair_uid'].nunique(),
        direction_accuracy=float(np.dot(aw,direction[active]==np.sign(real[active]))) if len(aw) else np.nan,
        zero_prediction_fraction=float(np.dot(aw,direction[active]==0)) if len(aw) else np.nan,
        relative_mae=float(np.dot(aw,np.abs(err[active])/np.abs(real[active]))) if len(aw) else np.nan,
        magnitude_ratio=float(np.dot(aw,np.abs(pred[active])/np.abs(real[active]))) if len(aw) else np.nan,
        trajectory_rmse=np.sqrt(avg((d.absolute_error_a.to_numpy()**2+d.absolute_error_b.to_numpy()**2)/2)),
        increment_rmse=np.sqrt(avg((ea**2+eb**2)/2)),
        cancellation_ratio=avg(err**2)/avg(ea**2+eb**2) if avg(ea**2+eb**2)>1e-15 else np.nan,
        causal_origin_rmse=np.sqrt(avg((pred-d.causal_real.to_numpy())**2)))


def response_rows(z,windows,pairs,cfg):
    rows=[];support=[];curve=[]
    models=[('GRU-Control',s) for s in cfg['seeds']]+[('GRU-NoControl',s) for s in cfg['seeds']]+[('Hold',0),('LinearTrend',0)]
    for pair in pairs.itertuples():
        a,b=int(pair.a),int(pair.b)
        na,nb=int(z['future_count'][a]),int(z['future_count'][b])
        ta=z['native_time_us'][a,:na+1].astype(float)*1e-6
        tb=z['native_time_us'][b,:nb+1].astype(float)*1e-6
        endpoint=min(.5,ta[-1],tb[-1])
        at=np.r_[np.arange(0,501,cfg['curve_grid_ms'])*1e-3,endpoint]
        kinds=['grid']*(len(at)-1)+['supported_endpoint']
        ya=np.vstack([z['truth_initial'][a],z['truth'][a,:na]])[:,CHANNELS]
        yb=np.vstack([z['truth_initial'][b],z['truth'][b,:nb]])[:,CHANNELS]
        real_a,sa=interpolate(ta,ya,at);real_b,sb=interpolate(tb,yb,at)
        valid=sa&sb
        dreal=(real_a-ya[0])-(real_b-yb[0])
        x0a=z['initial'][a,CHANNELS].astype(float);x0b=z['initial'][b,CHANNELS].astype(float)
        causal=(real_a-x0a)-(real_b-x0b)
        support.append(dict(pair_uid=pair.pair_uid,split=pair.split,pair_id=pair.pair_id,
            endpoint_a_s=ta[-1],endpoint_b_s=tb[-1],supported_endpoint_s=endpoint,exact_500ms_supported=bool(valid[-2]),
            p_initial_alignment_a=ya[0,3]-x0a[3],p_initial_alignment_b=yb[0,3]-x0b[3]))
        for model,seed in models:
            pred=z[f'{model}_{seed}']
            pa=np.vstack([z['initial'][a],pred[a,:na]])[:,CHANNELS].astype(float)
            pb=np.vstack([z['initial'][b],pred[b,:nb]])[:,CHANNELS].astype(float)
            pred_a,_=interpolate(ta,pa,at);pred_b,_=interpolate(tb,pb,at)
            dpred=(pred_a-x0a)-(pred_b-x0b)
            ea=(pred_a-x0a)-(real_a-ya[0]);eb=(pred_b-x0b)-(real_b-yb[0])
            for k,(t,kind) in enumerate(zip(at,kinds)):
                for j,var in enumerate(VARIABLES):
                    rows.append(dict(pair_uid=pair.pair_uid,pair_id=pair.pair_id,split=pair.split,
                        model=model,seed=seed,variable=var,time_ms=t*1000,time_kind=kind,supported=bool(valid[k]),
                        real=dreal[k,j],pred=dpred[k,j],error=dpred[k,j]-dreal[k,j],
                        real_increment_a=real_a[k,j]-ya[0,j],real_increment_b=real_b[k,j]-yb[0,j],
                        pred_increment_a=pred_a[k,j]-x0a[j],pred_increment_b=pred_b[k,j]-x0b[j],
                        increment_error_a=ea[k,j],increment_error_b=eb[k,j],
                        absolute_error_a=pred_a[k,j]-real_a[k,j],absolute_error_b=pred_b[k,j]-real_b[k,j],
                        causal_real=causal[k,j]))
            ix=np.flatnonzero(valid);ix=ix[np.argsort(at[ix],kind='stable')]
            _,first=np.unique(at[ix],return_index=True);ix=ix[first]
            t=at[ix];r=dreal[ix,3];p=dpred[ix,3];error=p-r
            real_rms=np.sqrt(np.trapz(r*r,t)/t[-1]);pred_rms=np.sqrt(np.trapz(p*p,t)/t[-1])
            ri=np.argmax(np.abs(r));pi=np.argmax(np.abs(p))
            peak_valid=real_rms>cfg['direction_floors']['p'] and pred_rms>1e-8
            curve.append(dict(pair_uid=pair.pair_uid,split=pair.split,model=model,seed=seed,
                duration_s=t[-1],curve_rmse=np.sqrt(np.trapz(error*error,t)/t[-1]),
                curve_bias=np.trapz(error,t)/t[-1],real_rms=real_rms,pred_rms=pred_rms,
                real_peak=r[ri],pred_peak=p[pi],real_peak_ms=t[ri]*1000,pred_peak_ms=t[pi]*1000,
                peak_lag_ms=(t[pi]-t[ri])*1000 if peak_valid else np.nan,
                correlation=np.corrcoef(p,r)[0,1] if np.std(r)>1e-8 and np.std(p)>1e-8 else np.nan))
    return pd.DataFrame(rows),pd.DataFrame(support),pd.DataFrame(curve)


def aggregate(responses,pairs,membership,cfg,out):
    points=responses[(responses.time_kind=='supported_endpoint')|((responses.time_kind=='grid')&responses.time_ms.round(6).isin(cfg['report_times_ms']))].copy()
    points['time_label']=np.where(points.time_kind=='supported_endpoint','supported_endpoint',points.time_ms.round().astype(int).astype(str))
    summaries=[];group_rows=[];leave=[];sensitivity=[]
    pairs=pairs.set_index('pair_uid',drop=False)
    for (split,cohort),mem in membership.groupby(['split','cohort'],sort=False):
        cp=pairs.loc[mem.pair_uid]
        d=points[points.pair_uid.isin(mem.pair_uid)]
        graph=labels(cp)
        for (model,var,tlabel),g in d.groupby(['model','variable','time_label'],sort=False):
            for seed in ['pooled']+sorted(g.seed.unique().tolist()):
                dd=g if seed=='pooled' else g[g.seed==seed]
                effective=cp.loc[cp.index.intersection(dd.loc[dd.supported,'pair_uid'].unique())]
                for weighting in ['pair','flight','component']:
                    if not len(effective):continue
                    wm=weights(effective,weighting)
                    ww=dd.pair_uid.map(wm).fillna(0).to_numpy()
                    summary=dict(split=split,cohort=cohort,model=model,seed=seed,variable=var,time_label=tlabel,
                        weighting=weighting,frozen_pairs=len(cp),supported_pairs=len(effective),
                        flights=len(set(effective.flight_a)|set(effective.flight_b)),components=len(set(labels(effective).values())),
                        **metrics(dd,ww,cfg['direction_floors'][var]))
                    summaries.append(summary)
                    if var=='p' and weighting=='flight' and seed=='pooled':
                        for scale in cfg['direction_sensitivity_multipliers']:
                            mm=metrics(dd,ww,cfg['direction_floors'][var]*scale)
                            sensitivity.append(dict(split=split,cohort=cohort,model=model,time_label=tlabel,scale=scale,
                                active_pairs=mm.get('active_pairs'),direction_accuracy=mm.get('direction_accuracy'),near_zero_fraction=mm.get('near_zero_fraction')))
                if var!='p' or tlabel!='supported_endpoint' or cohort not in ['main','small_pair']:continue
                # Report incident pairs within each flight/date/controller and connected component.
                for field,values in [('flight',sorted(set(cp.flight_a)|set(cp.flight_b))),
                                     ('date',sorted(set(cp.date_a)|set(cp.date_b))),
                                     ('controller',sorted(set(cp.controller_a)|set(cp.controller_b))),
                                     ('component',sorted(set(graph.values())))]:
                    for value in values:
                        if field=='component':uids=[u for u,c in graph.items() if c==value]
                        else:uids=cp.loc[(cp[field+'_a']==value)|(cp[field+'_b']==value),'pair_uid']
                        gd=dd[dd.pair_uid.isin(uids)]
                        group_rows.append(dict(split=split,cohort=cohort,model=model,seed=seed,grouping=field,group=value,
                            **metrics(gd,np.ones(len(gd)),cfg['direction_floors']['p'])))
                if cohort=='main' and split=='train' and seed=='pooled':
                    for fid in sorted(set(cp.flight_a)|set(cp.flight_b)):
                        pp=cp[(cp.flight_a!=fid)&(cp.flight_b!=fid)];ld=dd[dd.pair_uid.isin(pp.pair_uid)]
                        wm=weights(pp,'flight')
                        leave.append(dict(removed_flight=fid,model=model,**metrics(ld,ld.pair_uid.map(wm),cfg['direction_floors']['p'])))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(group_rows).to_csv(out/'per_group.csv',index=False)
    pd.DataFrame(leave).to_csv(out/'leave_one_flight_out.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'direction_sensitivity.csv',index=False)
    points.to_csv(out/'point_responses.csv.gz',index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=HERE/'results');a=ap.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());out=a.results
    z=np.load(out/'native_rollouts.npz');windows=pd.read_csv(out/'windows.csv');pairs=pd.read_csv(out/'pairs.csv');mem=pd.read_csv(out/'cohorts.csv')
    d,support,curves=response_rows(z,windows,pairs,cfg)
    d.to_csv(out/'response_curves.csv.gz',index=False);support.to_csv(out/'time_support.csv',index=False);curves.to_csv(out/'curve_metrics.csv',index=False)
    aggregate(d,pairs,mem,cfg,out)
    # Curve summaries average squared per-pair time-integrated errors, not predictions.
    records=[]
    for (split,cohort),g in mem.groupby(['split','cohort']):
        pp=pairs[pairs.pair_uid.isin(g.pair_uid)];cc=curves[curves.pair_uid.isin(g.pair_uid)]
        for model,c in cc.groupby('model'):
            for seed in ['pooled']+sorted(c.seed.unique().tolist()):
                sub=c if seed=='pooled' else c[c.seed==seed]
                for weighting in ['pair','flight','component']:
                    w=sub.pair_uid.map(weights(pp,weighting)).to_numpy();w=w/w.sum()
                    records.append(dict(split=split,cohort=cohort,model=model,seed=seed,weighting=weighting,pairs=len(pp),
                        curve_rmse=np.sqrt(np.dot(w,sub.curve_rmse**2)),curve_bias=np.dot(w,sub.curve_bias),
                        real_rms=np.sqrt(np.dot(w,sub.real_rms**2)),pred_rms=np.sqrt(np.dot(w,sub.pred_rms**2)),
                        median_peak_lag_ms=sub.peak_lag_ms.median(),median_correlation=sub.correlation.median()))
    pd.DataFrame(records).to_csv(out/'curve_summary.csv',index=False)
    print('response rows',len(d),'curve rows',len(curves),'all pairs',len(pairs),flush=True)
    print(pd.read_csv(out/'summary.csv').query('cohort=="main" and variable=="p" and seed=="pooled" and time_label=="supported_endpoint" and weighting=="flight"').to_string(index=False))


if __name__=='__main__':main()
