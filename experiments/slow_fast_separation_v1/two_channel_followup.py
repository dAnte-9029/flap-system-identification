"""vD/q follow-up. Reuses frozen data/QC and event identities; no test access."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from run import ROOT, DATA, prepare_phase
from methods import cell, fit_common

PREVIOUS = ROOT/'docs/analysis/results/slow_fast_separation_v1/final_run'
CHANNELS = {'vD': ('state_measurement_us', 'target_velocity_ned_m_s_z', .3),
            'q': ('target_rate_measurement_us', 'target_angular_velocity_body_rad_s_y', .2)}


def native_runs(s, pt, m, channel):
    tcol, col, _ = CHANNELS[channel]
    valid = s.valid_target_state & ~s.excluded & ~s.review & s.firmware_verified & ~s.provenance_review & s.valid_core
    a = s[[tcol, col, 'segment_id']].copy()
    a['valid'] = valid
    a = a.drop_duplicates(tcol).sort_values(tcol).reset_index(drop=True)
    tt = a[tcol].to_numpy()*1e-6
    yy = a[col].to_numpy()
    vv = a.valid.to_numpy() & np.isfinite(yy)
    seg = a.segment_id.to_numpy()
    av = tt  # unused by offline follow-up
    ts=pt.timestamp_us.to_numpy()*1e-6; ph=pt.phase_rad.to_numpy(); good=pt.phase_usable.to_numpy()
    cross=np.flatnonzero((np.diff(ph)<-np.pi)&good[:-1]&good[1:])+1
    bounds=ts[cross-1]+(2*np.pi-ph[cross-1])/(ph[cross]+2*np.pi-ph[cross-1])*(ts[cross]-ts[cross-1])
    runs=[];run=[];missing=0;cells=0;reasons=dict(phase=0,state=0,sparse=0);dur=[];obs=[]
    for k in range(len(bounds)-1):
        lo,hi=bounds[k:k+2];idx=np.flatnonzero((tt>=lo)&(tt<hi));reason=None
        if m['sampling_group']!='main_50hz':reason='sparse'
        elif not good[cross[k]-1:cross[k+1]+1].all():reason='phase'
        elif any(lo<=r*1e-6<=hi for r in m['reset_times_us']):reason='state'
        elif len(idx)<2 or not vv[idx].all() or len(np.unique(seg[idx]))!=1 or np.max(np.diff(np.r_[lo,tt[idx],hi]))>.04:reason='state'
        if reason:
            reasons[reason]+=1
            if run:runs.append(run);run=[]
            continue
        # Reconstruct only within this QC-admitted cycle, preserving Hall circular anchor.
        xp=ts[cross[k]-1:cross[k+1]+1];yp=np.unwrap(ph[cross[k]-1:cross[k+1]+1]);pp=np.interp(tt[idx],xp,yp);pp-=2*np.pi*np.floor(np.interp(lo,xp,yp)/(2*np.pi)+1e-7)
        if np.any(np.diff(pp)<=0) or np.min(pp)<-1e-6 or np.max(pp)>2*np.pi+1e-6:
            reasons['phase']+=1
            if run:runs.append(run);run=[]
            continue

        if run and run[-1]['segment']!=seg[idx[0]]:runs.append(run);run=[]
        c=cell(tt[idx],pp,yy[idx],lo,hi);c['indices']=idx;c['segment']=seg[idx[0]];c['availability']=av[idx];c['ready']=max(hi+.02,float(av[idx].max()));c['phase_idx']=(cross[k],cross[k+1]);run.append(c);missing+=np.isnan(c['g']).sum();cells+=8;dur.append(hi-lo);obs.append(len(idx))
    if run:runs.append(run)
    return a, runs


def crossing(t, y, baseline, change, start, end):
    if abs(change) < 1e-9:
        return np.nan
    ii = np.flatnonzero((t >= start) & (t <= end) & ((y-baseline)*np.sign(change) >= abs(change)*.5))
    return float(t[ii[0]]-start) if len(ii) else np.nan


def synthetic(cs, base, amplitude, rows):
    # One fixed central maneuver per admitted long run; no location search.
    n = len(cs)//2
    t0 = cs[n]['tau']
    if cs[25]['lo'] > t0-.5 or cs[-26]['hi'] < t0+1.05:
        return None
    changed=[]
    for c in cs:
        z=np.clip((c['t']-t0)/.5,0,1)
        changed.append(cell(c['t'],c['p'],c['y']+amplitude*z*z*(3-2*z),c['lo'],c['hi']))
    pieces=[]
    for i in range(25,len(cs)-25):
        c=cs[i]
        if c['hi']<t0-.5 or c['lo']>t0+1.05:
            continue
        _, f, l=fit_common(cs,i)
        _, fi, li=fit_common(changed,i)
        pieces.append(np.column_stack([c['t'], fi-f, li-l]))
    a=np.concatenate(pieces); t=a[:,0]; z=np.clip((t-t0)/.5,0,1); delta=amplitude*z*z*(3-2*z)
    complete=np.isfinite(a).all(axis=1)
    for j,method in enumerate(('F','L'),1):
        leak=a[:,j]; slow=delta-leak
        for stage,lo,hi in [('before',-.5,0),('start',0,.1),('during',0,.5),('end',.4,.5),('after',.5,1.0),('full',-.5,1.0)]:
            ok=(t>=t0+lo)&(t<t0+hi)&complete
            if not ok.any():continue
            rows.append(dict(**base,kind='synthetic',method=method,stage=stage,event_time_s=t0,amplitude=amplitude,n_observations=int(ok.sum()),coverage=float(complete[(t>=t0+lo)&(t<t0+hi)].mean()),slow_error_rmse=float(np.sqrt(np.mean(leak[ok]**2))),fast_leak_max=float(np.max(abs(leak[ok]))),raw_change=float(delta[ok].mean()),slow_change=float(slow[ok].mean()),fast_change=float(leak[ok].mean()),slow_half_change_delay_s=crossing(t[complete],slow[complete],0,amplitude,t0,t0+1),raw_half_change_delay_s=crossing(t[complete],delta[complete],0,amplitude,t0,t0+1)))
    return np.column_stack([t-t0,delta,delta-a[:,1],delta-a[:,2],a[:,1],a[:,2]])


def real_events(series, events, base, rows):
    t,raw,f,l=series.T
    for e in events.itertuples():
        et=e.event_time_s
        span=(t>=et-.5)&(t<=et+1)
        if t[0]>et-.5 or t[-1]<et+1 or span.sum()<5:
            continue
        # Do not silently bridge failed templates or gaps for event diagnostics.
        if not np.isfinite(series[span]).all() or np.max(np.diff(t[span]))>.04:
            continue
        before=(t>=et-.5)&(t<et); after=(t>=et+.5)&(t<=et+1)
        if min(before.sum(),after.sum())<5:continue
        rc=raw[after].mean()-raw[before].mean()
        rd=crossing(t,raw,raw[before].mean(),rc,et,et+1)
        for method,fast in [('F',f),('L',l)]:
            slow=raw-fast; sc=slow[after].mean()-slow[before].mean(); fc=fast[after].mean()-fast[before].mean()
            # Common raw net-change threshold avoids changing the timing target by method.
            sd=crossing(t,slow,slow[before].mean(),rc,et,et+1)
            legacy=crossing(t,abs(slow-slow[before].mean()),0,abs(sc),et,et+1)
            rows.append(dict(**base,kind='control_event',method=method,stage='before_after',event_time_s=et,control=e.control,command_change=e.command_change,raw_change=rc,slow_change=sc,fast_change=fc,abs_net_change_error=abs(sc-rc),raw_half_change_delay_s=rd,slow_half_change_delay_s=sd,relative_delay_s=sd-rd,legacy_slow_abs_half_delay_s=legacy,legacy_raw_abs_half_delay_s=e.raw_half_change_delay_s))


def summarize_comparison(cycles):
    rows=[]
    errors=['trend_mse','F_mse','L_mse','old_trend_mse','old_A_mse','old_B_mse']
    for (split,ch),allrows in cycles.groupby(['split','channel']):
        common=allrows[allrows.F_valid & allrows.L_valid]
        flights=common.groupby('flight_id')[errors].mean()
        coverage=dict(F_coverage=float(allrows.F_valid.mean()),L_coverage=float(allrows.L_valid.mean()),candidate_cycles=len(allrows),common_cycles=len(common),n_flights=len(flights))
        for level,data in [('cycle_equal',common[errors]),('flight_equal',flights)]:
            d=dict(split=split,channel=ch,aggregation=level,flight_id='ALL',**coverage)
            d.update(data.mean().to_dict())
            for key in errors:d[key.replace('mse','rmse')]=np.sqrt(d[key])
            for method in ('F','L'):
                d[f'{method}_vs_trend_abs']=d['trend_mse']-d[f'{method}_mse']
                d[f'{method}_vs_trend_relative']=1-d[f'{method}_mse']/d['trend_mse']
                d[f'{method}_improved_flights']=int((flights[f'{method}_mse']<flights.trend_mse).sum())
            d['L_vs_F_abs']=d['F_mse']-d['L_mse'];d['L_vs_F_relative']=1-d['L_mse']/d['F_mse']
            d['L_beats_F_flights']=int((flights.L_mse<flights.F_mse).sum())
            d['mean_flight_L_vs_F_relative']=float((1-flights.L_mse/flights.F_mse).mean())
            d['old_trend_gain']=d['old_trend_mse']-d['trend_mse']
            d['old_total_gain']=d['old_A_mse']-d['old_B_mse']
            d['old_incremental_periodic_gain']=d['old_total_gain']-d['old_trend_gain']
            rows.append(d)
        for fid,r in flights.iterrows():
            original=allrows[allrows.flight_id==fid]
            d=dict(split=split,channel=ch,aggregation='flight',flight_id=fid,**r.to_dict(),F_coverage=float(original.F_valid.mean()),L_coverage=float(original.L_valid.mean()),candidate_cycles=len(original))
            for key in errors:d[key.replace('mse','rmse')]=np.sqrt(d[key])
            d.update(L_vs_F_abs=r.F_mse-r.L_mse,L_vs_F_relative=1-r.L_mse/r.F_mse)
            rows.append(d)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'docs/analysis/results/slow_fast_separation_v1/two_channel_followup')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((PREVIOUS/'protocol.json').read_text())
    meta={x['flight_id']:x for x in json.loads((DATA/'flight_provenance.json').read_text()) if x['split'] in ('train','validation')}
    old=pd.read_csv(PREVIOUS/'cycle_prediction.csv')
    old=old[old.channel.isin(CHANNELS)].set_index(['flight_id','channel','run','cycle'])
    events=pd.read_csv(PREVIOUS/'motion_preservation.csv')
    events=events[(events.kind=='control_event')&(events.split=='validation')&events.channel.isin(CHANNELS)]
    predictions=[];maneuvers=[];coverage=[];plot_prediction={};plot_maneuver={};started=time.monotonic();checked=False;max_zero=0.;max_reconstruction=0.
    for split in ('train','validation'):
        columns=['flight_id','segment_id','valid_target_state','excluded','review','firmware_verified','provenance_review','valid_core']
        for tc,col,_ in CHANNELS.values():columns.extend([tc,col])
        samples=pd.read_parquet(DATA/f'samples_{split}.parquet',columns=columns)
        phase=pd.read_parquet(DATA/f'native_wing_phase_{split}.parquet')
        labels={ch:[] for ch in CHANNELS}
        for flight_index,(fid,s) in enumerate(samples.groupby('flight_id',sort=False)):
            m=meta[fid];pt=prepare_phase(phase[phase.flight_id==fid],m['transmission_ratio'],previous['phase_contract'])
            for channel,(tcol,col,amp) in CHANNELS.items():
                a,runs=native_runs(s,pt,m,channel)
                label=pd.DataFrame(dict(flight_id=fid,split=split,timestamp_us=a[tcol].to_numpy(),segment_id=a.segment_id.to_numpy(),raw=a[col].to_numpy(),method_F='F_H1_H3_shared_trend',method_L='L_Q8_P3_shared_trend'))
                for method in ('F','L'):
                    label[f'fast_{method}']=np.nan;label[f'slow_{method}']=np.nan;label[f'valid_{method}']=False
                for ri,cs in enumerate(runs):
                    if len(cs)<51:continue
                    base=dict(flight_id=fid,split=split,channel=channel,run=ri)
                    series=[]
                    for n in range(25,len(cs)-25):
                        c=cs[n];trend,f,l=fit_common(cs,n)
                        for method,fast in [('F',f),('L',l)]:
                            if not np.isfinite(fast).all():continue
                            label.loc[c['indices'],f'fast_{method}']=fast
                            label.loc[c['indices'],f'slow_{method}']=c['y']-fast
                            label.loc[c['indices'],f'valid_{method}']=True
                            max_zero=max(max_zero,abs(c['w']@fast));max_reconstruction=max(max_reconstruction,float(np.max(abs(c['y']-((c['y']-fast)+fast)))))
                        series.append(np.column_stack([c['t'],c['y'],f,l]))
                        if n%5!=2:continue
                        tr,ff,ll=fit_common(cs,n,True)
                        oldrow=old.loc[(fid,channel,ri,n)]
                        # Numerical equivalence with cached B confirms same native QC/window/trend.
                        if np.isfinite(ll).all():
                            np.testing.assert_allclose(np.mean((tr+ll-c['y'])**2),oldrow.B_mse,rtol=1e-8,atol=1e-11)
                            np.testing.assert_allclose(np.mean((tr-c['y'])**2),oldrow.B_trend_mse,rtol=1e-8,atol=1e-11)
                        predictions.append(dict(**base,cycle=n,n_observations=len(c['t']),F_valid=bool(np.isfinite(ff).all()),L_valid=bool(np.isfinite(ll).all()),trend_mse=float(np.mean((tr-c['y'])**2)),F_mse=float(np.mean((tr+ff-c['y'])**2)),L_mse=float(np.mean((tr+ll-c['y'])**2)),old_trend_mse=oldrow.trend_mse,old_A_mse=oldrow.A_mse,old_B_mse=oldrow.B_mse))
                        if not checked:
                            corrupted=[cell(z['t'],z['p'],z['y']+1000 if i%5==2 else z['y'],z['lo'],z['hi']) for i,z in enumerate(cs)]
                            for actual,expected in zip(fit_common(corrupted,n,True),(tr,ff,ll)):
                                np.testing.assert_allclose(actual,expected,rtol=0,atol=0)
                            checked=True
                        if split=='validation' and channel not in plot_prediction and np.isfinite(ll).all():
                            plot_prediction[channel]=np.column_stack([c['t']-c['lo'],c['y'],tr,tr+ff,tr+ll])
                    trace=synthetic(cs,base,amp,maneuvers)
                    if split=='validation' and trace is not None and channel not in plot_maneuver:plot_maneuver[channel]=trace
                    if split=='validation':real_events(np.concatenate(series),events[(events.flight_id==fid)&(events.channel==channel)],base,maneuvers)
                for method in ('F','L'):
                    coverage.append(dict(flight_id=fid,split=split,channel=channel,method=method,n_native=len(label),n_valid=int(label[f'valid_{method}'].sum())))
                labels[channel].append(label)
            if flight_index%10==0:print(f'{split} flight {flight_index+1}: {time.monotonic()-started:.1f}s',flush=True)
        for channel,frames in labels.items():
            pd.concat(frames,ignore_index=True).to_parquet(out/f'candidate_labels_{split}_{channel}.parquet',index=False)
    assert max_zero<1e-12 and max_reconstruction<1e-12 and checked
    comparison=summarize_comparison(pd.DataFrame(predictions));comparison.to_csv(out/'fair_comparison.csv',index=False)
    motion=pd.DataFrame(maneuvers);motion.to_csv(out/'maneuver_preservation.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'label_coverage.csv',index=False)
    config=dict(code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dataset='velocity_rate_multistep_v1',splits=['train','validation'],window=51,harmonics=[1,2,3],Q=8,P=3,holdout=previous['holdout'],phase_contract=previous['phase_contract'],event_thresholds_train_p90=previous['event_thresholds_train_p90'],events='reuse prior validation event identities',synthetic='one center per eligible run, 0.5 s smoothstep, vD=0.3 q=0.2',max_weighted_fast_mean=max_zero,max_reconstruction_error=max_reconstruction,holdout_invariance_checked=checked,elapsed_s=time.monotonic()-started)
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    from two_channel_report import write_report
    write_report(out,comparison,motion,pd.DataFrame(coverage),plot_prediction,plot_maneuver,config,events)
    print(comparison[comparison.aggregation=='flight_equal'].to_string(index=False),flush=True)


if __name__=='__main__':main()
