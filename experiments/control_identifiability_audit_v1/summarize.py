"""Derived diagnostics, selected-pair figures and Chinese report; no future state access."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audit import (HERE,ROOT,CHANNELS,V,Q,W,U,resolve,read_all,segment_intervals,event_indices,
                   disjoint_pairs,qstats,sha,write_json,Rotation,zoh)

OUT=HERE/'results'

def table(d):
    # Do not require optional tabulate dependency.
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.4g}'
        return str(x)
    rows=['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    rows += ['| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def diagnostics(cfg,flights,groups,natives):
    rows=[];deltas=[];eventrows=[];flightstats=[]
    for f in flights.itertuples():
        s=groups[f.flight_id];t,u,age=natives[f.flight_id];data=[]
        for lo,hi,seg in segment_intervals(s):
            ix=np.flatnonzero((t>=lo)&(t<hi)&(age<=160000))
            if len(ix)<2:continue
            tt=t[ix];uu=u[ix];dt=np.diff(tt)*1e-6
            for c,ch in enumerate(CHANNELS):
                ids=event_indices(tt,uu[:,c],cfg['event_threshold'][c],cfg['event_refractory_us'])
                prev=np.r_[0,ids[:-1]]
                for k,j in enumerate(ids):
                    eventrows.append(dict(flight_id=f.flight_id,channel=ch,time_us=tt[j],segment_id=seg,
                        change=uu[j,c]-uu[prev[k],c],elapsed_since_previous_s=(tt[j]-tt[prev[k]])*1e-6,
                        hold_until_next_event_s=((tt[ids[k+1]] if k+1<len(ids) else hi)-tt[j])*1e-6))
            dd=np.diff(uu,axis=0);data.append(dd)
            deltas.append((f.split,f.sampling_group,dd,dd/dt[:,None]))
        if data:
            d=np.concatenate(data)
            for c,ch in enumerate(CHANNELS):flightstats.append(dict(flight_id=f.flight_id,channel=ch,mean_abs_delta=float(np.mean(np.abs(d[:,c]))),
                p99_abs_delta=float(np.quantile(np.abs(d[:,c]),.99))))
    events=pd.DataFrame(eventrows);events.to_csv(OUT/'effective_events.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(flightstats).to_csv(OUT/'increment_per_flight.csv',index=False)
    corr=[];rank=[]
    for group in ['main_50hz','sparse_10hz']:
        for split in ['train','validation','all']:
            selected=[d for d in deltas if d[1]==group and (split=='all' or d[0]==split)]
            for meas,k in [('delta',2),('rate',3)]:
                data=np.concatenate([d[k] for d in selected])
                for i,ch in enumerate(CHANNELS):rows.append(dict(split=split,sampling_group=group,channel=ch,measure=meas,**qstats(np.abs(data[:,i]))))
            d=np.concatenate([d[2] for d in selected]);cc=np.corrcoef(d,rowvar=False)
            eig=np.linalg.eigvalsh(cc)
            rank.append(dict(split=split,sampling_group=group,min_eigenvalue=eig.min(),max_eigenvalue=eig.max(),condition=eig.max()/eig.min()))
            for c,ch in enumerate(CHANNELS):
                inv=np.linalg.inv(cc);corr.append(dict(split=split,sampling_group=group,channel=ch,
                    residual_variance_fraction=1/inv[c,c],**dict(zip(CHANNELS,cc[c]))))
    pd.DataFrame(rows).to_csv(OUT/'increment_rate_statistics.csv',index=False)
    pd.DataFrame(corr).to_csv(OUT/'increment_correlation_partial_variance.csv',index=False)
    pd.DataFrame(rank).to_csv(OUT/'increment_rank.csv',index=False)
    ec=pd.read_csv(OUT/'event_counts_per_flight.csv');primary=ec[ec.scale==1].groupby(['flight_id','channel']).events.sum()
    actual=events.groupby(['flight_id','channel']).size().reindex(primary.index,fill_value=0)
    np.testing.assert_array_equal(primary,actual)
    return events

def pair_plot(horizon,idx,p,w,groups,natives,path):
    pair=p.loc[idx];ma=w.iloc[int(pair.i)];mb=w.iloc[int(pair.j)]
    fig,axs=plt.subplots(4,2,figsize=(12,10),sharex='col')
    q0=groups[ma.flight_id].iloc[int(ma.origin_sample)][Q].to_numpy(float);q0/=np.linalg.norm(q0)
    for m,color,label in [(ma,'tab:blue','A'),(mb,'tab:orange','B')]:
        s=groups[m.flight_id];s=s.iloc[int(m.start_sample):int(m.origin_sample)+1]
        tt=(s.state_measurement_us-m.initial_measurement_us)*1e-6
        q=s[Q].to_numpy();q/=np.linalg.norm(q,axis=1,keepdims=True)
        vals=[np.linalg.norm(s[V],axis=1),np.linalg.norm(s[W],axis=1),s.flap_frequency_hz,
              np.rad2deg(2*np.arccos(np.abs(q@q0).clip(0,1)))]
        for ax,y in zip(axs[:,0],vals):ax.plot(tt,y,color=color,label=label)
        t,u,_=natives[m.flight_id];lo=m.decision_us-500000;hi=m.decision_us+horizon*1000
        a=np.searchsorted(t,lo,side='right')-1;b=np.searchsorted(t,hi,side='left')
        tx=np.r_[lo,t[a+1:b],hi];ux=np.vstack([u[a:b],u[b-1]])
        for c,ax in enumerate(axs[:,1]):ax.step((tx-m.decision_us)*1e-6,ux[:,c],where='post',color=color,label=label)
    for ax,name in zip(axs[:,0],['NED speed (m/s)','Body rate norm (rad/s)','Historical frequency (Hz)','Attitude distance to A origin (deg)']):ax.set_ylabel(name);ax.grid(alpha=.2);ax.legend()
    for ax,ch in zip(axs[:,1],CHANNELS):ax.set_ylabel(ch);ax.axvline(0,color='k',ls='--');ax.grid(alpha=.2);ax.legend()
    axs[-1,0].set_xlabel('History time relative to initial measurement (s)')
    axs[-1,1].set_xlabel('Command time relative to decision publication (s)')
    fig.suptitle(f'{horizon} ms, pair {idx}; history RMSE: v={pair.v_rmse:.2f}, R={pair.R_rmse_deg:.1f}deg, w={pair.w_rmse:.2f}, f={pair.f_rmse:.2f}\nA={ma.flight_id[:12]} ({ma.split}), B={mb.flight_id[:12]} ({mb.split}); no future outcome selection')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def pair_diagnostics(cfg,groups,natives):
    strata=[];quality=[];exports=[];support=[];examples=[];firm=[];strengthened=[];strong_export=[];eval_support=[]
    dc=json.loads((HERE/'diagnostic_config.json').read_text())
    for horizon in [500,1000]:
        p=pd.read_parquet(OUT/f'pairs_{horizon}ms.parquet');w=pd.read_parquet(OUT/f'windows_{horizon}ms.parquet')
        peak=np.zeros(len(w),bool)
        for fid,ww in w.groupby('flight_id'):
            s=groups[fid];cs=np.r_[0,np.cumsum(np.linalg.norm(s[W],axis=1)>2)]
            peak[ww.index]=cs[ww.origin_sample.to_numpy()+1]>cs[ww.start_sample.to_numpy()]
        p['historical_peak_dynamics']=peak[p.i]|peak[p.j]
        p['whole_window_review']=w.review.to_numpy()[p.i]|w.review.to_numpy()[p.j]
        p['historical_mode']=w.nav_state.to_numpy()[p.i]
        sel=pd.read_csv(OUT/f'selected_{horizon}ms.csv')
        for channel in ['any']+CHANNELS:
            ids=sel.loc[sel.channel==channel,'pair_index'].to_numpy(int);chosen=p.loc[ids]
            for row in chosen.itertuples():
                a,b=w.iloc[row.i],w.iloc[row.j]
                exports.append(dict(horizon_ms=horizon,channel=channel,pair_index=row.Index,
                    flight_a=a.flight_id,flight_b=b.flight_id,window_a=a.window_id,window_b=b.window_id,
                    start_a=a.history_start_us,end_a=a.pair_support_end_us,start_b=b.history_start_us,end_b=b.pair_support_end_us,
                    split_pair=row.split_pair,group=row.group,mode=a.nav_state,same_firmware=row.same_firmware,
                    history_review=row.any_review,history_mean_rate_above2=row.high_dynamics,
                    historical_peak_dynamics=row.historical_peak_dynamics,whole_window_review=row.whole_window_review,
                    **{k:getattr(row,k) for k in ['score','v_rmse','R_rmse_deg','w_rmse','f_rmse','initial_v_gap','initial_R_gap_deg','initial_w_gap','initial_f_gap']+['initial_u_gap_'+ch for ch in CHANNELS]+['future_gap_'+ch for ch in CHANNELS]}))
            endpoint=np.unique(chosen[['i','j']].to_numpy().ravel());m=w.iloc[endpoint]
            for col in ['history_speed','history_rate','history_frequency']+['history_'+ch for ch in CHANNELS]:
                support.append(dict(horizon_ms=horizon,channel=channel,feature=col,**qstats(m[col])))
            for col in ['v_rmse','R_rmse_deg','w_rmse','f_rmse','initial_v_gap','initial_R_gap_deg','initial_w_gap','initial_f_gap']+['history_gap_'+ch for ch in CHANNELS]+['future_gap_'+ch for ch in CHANNELS]:
                quality.append(dict(horizon_ms=horizon,channel=channel,feature=col,**qstats(chosen[col])))
        mask=p.primary&p[[f'contrast_{ch}' for ch in CHANNELS]].any(axis=1)
        qualified=p[mask]
        base=p[p.primary]
        for ratio in [0,2]:
            for ch in ['any']+CHANNELS:
                contrast=np.column_stack([base['contrast_'+c] & (base['future_gap_'+c]>=ratio*np.maximum(base['history_gap_'+c],base['initial_u_gap_'+c])) for c in CHANNELS])
                use=contrast.any(1) if ch=='any' else contrast[:,CHANNELS.index(ch)]&base['isolated_'+ch]
                for (group,sp),sub in base[use].groupby(['group','split_pair']):
                    ids=disjoint_pairs(sub,w);ends=np.unique(p.loc[ids,['i','j']].to_numpy().ravel())
                    eval_support.append(dict(horizon_ms=horizon,channel=ch,future_history_ratio=ratio,group=group,
                        split_pair=sp,raw_pairs=len(sub),nonoverlap_pairs=len(ids),flights=w.iloc[ends].flight_id.nunique()))
        for ratio in dc['future_to_history_gap_ratios']:
            for ch in ['any']+CHANNELS:
                contrast=np.column_stack([p['contrast_'+c] & (p['future_gap_'+c]>=ratio*np.maximum(p['history_gap_'+c],p['initial_u_gap_'+c])) for c in CHANNELS])
                use=contrast.any(1) if ch=='any' else contrast[:,CHANNELS.index(ch)]&p['isolated_'+ch]
                for split_pair in ['all','train-train','train-validation','validation-validation']:
                    sub=p[p.primary&use]
                    if split_pair!='all':sub=sub[sub.split_pair==split_pair]
                    ids=disjoint_pairs(sub,w);ends=np.unique(p.loc[ids,['i','j']].to_numpy().ravel())
                    strengthened.append(dict(horizon_ms=horizon,channel=ch,ratio=ratio,split_pair=split_pair,
                        raw_pairs=len(sub),nonoverlap_pairs=len(ids),flights=w.iloc[ends].flight_id.nunique(),
                        flight_disjoint_pairs=len(disjoint_pairs(sub,w,True))))
                    if ratio==2 and split_pair=='all':
                        for idx in ids:strong_export.append(dict(horizon_ms=horizon,channel=ch,pair_index=idx))
        # Each stratum receives its own nonoverlap packing, so stratum counts are NOT additive.
        for grouping in ['split_pair','group','any_review','historical_peak_dynamics','whole_window_review','historical_mode','same_firmware']:
            for label,g in qualified.groupby(grouping):
                ids=disjoint_pairs(g,w);endpoints=np.unique(p.loc[ids,['i','j']].to_numpy().ravel())
                strata.append(dict(horizon_ms=horizon,grouping=grouping,label=label,raw_pairs=len(g),
                    nonoverlap_pairs=len(ids),flights=w.iloc[endpoints].flight_id.nunique(),flight_disjoint_pairs=len(disjoint_pairs(g,w,True))))
        # Descriptive same-firmware sensitivity; no firmware-dependent outcome fitting.
        for ch in ['any']+CHANNELS:
            mask2=mask if ch=='any' else p.primary&p['isolated_'+ch]
            sub=p[mask2&p.same_firmware];ids=disjoint_pairs(sub,w)
            firm.append(dict(horizon_ms=horizon,channel=ch,raw_pairs=len(sub),nonoverlap_pairs=len(ids),
                candidate_flights=w.iloc[np.unique(sub[['i','j']].to_numpy().ravel())].flight_id.nunique(),
                nonoverlap_flights=w.iloc[np.unique(p.loc[ids,['i','j']].to_numpy().ravel())].flight_id.nunique()))
        selected=p.loc[sel[sel.channel=='any'].pair_index]
        endpoint=np.unique(selected[['i','j']].to_numpy().ravel());mc=w.iloc[endpoint]
        counts=mc.flight_id.value_counts().reindex(w.flight_id.unique(),fill_value=0)
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(counts,bins=25);ax.set(xlabel='Selected distinct windows per flight',ylabel='Flights',title=f'{horizon} ms: nonoverlap support distribution');fig.tight_layout();fig.savefig(OUT/f'figures/flight_support_{horizon}ms.png',dpi=160);plt.close(fig)
        # Choose only on history score and already-frozen control contrast category.
        picked=[];used=set()
        for ch in CHANNELS+['any']:
            choices=p.loc[sel[sel.channel==ch].pair_index].sort_values('score')
            for idx,row in choices.iterrows():
                if idx not in used:
                    picked.append((idx,ch));used.add(idx);break
        for idx,ch in picked[:4]:
            name=f'match_{horizon}ms_{ch}_{idx}.png';pair_plot(horizon,idx,p,w,groups,natives,OUT/'figures'/name)
            examples.append(dict(horizon_ms=horizon,channel=ch,pair_index=idx,figure='results/figures/'+name))
        strong_ids=[r['pair_index'] for r in strong_export if r['horizon_ms']==horizon and r['channel']=='motor']
        if strong_ids:
            idx=p.loc[strong_ids].sort_values('score').index[0]
            name=f'match_{horizon}ms_motor_strengthened_{idx}.png'
            pair_plot(horizon,idx,p,w,groups,natives,OUT/'figures'/name)
            examples.append(dict(horizon_ms=horizon,channel='motor_strengthened',pair_index=int(idx),figure='results/figures/'+name))
    pd.DataFrame(exports).to_csv(OUT/'selected_pairs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(quality).to_csv(OUT/'selected_pair_quality.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'selected_state_control_support.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'match_strata.csv',index=False)
    pd.DataFrame(firm).to_csv(OUT/'same_firmware_sensitivity.csv',index=False)
    pd.DataFrame(strengthened).to_csv(OUT/'strengthened_contrast_sensitivity.csv',index=False)
    pd.DataFrame(strong_export).to_csv(OUT/'strengthened_selected_pairs.csv',index=False)
    pd.DataFrame(eval_support).to_csv(OUT/'response_evaluation_support.csv',index=False)
    write_json(OUT/'examples.json',examples)
    summary=pd.read_csv(OUT/'match_sensitivity.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,h in zip(axs,[500,1000]):
        ss=summary.query('horizon_ms==@h and channel=="any" and group=="all"')
        for scale,g in ss.groupby('contrast_scale'):
            g=g.set_index('caliper').loc[['strict','primary','relaxed']]
            ax.plot(range(3),g.nonoverlap_pairs,'o-',label=f'Command threshold x{scale}')
        ax.set_xticks(range(3),['strict','primary','relaxed']);ax.set(title=f'{h} ms',ylabel='Nonoverlapping pairs',xlabel='History calipers');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figures/match_threshold_sensitivity.png',dpi=160);plt.close(fig)
    return pd.DataFrame(exports),pd.DataFrame(strata)

def main():
    cfg=json.loads((HERE/'config.json').read_text());root,flights,prov,manifest=resolve(cfg)
    groups,natives,windows=read_all(root,flights)
    keys=sorted({k for x in prov.values() for k in x['parameters'] if k.startswith(('PWM_MAIN_','CA_SV_CS'))})
    parameters=[]
    for key in keys:
        counts=pd.Series([x['parameters'].get(key) for x in prov.values()]).value_counts(dropna=False)
        parameters.append(dict(parameter=key,unique_values=len(counts),values_counts=str(counts.to_dict())))
    pd.DataFrame(parameters).to_csv(OUT/'output_parameter_coverage.csv',index=False)
    events=diagnostics(cfg,flights,groups,natives)
    exports,strata=pair_diagnostics(cfg,groups,natives)
    write_json(OUT/'summary_provenance.json',dict(code_sha256=sha(__file__),audit_code_sha256=sha(HERE/'audit.py'),
        diagnostic_config_sha256=sha(HERE/'diagnostic_config.json'),
        same_hardware_required=True,hardware_counts=flights.flight_id.map(lambda x:prov[x]['sys_uuid']).fillna('unknown').value_counts().to_dict(),
        firmware_count=len(set(prov[x]['firmware'] for x in flights.flight_id)),
        future_state_targets_read=False,future_outcomes_used_for_selection=False))
    print('Derived diagnostics and figures complete',flush=True)

if __name__=='__main__':main()
