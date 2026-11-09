"""Descriptive, flight/date-aware summaries of frozen controlled responses."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pacld_cir_v1 import OUT,TIMES
from system_identification.models.pahbd import geometry as so3

def clean(v):
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    return v

def table(d):
    def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['|'+'|'.join(d.columns)+'|','|'+'|'.join(['---']*len(d.columns))+'|']+['|'+'|'.join(fmt(x) for x in r)+'|' for r in d.itertuples(index=False,name=None)])

def main():
    d=pd.read_csv(OUT/'validation_responses.csv');selected=pd.read_csv(OUT/'selected_validation.csv');candidates=pd.read_csv(OUT/'validation_candidates.csv');protocol=json.load(open(OUT/'protocol.json'));checks=json.load(open(OUT/'checks.json'))
    verification=json.load(open(OUT/'verification.json'))
    resolution=max(1e-8,10*checks['batch_vs_single_state_max_error'],10*verification['batch3_vs64_max_physical_state_error'])
    d['asymmetry_relative']=np.where((abs(d.R_plus)+abs(d.R_minus))>resolution,abs(d.asymmetry_abs)/(abs(d.R_plus)+abs(d.R_minus)),np.nan)
    summaries=[]
    for (mult,time),a in d[d.time.isin([.2,.5])].groupby(['multiplier','time']):
        summaries.append(dict(multiplier=mult,time=time,windows=len(a),flights=a.flight.nunique(),dates=a.date.nunique(),G_p_median=a.G_p.median(),G_p_q05=a.G_p.quantile(.05),G_p_q95=a.G_p.quantile(.95),G_p_min=a.G_p.min(),G_p_max=a.G_p.max(),positive_fraction=(a.G_p>0).mean(),G_r_median=a.G_r.median(),G_r_q05=a.G_r.quantile(.05),G_r_q95=a.G_r.quantile(.95),asymmetry_median=a.asymmetry_relative.median(),asymmetry_q95=a.asymmetry_relative.quantile(.95)))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'gain_summary.csv',index=False)
    d['alpha_plus_increment']=d.alpha_p_plus-d.alpha_p0;d['alpha_central_gain']=(d.alpha_p_plus-d.alpha_p_minus)/(2*d.amplitude)
    main=d[d.multiplier==1].copy();h=main[main.time==.2]
    phase=h.groupby('phase_bin').agg(n=('G_p','size'),flights=('flight','nunique'),dates=('date','nunique'),G_p_median=('G_p','median'),G_p_min=('G_p','min'),G_p_max=('G_p','max'),G_r_median=('G_r','median'),speed_mean=('speed','mean'),speed_std=('speed','std'),freq_mean=('freq','mean'),asymmetry_median=('asymmetry_relative','median')).reset_index();phase.to_csv(OUT/'phase_summary.csv',index=False)
    flight=h.groupby(['date','flight']).agg(n=('G_p','size'),G_p_mean=('G_p','mean'),G_p_median=('G_p','median'),G_r_mean=('G_r','mean'),min=('G_p','min'),max=('G_p','max')).reset_index();flight.to_csv(OUT/'flight_summary.csv',index=False)
    date=flight.groupby('date').agg(flights=('flight','size'),G_p_mean=('G_p_mean','mean'),G_r_mean=('G_r_mean','mean'),min_flight_mean=('G_p_mean','min'),max_flight_mean=('G_p_mean','max')).reset_index();date.to_csv(OUT/'date_summary.csv',index=False)
    speed=h.groupby('speed_bin').G_p.agg(['count','mean','median','min','max']).reset_index();speed.to_csv(OUT/'speed_summary.csv',index=False)
    boot=np.random.default_rng(17).choice(date.G_p_mean.to_numpy(),(4000,len(date)),replace=True).mean(1)
    cache=torch.load(OUT/'validation_beliefs.pt',weights_only=False,map_location='cpu');conditions=[]
    for r in selected.itertuples():
        s=cache['states'][int(r.cache_index)];x=s['x'];vb=so3.rotate(so3.conjugate(x[6:10]),x[3:6]).numpy()
        conditions.append(dict(window_id=r.window_id,phase_bin=r.phase_bin,sin_phi=np.sin(r.phase),cos_phi=np.cos(r.phase),v_forward=vb[0],v_right=vb[1],v_down=vb[2],p=float(x[10]),q=float(x[11]),r=float(x[12]),frequency=float(s['f'][0])))
    cond=pd.DataFrame(conditions).merge(h[['window_id','G_p']],on='window_id',validate='one_to_one');cond.to_csv(OUT/'phase_state_conditions.csv',index=False);cond.select_dtypes('number').drop(columns='phase_bin').corr().to_csv(OUT/'phase_state_correlations.csv');cond.groupby('phase_bin').agg({k:['mean','std','min','max'] for k in ['v_forward','v_right','v_down','p','q','r','frequency']}).to_csv(OUT/'phase_state_distribution.csv')
    timing=[]
    for wid,a in main.groupby('window_id'):
        a=a.sort_values('time');y=a.R_central.to_numpy();times=a.time.to_numpy();mag=abs(y);peak=int(np.argmax(mag));interior=0<peak<len(y)-1 and mag[peak]>mag[peak-1]+resolution and mag[peak]>mag[peak+1]+resolution
        halflo=halfhi=np.nan
        if interior and np.all(np.diff(mag[:peak+1])>=-resolution) and mag[peak]>resolution:
            crossing=np.flatnonzero(mag[:peak+1]>=mag[peak]/2)[0]
            if crossing>0:halflo=times[crossing-1];halfhi=times[crossing]
        first=a[a.time<=.2];maximum=abs(first.R_central).max();net=abs(first[first.time==.2].R_central.iloc[0]);alpha=(first.alpha_p_plus-first.alpha_p_minus).to_numpy()/2
        timing.append(dict(window_id=wid,sampled_max_abs_response=mag[peak],sampled_max_time=times[peak],interior_sampled_peak=interior,peak_time=times[peak] if interior else np.nan,half_response_lower=halflo,half_response_upper=halfhi,net_to_max_0p2=net/maximum if maximum>resolution else np.nan,alpha_changes_sign_before_0p2=bool((alpha>resolution).any() and (alpha<-resolution).any())))
    timing=pd.DataFrame(timing);timing.to_csv(OUT/'sampled_response_timing.csv',index=False)
    mechanism=main.groupby('time')[['actuator_plus','actuator_minus','slow_plus_norm','fast_plus_norm','alpha_p_plus','alpha_p0','alpha_p_minus','phase_difference','frequency_difference','G_p','G_r','alpha_plus_increment','alpha_central_gain']].median().reset_index();mechanism.to_csv(OUT/'mechanism_time_summary.csv',index=False)
    pair=d[d.time==.2].pivot(index='window_id',columns='multiplier',values='G_p').dropna();amp=pd.DataFrame(dict(window_id=pair.index,G_half=pair[.5],G_main=pair[1.],G_double=pair[2.],half_minus_main=pair[.5]-pair[1.],double_minus_main=pair[2.]-pair[1.]));amp.to_csv(OUT/'amplitude_paired.csv',index=False)
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    for phasebin,g in main.groupby('phase_bin'):
        a=g.groupby(['flight','time'])[['R_plus','R_minus','G_p','G_r']].mean().groupby('time').mean()
        ax[0,0].plot(a.index,a.R_plus,color=f'C{int(phasebin)}',label=f'phase {int(phasebin)} +');ax[0,0].plot(a.index,a.R_minus,'--',color=f'C{int(phasebin)}',label=f'phase {int(phasebin)} -')
        ax[0,1].plot(a.index,a.G_p,'o-',label=f'phase {int(phasebin)}');ax[1,0].plot(a.index,a.G_r,'o-',label=f'phase {int(phasebin)}')
    for mult,g in d.groupby('multiplier'):
        a=g.groupby(['flight','time']).G_p.mean().groupby('time').mean();ax[1,1].plot(a.index,a,'o-',label=f'{mult}a')
    for a,title in zip(ax.flat,['Roll incremental responses (rad/s)','Roll finite-pulse gain','Yaw cross gain','Amplitude sensitivity (roll gain)']):a.set_title(title);a.set_xlabel('Time (s)');a.axhline(0,color='gray');a.axvline(.1,color='gray',linestyle=':');a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'response_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4));ax[0].boxplot([h[h.phase_bin==b].G_p for b in range(4)],tick_labels=['0','1','2','3']);ax[0].set_title('G_CF(0.2s) by natural phase');ax[0].set_xlabel('Phase quadrant');ax[1].scatter(cond.v_forward,cond.G_p,c=cond.phase_bin,cmap='viridis');ax[1].set_xlabel('Belief forward velocity (m/s)');ax[1].set_ylabel('G_CF');ax[2].scatter(cond.p,cond.G_p,c=cond.phase_bin,cmap='viridis');ax[2].set_xlabel('Belief p (rad/s)');fig.tight_layout();fig.savefig(OUT/'phase_conditions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(12,4));ax[0].plot(mechanism.time,mechanism.actuator_plus/protocol['amplitude'],'o-');ax[0].set_title('Actuator differential / a');ax[1].plot(mechanism.time,mechanism.slow_plus_norm,label='Slow');ax[1].plot(mechanism.time,mechanism.fast_plus_norm,label='Fast');ax[1].set_title('Latent displacement norms');ax[1].legend();ax[2].plot(mechanism.time,mechanism.alpha_plus_increment,'o-');ax[2].set_title('Median alpha+ minus alpha0');
    for a in ax:a.set_xlabel('Time (s)');a.axvline(.1,color='gray',linestyle=':')
    fig.tight_layout();fig.savefig(OUT/'mechanism.png',dpi=160);plt.close(fig)
    result=dict(selected_windows=len(selected),selected_flights=selected.flight.nunique(),dates=selected.date.nunique(),candidates=len(candidates),eligible=int(candidates.eligible.sum()),rejections=candidates[~candidates.eligible].reason.value_counts().to_dict(),missing_phase_speed_cells=int((pd.read_csv(OUT/'phase_speed_coverage.csv').selected==0).sum()),amplitude=protocol['amplitude'],gain_summary=summaries,date_balanced_mean=float(date.G_p_mean.mean()),date_bootstrap_interval=np.quantile(boot,[.025,.975]).tolist(),interior_peaks=int(timing.interior_sampled_peak.sum()),half_response_defined=int(timing.half_response_lower.notna().sum()),alpha_sign_changes_0p2=int(timing.alpha_changes_sign_before_0p2.sum()),net_to_max_median=float(timing.net_to_max_0p2.median()),nonlinearity_half_minus_main_abs_q95=float(abs(amp.half_minus_main).quantile(.95)),nonlinearity_double_minus_main_abs_q95=float(abs(amp.double_minus_main).quantile(.95)),numerical_resolution_used=resolution)
    (OUT/'summary.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False));print(summary.to_string(index=False));print(date.to_string(index=False));print(phase.to_string(index=False));print(json.dumps(clean(result),indent=2))
if __name__=='__main__':main()
