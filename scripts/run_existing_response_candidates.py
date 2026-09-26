"""Review fixed train-only rudder/common candidates; no model fitting/inference."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import run_response_timescale_diagnostic as d
m=d.m;ROOT=d.ROOT
OUT=ROOT/'docs/analysis/results/existing_response_candidates_v1'

def hold_flags(first,late,other_late,thresholds):
    sign=np.sign(first)
    return (np.mean(sign[:,None]*late>0,axis=1)>=.8)&(sign*late.mean(1)>=.5*np.abs(first))&(other_late<=thresholds).all(1)

def weighted(x,dt):return np.sum(x*dt[...,None],axis=1)/dt.sum(1)[:,None]

def feature_vector(t):
    q=t.truth.quaternion_nb[:,0];w,x,y,z=q.T
    gravity=np.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],axis=1)
    return np.c_[t.truth.velocity_n[:,0],t.truth.angular_velocity_b[:,0],gravity,t.truth.flap_frequency_hz[:,0],t.controls[:,0]]

def nearest_opposite(x,sign,flights,same):
    result=[]
    for i in range(len(x)):
        ok=(sign!=sign[i]) & ((flights==flights[i]) if same else (flights!=flights[i]))
        ids=np.flatnonzero(ok)
        if len(ids):
            distance=np.sqrt(np.mean((x[ids]-x[i])**2,axis=1));k=np.argmin(distance)
            result.append((i,int(ids[k]),float(distance[k])))
    return result

def run():
    assert not OUT.exists();OUT.mkdir(parents=True)
    source=ROOT/'docs/analysis/results/september_data_sufficiency_v1/nonoverlap_candidates.csv'
    c=pd.read_csv(source).query('partition=="train" and channel in ["rudder","common"]').sort_values(['channel','flight_id','origin_timestamp_us']).reset_index(drop=True)
    assert c.groupby('channel').size().to_dict()=={'common':68,'rudder':38}
    sp,batches,stats=d.r.source.inputs();b=batches['train'];t=b.trajectory
    ids={str(w):i for i,w in enumerate(t.window_ids)};ix=np.array([ids[w] for w in c.window_id]);assert len(set(ix))==len(ix)
    thpath=ROOT/'docs/analysis/results/response_timescale_diagnostic_v1/channel_thresholds.csv';th=pd.read_csv(thpath)
    features=feature_vector(t);scale=features.std(0,ddof=0);assert np.all(scale>0)
    pins={m.rel(p):m.file_hash(p) for p in [source,thpath,m.ART/'prepared.pt',Path(__file__),ROOT/'tests/test_existing_response_candidates.py']}
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inputs=pins,heldout_accessed=False,training=False,new_predictions=False,candidate_counts={'common':68,'rudder':38},
        selection='All prior fixed train nonoverlapping candidates, sorted identity; no outcome screening.',
        hold_diagnostic='Commands k25..49: >=80% have same departure sign as first25 mean; late signed mean>=0.5*abs(first25 mean); other-channel late standardized RMS departures<=frozen train q25. Descriptive only, not quality gate or physical steady state.',
        prehistory='H26 command RMS departure from origin <=0.5*abs(first25 mean) for target channel, descriptive stability flag.',
        responses='Original unfiltered p/q/r minus t0; dt-weighted intervals steps1..5,16..25,41..50 and endpoints5/10/25/50; mean response sign association is not causality or actuator sign correctness.',
        matching=dict(features='initial velocity_NED[3], rate_FRD[3], gravity_body[3], flap_frequency[1], initial command[4]',scale=scale.tolist(),
            rule='For each candidate, nearest opposite-sign candidate in same flight and separately different flight, RMS standardized distance; with replacement, no response used; no maximum-distance success threshold.',
            limitation='scales describe initial-state distances from existing train, never replace frozen model normalization. No matching of wind, actuator state, phase, controller integrators or history; not causal pairs.'),
        summaries='Candidate distributions and equal-flight sign-aligned mean response; no window-independent significance test; original finite candidates retained even if unfavorable.'))
    u=d.r.coordinates(t.controls[ix]);h=d.r.coordinates(b.history_controls[ix]);rates=t.truth.angular_velocity_b[ix];dt=t.dt_s[ix];elapsed=np.c_[np.zeros(len(ix)),np.cumsum(dt,axis=1)]
    du=u-u[:,:1];initial=[];response=[];trace=[]
    for ch in ['common','rudder']:
        jj=d.r.CHANNELS.index(ch);mask=(c.channel==ch).to_numpy();idx=np.flatnonzero(mask);others=np.arange(4)!=jj
        first=du[idx,:25,jj].mean(1);late=du[idx,25:,jj]
        other=np.sqrt(np.mean((du[idx,25:][:,:,others]/th.scale.to_numpy()[others])**2,axis=1))
        held=hold_flags(first,late,other,th.train_q25.to_numpy()[others])
        pre=np.sqrt(np.mean((h[idx,:,jj]-u[idx,:1,jj])**2,axis=1));stable=pre<=.5*np.abs(first)
        for k,i in enumerate(idx):
            initial.append(dict(**c.iloc[i].to_dict(),first_mean_departure=first[k],late_mean_departure=late[k].mean(),late_same_sign_fraction=float(np.mean(np.sign(first[k])*late[k]>0)),
                prehistory_rms_departure=pre[k],prehistory_stable=bool(stable[k]),holds_and_others_quiet_to_1s=bool(held[k]),both_flags=bool(stable[k]&held[k]),
                initial_speed=float(np.linalg.norm(t.truth.velocity_n[ix[i],0])),initial_rate_p=rates[i,0,0],initial_rate_q=rates[i,0,1],initial_rate_r=rates[i,0,2],frequency_hz=float(t.truth.flap_frequency_hz[ix[i],0]),actual_horizon_s=elapsed[i,-1]))
        for name,start,end in [('early_100ms',1,5),('late_500ms',16,25),('late_1s',41,50),('endpoint_100ms',5,5),('endpoint_200ms',10,10),('endpoint_500ms',25,25),('endpoint_1s',50,50)]:
            change=rates[idx,start:end+1]-rates[idx,:1]
            values=weighted(change,dt[idx,start-1:end])
            for k,i in enumerate(idx):
                for j,axis in enumerate('pqr'):
                    response.append(dict(channel=ch,flight_id=c.iloc[i].flight_id,window_id=c.iloc[i].window_id,sign=c.iloc[i].sign,interval=name,axis=axis,delta_rate=float(values[k,j]),sign_aligned_delta_rate=float(c.iloc[i].sign*values[k,j])))
    for i in range(len(ix)):
        for k in range(51):trace.append(dict(window_id=c.iloc[i].window_id,channel=c.iloc[i].channel,step=k,elapsed_s=elapsed[i,k],p=rates[i,k,0],q=rates[i,k,1],r=rates[i,k,2],**({ch:u[i,k,j] for j,ch in enumerate(d.r.CHANNELS)} if k<50 else {})))
    review=pd.DataFrame(initial);review.to_csv(OUT/'candidate_review.csv',index=False);pd.DataFrame(trace).to_csv(OUT/'all_traces.csv',index=False)
    resp=pd.DataFrame(response);resp.to_csv(OUT/'candidate_responses.csv',index=False)
    flight=resp.groupby(['channel','flight_id','interval','axis']).agg(n=('window_id','size'),sign_aligned_mean=('sign_aligned_delta_rate','mean')).reset_index();flight.to_csv(OUT/'response_per_flight.csv',index=False)
    summary=flight.groupby(['channel','interval','axis']).agg(flights=('flight_id','size'),equal_flight_mean=('sign_aligned_mean','mean'),flight_std=('sign_aligned_mean','std'),positive_flights=('sign_aligned_mean',lambda v:int((v>0).sum())),negative_flights=('sign_aligned_mean',lambda v:int((v<0).sum()))).reset_index();summary.to_csv(OUT/'response_summary.csv',index=False)
    pairs=[]
    for ch in ['common','rudder']:
        take=np.flatnonzero((c.channel==ch).to_numpy());x=features[ix[take]]/scale
        for same in [True,False]:
            for ia,ib,dist in nearest_opposite(x,c.iloc[take].sign.to_numpy(),c.iloc[take].flight_id.to_numpy(),same):
                i,j=take[ia],take[ib]
                pairs.append(dict(channel=ch,kind='same_flight' if same else 'cross_flight',window_id=c.iloc[i].window_id,opposite_window_id=c.iloc[j].window_id,distance=dist,
                    delta_speed=float(np.linalg.norm(t.truth.velocity_n[ix[i],0])-np.linalg.norm(t.truth.velocity_n[ix[j],0])),delta_frequency=float(t.truth.flap_frequency_hz[ix[i],0]-t.truth.flap_frequency_hz[ix[j],0])))
    pd.DataFrame(pairs).to_csv(OUT/'opposite_sign_neighbors.csv',index=False)
    statsout=review.groupby('channel').agg(candidates=('window_id','size'),flights=('flight_id','nunique'),prehistory_stable=('prehistory_stable','sum'),held_to_1s=('holds_and_others_quiet_to_1s','sum'),both=('both_flags','sum')).reset_index();statsout.to_csv(OUT/'coverage_summary.csv',index=False)
    # Entire fixed candidate set visualized; no selected favorable cases.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(OUT/'all_candidates.pdf') as pdf:
        for start in range(0,len(c),4):
            fig,axs=plt.subplots(4,2,figsize=(12,12))
            for row,i in enumerate(range(start,min(start+4,len(c)))):
                for j,ch in enumerate(d.r.CHANNELS):axs[row,0].step(elapsed[i,:50],du[i,:,j],where='post',label=ch)
                for j,axis in enumerate('pqr'):axs[row,1].plot(elapsed[i],rates[i,:,j]-rates[i,0,j],label=axis)
                axs[row,0].set_title(f'{i}: {c.iloc[i].channel} sign={c.iloc[i].sign:+.0f}',fontsize=9);axs[row,1].set_title(c.iloc[i].window_id,fontsize=6)
                axs[row,0].set_ylabel('Command departure');axs[row,1].set_ylabel('Rate departure [rad/s]')
                for ax in axs[row]:ax.axvline(.5,color='k',ls=':',lw=.5);ax.legend(fontsize=6);ax.set_xlabel('Actual elapsed time [s]');ax.grid(alpha=.2)
            for row in range(min(4,len(c)-start),4):
                for ax in axs[row]:ax.set_visible(False)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    assert np.isfinite(rates).all() and np.isfinite(u).all() and (dt>0).all()
    m.verify_pins(pins)
    m.write_json(OUT/'checks.json',dict(all106_candidates_retained=True,all_candidate_traces_rendered=True,heldout_accessed=False,no_model_inference=True,pins_unchanged=True,numerics_finite=True))
    print(statsout.to_string(index=False))

if __name__=='__main__':run()
