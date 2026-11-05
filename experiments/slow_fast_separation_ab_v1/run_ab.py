"""Frozen F mechanism validation. CPU, native observations, train/validation only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys, json, argparse, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/slow_fast_separation_v1'))
from two_channel_followup import native_runs, CHANNELS, DATA, prepare_phase
from methods import cell, fit_common
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PRIOR=ROOT/'docs/analysis/results/slow_fast_separation_v1/final_run'


def harmonics(p,h=3):
    return np.column_stack([f(k*p) for k in range(1,h+1) for f in (np.cos,np.sin)])


def smooth(x):
    z=np.clip(x,0,1)
    return z*z*(3-2*z)


def inject(c,t0,case,scale):
    t=c['t']-t0
    duration=.15 if case=='slow_015' else .30 if case=='slow_030' else .20
    s=scale['slow_amplitude']*np.sin(np.pi*np.clip(t/duration,0,1))**2 if case.startswith(('slow','both')) else np.zeros(len(t))
    if case.startswith(('fast','both')):
        d=2. if case.endswith('2s') else .5
        envelope=np.ones(len(t)) if case=='fast_constant' else .5+smooth(t/d)
        wave=np.sin(c['p'])+.35*np.sin(2*c['p']+.4)+.2*np.cos(3*c['p']-.2)
        f=scale['fast_rms']*wave/np.sqrt((1+.35**2+.2**2)/2)*envelope
        f-=c['w']@f
    else:f=np.zeros(len(t))
    return s,f


def residual_design(rec,h,shift=0.,orthogonal=False):
    c=rec['c']; u=(c['t']-c['tau'])/(c['hi']-c['lo'])
    z=harmonics(c['p']+shift,h)
    if orthogonal:
        b=np.column_stack([np.ones(len(u)),u,u*u])
        z-=b@np.linalg.lstsq(b*np.sqrt(c['w'])[:,None],z*np.sqrt(c['w'])[:,None],rcond=None)[0]
    return u,z


def fit_reg(X,y,w,ridge=1e-3):
    # Weights sum to one per flight, then globally normalized.
    w=w/w.sum(); return np.linalg.solve(X.T@(w[:,None]*X)+ridge*np.eye(X.shape[1]),X.T@(w*y))


def conditions(c,events):
    # Event window [-.5,+1] s; stable requires no event within +/-2 s.
    if np.any((events>=c['lo']-1)&(events<=c['hi']+.5)):return 'changing'
    if not np.any((events>=c['lo']-2)&(events<=c['hi']+2)):return 'stable'
    return 'buffer'


def collect(split,protocol):
    cols=['flight_id','segment_id','valid_target_state','excluded','review','firmware_verified','provenance_review','valid_core']
    for tc,col,_ in CHANNELS.values():cols.extend([tc,col])
    s=pd.read_parquet(DATA/f'samples_{split}.parquet',columns=cols)
    phase=pd.read_parquet(DATA/f'native_wing_phase_{split}.parquet')
    # Shared provenance registry only; retain requested split entries, never load test tables/labels.
    meta={m['flight_id']:m for m in json.loads((DATA/'flight_provenance.json').read_text()) if m['split']==split}
    ev=pd.read_csv(PRIOR/'motion_preservation.csv')
    ev=ev[(ev.split==split)&(ev.kind=='control_event')&ev.channel.isin(CHANNELS)]
    records=[]; banks={}; coverage=[]
    for j,(fid,g) in enumerate(s.groupby('flight_id',sort=True)):
        m=meta[fid]
        if m['sampling_group']!='main_50hz':continue
        pt=prepare_phase(phase[phase.flight_id==fid],m['transmission_ratio'],protocol['phase_contract'])
        for ch in CHANNELS:
            _,runs=native_runs(g,pt,m,ch); banks[(fid,ch)]=runs
            events=ev[(ev.flight_id==fid)&(ev.channel==ch)].event_time_s.to_numpy()
            count=0
            for ri,cs in enumerate(runs):
                for n in range(27,len(cs)-25,5):
                    c=cs[n];tr,f,_=fit_common(cs,n,True)
                    rec=dict(split=split,flight_id=fid,channel=ch,run=ri,cycle=n,c=c,trend=tr,fast=f,r=c['y']-tr-f,group=conditions(c,events))
                    records.append(rec);count+=1
            coverage.append(dict(split=split,flight_id=fid,channel=ch,valid_cycles=sum(map(len,runs)),heldout_cycles=count))
        if j%15==0:print(f'collect {split} flight {j+1}',flush=True)
    return records,banks,ev,coverage


def train_scales(records,events):
    scales={}
    for ch in CHANNELS:
        rr=[r for r in records if r['channel']==ch]
        ff=pd.DataFrame([dict(flight_id=r['flight_id'],rms=np.sqrt(r['c']['w']@(r['fast']**2)),period=r['c']['hi']-r['c']['lo']) for r in rr])
        fast=float(ff.groupby('flight_id').rms.median().median())
        e=events[events.channel==ch].copy(); e['abs_change']=abs(e.raw_change)
        motion=float(e.groupby('flight_id').abs_change.median().median())
        scales[ch]=dict(fast_rms=fast,event_motion_median=motion,slow_amplitude=max(motion,fast),period_median=float(ff.groupby('flight_id').period.median().median()))
    return scales


def a_evaluate(train,validation,out):
    rows=[]; models={}; rng=np.random.default_rng(20261105)
    # Fixed 16 independently shifted-per-beat null realizations, no time/value shuffling.
    shifts={id(r):rng.uniform(0,2*np.pi,16) for r in train+validation}
    for ch in CHANNELS:
      for group in ('all','stable','changing'):
        tr=[r for r in train if r['channel']==ch and (group=='all' or r['group']==group)]
        va=[r for r in validation if r['channel']==ch and (group=='all' or r['group']==group)]
        if not tr or not va:continue
        counts=pd.Series([r['flight_id'] for r in tr]).value_counts()
        # Context derived entirely from the held-out F predictor, never target y.
        def context(r):
            c=r['c'];return np.array([np.mean(r['trend']),(r['trend'][-1]-r['trend'][0])/(c['t'][-1]-c['t'][0]),np.sqrt(c['w']@(r['fast']**2)),1/(c['hi']-c['lo'])])
        ctx=np.array([context(r) for r in tr]);mean=ctx.mean(0);sd=ctx.std(0);sd=np.maximum(sd,1e-6)
        def design(r,h,shift,orth):
            u,z=residual_design(r,h,shift,orth);v=np.clip((context(r)-mean)/sd,-5,5)
            base=np.column_stack([np.ones(len(u)),u,u*u]+[np.full(len(u),x) for x in v])
            # Conditioned harmonic coefficients: intercept + four observable contexts.
            phase=np.concatenate([z]+[z*x for x in v],axis=1)
            return base,phase
        weights=np.concatenate([r['c']['w']/counts[r['flight_id']] for r in tr]);y=np.concatenate([r['r'] for r in tr])
        for orth in (False,True):
          for h in (1,2,3):
            b=np.concatenate([design(r,h,0,orth)[0] for r in tr]);bc=fit_reg(b,y,weights)
            for null in range(-1,16):
                def shift(r):return 0. if null<0 else shifts[id(r)][null]
                x=np.concatenate([np.column_stack(design(r,h,shift(r),orth)) for r in tr]);coef=fit_reg(x,y,weights)
                for split,rr in [('train',tr),('validation',va)]:
                  for r in rr:
                    b,z=design(r,h,shift(r),orth);base=b@bc;pred=np.column_stack([b,z])@coef;periodic=z@coef[b.shape[1]:];w=r['c']['w']
                    rows.append(dict(split=split,flight_id=r['flight_id'],channel=ch,group=group,harmonics=h,orthogonal_time=orth,null=null,run=r['run'],cycle=r['cycle'],baseline_mse=w@((r['r']-base)**2),model_mse=w@((r['r']-pred)**2),periodic_power=w@(periodic**2),fast_power=w@(r['fast']**2),residual_power=w@(r['r']**2)))
                    if null==-1 and h==3 and not orth and group!='all':r['phase_prediction']=periodic
            print(f'A {ch} {group} H{h} orthogonal={orth}',flush=True)
    df=pd.DataFrame(rows)
    keys=['split','flight_id','channel','group','harmonics','orthogonal_time','null']
    flights=df.groupby(keys).agg(n_cycles=('cycle','size'),baseline_mse=('baseline_mse','mean'),model_mse=('model_mse','mean'),periodic_power=('periodic_power','mean'),fast_power=('fast_power','mean'),residual_power=('residual_power','mean')).reset_index()
    flights['gain']=1-flights.model_mse/flights.baseline_mse
    flights['periodic_rms']=np.sqrt(flights.periodic_power);flights['relative_fast_rms']=np.sqrt(flights.periodic_power/flights.fast_power)
    flights.to_csv(out/'A_per_flight.csv',index=False)
    summary=flights.groupby([k for k in keys if k!='flight_id']).agg(n_flights=('flight_id','size'),n_cycles=('n_cycles','sum'),baseline_mse=('baseline_mse','mean'),model_mse=('model_mse','mean'),periodic_power=('periodic_power','mean'),fast_power=('fast_power','mean'),positive_flights=('gain',lambda s:int((s>0).sum()))).reset_index()
    summary['gain']=1-summary.model_mse/summary.baseline_mse;summary['periodic_rms']=np.sqrt(summary.periodic_power);summary['relative_fast_rms']=np.sqrt(summary.periodic_power/summary.fast_power)
    summary.to_csv(out/'A_summary.csv',index=False)
    return summary


def select_injection(banks,half):
    selected=[]
    # Same physical center for vD and q. Longest overlapping valid interior per flight.
    for fid in sorted({k[0] for k in banks}):
        options=[]
        for iv,v in enumerate(banks.get((fid,'vD'),[])):
          if len(v)<51:continue
          for iq,q in enumerate(banks.get((fid,'q'),[])):
            if len(q)<51:continue
            lo=max(v[25]['lo'],q[25]['lo']);hi=min(v[-26]['hi'],q[-26]['hi'])
            if hi-lo>=2*half+2:options.append((hi-lo,-iv,-iq,lo,hi))
        if options:
            _,iv,iq,lo,hi=max(options);t0=(lo+hi-2)/2
            selected.append((fid,-iv,-iq,t0))
    return selected


def b_evaluate(banks,split,scales,half,out,rows,selection):
    cases=['slow_015','slow_030','fast_constant','fast_05s','fast_2s','both_05s','both_2s']
    selected=select_injection(banks,half)
    for ix,(fid,iv,iq,t0) in enumerate(selected):
      selection.append(dict(split=split,flight_id=fid,vD_run=iv,q_run=iq,event_time_s=t0,half_window_s=half))
      for ch,ri in [('vD',iv),('q',iq)]:
        cs=banks[(fid,ch)][ri]
        ids=[i for i in range(25,len(cs)-25) if cs[i]['hi']>t0-half and cs[i]['lo']<t0+2+half]
        original={i:fit_common(cs,i,False)[1] for i in ids}
        for case in cases:
            sf=[inject(c,t0,case,scales[ch]) for c in cs]
            changed=[cell(c['t'],c['p'],c['y']+s+f,c['lo'],c['hi']) for c,(s,f) in zip(cs,sf)]
            pieces=[]
            for i in ids:
                c=cs[i];s,f=sf[i];new=fit_common(changed,i,False)[1];df=new-original[i];ds=s+f-df
                assert abs(c['w']@f)<1e-12
                pieces.append(np.column_stack([c['t']-t0,c['w']*(c['hi']-c['lo']),s,f,ds,df]))
            z=np.concatenate(pieces);t,w,s,f,ds,df=z.T
            # Select stage observations while retaining their original native quadrature.
            for stage,lo,hi in [('full',-half,2+half),('before',-half,0),('onset',0,.3),('changing',.3,2),('recovery',2,2+half),('late_recovery',1+half,2+half)]:
                mask=(t>=lo)&(t<hi);ww=w[mask];ww=ww/ww.sum();err=df[mask]-f[mask]
                sr=np.sqrt(ww@(s[mask]**2));fr=np.sqrt(ww@(f[mask]**2));rms=np.sqrt(ww@(err**2))
                rows.append(dict(split=split,flight_id=fid,channel=ch,case=case,stage=stage,slow_error_rmse=rms,fast_error_rmse=rms,injection_rms=np.sqrt(sr*sr+fr*fr),normalized_error=rms/np.sqrt(sr*sr+fr*fr) if sr*sr+fr*fr>1e-20 else np.nan,leak_max=np.max(abs(err)),slow_inj_rms=sr,fast_inj_rms=fr,n_observations=int(mask.sum())))
            if ix==0:
                pd.DataFrame(z,columns=['relative_time_s','time_weight_s','slow_truth','fast_truth','delta_slow','delta_fast']).to_csv(out/f'B_trace_{split}_{ch}_{case}.csv',index=False)
                if split=='validation':
                    fig,ax=plt.subplots(2,1,figsize=(11,5),sharex=True)
                    ax[0].plot(t,s,label='injected Slow');ax[0].plot(t,ds,label='delta Slow');ax[1].plot(t,f,label='injected Fast');ax[1].plot(t,df,label='delta Fast')
                    for a in ax:a.legend();a.grid(alpha=.2)
                    ax[1].set_xlabel('time from injection (s)');fig.suptitle(f'{ch}: {case} / {fid}');fig.tight_layout();fig.savefig(out/f'B_{ch}_{case}.png',dpi=130);plt.close(fig)
      print(f'B {split} {ix+1}/{len(selected)}',flush=True)


def plots_a(records,out):
    for ch in CHANNELS:
      for group in ('stable','changing'):
        rs=[r for r in records if r['channel']==ch and r['group']==group]
        if not rs:continue
        # Lexicographic flight/run, then middle eligible beat: independent of error.
        key=min((r['flight_id'],r['run']) for r in rs);rs=[r for r in rs if (r['flight_id'],r['run'])==key];center=rs[len(rs)//2]
        nearby=[r for r in records if r['channel']==ch and (r['flight_id'],r['run'])==key and abs(r['c']['tau']-center['c']['tau'])<2]
        fig,ax=plt.subplots(3,1,figsize=(11,7),sharex=True);traces=[]
        for r in nearby:
            c=r['c'];t=c['t']-center['c']['tau'];p=r.get('phase_prediction',np.full(len(t),np.nan))
            ax[0].plot(t,c['y'],color='gray');ax[0].plot(t,r['fast'],color='orange')
            ax[1].plot(t,c['y']-r['fast'],color='blue');ax[1].plot(t,r['trend'],color='green')
            ax[2].plot(t,r['r'],color='gray');ax[2].plot(t,p,color='red')
            traces.append(pd.DataFrame(dict(relative_time_s=t,raw=c['y'],fast=r['fast'],slow=c['y']-r['fast'],trend=r['trend'],residual=r['r'],phase_prediction=p)))
        for a,title in zip(ax,['raw (gray), held-out F Fast (orange)','Slow (blue), common trend (green)','residual (gray), train-fitted phase prediction (red)']):a.set_ylabel(ch);a.set_title(title);a.grid(alpha=.2)
        ax[-1].set_xlabel('time (s), gaps are non-target beats');fig.suptitle(f'{group} / {key}');fig.tight_layout();fig.savefig(out/f'A_{ch}_{group}.png',dpi=130);plt.close(fig)
        pd.concat(traces).to_csv(out/f'A_trace_{ch}_{group}.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'docs/analysis/results/slow_fast_separation_ab_v1');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();protocol=json.loads((PRIOR/'protocol.json').read_text())
    train,tbank,tev,tcov=collect('train',protocol);scales=train_scales(train,tev)
    half=25*max(x['period_median'] for x in scales.values())+1
    config=dict(base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dataset='velocity_rate_multistep_v1',splits=['train','validation'],phase_contract=protocol['phase_contract'],event_thresholds_train_p90=protocol['event_thresholds_train_p90'],event_identity='reuse original control_event rows for both splits',event_window_s=[-.5,1],stable_exclusion_s=[-2,2],scales=scales,half_window_s=half,seed=20261105,ridge=1e-3,harmonics=[1,2,3],null_realizations=16,selection='longest common native valid interior per flight; one physical center; require full event plus 25 cycles on both sides',cases=['slow_015','slow_030','fast_constant','fast_05s','fast_2s','both_05s','both_2s'],fast_wave='sin(phi)+0.35 sin(2phi+0.4)+0.2 cos(3phi-0.2); continuous RMS normalized then native beat weighted demean',fast_envelope='constant 1 or 0.5+smoothstep(t/d), d=0.5 or 2 s',slow_wave='amplitude*sin(pi*t/d)^2 within [0,d], else zero; d=.15,.30 or .20 for coupled cases')
    # Frozen before any validation table is opened.
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    validation,vbank,vev,vcov=collect('validation',protocol)
    pd.DataFrame(tcov+vcov).to_csv(out/'coverage.csv',index=False)
    a_evaluate(train,validation,out);plots_a(validation,out)
    rows=[];selection=[]
    b_evaluate(tbank,'train',scales,half,out,rows,selection);b_evaluate(vbank,'validation',scales,half,out,rows,selection)
    pd.DataFrame(selection).to_csv(out/'B_selection.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(out/'B_per_flight.csv',index=False)
    summary=df.groupby(['split','channel','case','stage']).agg(n_flights=('flight_id','nunique'),error_mse=('fast_error_rmse',lambda s:np.mean(s*s)),normalized_error_mean=('normalized_error','mean'),normalized_error_max=('normalized_error','max'),leak_max=('leak_max','max'),slow_inj_rms=('slow_inj_rms','mean'),fast_inj_rms=('fast_inj_rms','mean')).reset_index();summary['error_rmse']=np.sqrt(summary.error_mse);summary.to_csv(out/'B_summary.csv',index=False)
    (out/'runtime.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-start),indent=2))
    from report import main as render_report
    render_report(out)
    print('completed',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
