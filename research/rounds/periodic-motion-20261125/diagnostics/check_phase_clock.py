"""Test causal constant-frequency phase extrapolation against offline relative Hall phase."""
import json
import numpy as np
import pandas as pd
from prepare_reference import HERE, SOURCE, DATA, HISTORY, OLD, TAU, load_module


def main():
    contract=json.loads((SOURCE/'configs/pahbd/phase_reference.json').read_text())
    prepare=load_module('diag_phase',SOURCE/'src/system_identification/data/pahbd_phase.py').prepare_phase
    meta={r['flight_id']:r for r in json.loads((DATA/'flight_provenance.json').read_text()) if r['split']=='validation'}
    phase=pd.read_parquet(DATA/'native_wing_phase_validation.parquet').groupby('flight_id')
    wins=pd.read_parquet(HISTORY/'validation_windows.parquet').set_index('window_id')
    with np.load(OLD) as z:
        windows=wins.loc[z['window']].reset_index()
        history=z['hist'][:,-30:]
    with np.load(HERE/'reference.npz') as z:
        rr=dict(z)
    error=np.full(len(rr['time']),np.nan)
    reasons={}
    for fi in np.unique(rr['flight_index']):
        fid=rr['flight_ids'][fi]
        tdata=prepare(phase.get_group(fid),meta[fid]['transmission_ratio'],contract)
        t=tdata.timestamp_us.to_numpy()
        ph=tdata.phase_rad.to_numpy()
        good=tdata.phase_usable.to_numpy()
        bad_cum=np.r_[0,(~good).cumsum()]
        resets=np.asarray(meta[fid]['reset_times_us'])
        for wi in np.unique(rr['window_idx'][rr['flight_index']==fi]):
            origin=windows.decision_us.iloc[wi]
            ids=np.flatnonzero(rr['window_idx']==wi)
            assert windows.flight_id.iloc[wi]==fid
            l=np.searchsorted(t,origin,'right')-1; r=np.searchsorted(t,origin)
            fr=history[wi,15]
            if min(history[wi,21],history[wi,22])!=1 or not (0<fr<30):
                reasons['missing_model_phase_or_frequency']=reasons.get('missing_model_phase_or_frequency',0)+len(ids); continue
            if l<0 or r>=len(t) or not good[l:r+1].all():
                reasons['missing_offline_origin']=reasons.get('missing_offline_origin',0)+len(ids);continue
            rel=np.angle(np.exp(1j*(ph[r]-ph[l])))
            mixture=(origin-t[l])/max(t[r]-t[l],1)
            anchor=ph[l]+mixture*rel
            endpoint=origin+rr['time'][ids]*1e6
            end=np.searchsorted(t,endpoint).clip(max=len(t)-1)
            remain=(endpoint<=t[-1]) & ((bad_cum[end+1]-bad_cum[l])==0)
            remain &= np.searchsorted(resets,endpoint,'right')==np.searchsorted(resets,origin,'left')
            rejected=int((~remain).sum())
            reasons['intervening_phase_qc_or_reset']=reasons.get('intervening_phase_qc_or_reset',0)+rejected
            which=ids[remain]
            error[which]=np.angle(np.exp(1j*(anchor+TAU*fr*rr['time'][which]-rr['phase'][which])))
    ok=np.isfinite(error)
    frame=pd.DataFrame(dict(error=error[ok],time=rr['time'][ok],flight=rr['flight_index'][ok],
                            window=rr['window_idx'][ok],stream=rr['ch'][ok],block=rr['block'][ok],wt=rr['weight'][ok]))
    frame['lead']=np.digitize(frame.time,[.5,1.],right=True)
    frame['cos']=np.cos(frame.error);frame['sin']=np.sin(frame.error);frame['mse']=frame.error**2
    frame['within45']=(abs(frame.error)<np.pi/4).astype(float)
    out={}
    for (c,h),g in frame.groupby(['stream','lead']):
        names=['cos','sin','mse','within45']
        # A partial beat inside one lead bin contributes only its observed time.
        weight=g.wt/g.groupby(['flight','window']).wt.transform('sum')
        v=(g[names].mul(weight,axis=0)).groupby([g.flight,g.window]).sum().groupby('flight').mean()
        mean=v.mean()
        boot=np.random.default_rng(17).integers(len(v),size=(10000,len(v)))
        ci=np.quantile(v.cos.to_numpy()[boot].mean(1),[.025,.975]).tolist()
        res=dict(windows=int(g.window.nunique()),flights=int(g.flight.nunique()),native_points=len(g),
                 mean_cosine=mean['cos'],cosine_interval95=ci,circular_rms_deg=np.degrees(np.sqrt(mean.mse)),
                 circular_bias_deg=np.degrees(np.arctan2(mean.sin,mean['cos'])),within45_fraction=mean.within45)
        out.setdefault(['velocity','angular_rate'][c],{})[['0-0.5','0.5-1','1-2'][h]]=res
    (HERE/'phase_clock.json').write_text(json.dumps(dict(exclusions=reasons,results=out),indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
