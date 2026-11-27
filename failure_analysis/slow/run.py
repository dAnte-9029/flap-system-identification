"""Common-support slow angular-rate diagnosis. Run with the repository interpreter."""
from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIOR = ROOT / '.worktrees/anchored-slow-rate-b/experiments/anchored_rate'
sys.path.insert(0, str(PRIOR))
from labels import ref, CACHE
from train import data, args
from anchored_model import AnchoredRate, wave

DIAG = ROOT / 'research/rounds/periodic-motion-20261125/diagnostics'
OLD = ROOT / 'outputs/autopilot_flight_dynamics_20261124'
MODEL = ROOT / 'research/experiments/study-6a5d1c33c47e/exp-fa28c954ce7a/attempt-1/seed-17/output/model'


def linear_trend(frame, anchors):
    """Two consecutive measured beat means. Match published last-beat anchor exactly."""
    phase_fn = ref.load_module('phase_history', ref.SOURCE/'src/system_identification/data/pahbd_phase.py').prepare_phase
    contract = json.loads((ref.SOURCE/'configs/pahbd/phase_reference.json').read_text())
    meta = {m['flight_id']:m for m in json.loads((ref.DATA/'flight_provenance.json').read_text()) if m['split']=='validation'}
    groups = [pd.read_parquet(ref.DATA/(n+'_validation.parquet')).groupby('flight_id') for n in ('samples','native_wing_phase','native_vehicle_angular_velocity')]
    windows = pd.read_parquet(ref.HISTORY/'validation_windows.parquet').set_index('window_id')
    trend = np.full((len(frame),3), np.nan)
    for fid, sub in frame.groupby('flight'):
        state, ph, rates = [g.get_group(fid) for g in groups]
        phase = phase_fn(ph,meta[fid]['transmission_ratio'],contract)
        cycles = ref.make_cycles(phase,meta[fid],state,rates,False,Counter())
        rt = rates.loc[rates.in_reviewed_flight].drop_duplicates('timestamp_sample_us').set_index('timestamp_sample_us')
        st = state.state_measurement_us.to_numpy()
        pub = state.state_publication_us.to_numpy()
        beats = {}
        for i,c in cycles.items():
            t = np.rint(c['t']*1e6).astype(int)
            j = np.searchsorted(st,t)
            k = np.where(st[j]==t,j,(j-1).clip(0))
            phase_edge = phase.timestamp_us.searchsorted(c['right'])
            ready = max(pub[k].max(),pub[j].max(),rt.loc[t,'timestamp_us'].max(),phase.availability_us.iloc[phase_edge])
            beats[i] = (c['left'],c['right'],ready,c['weight']@c['data'],c['segment'])
        for wi,group in sub.groupby('window'):
            now = windows.loc[anchors['window'][wi],'decision_us']
            candidates = [i for i,(l,r,p,v,s) in beats.items() if l>=now-500000 and r<=now and p<=now and np.allclose(v,anchors['anchor'][wi],rtol=1e-6,atol=1e-7)]
            assert candidates
            i = candidates[-1]
            if i-1 not in beats: continue
            l,r,_,v,s = beats[i]
            pl,pr,ready,prev,ps = beats[i-1]
            if ps!=s or pr!=l or ready>now: continue
            slope = (v-prev)*2e6/(l+r-pl-pr)
            trend[group.index] = v + (group.lead.to_numpy()+(now-(l+r)/2)*1e-6)[:,None]*slope
    return trend


def design(anchor,origin,lead,commands=None):
    values = [np.ones((len(lead),1)),lead[:,None],anchor,origin-anchor]
    if commands is not None: values.append(commands)
    return np.column_stack(values)


def fit_predict(b,wi,t):
    with np.load(CACHE/'train.npz') as z, np.load(OLD/'train.npz') as raw, np.load(OLD/'validation.npz') as dev:
        np.testing.assert_array_equal(z['window'],raw['window'])
        np.testing.assert_array_equal(b['window'],dev['window'])
        idx = z['wi']
        offsets = np.r_[0,z['lengths'].cumsum()[:-1]]
        times = np.add.reduceat(z['t']*z['weight'],offsets)
        anchor,origin,commands = z['anchor'][idx],z['origin'][idx],raw['u0'][idx]
        y = z['target']-anchor
        out = {}
        for use in (False,True):
            x = design(anchor,origin,times,commands if use else None)
            v = design(b['anchor'][wi],b['origin'][wi],t,dev['u0'][wi] if use else None)
            w = np.linalg.lstsq(x,y,rcond=1e-8)[0]
            out['linear_motion_cmd' if use else 'linear_motion'] = b['anchor'][wi]+v@w
        return out


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    net = AnchoredRate().eval()
    net.weights.copy_(torch.load(MODEL,weights_only=True,map_location='cpu')['weights'])
    d,_,_ = data('validation')
    with np.load(CACHE/'validation.npz') as z: b=dict(z)
    directory,refarr,times,pool,_ = wave.native_cache(ROOT,'cpu')
    sys.path.insert(0,str(HERE.parent/'models'))
    from common_paths import TMP
    bare=np.load(TMP/'bare17.npy')
    assert bare.shape==refarr['origin'].shape
    first = np.r_[0,np.flatnonzero(np.diff(refarr['block']))+1]
    def mean(a):
        weight=refarr['weight'].reshape((-1,)+(1,)*(a.ndim-1))
        return np.add.reduceat(weight*a,first)
    ids=refarr['window_idx']
    branch=[]
    for j in range(0,len(ids),512):
        sl=slice(j,j+512)
        ii=ids[sl]
        h,p,u,ct,_=args(d,ii)
        a=torch.from_numpy(b['anchor'][ii])
        branch.append((a+ (net.features(net.states(h,p,u,ct),ct[:,:,1],times[sl,:,1],pool[sl,:,1],a)@net.weights)[:,0]).numpy())
    branch=np.concatenate(branch)
    # Weighted nested fit. Improvement after local affine trend measures phase-locked
    # within-beat variation that the slow output still contains.
    def phase_rms2(points):
        projected=[]
        for a,e in zip(first,np.r_[first[1:],len(ids)]):
            sl=slice(a,e)
            t=refarr['time'][sl]
            phi=refarr['phase'][sl]
            x=np.column_stack([np.ones(len(t)),(t-t.mean())/np.ptp(t),np.cos(phi),np.sin(phi)])
            w=np.sqrt(refarr['weight'][sl])[:,None]
            target=w*points[sl]
            design=w*x
            sq=[]
            for n in (2,4):
                m=design[:,:n]
                sq.append(np.sum((target-m@np.linalg.lstsq(m,target,rcond=None)[0])**2))
            projected.append(max(0,sq[0]-sq[1]))
        return np.array(projected)
    eligible=(refarr['ch'][first]==1)&b['ok'][ids[first]]
    wi=ids[first][eligible]
    flight=refarr['flight_ids'][refarr['flight_index'][first]][eligible]
    lead=mean(refarr['time'])[eligible]
    frame=pd.DataFrame(dict(window=wi,flight=flight,lead=lead))
    y=mean(refarr['slow'])[eligible]
    forecasts={'instant':b['origin'][wi],'last_beat':b['anchor'][wi],'two_beat_linear':linear_trend(frame,b),
               'bare_s17':mean(bare)[eligible],'anchored_B_s17':mean(branch)[eligible],**fit_predict(b,wi,lead)}
    leak={'B_sync_after_trend':phase_rms2(branch)[eligible],'bare_sync_after_trend':phase_rms2(bare)[eligible]}
    for seed in (17,29,43):
        with np.load(directory/f'original_corrected{seed}_points.npz') as z:
            leak[f'original_slow_sync_s{seed}']=phase_rms2(z['slow_abs'])[eligible]
            forecasts[f'original_slow_s{seed}']=mean(z['slow_abs'])[eligible]
    def energy(values):
        return mean((values**2).sum(axis=-1))[eligible]
    stats={**leak,'sync_energy':energy(refarr['fast']),'reference_residual':energy(refarr['data']-refarr['slow']-refarr['fast']),
           'beatmean_reference_disagreement':((mean(refarr['data'])[eligible]-y)**2).sum(1)}
    # An independent reference may still reuse the last historical beat in its four neighbors.
    # Score actual native target-beat means as a second outcome.
    observed=mean(refarr['data'])[eligible]
    for name,v in forecasts.items():
        delta_frame=pd.DataFrame(v-y)
        first_err=delta_frame.groupby(frame.window).transform('first').to_numpy()
        evolution=(v-y-first_err)
        later=frame.groupby('window').cumcount().to_numpy()>0
        stats[name+'_evolution']=np.where(later,(evolution**2).sum(1),np.nan)
        err=v-y
        stats[name]=(err**2).sum(1)
        stats[name+'_vs_measured']=((v-observed)**2).sum(1)
        for i,axis in enumerate('pqr'): stats[name+'_'+axis]=err[:,i]**2
    frame=frame.assign(**stats)
    common=np.isfinite(frame.two_beat_linear)
    mags=np.linalg.norm(y-b['anchor'][wi],axis=1)
    # Define conditions by reference change, independently of model errors.
    thresh=float(np.quantile(mags,.75))
    tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/validation_commands.npy',mmap_mode='r')
    with np.load(OLD/'validation.npz') as raw:
        u0=raw['u0'][wi]
        np.testing.assert_array_equal(raw['window'],b['window'])
    activity=[]
    for w,h,initial in zip(wi,lead,u0):
        cmd=tape[w].reshape(-1,5)
        dt=cmd[:,4]*1e-6
        weight=np.minimum(dt,np.maximum(0,h-(dt.cumsum()-dt)))
        activity.append(np.sqrt(weight@((cmd[:,:4]-initial)**2).sum(1)/h))
    activity=np.array(activity)
    low,high=np.quantile(activity,[.25,.75])
    print('command RMS thresholds',low,high)
    subsets={'quiet_commands':activity<=low,'active_commands':activity>=high,'anchor' :np.ones(len(frame),bool),'two_consecutive':common,'lower_change':mags<thresh,'upper_change':mags>=thresh}
    rows=[];flights=[]
    for label,mask in subsets.items():
        for horizon,lo,hi in [('all',0,2),('0-.5',0,.5),('.5-1',.5,1),('1-2',1,2)]:
            f=frame[mask&(lead>lo)&(lead<=hi)]
            if f.empty: continue
            m=f.groupby(['flight','window'])[list(stats)].mean().groupby('flight').mean()
            if label!='two_consecutive': m=m.drop(columns=[c for c in m if c.startswith('two_beat_linear')])
            flights.append(m.assign(subset=label,horizon=horizon))
            rows.append(dict(subset=label,horizon=horizon,flights=len(m),windows=f.window.nunique(),beats=len(f),**np.sqrt(m.mean()).to_dict()))
    pd.DataFrame(rows).to_csv(HERE/'rmse.csv',index=False)
    pd.concat(flights).to_csv(HERE/'flight_mse.csv')
    print(pd.DataFrame(rows)[['subset','horizon','flights','beats',*forecasts.keys(),'sync_energy','reference_residual']].to_string(index=False))
    print('upper-change threshold rad/s',thresh)
    anchor_row=next(r for r in rows if r['subset']=='anchor' and r['horizon']=='all')
    np.testing.assert_allclose(anchor_row['anchored_B_s17'],.18278,atol=.00001)

if __name__=='__main__': main()
