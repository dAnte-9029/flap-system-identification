"""Known-component injection on archived validation flight centers and native clocks."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parents[2] / 'slow-fast-separation-validation-ab-v1'
sys.path.insert(0, str(PRIOR / 'experiments/slow_fast_separation_v1'))
from two_channel_followup import CHANNELS, DATA, native_runs, prepare_phase
from methods import weights, linear

ARCHIVE = PRIOR / 'docs/analysis/results/slow_fast_separation_ab_v1'
CASES = ['constant', 'varying_amplitude', 'varying_frequency', 'transient', 'combined']


def design(phi):
    return np.column_stack([f(k * phi) for k in (1, 2, 3) for f in (np.cos, np.sin)])


def evaluate(cycles, center, stride, case):
    # All operators below are linear in motion. Applying them to the known
    # increment is exactly the change between injected and original outputs.
    beats = []
    for c in cycles:
        t = c['t'][::stride]
        x = t - center
        phase = c['p'][::stride].copy()
        if case == 'varying_frequency':
            # Known local frequency modulation, monotone phase, same beat boundaries.
            phase += .42 * np.sin(phase + .4 * np.sin(x)) - .42 * np.sin(.4 * np.sin(x))
        w = weights(t, c['lo'], c['hi'])
        envelope = .5 + np.clip(x / .5, 0, 1) if case in ('varying_amplitude', 'combined') else 1
        f = envelope * (np.sin(phase) + .35*np.sin(2*phase+.4) + .2*np.cos(3*phase-.2))
        f -= w @ f
        if case == 'transient':
            f[:] = 0
        s = np.sin(np.pi * np.clip(x/.2, 0, 1))**2 if case in ('transient','combined') else np.zeros_like(x)
        beats.append((t, design(phase), w, s, f, c['hi']-c['lo']))
    results = []
    for radius, local in [(25,False),(5,False),(5,True),(2,False),(2,True)]:
        sums = np.zeros(7)
        for i in range(25,len(beats)-25):
            t,z,w,s,f,duration = beats[i]
            if t[-1]<center-2 or t[0]>center+2: continue
            span=beats[i-radius:i+radius+1]
            tau=np.array([q[2]@q[0] for q in span])
            means=np.array([q[2]@(q[3]+q[4]) for q in span])
            time=np.concatenate([q[0] for q in span])
            y=np.concatenate([q[3]+q[4] for q in span])-linear(time,tau,means)
            X=np.vstack([q[1] for q in span])
            if local:
                X=np.column_stack([X, X*(time-tau[radius])[:,None]])
            X=np.column_stack([np.ones(len(time)),X])
            a=np.concatenate([np.full(len(q[0]),1/np.sqrt(len(q[0]))) for q in span])
            beta=np.linalg.lstsq(X*a[:,None],y*a,rcond=None)[0]
            pred=z@beta[1:7]
            if local: pred+=(z@beta[7:])*(t-tau[radius])
            pred-=w@pred
            wt=w*duration*(abs(t-center)<=2)
            sums+=np.array([wt@((pred-f)**2),wt@(s*s+f*f),wt@(s*(s+f-pred)),wt@(s*s),wt.sum(), wt@(pred*f), wt@(f*f)])
        err,power,dot,sp,ws,fast_dot,fp=sums
        results.append(dict(beats=radius*2+1,local_linear=local,nrmse=np.sqrt(err/power),rmse=np.sqrt(err/ws),pulse_gain=dot/sp if sp else np.nan, waveform_gain=fast_dot/fp if fp else np.nan))
    return results


def main():
    selected=pd.read_csv(ARCHIVE/'B_selection.csv').query("split == 'validation'")
    meta={x['flight_id']:x for x in json.loads((DATA/'flight_provenance.json').read_text()) if x['split']=='validation'}
    pc=json.loads((PRIOR/'configs/pahbd/phase_reference.json').read_text())
    s=pd.read_parquet(DATA/'samples_validation.parquet',filters=[('flight_id','in',selected.flight_id.tolist())],
                      columns=['flight_id','segment_id','valid_target_state','excluded','review','firmware_verified','provenance_review','valid_core',*[v for a in CHANNELS.values() for v in a[:2]]])
    ph=pd.read_parquet(DATA/'native_wing_phase_validation.parquet')
    rows=[]
    for r in selected.itertuples():
        m=meta[r.flight_id]
        pt=prepare_phase(ph[ph.flight_id==r.flight_id],m['transmission_ratio'],pc)
        for ch in CHANNELS:
            _,runs=native_runs(s[s.flight_id==r.flight_id],pt,m,ch)
            run=runs[getattr(r,ch+'_run')]
            # retain full supports spanning +/- 2s and 25 neighbors.
            ids=[i for i,c in enumerate(run) if c['hi']>=r.event_time_s-2 and c['lo']<=r.event_time_s+2]
            run=run[min(ids)-25:max(ids)+26]
            for stride in (1,2):
                for case in CASES:
                    for stat in evaluate(run,r.event_time_s,stride,case):
                        rows.append(dict(flight=r.flight_id,channel=ch,stride=stride,case=case,**stat))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'per_flight.csv',index=False)
    summary=df.groupby(['channel','stride','case','beats','local_linear'])[['nrmse','pulse_gain','waveform_gain']].mean()
    summary.to_csv(OUT/'summary.csv')
    print(summary.to_string())


if __name__=='__main__': main()
