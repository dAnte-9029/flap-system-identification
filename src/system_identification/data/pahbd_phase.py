"""Hall circular reference admission, based only on sensor data and frozen boundaries."""
import numpy as np


def prepare_phase(table, ratio, contract):
    t=table.sort_values('timestamp_us').copy()
    ts=t.timestamp_us.to_numpy();p=t.phase_rad.to_numpy();c=t.encoder_total_count.to_numpy(dtype=float)
    dt=np.diff(ts,prepend=ts[0]);dc=np.diff(c,prepend=c[0])
    residual=np.angle(np.exp(1j*(np.diff(p,prepend=p[0])-dc*2*np.pi/(4096*ratio))))
    bad=(~t.phase_valid.astype(bool).to_numpy() | ~np.isfinite(p) | ~np.isfinite(c)
         | (p<0) | (p>=2*np.pi) | (dt<=0) | (dt>contract['max_gap_us']) | (dc<0)
         | ~np.isfinite(residual) | (dc*1e6/(4096*ratio)>contract['max_encoder_frequency_hz']*dt)
         | (np.abs(residual)>contract['max_reference_error_rad']))
    bad[0]=True
    last_bad=np.maximum.accumulate(np.where(bad,ts,ts[0]))
    t['phase_usable']=(ts-last_bad>=contract['recovery_us']) & t.in_reviewed_flight
    t['reference_error_rad']=residual
    t['phase_anomaly']=bad
    t['availability_us']=ts+contract['availability_delay_us']
    return t


def annotate_hall(windows,provenance,tables,starts,contract):
    out=windows.copy()
    out['shared_periodic_eligible']=False;out['phase_exclusion']='missing_phase'
    for fid,ids in out.groupby('flight_id',sort=False).groups.items():
        t=tables[fid];g=out.loc[ids];ts=t.timestamp_us.to_numpy();av=t.availability_us.to_numpy()
        good=t.phase_usable.to_numpy();badts=ts[~good]
        begin=starts[fid][g.start_sample.to_numpy(int)]
        # Initialize phase at the first reliable historical measurement, replayed
        # only once its guarded arrival precedes the decision. No older controls
        # outside the frozen support are needed for a pre-history phase seed.
        j=np.minimum(np.searchsorted(ts,begin,side='left'),len(ts)-1)
        bootstrap=good[j]&(av[j]<=g.decision_us.to_numpy())&(ts[j]-begin<=contract['max_gap_us'])
        a=np.searchsorted(badts,begin,side='left');b=np.searchsorted(badts,g.support_end_us,side='right')
        covered=(a==b)&(ts[-1]>=g.support_end_us.to_numpy()-contract['max_gap_us'])
        resets=np.asarray(provenance[fid]['reset_times_us'])
        reset=(np.searchsorted(resets,g.support_end_us,side='right')-np.searchsorted(resets,g.support_start_us,side='left'))>0
        ok=bootstrap&covered&g.complete_phase_frequency.to_numpy()&~reset
        out.loc[ids,'shared_periodic_eligible']=ok
        out.loc[ids,'phase_exclusion']=np.select([~bootstrap,~covered,reset,~g.complete_phase_frequency.to_numpy()],['no_causal_phase_bootstrap','phase_anomaly_or_recovery','frozen_reset','incomplete_frequency'],default='eligible')
    out['phase_mode']=np.where(out.shared_periodic_eligible,'shared_hall','average_only')
    out['phase_reference_id']=contract['reference_id']
    return out
