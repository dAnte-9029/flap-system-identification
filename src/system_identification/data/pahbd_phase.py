"""Interval admission from frozen contracts and existing diagnostics, never model error.

No shared index has been certified in the current evidence. Hall existence and high
within-log resultant are intentionally NOT a certificate of a shared reference.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_EVIDENCE=Path(__file__).resolve().parents[3]/'experiments/phase_coupled_mpc_v1'


def annotate_phase(windows,provenance,evidence=DEFAULT_EVIDENCE,hall_events=None):
    diagnostics=pd.read_csv(Path(evidence)/'phase_per_flight.csv').set_index('flight_id')
    resets=pd.read_csv(Path(evidence)/'phase_resets.csv.gz')
    out=windows.copy()
    out['phase_mode']='average_only'
    out['shared_periodic_eligible']=False
    out['mechanical_zero_verified']=False
    out['sensor_index_shared_verified']=False
    out['hall_event_traceable']=False
    out['hall_log_recorded']=False
    out['hall_publication_verified']=False
    out['reference_change_max_rad']=0.
    out['reference_updates']=0
    out['relative_phase_eligible']=False
    # No new empirical cutoffs: inherit complete phase/frequency and reviewed
    # segment rules, and count ALL recorded software-reference changes.
    for fid,ids in out.groupby('flight_id',sort=False).groups.items():
        if fid not in diagnostics.index:continue
        d=diagnostics.loc[fid];g=out.loc[ids]
        events=resets[resets.flight_id==fid].sort_values('timestamp_us')
        rt=events.timestamp_us.to_numpy()
        left=np.searchsorted(rt,g.support_start_us,side='left');right=np.searchsorted(rt,g.support_end_us,side='right')
        counts=right-left
        correction=np.abs(np.angle(np.exp(1j*events.reset_correction_rad.to_numpy())))
        out.loc[ids,'reference_change_max_rad']=[float(correction[a:b].max()) if b>a else 0. for a,b in zip(left,right)]
        out.loc[ids,'reference_updates']=counts
        out.loc[ids,'hall_log_recorded']=bool(d.hall_rows>0)
        if hall_events is not None:
            ht=np.sort(hall_events.loc[hall_events.flight_id==fid,'timestamp_us'].to_numpy())
            out.loc[ids,'hall_event_traceable']=(np.searchsorted(ht,g.support_end_us,side='right')-np.searchsorted(ht,g.support_start_us,side='left'))>0
        # Frozen reset_times include EKF resets and backwards encoder counts, unlike software
        # Hall reference updates in phase_resets.csv.gz.
        bad=np.sort(provenance[fid]['reset_times_us'])
        crossings=np.searchsorted(bad,g.support_end_us,side='right')-np.searchsorted(bad,g.support_start_us,side='left')
        relative=g.complete_phase_frequency.to_numpy() & (crossings==0) & ~g.review.to_numpy()
        out.loc[ids,'relative_phase_eligible']=relative
        out.loc[ids,'phase_mode']=np.where(relative,'log_relative','average_only')
    out['software_reference_unchanged']=out.reference_updates==0
    out['phase_reference_id']=out.flight_id.str.split(':').str[0]  # unknown independent log origins
    return out


def apply_shared_reference(frame,provenance,contract):
    """Explicit evidence-backed certificates, never inferred from phase concentration.

    Each count anchor must already be published before the entire support interval.
    Empty certificates are the current conservative contract. Certificates are
    supplied by a physical/source verification task, not fitted by this loader.
    """
    import json
    import math
    obj=json.loads(Path(contract).read_text())
    if obj['schema']!='pahbd_phase_reference_v1':raise ValueError('phase certificate schema')
    out=frame.copy();out['log_reference_id']=out.phase_reference_id
    out['reference_count']=np.nan
    for item in obj['certified_intervals']:
        if not all(item.get(k) is True for k in ('sensor_index_shared_verified','causal_publication_verified','continuity_verified')):
            raise ValueError('incomplete shared index certificate')
        if not item.get('evidence') or not item.get('reference_id') or not math.isfinite(item['reference_count']):
            raise ValueError('missing phase reference evidence/count')
        if not item['reference_measurement_us']<=item['reference_availability_us']<=item['start_us']<item['end_us']:
            raise ValueError('noncausal or reversed reference interval')
        fid=item['flight_id']
        if fid not in provenance:continue  # certificates may contain the other allowed split
        meta=provenance[fid]
        if not meta['start_us']<=item['start_us']<item['end_us']<=meta['end_us']:
            raise ValueError('certificate outside reviewed flight')
        take=(out.flight_id==fid)&out.relative_phase_eligible&(out.support_start_us>=item['start_us'])&(out.support_end_us<=item['end_us'])
        if out.loc[take,'shared_periodic_eligible'].any():raise ValueError('ambiguous overlapping certificates')
        out.loc[take,'shared_periodic_eligible']=True
        out.loc[take,'sensor_index_shared_verified']=True
        out.loc[take,'hall_publication_verified']=True
        out.loc[take,'mechanical_zero_verified']=bool(item['mechanical_zero_verified'])
        out.loc[take,'reference_count']=float(item['reference_count'])
        out.loc[take,'phase_reference_id']=item['reference_id']
        out.loc[take,'phase_mode']='shared_sensor_index'
    return out
