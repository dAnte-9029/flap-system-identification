"""Independently check every retained observation against its authorized raw ULog."""
import numpy as np
import pandas as pd
from pyulog import ULog
from audit_replay_preflight_v37 import OUT,TOPICS,read
import train_control_windows_v2 as base


def main():
    records=read(OUT/'origin_observations.json');manifest=read(OUT/'dataset_manifest.json');verified=0
    for entry in manifest['sources']:
        source=entry['source'];assert base.sha(source['path'])==source['sha256']
        u=ULog(source['path'],message_name_filter_list=TOPICS)
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        for r in records:
            if r['log_id']!=entry['log_id']:continue
            for name,obs in r['topics'].items():
                d=ds[name];t=np.asarray(d['timestamp'],np.int64);sample=np.asarray(d.get('timestamp_sample',t),np.int64)
                ix=np.flatnonzero((t<=r['origin_us'])&(sample<=r['origin_us']))
                if not len(ix):assert not obs['available'];continue
                idx=int(ix[np.argmax(t[ix])]);assert idx==obs['source_index']
                assert obs['publication_us']==t[idx] and obs['sample_us']==sample[idx]
                for field,value in obs['values'].items():
                    raw=d[field][idx]
                    if value is None:assert not np.isfinite(raw) and field in obs['nonfinite_fields']
                    else:assert value==raw
                verified+=1
    c=pd.read_csv(OUT/'conditional_clock_estimates.csv')
    pars={p['log_id']:p['values'] for p in read(OUT/'parameters_by_flight.json')}
    for (log,axis),g in c.groupby(['log_id','axis']):
        k={'roll':'R','pitch':'P','yaw':'Y'}[axis];den=pars[log][f'FW_{k}R_I']*g.error*np.maximum(0,1-(g.error/np.deg2rad(400))**2)
        dt=(g.post_integral-g.pre_integral)/den
        np.testing.assert_allclose(dt,g.dt_s,rtol=1e-8,atol=1e-12)
    p=c.pivot(index=['log_id','terms_publication_us'],columns='axis',values='dt_s')
    multiple=p[p.notna().sum(1)>=2];spread=multiple.max(1)-multiple.min(1)
    base.write(OUT/'independent_verification.json',dict(status='passed',snapshots=407,topic_observations_verified=verified,
        nonfinite_values_preserved_as_null=True,clock_estimates_recomputed=len(c),
        multiaxis_clock_rows=len(multiple),multiaxis_dt_spread_median_us=float(spread.median()*1e6),
        multiaxis_dt_spread_p95_us=float(spread.quantile(.95)*1e6),
        exact_controller_clock_not_claimed=True))
    print(f'Raw snapshot verification passed: {verified} observations; {len(c)} conditional dt estimates.',flush=True)


if __name__=='__main__':main()
