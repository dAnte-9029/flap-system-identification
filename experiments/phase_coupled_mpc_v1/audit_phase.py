"""Read-only train/validation mechanical phase audit; no test parquet access."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
OUT = Path(__file__).resolve().parent


def quant(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return dict(zip(['p01', 'p50', 'p99'], map(float, np.quantile(x, [.01, .5, .99])))) if len(x) else {}


def run():
    meta = {x['flight_id']: x for x in json.loads((ROOT/'flight_provenance.json').read_text())
            if x['split'] in ('train', 'validation')}
    rows, events = [], []
    for split in ('train', 'validation'):
        tables = {t: pd.read_parquet(ROOT/f'native_{t}_{split}.parquet')
                  for t in ('wing_phase', 'hall_event', 'encoder_count')}
        groups = {t: dict(tuple(df.groupby('flight_id', sort=False))) for t, df in tables.items()}
        for fid, m in meta.items():
            if m['split'] != split: continue
            w = groups['wing_phase'][fid].sort_values('timestamp_us')
            w = w[w.in_reviewed_flight]
            v = w[w.phase_valid.astype(bool) & np.isfinite(w.phase_unwrapped_rad)]
            e = groups['encoder_count'][fid].sort_values('timestamp_us')
            h = groups['hall_event'].get(fid, pd.DataFrame())
            ratio = m['transmission_ratio']; scale = 4096*ratio/(2*np.pi)
            row = dict(flight_id=fid, split=split, date=m['flight_date'], log_path=m['log_path'],
                       firmware=m['firmware'], hardware=m['sys_uuid'], sampling_group=m['sampling_group'],
                       wing_rows=len(w), valid_rows=len(v), hall_rows=len(h), configured_ratio=ratio)
            c=v.encoder_total_count.to_numpy(); p=v.phase_unwrapped_rad.to_numpy(); t=v.timestamp_us.to_numpy()
            zero=c-p*scale
            reset=np.flatnonzero(np.abs(np.diff(zero))>1000)+1
            row['reset_count']=len(reset)
            row['relative_vs_hall_resultant']=float(abs(np.mean(np.exp(1j*(c/scale-p))))) if len(v) else None
            delta=(c-m['phase_origin_count'])/scale-p
            row['relative_origin_minus_hall_rad']=float(np.angle(np.mean(np.exp(1j*delta)))) if len(v) else None
            for harmonic in (2,3):
                row[f'relative_vs_hall_resultant_h{harmonic}']=float(abs(np.mean(np.exp(1j*harmonic*delta)))) if len(v) else None
            row['reset_ratio_median']=float(np.median(np.diff(zero[reset])/4096)) if len(reset)>1 else None
            for i in reset:
                events.append(dict(flight_id=fid, timestamp_us=int(t[i]),
                                   reset_correction_rad=float(-(zero[i]-zero[i-1])/scale+2*np.pi),
                                   count_at_hall_estimate=float(zero[i])))
            if len(h)>1:
                ht=h.timestamp_us.to_numpy(); ec=e.total_count.to_numpy(); et=e.timestamp_us.to_numpy()
                j=np.searchsorted(et,ht); ok=(j>0)&(j<len(et))
                jj=np.clip(j,1,len(et)-1)
                ok &= (et[jj]-et[jj-1])<=30000
                hc=np.interp(ht,et,ec); dh=np.diff(h.pulse_count.to_numpy().astype(np.int64))
                good=ok[1:]&ok[:-1]&(dh==1)&(np.diff(ht)>100000)&(np.diff(ht)<1000000)
                ratios=np.diff(hc)[good]/4096
                row['hall_ratio_pairs']=int(good.sum())
                row.update({'hall_ratio_'+k:x for k,x in quant(ratios).items()})
                row['hall_interval_median_s']=float(np.median(np.diff(ht)[good])*1e-6) if good.any() else None
            rows.append(row)
        print('completed',split,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'phase_per_flight.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'phase_resets.csv.gz',index=False)
    summary={}
    for s,g in df.groupby('split'):
        summary[s]=dict(flights=len(g),valid_phase_flights=int((g.valid_rows>0).sum()),
                        hall_flights=int((g.hall_rows>0).sum()),
                        valid_phase_rows=int(g.valid_rows.sum()), wing_rows=int(g.wing_rows.sum()),
                        reset_count=int(g.reset_count.sum()),
                        relative_vs_hall_resultant=quant(g.relative_vs_hall_resultant),
                        hall_ratio_per_flight_median=quant(g.hall_ratio_p50),
                        reset_ratio_per_flight_median=quant(g.reset_ratio_median))
    summary['dataset_manifest_sha256']=hashlib.sha256((ROOT/'dataset_manifest.json').read_bytes()).hexdigest()
    summary['method']='Hall count interpolation is offline source validation, never deployable future input; no IMU alignment.'
    (OUT/'phase_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': run()
