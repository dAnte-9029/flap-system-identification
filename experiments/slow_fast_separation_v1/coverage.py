"""Read only train/validation coverage; no training or test loading."""
import sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from system_identification.data.pahbd_phase import prepare_phase
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
def main():
    contract=json.loads((ROOT/'configs/pahbd/phase_reference.json').read_text())
    meta={x['flight_id']:x for x in json.loads((DATA/'flight_provenance.json').read_text()) if x['split'] in ('train','validation')}
    rows=[];started=time.monotonic()
    for split in ('train','validation'):
        samples=pd.read_parquet(DATA/f'samples_{split}.parquet')
        phase=pd.read_parquet(DATA/f'native_wing_phase_{split}.parquet')
        for fid,s in samples.groupby('flight_id',sort=False):
            m=meta[fid];p=prepare_phase(phase[phase.flight_id==fid],m['transmission_ratio'],contract)
            t=p.timestamp_us.to_numpy();a=p.phase_rad.to_numpy();good=p.phase_usable.to_numpy()
            # Circular crossings bracket a full Hall cycle only within contiguous QC runs.
            cross=np.flatnonzero((np.diff(a)<-np.pi)&good[1:]&good[:-1])+1
            boundaries=t[cross-1]+(2*np.pi-a[cross-1])/(a[cross]+2*np.pi-a[cross-1])*(t[cross]-t[cross-1])
            st=s.state_measurement_us.to_numpy();valid=(s.valid_core&~s.review&~s.excluded&s.firmware_verified&~s.provenance_review).to_numpy()
            durations=[];counts=[];runs=[];run=0
            for k in range(len(boundaries)-1):
                lo,hi=boundaries[k:k+2];mask=(st>=lo)&(st<hi);indices=np.flatnonzero(mask)
                ok=good[cross[k]-1:cross[k+1]+1].all() and len(indices)>=2 and valid[mask].all() and m['sampling_group']=='main_50hz'
                if len(indices):
                    q=s.iloc[indices];ok=ok and q.segment_id.nunique()==1 and np.max(np.diff(np.r_[lo,st[mask],hi]))<=40000
                if ok:durations.append((hi-lo)*1e-6);counts.append(len(indices));run+=1
                else:runs.append(run);run=0
            runs.append(run)
            rows.append(dict(flight_id=fid,split=split,date=m['flight_date'],sampling_group=m['sampling_group'],samples=len(s),valid_samples=int(valid.sum()),hall_usable_fraction=float(good.mean()),candidate_cycles=max(0,len(boundaries)-1),valid_cycles=len(durations),max_consecutive_cycles=max(runs),windows_51=sum(max(0,n-50) for n in runs),valid_duration_s=sum(durations),median_frequency_hz=float(1/np.median(durations)) if durations else np.nan,median_observations_per_cycle=float(np.median(counts)) if counts else np.nan))
            done=len(rows);elapsed=time.monotonic()-started
            print(f'coverage {done}/{len(meta)} elapsed={elapsed:.1f}s ETA={elapsed/done*(len(meta)-done):.1f}s',flush=True)
    out=ROOT/'docs/analysis/results/slow_fast_separation_v1/coverage_preliminary.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).groupby(['split','sampling_group'])[['valid_cycles','windows_51','valid_duration_s']].sum())
if __name__=='__main__':main()
