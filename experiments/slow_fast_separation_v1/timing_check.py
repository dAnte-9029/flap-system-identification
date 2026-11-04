"""Native observation offsets; only explicitly allowed partitions."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
rows=[]
for split in ('train','validation'):
    s=pd.read_parquet(DATA/f'samples_{split}.parquet')
    for group,g in s.groupby('sampling_group'):
        for signal,col in [('rate','target_rate_measurement_us'),('attitude','target_attitude_measurement_us')]:
            mask=g.valid_target_state & ~g.review & ~g.excluded & g.firmware_verified & ~g.provenance_review
            offset=(g.loc[mask,col]-g.loc[mask,'state_measurement_us']).to_numpy()/1000
            rows.append(dict(split=split,sampling_group=group,signal=signal,count=len(offset),abs_offset_median_ms=np.median(abs(offset)),abs_offset_p95_ms=np.percentile(abs(offset),95),abs_offset_max_ms=max(abs(offset)),exact_fraction=np.mean(offset==0)))
out=ROOT/'docs/analysis/results/slow_fast_separation_v1/native_timing.csv'
pd.DataFrame(rows).to_csv(out,index=False)
print(pd.DataFrame(rows).to_string(index=False))
