"""Flight-paired descriptive differences for the single frozen comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

study=Path('/home/zn/flap-system-identification/research/experiments/study-6a5d1c33c47e')
out=Path(__file__).resolve().parent
dirs=[study/s/'attempt-1/seed-17/output' for s in ('baseline','exp-fa28c954ce7a')]
def diffs(a,b,kind):
    a,b=a.align(b,join='inner',axis=0)
    assert len(a)==len(b) and len(a)>0
    idx=np.random.default_rng(17).integers(len(a),size=(10000,len(a)))
    def stat(df):
        if kind=='rmse':
            values=df.to_numpy().ravel()
            return np.sqrt(values.mean()),np.sqrt(values[idx].mean(1)),np.sqrt(values)
        cov,p2,r2=df[['dot','p2','r2']].to_numpy().T
        return (cov.mean()/np.sqrt(p2.mean()*r2.mean()),cov[idx].mean(1)/np.sqrt(p2[idx].mean(1)*r2[idx].mean(1)),cov/np.sqrt(p2*r2))
    x,xx,fx=stat(a);y,yy,fy=stat(b)
    return dict(reference=x,candidate=y,candidate_minus_reference=y-x,interval95=np.quantile(yy-xx,[.025,.975]).tolist(),
                improved=int((fy<fx if kind=='rmse' else fy>fx).sum()),flights=len(a))
ends=[pd.read_csv(d/'endpoints_by_flight.csv').set_index('flight') for d in dirs]
beats=[pd.read_csv(d/'beats_by_flight.csv').set_index(['group','flight']) for d in dirs]
r={k:diffs(ends[0][k],ends[1][k],'rmse') for k in ends[0]}
for key in ['full_all','full_high','retrospective_rate','retrospective_speed']:
    a,b=[f.loc[key] for f in beats]
    r[key]=dict(versus_AC=diffs(a.error,b.error,'rmse'),versus_persistence=diffs(b.r2,b.error,'rmse'),
                direction=diffs(a,b,'direction'))
for key in ['all','high']:
    b=beats[1].loc['slow_readout_'+key]
    r['slow_readout_'+key]=diffs(b.r2,b.error,'rmse')
(out/'paired.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
