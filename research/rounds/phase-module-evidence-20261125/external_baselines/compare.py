"""Whole-flight paired comparisons using only the common seed-17 development set."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
HERE=Path(__file__).resolve().parent
KEYS=('v05','w05','v1','w1','v2','w2')


def pair(a,b,indices):
    def score(x):
        return np.sqrt(x.mean(axis=-2))
    pa,pb=score(a),score(b)
    draws=score(a[indices])-score(b[indices])
    return {k:dict(gain=float((pa-pb)[j]),interval95=np.quantile(draws[:,j],[.025,.975]).tolist(),
                   improved_flights=int((b[:,j]<a[:,j]).sum())) for j,k in enumerate(KEYS)}


def main():
    out={}
    ids=None
    for name in ('archived','best','terminal'):
        s=json.loads((HERE/f'{name}_flight_mse.json').read_text())
        if ids is None:
            ids=sorted(s['v2'])
        assert all(sorted(s[k])==ids for k in KEYS)
        out['tcn_'+name]=np.array([[s[k][f] for k in KEYS] for f in ids])
    for src,arm,title in [('waveform','corrected','engineering_corrected'),('waveform','full','waveform_full'),
                           ('waveform/coherence','tapered_round1','waveform_tapered')]:
        f=pd.read_csv(HERE.parent/src/'flight_mse.csv')
        f=f[(f.seed==17)&(f.arm==arm)].set_index('flight')
        assert set(f.index)==set(ids) and len(f)==39
        out[title]=f.loc[ids,list(KEYS)].to_numpy()
    metrics={arm:dict(zip(KEYS,np.sqrt(a.mean(0)).tolist())) for arm,a in out.items()}
    pairs=[('tcn_archived','tcn_best'),('tcn_terminal','tcn_best'),
           ('tcn_best','engineering_corrected'),('tcn_best','waveform_full'),('tcn_best','waveform_tapered')]
    ix=np.random.default_rng(17).integers(0,len(ids),(10000,len(ids)))
    comp={arm+'_vs_'+ref:pair(out[ref],out[arm],ix) for ref,arm in pairs}
    (HERE/'comparison.json').write_text(json.dumps(dict(seed=17,metrics=metrics,pairs=comp),indent=2)+'\n')


if __name__=='__main__':
    main()
