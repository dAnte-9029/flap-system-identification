"""Independent numerical acceptance checks for the completed train-only CRM run."""
import json
import hashlib
import numpy as np
import pandas as pd
from pacld_crm_v1 import OUT, CHECKPOINT, SHA, save

def main():
    d=pd.read_csv(OUT/'windows.csv'); audit=json.loads((OUT/'audit.json').read_text())
    assert len(d)==11221 and d.date.nunique()==17 and d.flight.nunique()==114
    assert ((d.query_s>0)&(d.query_s<.25)).all()
    assert not d.duplicated(['flight','t']).any()
    assert d.sort_values('t').groupby('flight').t.diff().dropna().min()>=1000000
    np.testing.assert_allclose(d.E_endpoint,d.E_increment+d.E_initial,atol=2e-14,rtol=0)
    np.testing.assert_allclose(d.moment_obs-d.moment_model,d.moment_normal,atol=2e-14,rtol=0)
    np.testing.assert_allclose(d.moment_endpoint,d.moment_normal+d.moment_initial,atol=2e-14,rtol=0)
    # Independent explicit hierarchy; invariance to window order and flight duplication.
    cols=[c for c in d if c.startswith('moment_')]
    def reduce(frame):
        return np.mean([np.mean([g[cols].mean().to_numpy() for _,g in day.groupby('flight')],axis=0) for _,day in frame.groupby('date')],axis=0)
    value=reduce(d);np.testing.assert_allclose(value,reduce(d.sample(frac=1,random_state=81)),rtol=0,atol=1e-14)
    one=d[d.flight==d.flight.iloc[0]];np.testing.assert_allclose(value,reduce(pd.concat([d,one])),rtol=0,atol=1e-14)
    target=json.loads((OUT/'moments.json').read_text());np.testing.assert_allclose(value,[target['moments'][c]['M'] for c in cols],atol=1e-14,rtol=0)
    for p in ['batch_statistics.csv','batch_stratified.csv']:
        b=pd.read_csv(OUT/p);np.testing.assert_allclose(b.M2_mean,b['mean']**2+b.sd**2,rtol=0,atol=1e-14)
    folds=pd.read_csv(OUT/'folds.csv');assert len(folds)==17 and (folds.train_dates==16).all() and (folds.evaluation_dates==1).all()
    g=pd.read_csv(OUT/'gradient_statistics.csv');assert g.finite.all() and (g.rms>=0).all()
    stop=g[(g.module=='belief')&g.objective.isin(['M_stop','M2_stop'])];assert len(stop)==8 and (stop.l2==0).all()
    for module in ['coupling','equilibrium','relaxation','slow','decoder']:
        assert (g[(g.module==module)&(g.objective=='M_stop')].l2>0).all()
    checks=json.loads((OUT/'gradient_checks.json').read_text())
    assert checks['parameters_unchanged'] and not checks['optimizer_created']
    assert max(x['factual_parity_max'] for x in checks['batches'])<2e-5
    assert max(x['autograd_M2_chain_max'] for x in checks['batches'])<2e-5
    for bid in range(4):
        v=np.load(OUT/f'gradient_vectors_batch{bid}.npz')
        np.testing.assert_allclose(v['M2'],2*v['moment_value']*v['M'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(v['M2_stop'],2*v['moment_value']*v['M_stop'],atol=1e-14,rtol=0)
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA
    for name in ['date_moments.png','negative_controls.png','batch_gradient_stability.png']:
        assert (OUT/name).stat().st_size>1000
    save('verification.json',dict(status='PASS',checks=['train cohort/unique key/spacing','three error and moment identities','explicit hierarchy and permutation/flight duplication invariance','M_batch square variance identity','17 leave-one-date-out folds','finite module gradients','stopped Belief zero gradient; future dynamics reachable','factual inference parity','M2 autograd chain','parameters and checkpoint unchanged','three figures'],validation_test_data_opened=False))
    print('PASS: CRM cohort, numerical identities, gradients, checkpoint, figures')

if __name__=='__main__':main()
