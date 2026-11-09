"""Targeted tests: train-only local regression, causal ZOH, residual score."""
import numpy as np
from diagnose_pacld_a2_v2 import zoh_mean, crossflight_pairs, local_fit
from diagnose_pacld_a2 import ScaledRidge

from score_pacld_a2_v2 import response_score
from validation_gain_statistics_a2_v2 import ridge_statistics

def run():
    assert np.allclose(zoh_mean(np.array([0,50000,100000]),np.array([[0.],[1.],[0.]]),0,100000),[.5])
    rng=np.random.default_rng(17);n=900;z=rng.normal(size=(n,3));u=rng.normal(size=(n,4));A=rng.normal(size=(2,3));G=np.array([[0,3,0,.2],[0,.4,0,-2.]])
    y=z@A.T+u@G.T;flight=np.repeat(np.arange(9),100);train=np.arange(800);query=np.arange(800,900)
    pairs=crossflight_pairs(z,flight,train)
    assert not np.isin(pairs,query).any() and (flight[pairs[:,0]]!=flight[pairs[:,1]]).all()
    yt=y.copy();yt[query]=np.nan
    m,_,info,gain,_=local_fit(z,u,yt,pairs,z[query],1.,np.ones(4))
    assert info['identified'] and np.allclose(gain,G,atol=.1)
    poisoned=yt.copy();poisoned[query]=1e9
    _,_,_,g2,_=local_fit(z,u,poisoned,pairs,z[query],1.,np.ones(4))
    assert np.array_equal(gain,g2)
    baseline=ScaledRidge(1).fit(z[train],y[train]);res=y[query]-baseline.predict(z[query])
    assert np.allclose(response_score(res,np.zeros_like(res)),0)
    assert np.allclose(response_score(res,res),1)
    assert np.isfinite(m.predict(np.c_[z[query],u[query]])).all()
    grams=np.stack([u[k:k+100].T@u[k:k+100] for k in range(0,400,100)])
    crosses=np.stack([u[k:k+100].T@y[k:k+100] for k in range(0,400,100)])
    b=ridge_statistics(grams,crosses,np.ones((1,4)))[0]
    assert np.allclose(b,np.linalg.solve(u[:400].T@u[:400]+10*np.eye(4),u[:400].T@y[:400]))
    print('PASS date sufficient-statistic regression,  ZOH integration, cross-flight train-only pairs, synthetic G recovery, held-out-label poisoning invariance, zero/perfect response scores')
if __name__=='__main__':run()

# When execution artifacts exist, independently recompute the published date-balanced score.
from pathlib import Path
import pandas as pd
from diagnose_pacld_a2_v2 import OUT
if (OUT/'validation_summary.csv').exists():
    e=pd.read_csv(OUT/'validation_flight_scores.csv')
    s=pd.read_csv(OUT/'validation_summary.csv')
    for axis in 'pr':
        a=e[e.axis==axis].groupby('date')[['zero_response_mse','M1_response_mse','PACLD_response_mse']].mean()
        assert len(a)==4
        for model in ['M1','PACLD']:
            actual=1-a[model+'_response_mse'].mean()/a.zero_response_mse.mean()
            expected=s[(s.axis==axis)&(s.model==model)].response_score.iloc[0]
            assert abs(actual-expected)<1e-12
    train=pd.read_csv(OUT/'pacld_train.csv');val=pd.read_csv(OUT/'pacld_validation.csv')
    assert not set(train.flight)&set(val.flight) and not set(train.date)&set(val.date)
    assert len(train)==11221 and len(val)==3657
    assert train.flight.nunique()==114 and val.flight.nunique()==39
    assert not train.duplicated(['flight','t']).any() and not val.duplicated(['flight','t']).any()
    print('PASS actual score aggregation, completed factual coverage, split/date disjointness, unique decisions')
