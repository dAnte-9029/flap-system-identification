import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_condition_diagnostics_v12 import covariates,weighted_terciles,bins,waveform,aggregate,repeatability


def test_conditions_ignore_all_future_inputs():
    rng=np.random.default_rng(12);x=dict(history_state=rng.normal(size=(6,26,14)),history_reference=rng.normal(size=(6,26,5)),history_actuator=rng.normal(size=(6,26,4)),future_reference=np.zeros((6,50,5)))
    a=covariates(x,0);x['future_reference']+=1e6
    pd.testing.assert_frame_equal(a,covariates(x,0))
    assert list(a.columns)==['body_speed','frequency_mean','frequency_std','rate_std','rate_abs_mean','reference_std','reference_step_rms','actuator_step_rms']


def test_flight_balanced_quantiles_and_ties():
    a=weighted_terciles(np.array([0,10,20]),np.array(['a','b','c']))
    b=weighted_terciles(np.array([0]*9+[10,20]),np.array(['a']*9+['b','c']))
    np.testing.assert_array_equal(a,b)
    assert set(bins(np.zeros(10),[0,0]))=={'low'}
    with pytest.raises(ValueError):weighted_terciles([np.nan],['a'])


def test_waveform_decomposition_and_mean_invariance():
    t=np.sin(np.arange(50))[None,:];p=.5*t+2
    d=waveform(p,t,t)
    np.testing.assert_allclose(d['total_mse'],d['bias_mse']+d['amplitude_mse']+d['shape_mse'])
    np.testing.assert_allclose(d['correlation'],1)
    out=aggregate(pd.DataFrame(d));assert np.isnan(out.reference_utility)
    assert aggregate(pd.DataFrame(waveform(p,t,t+.1))).reference_utility<0
    np.testing.assert_allclose(out.normalized_centered,.5)
    z=waveform(np.ones_like(t),t,t);assert np.isnan(z['correlation']).all()


def test_repeatability_requires_flight_support_and_all_seeds():
    rows=[]
    for seed in [17,23,42]:
        for log in range(5):
            for stratum in ['low','high']:
                rows.append(dict(axis='p',condition='rate_std',cohort='oof',horizon_s=1.,seed=seed,log_id=str(log),stratum=stratum,
                    n=10,normalized_centered=1 if stratum=='low' else 1.4,rmse=1 if stratum=='low' else 2.,mean_correlation=.8 if stratum=='low' else .4))
    a=pd.DataFrame(rows);result,_=repeatability(a);assert result.repeatable_screen.iloc[0]
    a.loc[(a.seed==42)&(a.stratum=='high'),'normalized_centered']=.5
    result,_=repeatability(a);assert not result.repeatable_screen.iloc[0]
    a.loc[a.log_id.isin(['0','1']),'n']=2
    result,_=repeatability(a);assert not result.supported.iloc[0]
