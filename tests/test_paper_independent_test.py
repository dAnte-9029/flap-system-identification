"""Pre-opening synthetic/validation-only registration and statistical contracts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import pytest
from paper_independent_test_core import aggregate,paired,endpoint_adapter,METRICS,KEYS
from system_identification.evaluation.future_control_diagnostic import assign_groups,excitation,hold_controls,interval_mse
from run_paper_history_length_ablation import truncate_history
from build_expanded_september import windows


def test_frozen_windows_require_longest_context_and_same_four_horizons():
    samples=pd.DataFrame(dict(valid_core=[True]*176,log_id=['flight']*176,segment_id=[0]*176,sample_in_segment=np.arange(176)))
    w=windows(samples,'Sep8',50)
    assert list(w.start_sample_in_segment)==[25,75,125]
    assert (w.state_sample_count==51).all()
    assert all(x.startswith('Sep8:') for x in w.window_id)
    samples.loc[30,'sample_in_segment']=400
    with pytest.raises(ValueError):windows(samples,'Sep8',50)


def test_train_threshold_boundaries_and_no_test_quantile_refit():
    np.testing.assert_array_equal(assign_groups(np.array([0,.1,.2,.3]),.1,.2),['low','low','middle','high'])
    assert set(assign_groups(np.zeros(7),.1,.2))=={'low'}
    assert set(assign_groups(np.zeros(7),.1,.1))=={'unavailable'}
    controls=np.zeros((3,50,4));controls[1,1:]=1
    e=excitation(controls,np.ones(4),5)
    assert e[0]==e[2]==0 and e[1]==np.sqrt(4/5)


def test_date_flight_macro_sample_sd_and_pair_signs():
    rows=[]
    for date in ['Sep8','Sep19']:
        for model in ['B0','MLP','H1','H13','H26']:
            for seed in ([None] if model=='B0' else [17,23,42]):
                for condition in (['Actual','Hold'] if model=='H26' else ['Actual']):
                    for flight,n,base in [('a',1,2),('b',99,8)]:
                        error=base+(0 if seed is None else {17:0,23:1,42:2}[seed])+(1 if model=='MLP' or condition=='Hold' else 0)
                        for metric in METRICS:rows.append(dict(date=date,model=model,history_steps=26 if model=='H26' else 1,condition=condition,
                            kind='endpoint',group='ALL',step=25,horizon_s=.5,metric=metric,seed=seed,flight_id=date+flight,n_origins=n,value=error))
    per=pd.DataFrame(rows);seed,multi=aggregate(per)
    h=multi[(multi.model=='H26')&(multi.condition=='Actual')].iloc[0]
    assert h['mean']==6 and h['std']==1 and h.n_origins==100
    assert set(multi.date)=={'Sep8','Sep19'} and multi[multi.model=='B0']['std'].isna().all()
    ps,pf,g=paired(seed,per)
    assert set(g.query('comparison=="GRU_vs_MLP"').relative_gain_pct)=={100/7}
    assert set(g.query('comparison=="GRU_vs_MLP"').flights_improved)=={2}
    assert set(g.query('comparison=="Actual_vs_Hold"').seeds_improved)=={3}
    missing=per[~((per.model=='MLP')&(per.seed==42))]
    _,m=aggregate(missing)
    assert m[m.model=='MLP']['mean'].isna().all() and (m[m.model=='MLP'].status=='incomplete_seeds').all()


def test_short_history_exact_suffix_and_same_future():
    from test_paper_baselines import make_batch
    from system_identification.training.trajectory_main_v1 import HistoryTrajectoryWindowBatch
    rng=np.random.default_rng(19);t=make_batch()
    b=HistoryTrajectoryWindowBatch(t,rng.normal(size=(2,26,12)),rng.normal(size=(2,26,4)),np.ones((2,26),bool))
    for h in [1,5,13,26]:
        short=truncate_history(b,h)
        assert short.trajectory is b.trajectory
        np.testing.assert_array_equal(short.history_state_features,b.history_state_features[:,-h:])
        np.testing.assert_array_equal(short.history_controls,b.history_controls[:,-h:])


def test_adapter_uses_explicit_partition_and_matches_original_numeric_errors():
    from test_paper_baselines import make_batch
    from types import SimpleNamespace
    from system_identification.models.trajectory import ConstantTwistPredictor
    from system_identification.evaluation.paper_baselines import endpoint_metrics
    b=make_batch();b.log_ids=np.array(['2026.9.7/a','2026.9.7/b']);b.controls=np.zeros((2,50,4))
    from system_identification.models.trajectory import InitialTrajectoryState
    state=InitialTrajectoryState(**{k:v[:,0].copy() for k,v in vars(b.truth).items()})
    pred=ConstantTwistPredictor().rollout(state,b.controls,b.dt_s)
    orig=endpoint_metrics(pred,b,model='B0')
    frame,_=endpoint_adapter(pred,SimpleNamespace(trajectory=b),model='B0',seed=None,date='Sep8',allowlist=b.log_ids)
    cols=['position_error_m','velocity_error_m_s','attitude_error_deg','body_rate_error_rad_s']
    np.testing.assert_array_equal(frame[cols],orig[cols])
    assert set(frame.split)=={'Sep8'}
    with pytest.raises(ValueError):endpoint_adapter(pred,SimpleNamespace(trajectory=b),model='B0',seed=None,date='Sep8',allowlist=['different'])
