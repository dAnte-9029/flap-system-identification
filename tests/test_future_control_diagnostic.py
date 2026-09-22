"""Information-control contract, native-time errors, fair paired statistics."""
from dataclasses import dataclass
from types import SimpleNamespace
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from system_identification.evaluation.future_control_diagnostic import (
    hold_controls,excitation,train_thresholds,assign_groups,squared_errors,interval_mse,
    flight_metrics,aggregate,paired_comparisons,gains,METRICS,
)
from system_identification.training.trajectory_main_v1 import HistoryTrajectoryWindowBatch
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from test_paper_baselines import inputs,stats,make_batch


@dataclass(frozen=True)
class Tape:
    controls: np.ndarray
    dt_s: np.ndarray
    truth: object


def test_hold_exact_origin_and_no_mutation():
    rng=np.random.default_rng(8);u=rng.normal(size=(2,50,4))
    batch=HistoryTrajectoryWindowBatch(Tape(u,np.ones((2,50))*.02,object()),rng.normal(size=(2,26,12)),rng.normal(size=(2,26,4)),np.ones((2,26),bool))
    batch.history_controls[:,-1]=u[:,0];old=u.copy()
    held=hold_controls(batch)
    np.testing.assert_array_equal(held.trajectory.controls,np.repeat(u[:,:1],50,axis=1))
    np.testing.assert_array_equal(held.trajectory.controls[:,0],old[:,0])
    np.testing.assert_array_equal(u,old)
    assert held.history_state_features is batch.history_state_features
    assert held.history_controls is batch.history_controls
    assert held.history_mask is batch.history_mask
    assert held.trajectory.truth is batch.trajectory.truth
    assert held.trajectory.dt_s is batch.trajectory.dt_s
    assert not np.shares_memory(held.trajectory.controls,u)


def test_first_transition_and_constant_control_full_prediction():
    torch.manual_seed(17);model=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True,**stats())
    torch.nn.init.normal_(model.derivative_head[-1].weight,std=.02)
    x=inputs();x['history_controls'][:,-1]=x['future_controls'][:,0]
    y=copy.deepcopy(x);y['future_controls']=x['future_controls'][:,:1].expand(-1,50,-1).clone()
    state={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():a=model(**x);b=model(**y);c=model(**y)
    for p,q,r in zip(a,b,c):
        assert torch.equal(p[:,:2],q[:,:2])
        assert torch.equal(q,r)
    assert not torch.equal(a.velocity_n[:,2:],b.velocity_n[:,2:])
    for k,v in model.state_dict().items():assert torch.equal(v,state[k])


def test_excitation_train_quantiles_ties_and_no_error_input():
    u=np.zeros((4,5,4));u[:,1:,0]=np.arange(4)[:,None]
    result=excitation(u,np.array([2.,1,1,1]),5)
    np.testing.assert_allclose(result,np.arange(4)/np.sqrt(20))
    lo,hi,ok=train_thresholds(result);assert ok
    np.testing.assert_array_equal(assign_groups(np.array([lo,hi,hi+.01]),lo,hi),['low','middle','high'])
    assert train_thresholds(np.zeros(10))==(0.,0.,False)
    assert set(assign_groups(np.ones(4),0.,0.))=={'unavailable'}
    with pytest.raises(ValueError):excitation(u,np.zeros(4),5)
    np.testing.assert_array_equal(excitation(np.ones((3,50,4))*7,np.ones(4),25),0)


def test_native_interval_and_geodesic_sign():
    err=np.array([[[4.]*4,[16.]*4]])
    np.testing.assert_allclose(interval_mse(err,np.array([[1.,3.]]),2),13)
    np.testing.assert_allclose(interval_mse(err,np.array([[1.,3.]]),1),4)
    with pytest.raises(ValueError):interval_mse(err,np.zeros((1,2)),2)
    b=make_batch();p=copy.deepcopy(b.truth);p.quaternion_nb*=-1
    e=squared_errors(p,b.truth)
    np.testing.assert_allclose(e,0,atol=1e-10)


def fixture_flights():
    logs=np.array(['a']*100+['b']);cohorts=np.full(101,'Sep7');rows=[]
    for seed in [17,23,42]:
        for condition in ['Actual','Hold']:
            error=np.array([1.]*100+[3.])+(1 if condition=='Hold' else 0)
            rows+=flight_metrics(np.repeat(error[:,None]**2,4,axis=1),logs,cohorts,np.ones(101,bool),
                kind='endpoint',condition=condition,seed=seed,group='ALL',group_rule='test',step=25,horizon_s=.5)
    return pd.DataFrame(rows)


def test_flight_equal_weight_seed_pair_sign_and_zero_denominator():
    f=fixture_flights();s,m=aggregate(f)
    np.testing.assert_allclose(s.query('condition=="Actual"').value,2)
    np.testing.assert_allclose(s.query('condition=="Hold"').value,3)
    assert (m.n_seeds==3).all() and (m['std']==0).all()
    p=paired_comparisons(s,f);macro=p.query('level=="macro"')
    np.testing.assert_allclose(macro.absolute_gain,1)
    np.testing.assert_allclose(macro.relative_gain_pct,100/3)
    assert (macro.seeds_improved==3).all() and (macro.flights_improved==2).all()
    delta,pct=gains(np.array([1.,2.]),np.array([0.,1.]))
    assert np.isnan(pct[0]) and pct[1]==-100 and delta[0]==-1
    with pytest.raises(ValueError):aggregate(f[f.seed!=42])
    with pytest.raises(ValueError):paired_comparisons(s.iloc[1:],f)


def test_interval_equal_origin_then_flight_not_pooled_duration():
    # Each origin has its own duration denominator before origin averaging.
    e=np.array([[[1.]*4,[1.]*4],[[9.]*4,[9.]*4]])
    mse=interval_mse(e,np.array([[1.,1.],[10.,10.]]),2)
    rows=flight_metrics(mse,np.array(['a','a']),np.array(['Sep7','Sep7']),np.ones(2,bool))
    assert all(r['value']==pytest.approx(np.sqrt(5)) for r in rows)


def test_report_end_to_end(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from report_paper_dynamics_input_diagnostic import generate
    out=tmp_path/'out';art=tmp_path/'art';out.mkdir();art.mkdir()
    b=make_batch();b.controls=np.zeros((2,50,4))
    b.window_ids=np.array(['fixture_a','fixture_b'])
    b.log_ids=np.array(['2026.9.7/a','9.17数据/b'])
    batch=SimpleNamespace(trajectory=b)
    actual=copy.deepcopy(b.truth);hold=copy.deepcopy(b.truth);hold.velocity_n[:,2:]+=.1
    actual_path=art/'actual.npz'
    np.savez(actual_path,**vars(actual),window_ids=b.window_ids)
    np.savez(art/'Hold_seed17_predictions.npz',**vars(hold),window_ids=b.window_ids)
    rows=[];curves=[];times=[];coverage=[]
    for seed in [17,23,42]:
        for condition,pred in [('Actual',actual),('Hold',hold)]:
            e=squared_errors(pred,b.truth)
            for step in range(1,51):
                for group in ['ALL','low','middle','high']:
                    meta=dict(condition=condition,seed=seed,group=group,step=step,horizon_s=step*.02)
                    curves+=flight_metrics(e[:,step-1],b.log_ids,np.array(['Sep7','Sep17']),np.ones(2,bool),kind='evolution',group_rule='fixed',**meta)
                    if step in [5,10,25,50]:
                        for kind,mse in [('endpoint',e[:,step-1]),('interval',interval_mse(e,b.dt_s,step))]:
                            rows+=flight_metrics(mse,b.log_ids,np.array(['Sep7','Sep17']),np.ones(2,bool),kind=kind,group_rule='native',**meta)
    f=pd.DataFrame(rows);s,m=aggregate(f)
    _,cm=aggregate(pd.DataFrame(curves))
    cm['elapsed_median_s']=cm.step*.02;cm['elapsed_min_s']=cm.step*.02;cm['elapsed_max_s']=cm.step*.02
    m.to_csv(out/'multiseed_summary.csv',index=False);cm.to_csv(out/'error_evolution.csv',index=False)
    paired_comparisons(s,f).to_csv(out/'paired_differences.csv',index=False)
    for c in ['ALL','Sep7','Sep17']:
        for group in ['ALL','low','middle','high']:
            coverage.append(dict(step=25,cohort=c,group=group,n_origins=2 if c=='ALL' else 1,n_flights=2 if c=='ALL' else 1,E_median=1.,E_p95=2.))
    pd.DataFrame(coverage).to_csv(out/'control_group_coverage.csv',index=False)
    pd.DataFrame([dict(step=k,train_q25=.2,train_q75=.8) for k in [5,10,25,50]]).to_csv(out/'control_group_thresholds.csv',index=False)
    pd.DataFrame([dict(step=1)]).to_csv(out/'native_elapsed_time.csv',index=False)
    p=dict(actual_predictions={'17':str(actual_path)},case_selection=dict(cases=[dict(name='fixture',window_id='fixture_a',source_index=0)]))
    generate(out,art,p,batch)
    assert (out/'review_report.md').exists() and (out/'report.md').exists()
    assert len(list(out.glob('*.png')))==10
    assert len(list(out.glob('*.pdf')))==10
    assert len(pd.read_csv(out/'paper_table_500ms.csv'))==24
