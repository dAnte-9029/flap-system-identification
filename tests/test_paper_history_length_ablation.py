"""Critical fairness and end-to-end reporting checks without experiment training."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_paper_history_length_ablation import truncate_history,HS,duration_stats,schedule
from system_identification.training.trajectory_main_v1 import HistoryTrajectoryWindowBatch
from report_paper_history_length_ablation import summarize,relative_gains,generate,METRICS


def batch():
    rng=np.random.default_rng(7)
    return HistoryTrajectoryWindowBatch(trajectory=SimpleNamespace(
        window_ids=np.array(['a','b']),controls=rng.normal(size=(2,50,4)),
        dt_s=np.full((2,50),.02),truth=SimpleNamespace(velocity=rng.normal(size=(2,51,3)))),
        history_state_features=rng.normal(size=(2,26,12)),history_controls=rng.normal(size=(2,26,4)),
        history_mask=np.ones((2,26),dtype=bool))


@pytest.mark.parametrize('h',[1,5,13,26])
def test_suffix_and_future_identity(h):
    full=batch();before=copy.deepcopy(full)
    short=truncate_history(full,h)
    assert short.trajectory is full.trajectory
    for key in ['history_state_features','history_controls','history_mask']:
        np.testing.assert_array_equal(getattr(short,key),getattr(before,key)[:,-h:])
        np.testing.assert_array_equal(getattr(full,key),getattr(before,key))
    assert short.history_state_features.shape==(2,h,12)
    assert short.history_controls.shape==(2,h,4)
    np.testing.assert_array_equal(short.history_state_features[:,-1],full.history_state_features[:,-1])


def test_reject_incomplete_history_or_undeclared_h():
    full=batch()
    with pytest.raises(ValueError):truncate_history(full,2)
    full.history_mask[0,0]=False
    with pytest.raises(ValueError):truncate_history(full,5)


def test_native_duration_and_seed_contract():
    assert duration_stats([0,0])['max_s']==0
    assert duration_stats([.077,.082,.085])['median_s']==.082
    from run_paper_standard_gru_multiseed import read_json,OUT
    p=read_json(OUT/'protocol.json')
    for seed in [17,23,42]:
        stage=schedule(p,seed)
        assert [s[2] for s in stage]==[seed,seed+12]
        assert [s[1] for s in stage]==[40,25]


def test_same_parameterization_and_h1_current_token_encoding():
    from run_paper_standard_gru_multiseed import build_model
    stats=dict(feature_mean=np.zeros(12),feature_std=np.ones(12),control_mean=np.zeros(4),control_std=np.ones(4),
        derivative_mean=np.zeros(7),derivative_std=np.ones(7))
    reference=None
    for h in HS:
        model=build_model(17,stats)
        assert sum(p.numel() for p in model.parameters())==21383
        if reference is not None:
            for k,v in model.state_dict().items():assert torch.equal(v,reference[k])
        reference=copy.deepcopy(model.state_dict())
        cut=truncate_history(batch(),h)
        state=torch.tensor(cut.history_state_features,dtype=torch.float32)
        control=torch.tensor(cut.history_controls,dtype=torch.float32)
        hidden=model._encode_history(state,control,torch.ones((2,h),dtype=torch.bool))
        if h==1:
            direct=model.recurrent_cell(model._model_input(state[:,0],control[:,0]),torch.zeros(2,64))
            assert torch.equal(hidden,direct)


def fixture_frames():
    summaries=[];per=[]
    for h in HS:
        for seed,offset in [(17,0),(23,.01),(42,-.01)]:
            for horizon in [.1,.2,.5,1.]:
                for c in ['ALL','Sep7','Sep17']:
                    value=(2-h*.01+offset)*horizon
                    summaries.append(dict(history_steps=h,seed=seed,cohort=c,horizon_s=horizon,**{m:value for m in METRICS}))
                    if c!='ALL':
                        for log in ['a','b']:
                            per.append(dict(history_steps=h,seed=seed,cohort=c,log_id=c+log,horizon_s=horizon,**{m:value for m in METRICS}))
    return pd.DataFrame(summaries),pd.DataFrame(per)


def test_complete_grid_and_relative_gain():
    summary,_=fixture_frames();multi=summarize(summary)
    assert len(multi)==192
    with pytest.raises(ValueError):summarize(summary.iloc[1:])
    gains=relative_gains(multi)
    row=gains.query('reference_history_steps==13 and history_steps==26').iloc[0]
    assert row.relative_improvement_pct==pytest.approx((1.87-1.74)/1.87*100)


def test_full_report_pipeline(tmp_path):
    summary,per=fixture_frames()
    summary.to_csv(tmp_path/'per_seed_summary.csv',index=False)
    per.to_csv(tmp_path/'per_flight_wide.csv',index=False)
    pd.DataFrame([dict(history_steps=h,seed=s,status='fixture') for h in HS for s in [17,23,42]]).to_csv(tmp_path/'training_runs.csv',index=False)
    duration=[]
    for h in HS:
        for partition,c in [('train','ALL'),('validation','ALL'),('validation','Sep7'),('validation','Sep17')]:
            duration.append(dict(history_steps=h,partition=partition,cohort=c,**duration_stats([.02*(h-1)]*2)))
    pd.DataFrame(duration).to_csv(tmp_path/'history_duration_summary.csv',index=False)
    examples=dict(time_s=np.arange(26)*.02)
    for name in ['truth','H1','H13','H26']:
        for field in ['velocity_n','angular_velocity_b']:examples[f'{name}_{field}']=np.zeros((26,3))
        examples[f'{name}_quaternion_nb']=np.tile([1.,0,0,0],(26,1))
    np.savez(tmp_path/'representative_prediction.npz',**examples)
    protocol=dict(source_frozen_contract={},normalization_sha256='fixture',train_window_sha256='fixture',
        validation_origin_sha256='fixture',source_main_model_commit='fixture')
    generate(tmp_path,protocol)
    assert len(list(tmp_path.glob('*.png')))==11
    assert len(list(tmp_path.glob('*.pdf')))==11
    assert (tmp_path/'report.md').stat().st_size>1000
    assert len(pd.read_csv(tmp_path/'multiseed_summary.csv'))==192
    assert (tmp_path/'paired_flight_directions.csv').exists()
