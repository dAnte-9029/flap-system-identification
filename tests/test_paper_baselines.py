"""Critical benchmark contracts: causality, integration and flight aggregation."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from system_identification.evaluation.paper_baselines import (
    HORIZONS, METRIC_MAP, aggregate_flights, cohort, endpoint_metrics,
)
from system_identification.models.paper_baselines import MemorylessTrajectoryModel
from system_identification.models.trajectory import ConstantTwistPredictor, InitialTrajectoryState
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel


def stats():
    return dict(feature_mean=np.zeros(12), feature_std=np.ones(12),
                control_mean=np.zeros(4), control_std=np.ones(4),
                derivative_mean=np.zeros(7), derivative_std=np.ones(7))


def inputs():
    return dict(history_state_features=torch.randn(2,26,12),
                history_controls=torch.randn(2,26,4), history_mask=torch.ones(2,26,dtype=torch.bool),
                position_n=torch.zeros(2,3), velocity_n=torch.ones(2,3),
                quaternion_nb=torch.tensor([[1.,0,0,0]]).repeat(2,1),
                angular_velocity_b=torch.tensor([[0.,0,1.]]).repeat(2,1),
                relative_phase_rad=torch.zeros(2), flap_frequency_hz=torch.ones(2)*5,
                future_controls=torch.randn(2,50,4), dt_s=torch.ones(2,50)*.02)


def test_memoryless_ignores_all_history_with_nonzero_head():
    model = MemorylessTrajectoryModel(**stats())
    torch.nn.init.normal_(model.derivative_head.mlp.network[-1].weight, std=.01)
    x = inputs()
    before = model(**x)
    x['history_state_features'].fill_(float('nan'))
    x['history_controls'].fill_(float('nan'))
    x['history_mask'].fill_(False)
    after = model(**x)
    for a,b in zip(before,after):
        assert torch.equal(a,b)
    assert sum(p.numel() for p in model.parameters()) == 5703


@pytest.mark.parametrize('kind', ['mlp','gru'])
def test_zero_derivative_matches_analytic_constant_twist(kind):
    model = MemorylessTrajectoryModel(**stats()) if kind=='mlp' else CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True,**stats())
    x=inputs()
    prediction=model(**x).to_numpy()
    initial=InitialTrajectoryState(**{k:x[k].numpy() for k in vars(prediction)})
    expected=ConstantTwistPredictor().rollout(initial,x['future_controls'].numpy(),x['dt_s'].numpy())
    for key in vars(expected):
        if key == 'relative_phase_rad':
            delta=getattr(prediction,key)-getattr(expected,key)
            np.testing.assert_allclose(np.arctan2(np.sin(delta),np.cos(delta)),0.,atol=1e-5)
        else:
            np.testing.assert_allclose(getattr(prediction,key),getattr(expected,key),atol=1e-5)


def test_flight_macro_does_not_weight_by_window_count():
    rows=[]
    for log,n,error in [('a',100,1.),('b',1,3.)]:
        for i in range(n):
            rows.append(dict(model='B',seed=17,cohort='Sep7',log_id=log,horizon_s=.5,
                             **{k:error for k in METRIC_MAP}))
    per,summary,agg=aggregate_flights(pd.DataFrame(rows))
    overall=summary.loc[summary.cohort=='ALL'].iloc[0]
    assert overall.position_rmse_m==2.
    assert overall.n_flights==2 and overall.n_windows==101
    assert agg.iloc[0].position_rmse_m_std_across_flights==pytest.approx(np.sqrt(2))


def make_batch():
    x=inputs()
    initial=InitialTrajectoryState(**{k:x[k].numpy().astype(float) for k in
        ['position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz']})
    dt=np.full((2,50),.02)
    truth=ConstantTwistPredictor().rollout(initial,np.zeros((2,50,4)),dt)
    return SimpleNamespace(truth=truth,dt_s=dt,window_ids=np.array(['a','b']),
        log_ids=np.array(['2026.9.7/a','9.17数据/b']),segment_ids=np.array([0,0]))


def test_increment_identity_axis_metrics_and_quaternion_sign():
    b=make_batch(); p=copy.deepcopy(b.truth)
    p.velocity_n[:,1:]+=np.array([3.,4.,0.])
    p.angular_velocity_b[:,1:]+=np.array([0.,0.,2.])
    p.quaternion_nb[:,1:]*=-1
    rows=endpoint_metrics(p,b,model='test')
    np.testing.assert_allclose(rows.velocity_error_m_s,5.)
    np.testing.assert_allclose(rows.delta_v_error_m_s,rows.velocity_error_m_s)
    np.testing.assert_allclose(rows.delta_omega_error_rad_s,rows.body_rate_error_rad_s)
    np.testing.assert_allclose(rows.velocity_x_error_m_s,3.)
    np.testing.assert_allclose(rows.attitude_error_deg,0.,atol=2e-6)
    assert len(rows)==2*len(HORIZONS)


def test_metric_rejects_wrong_timestamps_nonfinite_and_incomplete_rollouts():
    b=make_batch(); b.dt_s[0,0]=-.02
    with pytest.raises(ValueError,match='timestamps'):
        endpoint_metrics(b.truth,b,model='bad')
    b=make_batch(); p=copy.deepcopy(b.truth); p.velocity_n[0,5,0]=np.nan
    with pytest.raises(ValueError,match='shape/values'):
        endpoint_metrics(p,b,model='bad')
    p=copy.deepcopy(b.truth); p.position_n=p.position_n[:,:26]
    with pytest.raises(ValueError,match='shape/values'):
        endpoint_metrics(p,b,model='bad')


def test_unknown_or_sealed_cohort_is_rejected():
    for name in ['2026.9.8/log.ulg','9.19数据/log.ulg','9.18数据/log.ulg']:
        with pytest.raises(ValueError,match='outside'):
            cohort(name)


@pytest.mark.parametrize('kind', ['mlp','gru'])
@pytest.mark.parametrize('frequency', [False,True])
def test_existing_training_loop_accepts_baselines_and_frequency_stage(kind,frequency):
    from scripts.run_paper_baseline_comparison import _FrequencyLossAdapter
    from system_identification.training.main_v2_increment import train_increment_stage
    b=make_batch()
    b.controls=np.zeros((2,50,4))
    batch=SimpleNamespace(trajectory=b,history_state_features=np.zeros((2,26,12)),
                          history_controls=np.zeros((2,26,4)),history_mask=np.ones((2,26),bool))
    model=MemorylessTrajectoryModel(**stats()) if kind=='mlp' else CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True,**stats())
    trained=_FrequencyLossAdapter(model) if frequency else model
    before={k:v.clone() for k,v in model.state_dict().items()}
    trained,history=train_increment_stage(trained,batch,scales=[1.,1.],weights=[.1,.2],
        device='cpu',epochs=1,seed=17,learning_rate=.0003,actuator=frequency)
    assert len(history)==1 and np.isfinite(history.loss).all()
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_causality_probe_preserves_frozen_batch_and_uses_no_future_labels():
    from scripts.run_paper_baseline_comparison import causality_check
    from system_identification.evaluation.trajectory import TrajectoryWindowBatch
    from system_identification.training.trajectory_main_v1 import HistoryTrajectoryWindowBatch
    b=make_batch()
    trajectory=TrajectoryWindowBatch(**vars(b),controls=np.zeros((2,50,4)))
    batch=HistoryTrajectoryWindowBatch(trajectory=trajectory,history_state_features=np.zeros((2,26,12)),
        history_controls=np.zeros((2,26,4)),history_mask=np.ones((2,26),bool))
    model=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True,**stats())
    result=causality_check(model,batch,'cpu')
    assert result['future_label_poisoning_bitwise_equal']
    assert np.isfinite(batch.trajectory.truth.velocity_n).all()


@pytest.mark.parametrize('problem', ['overlap','extra_artifact','duplicate_hash'])
def test_dataset_gate_rejects_unsafe_manifest_before_any_parquet_read(tmp_path,monkeypatch,problem):
    import hashlib
    import json
    import yaml
    from scripts import run_paper_baseline_comparison as runner
    manifest=dict(split_contract=dict(sealed_test_opened=False,assignments={
        'train':['2026.9.6/a'],'validation':['2026.9.7/b'],
        'sealed_test':['2026.9.8/c'],'reserved_evaluation':['9.19数据/d']}),
        source=dict(ulog_sha256={'2026.9.6/a':'a','2026.9.7/b':'b'}),
        artifact_sha256={f'{kind}_{part}.parquet':'unused' for kind in ['samples','windows'] for part in ['train','validation']})
    if problem=='overlap':
        manifest['split_contract']['assignments']['sealed_test'].append('2026.9.7/b')
    elif problem=='extra_artifact':
        manifest['artifact_sha256']['samples_sealed_test.parquet']='NEVER_READ'
    else:
        manifest['source']['ulog_sha256']['2026.9.7/b']='a'
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    reg=tmp_path/'configs/data/trajectory_dataset_registry.yaml';reg.parent.mkdir(parents=True)
    reg.write_text(yaml.safe_dump(dict(default_dataset_id=runner.DATASET_ID,datasets={runner.DATASET_ID:dict(
        manifest_path='manifest.json',manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest())})))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    def forbidden(*args,**kwargs):
        pytest.fail('unsafe manifest reached sample loading')
    monkeypatch.setattr(pd,'read_parquet',forbidden)
    with pytest.raises(ValueError,match='overlap|unexpected artifact|duplicate source'):
        runner.prepare()
