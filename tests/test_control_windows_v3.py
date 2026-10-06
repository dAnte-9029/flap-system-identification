import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from control_windows_v3 import input_features
from control_windows_v3_ridge import fit_predict,features
from test_control_windows_v2 import model_and_input


def test_input_score_prefers_sustained_change_at_equal_range():
    refs=np.zeros((2,50,5))
    refs[0,:,0]=np.linspace(0,.1,50)
    refs[1,:,0]=np.tile([0,.1],25)
    score,*_=input_features(refs)
    assert score[0]>score[1]
    assert input_features(np.zeros((1,50,5)))[0][0]==0


def test_ridge_scaling_does_not_fit_query_rows():
    rng=np.random.default_rng(13);x=rng.normal(size=(12,3));y=x[:,:1]*2
    train=np.arange(8);query=np.arange(8,12);logs=np.array(['a']*4+['b']*4+['v']*4)
    _,first=fit_predict(x,y,train,query,logs)
    x[8:]+=10000;y[8:]-=10000
    _,second=fit_predict(x,y,train,query,logs)
    for key in first:np.testing.assert_array_equal(first[key],second[key])


def test_ridge_endpoint_cannot_read_later_reference():
    _,x=model_and_input()
    first=features(x,5)
    x['future_reference'][:,5:]+=1000
    np.testing.assert_array_equal(first,features(x,5))


def test_response_smoothing_is_causal_and_identical_waveforms_match():
    from control_windows_v3_response import causal_mean,waveform_summary
    value=np.zeros((100,3));value[30:60]=.2
    first=causal_mean(value)
    changed=value.copy();changed[80:]+=10
    np.testing.assert_array_equal(first[:71],causal_mean(changed)[:71])
    summary,observed,predicted=waveform_summary(value,value,np.zeros((26,3)),np.zeros((50,3)))
    np.testing.assert_array_equal(observed,predicted)
    for row in summary:
        assert row['measurable']
        assert row['amplitude_error']==row['peak_time_error_s']==row['waveform_rmse']==0


def test_decoder_gravity_matches_dataset_unit_direction_for_roll():
    import torch
    from system_identification.models.reference_gru_unit_gravity import unit_state_features
    q=torch.tensor([[1.,0,0,0],[2**-.5,2**-.5,0,0]])
    zero=torch.zeros(2);v=torch.zeros(2,3)
    state=unit_state_features(v,q,v,zero,zero,torch.ones(2)*4)
    torch.testing.assert_close(state[:,6:9],torch.tensor([[0.,0,1.],[0.,1.,0.]]),atol=2e-6,rtol=0)
    torch.testing.assert_close(state[:,9:11],torch.tensor([[0.,1.],[0.,1.]]))


def test_unit_fix_preserves_initial_parameters_and_causality():
    import torch
    from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity
    old,x=model_and_input()
    stats={k:v for k,v in old.state_dict().items() if k.endswith('_mean') or k.endswith('_scale')}
    new=ReferenceGRUUnitGravity(stats);new.load_state_dict(old.state_dict());new.eval()
    with torch.no_grad():
        before=new(**x);x['future_reference'][:,5:]+=100;after=new(**x)
    for key in before:torch.testing.assert_close(before[key][:,:5],after[key][:,:5],rtol=0,atol=0)


def test_fixed_checkpoint_loader_rejects_legacy_unit_contract(tmp_path):
    import torch
    import pytest
    from train_reference_unit_gravity import model_from
    path=tmp_path/'legacy.pt';torch.save({'feature_contract':'legacy'},path)
    with pytest.raises(ValueError,match='gravity contract'):model_from(path)
