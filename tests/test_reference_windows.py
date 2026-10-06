import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_reference_dynamics_data import chunks,window_origins,configuration_id
from system_identification.data.reference_windows import extract_window


def fixture_arrays():
    rng=np.random.default_rng(41);n=300
    return dict(timestamp_us=np.arange(n)*20000,valid=np.ones(n,bool),reset=np.zeros(n,bool),nav_state=np.full(n,3),
        phase_rad=np.arange(n)*.1,state_features=rng.normal(size=(n,14)),references=rng.normal(size=(n,5)),
        actuator_commands=rng.normal(size=(n,4)),quaternion=rng.normal(size=(n,4)),
        velocity_n=rng.normal(size=(n,3)),rates_b=rng.normal(size=(n,3)))


def test_future_measurements_cannot_change_inputs():
    a=fixture_arrays();x,_=extract_window(a,100)
    for name in ['state_features','phase_rad','quaternion','velocity_n','rates_b']:
        a[name][101:]=999
    a['actuator_commands'][100:]=-999
    y,_=extract_window(a,100)
    for k in x:np.testing.assert_array_equal(x[k],y[k])
    np.testing.assert_array_equal(x['history_actuator'],fixture_arrays()['actuator_commands'][74:100])
    np.testing.assert_array_equal(x['current_mode'],[1,0])
    assert x['future_reference'].shape==(50,5)


def test_short_reference_prefix_is_invariant_and_phase_zero():
    a=fixture_arrays();x,_=extract_window(a,100,horizon=5)
    a['references'][105:]=999
    y,_=extract_window(a,100,horizon=5)
    for k in x:np.testing.assert_array_equal(x[k],y[k])
    np.testing.assert_allclose(x['history_state'][-1,9:11],[0,1])


@pytest.mark.parametrize('change',['mode','gap','invalid','reset'])
def test_boundaries_rejected(change):
    a=fixture_arrays()
    if change=='mode':a['nav_state'][110:]=15
    elif change=='gap':a['timestamp_us'][110:]+=10000
    elif change=='reset':a['reset'][110]=True
    else:a['valid'][110]=False
    with pytest.raises(ValueError):extract_window(a,100)


def test_chunks_split_resets_and_modes_and_keep_context_inside():
    valid=np.ones(800,bool);mode=np.full(800,3);mode[400:]=15
    reset=np.zeros(800,bool);reset[200]=True
    blocks=chunks(valid,mode,reset)
    assert blocks==[(25,175),(225,375),(425,775)]
    for a,b in blocks:
        for i in window_origins(a,b):assert i-26>=a and i+50<b


def test_config_group_separates_missing_parameters_and_firmware():
    a=configuration_id('a',{'P':1})
    assert a!=configuration_id('b',{'P':1})
    assert a!=configuration_id('a',{'P':1,'Q':0})
