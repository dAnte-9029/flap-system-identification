import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_reference_windows import fixture_arrays
from reference_refresh_v15 import refresh_inputs,stitch,persistence,paired_local_age


def test_refresh_indices_reference_and_phase_anchor():
    a=fixture_arrays();x=refresh_inputs(a,100,5)
    assert x['history_state'].shape==(10,26,14)
    np.testing.assert_allclose(x['history_state'][:,-1,9:11],np.tile([0,1],(10,1)),atol=1e-7)
    for i in range(10):
        o=100+5*i
        np.testing.assert_array_equal(x['history_actuator'][i],a['actuator_commands'][o-26:o])
        np.testing.assert_array_equal(x['future_reference'][i],a['references'][o:o+5])
        np.testing.assert_array_equal(x['initial_rates_b'][i],a['rates_b'][o])
    np.testing.assert_array_equal(stitch(x['future_reference'],1,5)[0],a['references'][100:150])


def test_no_later_measurements_in_current_refreshed_forecast():
    a=fixture_arrays();before=refresh_inputs(a,100,10)
    for key in ['state_features','phase_rad','quaternion','velocity_n','rates_b']:a[key][121:]=1234
    a['actuator_commands'][120:]=5678
    after=refresh_inputs(a,100,10)
    for k in before:np.testing.assert_array_equal(before[k][:3],after[k][:3])
    assert not np.array_equal(before['initial_rates_b'][3],after['initial_rates_b'][3])


def test_stitch_and_matched_age_keep_exact_target_order():
    v=np.arange(100).reshape(20,5,1);result=stitch(v,2,5)
    np.testing.assert_array_equal(result[:,:,0],np.arange(100).reshape(2,50))
    e=np.broadcast_to(np.arange(50)[None,:,None],(2,50,5)).copy()
    local=paired_local_age(e,5)
    np.testing.assert_array_equal(local[0,:,0],np.arange(5)+22.5)
    with pytest.raises(ValueError):stitch(v,2,10)


def test_persistence_receives_same_updates_and_no_prediction_targets():
    x=refresh_inputs(fixture_arrays(),100,10);p=persistence(x)
    assert p['rates_b'].shape==(5,10,3)
    np.testing.assert_array_equal(p['rates_b'][:,9],x['initial_rates_b'])
    np.testing.assert_array_equal(stitch(p['rates_b'],1,10)[0,10],x['initial_rates_b'][1])
