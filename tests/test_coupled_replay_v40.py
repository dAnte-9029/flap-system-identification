import json
import sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from coupled_replay_v40 import interval, controllers, rollout, VARIANTS
from test_actuator_command_step import fixture
from test_px4_c9af import parameters
from system_identification.control.px4_c9af import C9Controller


def setup():
    model, inputs, _, _ = fixture()
    c=C9Controller(parameters());c.state.airspeed_filter=8
    snaps=[c.snapshot(),c.snapshot()]
    tape=np.zeros((2,400,4));tape[:,:,2]=.4;tape[:,:,0]=.1
    support=dict(state_min=np.full(14,-1e5),state_max=np.full(14,1e5),
                 command_min=np.array([0,-1,-1,-1]),command_max=np.ones(4))
    return model,inputs,snaps,tape,np.full(2,8.),np.array([True,False]),support


def test_left_endpoint_and_subcycle_actions_are_distinct():
    _,_,snaps,tape,air,manual,_=setup()
    tape[:,1:8,2]=.8
    out=interval(controllers(snaps),np.array([[1.,0,0,0]]*2),np.zeros((2,3)),tape[:,:8],air,manual,2500,'source_no_interleave')
    np.testing.assert_array_equal(out['command'][:,0], [.4,.4])
    np.testing.assert_array_equal(out['micro_commands'][:,1:,0],np.full((2,7),.8))
    assert out['micro_commands'].shape==(2,8,4)


def test_coupling_prefix_and_reentrant_state_isolation():
    model,inputs,snaps,tape,air,manual,support=setup();before=json.dumps(snaps)
    a=rollout(model,inputs,snaps,tape,air,manual,VARIANTS[0],support)
    changed=tape.copy();changed[:,200:,0]+=.1
    b=rollout(model,inputs,snaps,changed,air,manual,VARIANTS[0],support)
    repeat=rollout(model,inputs,snaps,tape,air,manual,VARIANTS[0],support)
    for key in a:
        np.testing.assert_array_equal(a[key],repeat[key])
        np.testing.assert_array_equal(a[key][:,:25],b[key][:,:25])
    assert np.any(a['commands'][:,25:]!=b['commands'][:,25:])
    assert json.dumps(snaps)==before
    assert np.isfinite(a['rates_b']).all()


def test_proxy_and_clock_contract_rejections():
    _,_,snaps,tape,air,manual,_=setup()
    snaps[0]['parameters']['FW_RR_D']=.01
    with pytest.raises(AssertionError):controllers(snaps)
    with pytest.raises(ValueError):interval([],np.zeros((0,4)),np.zeros((0,3)),tape,air,manual,1000,'source_no_interleave')
