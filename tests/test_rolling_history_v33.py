import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rolling_history_v33 import rolling_windows,crossed_step,VARIANTS
from test_rollout_trace_v32 import case


def arrays():
    n=77
    return dict(timestamp_us=np.arange(n)*20000,valid=np.ones(n,bool),reset=np.zeros(n,bool),nav_state=np.full(n,15),
        phase_rad=np.arange(n)*.4,state_features=np.zeros((n,14),np.float32),references=np.zeros((n,5),np.float32),
        actuator_commands=np.arange(n*4,dtype=np.float32).reshape(n,4),
        quaternion=np.tile([1.,0,0,0],(n,1)).astype(np.float32),velocity_n=np.zeros((n,3),np.float32),rates_b=np.zeros((n,3),np.float32))


def test_rolling_alignment_reanchoring_and_support_boundary():
    a=arrays();x,y=rolling_windows(a,26)
    assert x['history_state'].shape==(50,26,14)
    np.testing.assert_array_equal(x['history_actuator'][49],a['actuator_commands'][49:75])
    np.testing.assert_array_equal(x['history_command'][49],a['actuator_commands'][50:76])
    np.testing.assert_array_equal(x['future_command'][49,0],a['actuator_commands'][75])
    np.testing.assert_array_equal(x['history_state'][:,-1,9:11],np.tile([0,1],(50,1)))
    np.testing.assert_array_equal(y['rates_b'][-1,0],a['rates_b'][76])
    with pytest.raises(ValueError):rolling_windows(a,27)


def test_future_records_do_not_enter_earlier_rolling_inputs():
    a=arrays();x,y=rolling_windows(a,26)
    a['state_features'][40:]+=10;a['references'][40:]+=5;a['phase_rad'][40:]+=1
    changed,_=rolling_windows(a,26)
    for key in x:np.testing.assert_array_equal(x[key][:14],changed[key][:14])
    assert not np.array_equal(x['history_state'][14],changed['history_state'][14])


def test_crossed_branches_equal_when_hidden_and_context_are_equal():
    model,x,y=case();x['future_command']=x['future_command'][:,:1];x['dt_s']=x['dt_s'][:,:1]
    captured={}
    def eh(m,a,o):captured['h']=o[1][0]
    def ch(m,a,o):captured['i']=a[0]
    hooks=[model.encoder.register_forward_hook(eh),model.cell.register_forward_hook(ch)]
    with torch.no_grad():model(**x)
    for h in hooks:h.remove()
    r=crossed_step(model,x,captured['h'],captured['i'])
    for name in VARIANTS[1:]:
        for key in ['rates_b','derivative','raw_head','hidden_after']:
            np.testing.assert_array_equal(r[name+'_'+key],r['rolling_current_'+key])
