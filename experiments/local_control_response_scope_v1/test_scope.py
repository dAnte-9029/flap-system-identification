import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scope', HERE/'analyze.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CFG=json.loads((HERE/'config.json').read_text())


def test_command_coordinates_sign_and_inverse():
    u=np.array([[.7,-.3,.5,.1],[.4,.2,.2,-.1]])
    c=m.coords(u)
    assert np.allclose(c,[[.7,.1,.4,.1],[.4,.2,0,-.1]])
    assert np.allclose(np.c_[c[:,0],c[:,1]-c[:,2],c[:,1]+c[:,2],c[:,3]],u)


def test_zoh_partition_invariant_statistics():
    a=m.integral_stats(np.array([0,200000,500000]),np.array([[0,0,0,0],[0,0,.2,0]]),CFG['change_thresholds'])
    b=m.integral_stats(np.array([0,100000,200000,300000,500000]),np.array([[0,0,0,0],[0,0,0,0],[0,0,.2,0],[0,0,.2,0]]),CFG['change_thresholds'])
    for x,y in zip(a,b):
        np.testing.assert_allclose(x,y)
    assert a[3][2]==300000
    assert m.direction_flags(a,CFG)[0].tolist()==[False,False,True,False]


def test_reversals_are_not_a_sustained_signed_direction():
    a=m.integral_stats(np.array([0,250000,500000]),np.array([[0,0,.2,0],[0,0,-.2,0]]),CFG['change_thresholds'])
    assert not m.direction_flags(a,CFG)[0].any()
    assert a[3][2]==250000


def test_contiguous_dwell_does_not_sum_separate_pulses():
    t=np.array([0,80000,200000,280000,500000])
    a=m.integral_stats(t,np.array([[0,0,.3,0],[0,0,0,0],[0,0,.3,0],[0,0,0,0]]),CFG['change_thresholds'])
    assert a[3][2]==80000
    assert not m.direction_flags(a,CFG)[0].any()


def test_joint_energy_not_independent_channels():
    a=m.integral_stats(np.array([0,500000]),np.array([[.1,0,.2,0]]),CFG['change_thresholds'])
    assert not m.direction_flags(a,CFG)[0].any()


def test_cut_half_open_and_no_future_boundary_value():
    native=(np.array([0,100,200,300]),np.tile(np.arange(4)[:,None],(1,4)),np.zeros(4))
    t,u=m.cut_tape(native,50,200)
    assert t.tolist()==[0,50,150]
    assert u[:,0].tolist()==[0,1]


def test_exact_pair_contrast_and_swapped_sign():
    a=(np.array([0,100,500]),np.array([[0,0,0,0],[0,0,.2,0]]))
    b=(np.array([0,300,500]),np.array([[0,0,0,0],[0,0,.1,0]]))
    t,u=m.contrast(a,b)
    tt,uu=m.contrast(b,a)
    assert t.tolist()==[0,100,300,500]
    np.testing.assert_equal(t,tt)
    np.testing.assert_allclose(u,-uu)


def test_sealed_split_rejected_before_read():
    with pytest.raises(ValueError,match='test is sealed'):
        m.load_data(Path('/nonexistent'),'test',CFG,Path('/nonexistent'))
    with pytest.raises(ValueError,match='future outcome forbidden'):
        m.old.load_parquet(Path('/nonexistent'),'samples','train',['target_velocity_ned_m_s_x'])


def test_closed_support_overlap_and_union():
    d=pd.DataFrame(dict(log_sha=['a','a','a'],decision_us=[500,700,1600],command_end_us=[1000,1200,2100],
                        history_start_us=[0,200,1100],pair_support_end_us=[1100,1300,2200],window_id=['a','b','c']))
    assert m.nonoverlap_windows(d)==[0]
    assert m.union_seconds(d)==pytest.approx(.0012)


def test_shared_nodes_flights_and_components_separate():
    d=pd.DataFrame(dict(log_sha=['a','b','b','c','d']))
    p=pd.DataFrame(dict(i=[0,0,2],j=[1,3,4]))
    result=m.graph_counts(p,d)
    assert result['unique_nodes']==5
    assert result['shared_node_uses']==1
    assert result['flights']==4
    assert result['flight_components']==1
