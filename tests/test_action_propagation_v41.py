import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from action_propagation_v41 import probes,BRANCHES,one_tick,copy_controller,vector,from_vector
from coupled_replay_v40 import interval
from system_identification.control.px4_c9af import C9Controller
from test_px4_c9af import parameters


def pair():
    a=C9Controller(parameters());a.state.airspeed_filter=8.;a.state.integral=[.01,.02,.03]
    b=copy_controller(a);b.state.integral=[-.01,-.02,-.03];b.state.cached_gain=[.6,.7,.8]
    return a,b


def test_probe_isolation_and_exact_factorial_identity():
    a,b=pair();av=vector(a);bv=vector(b)
    commands,_=probes(a,b,[.1,.2],[.3,.2,.1],[0.,0.],[.1,0.,0.],np.array([.1,.1,.4,.2]),8,True,2500,'source_no_interleave')
    mm,mt,tm,tt=commands[:4]
    np.testing.assert_allclose(mm-tt,(mm-mt)+(mt-tt),rtol=0,atol=1e-16)
    np.testing.assert_allclose((mm-mt)-(tm-tt),mm-mt-tm+tt,rtol=0,atol=1e-16)
    np.testing.assert_array_equal(vector(a),av);np.testing.assert_array_equal(vector(b),bv)
    assert not np.array_equal(mm,commands[BRANCHES.index('integral')])


def test_one_tick_matches_full_interval_left_endpoint():
    a,_=pair();reference=np.array([.1,.1,.4,.2]);q=np.array([[1.,0,0,0.]])
    command,sat=one_tick(copy_controller(a),[0.,0.],[.1,.2,.3],reference,8,True,2500,'source_no_interleave')
    full=interval([copy_controller(a)],q,np.array([[.1,.2,.3]]),np.tile(reference,(1,8,1)),np.array([8.]),np.array([True]),2500,'source_no_interleave')
    np.testing.assert_array_equal(command,full['command'][0]);np.testing.assert_array_equal(sat,full['surface_clipped'][0,0])


def test_snapshot_roundtrip_and_unobserved_filter_no_immediate_output_effect():
    a,b=pair();restored=from_vector(a,vector(a))
    assert restored.snapshot()==a.snapshot()
    b=copy_controller(a);b.state.hpf=[2.]*3;b.state.previous_effort=[3.]*3
    args=([0.,0.],[.1,.2,.3],np.array([.1,.1,.4,.2]),8,False,2500,'source_no_interleave')
    x,_=one_tick(a,*args);y,_=one_tick(b,*args)
    np.testing.assert_array_equal(x,y)
    assert a.state.cached_gain!=b.state.cached_gain  # affects subsequent ticks
