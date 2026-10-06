import json
from pathlib import Path
import numpy as np
import pytest
from system_identification.control.px4_c9af import C9Controller


def parameters():
    return json.loads((Path(__file__).resolve().parents[1]/'docs/analysis/results/control_windows_v37/parameters_by_flight.json').read_text())[0]['values']


def step(c,**extra):
    args=dict(rates=[.1,.2,-.1],setpoints=[.3,-.2,.2],acceleration=[0.]*3,airspeed=8.,air_age=0.,motor=.4,dt=.0025)
    args.update(extra);return c.rate(**args)


def test_snapshot_complete_and_resume_exact():
    c=C9Controller(parameters());c.state.airspeed_filter=8
    for _ in range(7):step(c)
    c.attitude(.2,.1,.3,.2,8,0,manual=True,yaw_stick=.4)
    snapshot=c.snapshot();other=C9Controller(parameters());other.restore(snapshot)
    assert step(c)==step(other) and c.snapshot()==other.snapshot()
    assert snapshot!=c.snapshot()
    snapshot['firmware']='wrong'
    with pytest.raises(ValueError):other.restore(snapshot)


def test_pre_integral_and_cached_gain_are_used_before_updates():
    c=C9Controller(parameters());s=c.state;s.airspeed_filter=8;s.integral=[.02]*3;s.cached_gain=[.6]*3;s.gain=[.6]*3
    out=step(c)
    assert out['integral_used']==[.02]*3 and out['gain_used']==[.6]*3
    assert s.integral!=out['integral_used'] and s.cached_gain!=out['gain_used']


def test_manual_yaw_is_added_after_body_limit_and_old_yaw_used():
    c=C9Controller(parameters());c.state.yaw_euler=.8
    rates=c.attitude(.2,.1,.3,.2,8,0,manual=True,yaw_stick=1)
    p=parameters();expect=(.3-.2)/p['FW_R_TC']-np.sin(.1)*.8
    np.testing.assert_allclose(rates[0],expect)
    np.testing.assert_allclose(rates[2]-c.state.yaw_body,np.deg2rad(p['FW_MAN_YR_MAX']))


def test_feedback_order_and_non_vtol_subscription_approximation():
    a=C9Controller(parameters());a.state.airspeed_filter=8;a.state.unallocated=[1.,1.,1.]
    b=C9Controller(parameters());b.restore(a.snapshot())
    step(a,feedback='source_no_interleave');step(b,feedback='delayed_allocation')
    assert a.state.integral[0]>0 and b.state.integral[0]==0


def test_signed_highpass_ambiguity_changes_next_gain():
    a=C9Controller(parameters());a.state.airspeed_filter=8;a.state.hpf=[.2]*3
    b=C9Controller(parameters());b.restore(a.snapshot());b.state.hpf=[-.2]*3
    step(a);step(b)
    assert a.state.gain!=b.state.gain


def test_source_reset_does_not_clear_filters_or_cached_gain():
    c=C9Controller(parameters());c.state.airspeed_filter=8;c.state.cached_gain=[.5]*3;c.state.gain=[.5]*3;c.state.energy=[.2]*3
    out=step(c,landed=True)
    assert out['gain_used']==[.5]*3 and out['integral_used']==[0.]*3
    assert all(v>0 for v in c.state.energy)
