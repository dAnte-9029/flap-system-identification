import copy
from dataclasses import replace
import math
import pytest
from system_identification.control.px4_pitch import PitchConfig,PitchInput,PX4PitchRateController


def config(**overrides):
    c=PitchConfig(.06,.1,0.,.5,.4,8.,3.,4.,20.,False,False,False,.3)
    return replace(c,**overrides)


def sample(**overrides):
    return replace(PitchInput(0.,1.,0.,.01,8.,0.),**overrides)


def test_integral_used_before_update_and_positive_antiwindup():
    c=PX4PitchRateController(config())
    first=c.step(sample()); assert first.torque==pytest.approx(.56)
    assert c.state.integral==pytest.approx(.001*(1-(1/math.radians(400))**2))
    before=c.state.integral
    c.step(sample(saturation_positive=True)); assert c.state.integral==before
    c.step(sample(rate_setpoint=-1.,saturation_positive=True)); assert c.state.integral<before


def test_reset_integral_does_not_reset_gain_or_filters():
    c=PX4PitchRateController(config(compress_gain=True))
    for i in range(100): c.step(sample(rate_setpoint=(-1)**i))
    previous=c.snapshot()
    c.step(sample(reset_integral=True))
    assert c.state.energy>0
    # Integral reset must not replace the cached adaptive gain with one.
    reference=PX4PitchRateController(c.config); reference.restore(previous)
    reference.state.integral=0
    reference.step(sample())
    assert c.snapshot()==reference.snapshot()


def test_snapshot_restores_every_hidden_state_and_is_not_aliased():
    c=PX4PitchRateController(config(compress_gain=True,use_airspeed=True,scale_airspeed=True))
    for i in range(100): c.step(sample(rate=math.sin(i),calibrated_airspeed=6.))
    snap=c.snapshot(); before=copy.deepcopy(snap)
    expected=[c.step(sample(rate=.3)) for _ in range(80)]
    clone=PX4PitchRateController(c.config); clone.restore(snap)
    actual=[clone.step(sample(rate=.3)) for _ in range(80)]
    assert actual==expected and snap==before and clone.snapshot()==c.snapshot()
    with pytest.raises(ValueError): PX4PitchRateController(config()).restore(snap)


def test_disabled_does_not_invent_manual_output_and_stale_air_falls_back():
    c=PX4PitchRateController(config(use_airspeed=True,scale_airspeed=True))
    c.step(sample())
    assert c.step(sample(rates_enabled=False)).torque is None
    assert c.step(sample(calibrated_airspeed=float('nan'),airspeed_age_s=float('inf'))).airspeed_scale==1


@pytest.mark.parametrize('dt',[0,.001,.041,float('nan')])
def test_invalid_clock_is_rejected(dt):
    with pytest.raises(ValueError): PX4PitchRateController(config()).step(sample(dt_s=dt))
