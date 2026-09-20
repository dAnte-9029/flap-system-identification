"""Pitch rate arithmetic matching the e624 PX4 fixed-wing implementation.

This component is not the complete PX4 controller. Roll/B2b, attitude guidance,
allocation, transport and sensor clocks belong to their respective adapters.
Call once per actual controller update, using only observations available then.
"""
from dataclasses import asdict, dataclass, fields
import math

FIRMWARE_COMMIT = 'e624a99f2955addbf76681e636c44162a0c03055'


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


@dataclass(frozen=True)
class PitchConfig:
    p: float
    i: float
    d: float
    ff: float
    imax: float
    airspeed_trim: float
    airspeed_stall: float
    airspeed_min: float
    airspeed_max: float
    use_airspeed: bool
    scale_airspeed: bool
    compress_gain: bool
    gain_min: float
    trim: float = 0.
    dtrim_min: float = 0.
    dtrim_max: float = 0.

    def __post_init__(self):
        numeric=[v for v in asdict(self).values() if not isinstance(v,bool)]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('Nonfinite controller parameter')
        if not 0 < self.airspeed_min < self.airspeed_trim < self.airspeed_max or self.airspeed_stall<=0:
            raise ValueError('Invalid airspeed range')
        if min(self.p,self.i,self.d,self.ff,self.imax)<0 or not 0<self.gain_min<=1:
            raise ValueError('Invalid controller gains')

    @classmethod
    def from_parameters(cls, p):
        # Explicit keys: missing flight parameters must not silently use defaults.
        return cls(*(float(p[k]) for k in ['FW_PR_P','FW_PR_I','FW_PR_D','FW_PR_FF','FW_PR_IMAX',
                                          'FW_AIRSPD_TRIM','FW_AIRSPD_STALL','FW_AIRSPD_MIN','FW_AIRSPD_MAX']),
                   *(bool(p[k]) for k in ['FW_USE_AIRSPD','FW_ARSP_SCALE_EN','FW_GC_EN']),
                   float(p['FW_GC_GAIN_MIN']),float(p['TRIM_PITCH']),float(p['FW_DTRIM_P_VMIN']),float(p['FW_DTRIM_P_VMAX']))


@dataclass
class PitchState:
    integral: float = 0.
    airspeed_filtered: float = 0.
    compression_gain: float = 1.
    cached_gain: float = 1.
    highpass: float = 0.
    energy: float = 0.
    previous_effort: float = 0.


@dataclass(frozen=True)
class PitchInput:
    rate: float
    rate_setpoint: float
    angular_acceleration: float
    dt_s: float
    calibrated_airspeed: float
    airspeed_age_s: float
    landed: bool = False
    rates_enabled: bool = True
    reset_integral: bool = False
    saturation_positive: bool = False
    saturation_negative: bool = False


@dataclass(frozen=True)
class PitchOutput:
    torque: float | None
    integral_used: float
    gain_used: float
    airspeed_scale: float | None
    p_term: float | None
    ff_term: float | None
    clipped: bool


class PX4PitchRateController:
    """Own all pitch filter/integrator state; never ingest logged target outputs."""
    def __init__(self, config: PitchConfig):
        self.config=config
        self.state=PitchState()

    def snapshot(self):
        return {'version':1,'firmware':FIRMWARE_COMMIT,'config':asdict(self.config),'state':asdict(self.state)}

    def restore(self, snapshot):
        if snapshot.get('version')!=1 or snapshot.get('firmware')!=FIRMWARE_COMMIT or snapshot.get('config')!=asdict(self.config):
            raise ValueError('Controller state contract mismatch')
        state=snapshot.get('state',{})
        if set(state)!={f.name for f in fields(PitchState)} or not all(math.isfinite(v) for v in state.values()):
            raise ValueError('Invalid controller state')
        if abs(state['integral'])>self.config.imax or not self.config.gain_min<=state['compression_gain']<=1 or not self.config.gain_min<=state['cached_gain']<=1 or state['energy']<0:
            raise ValueError('Controller state outside bounds')
        self.state=PitchState(**state)

    def warm_start(self, integral, airspeed_filtered):
        """Explicit initial-condition assumption. Use restore for complete history."""
        if not math.isfinite(integral) or not math.isfinite(airspeed_filtered) or abs(integral)>self.config.imax:
            raise ValueError('Invalid warm start')
        self.state=PitchState(integral=integral,airspeed_filtered=airspeed_filtered)

    def step(self, x: PitchInput):
        if not math.isfinite(x.dt_s) or not .002<=x.dt_s<=.04:
            raise ValueError('dt must be an actual controller update in [0.002,0.04] s; schedule substeps outside this class')
        if not all(math.isfinite(v) for v in [x.rate,x.rate_setpoint,x.angular_acceleration]):
            raise ValueError('Nonfinite rate input')
        if math.isnan(x.airspeed_age_s) or x.airspeed_age_s<0:
            raise ValueError('Invalid airspeed age')
        s=self.state; c=self.config; dt=x.dt_s
        if not x.rates_enabled:
            s.integral=0.; s.compression_gain=1.
            return PitchOutput(None,0.,s.cached_gain,None,None,None,False)
        speed=c.airspeed_trim
        if c.use_airspeed and math.isfinite(x.calibrated_airspeed) and x.airspeed_age_s<1:
            s.airspeed_filtered+=dt/(1+dt)*(max(.5,x.calibrated_airspeed)-s.airspeed_filtered)
            speed=s.airspeed_filtered
        scale=c.airspeed_trim/max(speed,c.airspeed_stall,.1) if c.scale_airspeed else 1.
        if x.reset_integral: s.integral=0.
        if x.landed:
            s.integral=0.; s.compression_gain=1.
        p=c.p*(x.rate_setpoint-x.rate); ff=c.ff/scale*x.rate_setpoint
        used_i=s.integral; used_gain=s.cached_gain
        effort=used_gain*(p+used_i-c.d*x.angular_acceleration+ff)*scale**2
        if not x.landed:
            error=x.rate_setpoint-x.rate
            if x.saturation_positive: error=min(error,0.)
            if x.saturation_negative: error=max(error,0.)
            factor=max(0.,1-(error/math.radians(400))**2)
            s.integral=clamp(s.integral+factor*c.i*error*dt,-c.imax,c.imax)
        if c.compress_gain:
            alpha=1/(1+2*math.pi*10*dt)
            s.highpass=alpha*(s.highpass+effort-s.previous_effort)
            s.previous_effort=effort
            s.energy+=dt/(1/(2*math.pi*5)+dt)*(s.highpass**2-s.energy)
            s.compression_gain=clamp(s.compression_gain+(-200*max(s.compression_gain-c.gain_min,0)*s.energy+.1*(1-s.compression_gain))*dt,c.gain_min,1.)
            s.cached_gain=s.compression_gain
        else:
            s.compression_gain=1.; s.cached_gain=1.
        trim=c.trim*scale**2
        if speed<c.airspeed_trim:
            trim+=c.dtrim_min*(1-clamp((speed-c.airspeed_min)/(c.airspeed_trim-c.airspeed_min),0.,1.))
        else:
            trim+=c.dtrim_max*clamp((speed-c.airspeed_trim)/(c.airspeed_max-c.airspeed_trim),0.,1.)
        output=effort+trim
        return PitchOutput(clamp(output,-1.,1.),used_i,used_gain,scale,p,ff,abs(output)>1)
