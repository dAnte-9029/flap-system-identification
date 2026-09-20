"""e624 normal fixed-wing attitude/rate/allocation chain in NED/FRD.

Baseline A only (FLAP_SLOW_EN=0); no manual inputs, autotune, VTOL or wheel.
Sensor rates/derivatives are supplied by the caller, not reconstructed here.
"""
from dataclasses import dataclass
import math
import numpy as np
from .px4_pitch import PitchConfig, PitchInput, PX4PitchRateController, clamp


@dataclass(frozen=True)
class Allocation:
    command: np.ndarray  # motor, left elevon, right elevon, rudder (before PWM)
    raw_surfaces: np.ndarray
    unallocated_torque: np.ndarray


class FlightAllocator:
    def __init__(self, parameters):
        p = parameters
        if p['CA_AIRFRAME'] != 1 or p['CA_METHOD'] != 2 or p['CA_SV_CS_COUNT'] != 3:
            raise ValueError('Only audited fixed-wing AUTO allocation is supported')
        for j in range(3):
            if any(p[f'CA_SV_CS{j}_{suffix}'] != 0 for suffix in ['TRIM','FLAP','SPOIL']) or p[f'CA_SV{j}_SLEW'] != 0:
                raise ValueError('Nonzero trim/flap/spoiler/slew requires another allocation contract')
        self.effectiveness = np.array([[p[f'CA_SV_CS{j}_TRQ_{axis}'] for j in range(3)] for axis in 'RPY'])
        # Fixed-wing AUTO uses an unnormalized pseudo-inverse.
        self.mix = np.linalg.pinv(self.effectiveness)

    def allocate(self, torque, motor):
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (3,) or not np.all(np.isfinite(torque)) or not math.isfinite(motor):
            raise ValueError('Invalid actuator demand')
        raw = self.mix @ torque
        surfaces = raw.clip(-1, 1)
        return Allocation(np.r_[clamp(motor, 0, 1), surfaces], raw,
                          torque - self.effectiveness @ surfaces)


class FlightAttitudeController:
    def __init__(self, parameters):
        self.p = dict(parameters)
        self.yaw_euler = self.yaw_body = 0.

    def step(self, roll, pitch, roll_sp, pitch_sp, airspeed, airspeed_age=0.):
        if not all(math.isfinite(x) for x in [roll,pitch,roll_sp,pitch_sp]):
            raise ValueError('Invalid attitude')
        p = self.p
        pitch_euler = (pitch_sp-pitch)/p['FW_P_TC']
        pr = clamp((roll_sp-roll)/p['FW_R_TC']-math.sin(pitch)*self.yaw_euler,
                   -math.radians(p['FW_R_RMAX']), math.radians(p['FW_R_RMAX']))
        qr = clamp(math.cos(roll)*pitch_euler+math.cos(pitch)*math.sin(roll)*self.yaw_euler,
                   -math.radians(p['FW_P_RMAX_NEG']), math.radians(p['FW_P_RMAX_POS']))
        if abs(roll) < math.pi/2:
            rc = clamp(clamp(roll, -math.radians(80), math.radians(80)), -abs(roll_sp), abs(roll_sp))
            speed = airspeed if p['FW_USE_AIRSPD'] and math.isfinite(airspeed) and airspeed_age < 1 else p['FW_AIRSPD_TRIM']
            speed = clamp(max(.5,speed),p['FW_AIRSPD_STALL'],p['FW_AIRSPD_MAX'])
            self.yaw_euler = math.tan(rc)*math.cos(pitch)*9.80665/speed
            self.yaw_body = clamp(-math.sin(roll)*pitch_euler+math.cos(roll)*math.cos(pitch)*self.yaw_euler,
                                  -math.radians(p['FW_Y_RMAX']), math.radians(p['FW_Y_RMAX']))
        return np.array([pr,qr,self.yaw_body])


class FlightInnerLoop:
    def __init__(self, parameters):
        p = dict(parameters)
        if p['FLAP_SLOW_EN'] != 0:
            raise ValueError('B2b roll integral transfer is not supported by baseline A')
        if p['FW_BAT_SCALE_EN'] != 0:
            raise ValueError('Battery throttle scaling requires battery observations')
        self.attitude = FlightAttitudeController(p)
        self.allocator = FlightAllocator(p)
        self.axes = []
        for axis,name in zip('RPY',['ROLL','PITCH','YAW']):
            c = PitchConfig(*(float(p[f'FW_{axis}R_{k}']) for k in ['P','I','D','FF','IMAX']),
                *(float(p[f'FW_AIRSPD_{k}']) for k in ['TRIM','STALL','MIN','MAX']),
                *(bool(p[k]) for k in ['FW_USE_AIRSPD','FW_ARSP_SCALE_EN','FW_GC_EN']),
                float(p['FW_GC_GAIN_MIN']),float(p[f'TRIM_{name}']),
                float(p[f'FW_DTRIM_{axis}_VMIN']),float(p[f'FW_DTRIM_{axis}_VMAX']))
            self.axes.append(PX4PitchRateController(c))
        self.unallocated = np.zeros(3)

    def step_rates(self, rates, rate_sp, acceleration, airspeed, motor, dt=.0025,
                   airspeed_age=0., landed=False, reset_integral=False):
        # PX4 consumes allocator feedback from the preceding completed allocation.
        outputs = [c.step(PitchInput(float(rates[j]),float(rate_sp[j]),float(acceleration[j]),dt,
                    airspeed,airspeed_age,landed=landed,reset_integral=reset_integral,
                    saturation_positive=self.unallocated[j]>np.finfo(np.float32).eps,
                    saturation_negative=self.unallocated[j]<-np.finfo(np.float32).eps)) for j,c in enumerate(self.axes)]
        allocation = self.allocator.allocate([o.torque for o in outputs],motor)
        self.unallocated = allocation.unallocated_torque.copy()
        return allocation, outputs
