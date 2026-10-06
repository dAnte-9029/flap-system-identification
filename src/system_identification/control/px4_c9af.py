"""c9af fixed-wing component arithmetic, explicit state, NED/FRD.

Source snapshots: control_windows_v37/source. Finite-input, non-VTOL attitude
control only; no autotune, wheel, battery scaling, manual-direct or navigation.
Scheduling/subscription delivery is the caller's explicit approximation.
Float64 implementation, not a bit-exact firmware emulator.
"""
from dataclasses import dataclass, field, asdict
import copy
import math
import numpy as np

FIRMWARE = 'c9af571575af33d5403fbb479256848797c801e2'


def clip(x, lo, hi): return min(max(x, lo), hi)


@dataclass
class ControllerState:
    integral: list = field(default_factory=lambda: [0.]*3)
    gain: list = field(default_factory=lambda: [1.]*3)
    cached_gain: list = field(default_factory=lambda: [1.]*3)
    hpf: list = field(default_factory=lambda: [0.]*3)
    energy: list = field(default_factory=lambda: [0.]*3)
    previous_effort: list = field(default_factory=lambda: [0.]*3)
    positive: list = field(default_factory=lambda: [False]*3)
    negative: list = field(default_factory=lambda: [False]*3)
    airspeed_filter: float = 0.
    yaw_euler: float = 0.
    yaw_body: float = 0.
    last_rate_sp: list = field(default_factory=lambda: [0.]*3)
    unallocated: list = field(default_factory=lambda: [0.]*3)


class C9Controller:
    def __init__(self, parameters):
        self.p = dict(parameters); p = self.p; self.state = ControllerState()
        if p['FW_BAT_SCALE_EN'] or p['FW_W_EN']:
            raise ValueError('Unsupported battery/wheel path')
        if p['CA_AIRFRAME'] != 1 or p['CA_METHOD'] != 2 or p['CA_SV_CS_COUNT'] != 3:
            raise ValueError('Unsupported allocation')
        for j in range(3):
            if any(p[f'CA_SV_CS{j}_{k}'] != 0 for k in ['TRIM','FLAP','SPOIL']) or p[f'CA_SV{j}_SLEW'] != 0:
                raise ValueError('Unsupported surface trim/slew')
        self.eff = np.array([[p[f'CA_SV_CS{j}_TRQ_{a}'] for j in range(3)] for a in 'RPY'])
        self.mix = np.linalg.pinv(self.eff)
        self.gains = [[p[f'FW_{a}R_{k}'] for k in ['P','I','D','FF','IMAX']] for a in 'RPY']
        self.trims = [[p[f'TRIM_{name}'],p[f'FW_DTRIM_{a}_VMIN'],p[f'FW_DTRIM_{a}_VMAX']]
                      for a,name in zip('RPY',['ROLL','PITCH','YAW'])]

    def snapshot(self):
        return dict(version=1, firmware=FIRMWARE, parameters=copy.deepcopy(self.p), state=asdict(self.state))

    def restore(self, value):
        if value.get('version') != 1 or value.get('firmware') != FIRMWARE or value.get('parameters') != self.p:
            raise ValueError('State contract mismatch')
        d=copy.deepcopy(value['state'])
        if set(d)!=set(asdict(ControllerState())): raise ValueError('Incomplete state')
        for k,v in d.items():
            expected=asdict(ControllerState())[k]
            if isinstance(expected,list) and (not isinstance(v,list) or len(v)!=3):raise ValueError('State shape')
            if not np.isfinite(v).all():raise ValueError('Nonfinite state')
        for i,g in enumerate(self.gains):
            if abs(d['integral'][i])>g[4] or d['energy'][i]<0:raise ValueError('State bounds')
            if not self.p['FW_GC_GAIN_MIN']<=d['gain'][i]<=1 or not self.p['FW_GC_GAIN_MIN']<=d['cached_gain'][i]<=1:raise ValueError('Gain bounds')
        self.state=ControllerState(**d)

    def attitude(self, roll, pitch, roll_sp, pitch_sp, airspeed, air_age,
                 manual=False, yaw_stick=0.):
        if not all(math.isfinite(v) for v in [roll,pitch,roll_sp,pitch_sp,yaw_stick]):raise ValueError('Nonfinite attitude input')
        p=self.p;s=self.state;pe=(pitch_sp-pitch)/p['FW_P_TC'];old_yaw=s.yaw_euler
        pr=clip((roll_sp-roll)/p['FW_R_TC']-math.sin(pitch)*old_yaw,-math.radians(p['FW_R_RMAX']),math.radians(p['FW_R_RMAX']))
        qr=clip(math.cos(roll)*pe+math.cos(pitch)*math.sin(roll)*old_yaw,-math.radians(p['FW_P_RMAX_NEG']),math.radians(p['FW_P_RMAX_POS']))
        speed=airspeed if p['FW_USE_AIRSPD'] and math.isfinite(airspeed) and air_age<1 else p['FW_AIRSPD_TRIM']
        speed=clip(max(.5,speed),p['FW_AIRSPD_STALL'],p['FW_AIRSPD_MAX'])
        if abs(roll)<math.pi/2:
            rc=clip(clip(roll,-math.radians(80),math.radians(80)),-abs(roll_sp),abs(roll_sp))
            s.yaw_euler=math.tan(rc)*math.cos(pitch)*9.80665/speed
            s.yaw_body=clip(-math.sin(roll)*pe+math.cos(roll)*math.cos(pitch)*s.yaw_euler,-math.radians(p['FW_Y_RMAX']),math.radians(p['FW_Y_RMAX']))
        yr=s.yaw_body
        if manual:yr+=clip(yaw_stick*math.radians(p['FW_MAN_YR_MAX']),-math.radians(p['FW_Y_RMAX']),math.radians(p['FW_Y_RMAX']))
        s.last_rate_sp=[pr,qr,yr];return s.last_rate_sp.copy()

    def rate(self, rates, setpoints, acceleration, airspeed, air_age, motor, dt,
             landed=False, reset_integral=False, feedback='source_no_interleave'):
        if not .002<=dt<=.04:raise ValueError('dt outside source bounds')
        if not np.isfinite([*rates,*setpoints,*acceleration,motor]).all():raise ValueError('Nonfinite rate input')
        if feedback not in ['source_no_interleave','delayed_allocation']:raise ValueError('feedback policy')
        p=self.p;s=self.state
        # Both source subscriptions refer to index0 in non-VTOL. The first update
        # consumes it for differential-thrust (disabled here); without a new
        # publication between calls, the surface update sees no new message.
        if feedback=='delayed_allocation':
            s.positive=[v>np.finfo(np.float32).eps for v in s.unallocated]
            s.negative=[v<-np.finfo(np.float32).eps for v in s.unallocated]
        speed=p['FW_AIRSPD_TRIM']
        if p['FW_USE_AIRSPD'] and math.isfinite(airspeed) and air_age<1:
            s.airspeed_filter+=dt/(1+dt)*(max(.5,airspeed)-s.airspeed_filter)
            speed=s.airspeed_filter
        scale=p['FW_AIRSPD_TRIM']/max(speed,p['FW_AIRSPD_STALL'],.1) if p['FW_ARSP_SCALE_EN'] else 1.
        if landed or reset_integral:s.integral=[0.]*3
        if landed:s.gain=[1.]*3  # reset() does not clear cached gains or filters.
        torque=[];efforts=[];used_i=s.integral.copy();used_g=s.cached_gain.copy()
        for j,((kp,ki,kd,kff,imax),(trim,tmin,tmax)) in enumerate(zip(self.gains,self.trims)):
            error=setpoints[j]-rates[j]
            total=kp*error+s.integral[j]-kd*acceleration[j]+kff/scale*setpoints[j]
            effort=s.cached_gain[j]*total*scale*scale;efforts.append(effort)
            if not landed:
                if s.positive[j]:error=min(error,0.)
                if s.negative[j]:error=max(error,0.)
                factor=max(0.,1-(error/math.radians(400))**2)
                s.integral[j]=clip(s.integral[j]+factor*ki*error*dt,-imax,imax)
            if p['FW_GC_EN']:
                alpha=1/(1+2*math.pi*10*dt)
                s.hpf[j]=alpha*s.hpf[j]+alpha*(effort-s.previous_effort[j])
                s.energy[j]+=dt/(1/(2*math.pi*5)+dt)*(s.hpf[j]**2-s.energy[j])
                s.previous_effort[j]=effort
                s.gain[j]=clip(s.gain[j]+(-200*max(s.gain[j]-p['FW_GC_GAIN_MIN'],0)*s.energy[j]+.1*(1-s.gain[j]))*dt,p['FW_GC_GAIN_MIN'],1)
                s.cached_gain[j]=s.gain[j]
            else:s.gain[j]=s.cached_gain[j]=1.
            trim*=scale*scale
            if speed<p['FW_AIRSPD_TRIM']:trim+=tmin*(1-clip((speed-p['FW_AIRSPD_MIN'])/(p['FW_AIRSPD_TRIM']-p['FW_AIRSPD_MIN']),0,1))
            else:trim+=tmax*clip((speed-p['FW_AIRSPD_TRIM'])/(p['FW_AIRSPD_MAX']-p['FW_AIRSPD_TRIM']),0,1)
            torque.append(clip(effort+trim,-1,1))
        torque[2]=clip(torque[2]+p['FW_RLL_TO_YAW_FF']*torque[0],-1,1)
        raw=self.mix@torque;surfaces=np.clip(raw,-1,1)
        s.unallocated=(np.asarray(torque)-self.eff@surfaces).tolist()
        return dict(torque=torque,commands=[clip(motor,0,1),*surfaces],raw_surfaces=raw.tolist(),
                    integral_used=used_i,gain_used=used_g,effort=efforts,airspeed_scale=scale)
