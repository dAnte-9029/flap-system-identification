"""Native e624 outer libraries with an explicit straight-cruise wrapper.

Zero wind, sea-level density, no flaps, weight ratio one, valid navigation.
Not a flight-mode manager: takeoff/landing/manual transitions are outside scope.
"""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .px4_pitch import FIRMWARE_COMMIT, clamp


class NativeOuter:
    def __init__(self, build_dir, parameters_path, *, origin_ne, heading, altitude, speed):
        build_dir, parameters_path = Path(build_dir), Path(parameters_path)
        manifest = json.loads((build_dir/'manifest.json').read_text())
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if manifest['firmware_commit'] != FIRMWARE_COMMIT or digest(parameters_path) != manifest['parameter_sha256']:
            raise ValueError('Native library flight configuration mismatch')
        for name in ['libpx4_outer.so','wrapper.cpp']:
            if digest(build_dir/name) != manifest['sources'][name]:
                raise ValueError('Native library hash mismatch')
        self.p = json.loads(parameters_path.read_text())
        p=self.p
        if p['WEIGHT_BASE']>0 or p['WEIGHT_GROSS']>0 or p['FW_SERVICE_CEIL']>0 or p['FW_WIND_ARSP_SC']!=0 or p['FW_LND_THRTC_SC']!=1:
            raise ValueError('Configuration outside audited straight-cruise wrapper')
        self.lib = ctypes.CDLL(str(build_dir/'libpx4_outer.so'))
        self.lib.create_chain.restype = ctypes.c_void_p
        self.lib.destroy_chain.argtypes = [ctypes.c_void_p]
        fp = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
        self.lib.outer_step.argtypes = [ctypes.c_void_p,ctypes.c_uint64,fp,fp]
        self.lib.outer_step.restype = None
        self.ptr = self.lib.create_chain()
        self.origin = np.asarray(origin_ne,dtype=float)
        self.tangent = np.array([math.cos(heading),math.sin(heading)])
        self.altitude, self.speed = altitude,speed
        self.airspeed_sp = None
        self.roll_sp = None
        self.time_us = 1000000
        self.tecs_min_speed = p['FW_AIRSPD_MIN']  # parameters updated before cruise, bank=0

    def close(self):
        if self.ptr:
            self.lib.destroy_chain(self.ptr);self.ptr=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def step(self, *, position_n, velocity_n, roll, pitch, dt=.02):
        if self.ptr is None:raise RuntimeError('Controller is closed')
        if not 0.001<=dt<=.1:raise ValueError('Outer update outside PX4 clock range')
        p=self.p
        speed=float(np.linalg.norm(velocity_n[:2])) # declared ideal horizontal CAS, zero wind
        load=1/max(math.cos(roll),np.finfo(np.float32).eps)
        low=p['FW_AIRSPD_MIN']*math.sqrt(load);high=p['FW_AIRSPD_MAX']
        target=clamp(max(self.speed,p['FW_GND_SPD_MIN']),low,high)
        if self.airspeed_sp is None:self.airspeed_sp=clamp(speed,low,high)
        elif self.airspeed_sp<low:self.airspeed_sp=low
        elif self.airspeed_sp>high:self.airspeed_sp=high
        else:self.airspeed_sp+=clamp(target-self.airspeed_sp,-dt,dt)
        trim=p['FW_THR_TRIM']
        if self.airspeed_sp<p['FW_AIRSPD_TRIM'] and p['FW_THR_ASPD_MIN']>np.finfo(np.float32).eps:
            trim-=(p['FW_THR_TRIM']-p['FW_THR_ASPD_MIN'])/(p['FW_AIRSPD_TRIM']-p['FW_AIRSPD_MIN'])*(p['FW_AIRSPD_TRIM']-self.airspeed_sp)
        elif self.airspeed_sp>p['FW_AIRSPD_TRIM'] and p['FW_THR_ASPD_MAX']>np.finfo(np.float32).eps:
            trim+=(p['FW_THR_ASPD_MAX']-p['FW_THR_TRIM'])/(p['FW_AIRSPD_MAX']-p['FW_AIRSPD_TRIM'])*(self.airspeed_sp-p['FW_AIRSPD_TRIM'])
        offset=math.radians(p['FW_PSP_OFF'])
        x=np.array([pitch-offset,-position_n[2],self.altitude,self.airspeed_sp,speed,1.,
             p['FW_THR_MIN'],p['FW_THR_MAX'],clamp(trim,p['FW_THR_MIN'],p['FW_THR_MAX']),
             math.radians(p['FW_P_LIM_MIN'])-offset,math.radians(p['FW_P_LIM_MAX'])-offset,
             p['FW_T_CLMB_R_SP'],p['FW_T_SINK_R_SP'],-velocity_n[2],load,self.tecs_min_speed,
             *position_n[:2],*velocity_n[:2],0.,0.,*self.tangent,*self.origin],dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite outer state')
        self.time_us+=round(dt*1e6)
        y=np.zeros(7,dtype=np.float32);self.lib.outer_step(self.ptr,self.time_us,x,y)
        if not np.isfinite(y).all():raise RuntimeError('Nonfinite native output')
        desired=clamp(math.atan(float(y[2])/9.80665),-math.radians(p['FW_R_LIM']),math.radians(p['FW_R_LIM']))
        # SlewRate starts NaN in PX4: first finite setpoint is accepted as-is.
        if self.roll_sp is None:self.roll_sp=desired
        else:self.roll_sp+=clamp(desired-self.roll_sp,-math.radians(p['FW_PN_R_SLEW_MAX'])*dt,math.radians(p['FW_PN_R_SLEW_MAX'])*dt)
        return dict(roll_sp=self.roll_sp,pitch_sp=float(y[0])+offset,motor=float(y[1]),
                    course_sp=float(y[3]),npfg_period=float(y[4]),tecs_filtered_speed=float(y[5]),
                    altitude_reference=float(y[6]),airspeed_sp=self.airspeed_sp)
