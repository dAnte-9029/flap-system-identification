"""Single-aircraft, CPU adapter from NWU/FLU observations to e624 cruise control.

Outer/attitude at model 50 Hz; rate/allocation at 400 Hz on held observations.
The plant receives the last inner command for its 20 ms step. This explicit
sample/hold approximation does not recreate unlogged 400 Hz sensor dynamics.
"""
import math
import numpy as np
import torch
from system_identification.control.px4_native_outer import NativeOuter
from system_identification.control.px4_flight_chain import FlightInnerLoop


class FlightStraightAdapter:
    def __init__(self, build_dir, parameters_path, *, position_n, velocity_n):
        speed=float(np.linalg.norm(velocity_n[:2]));heading=math.atan2(velocity_n[1],velocity_n[0])
        self.outer=NativeOuter(build_dir,parameters_path,origin_ne=position_n[:2],heading=heading,
                               altitude=-position_n[2],speed=speed)
        self.inner=FlightInnerLoop(self.outer.p)
        # Airspeed filter is initialized from the available observation, not zero.
        # All rate integrals/gain-compression states start at their reset values.
        for axis in self.inner.axes:axis.warm_start(0.,speed)
        if any(axis.config.d!=0 for axis in self.inner.axes):
            raise ValueError('Nonzero D requires a validated gyro derivative adapter')

    def close(self):self.outer.close()

    def compute_actions(self, *, pos_local,ground_vel_local,roll,pitch,yaw,ang_vel_body):
        if pos_local.shape!=(1,3) or pos_local.device.type!='cpu':
            raise ValueError('Flight alignment adapter requires one CPU aircraft')
        sign=np.array([1.,-1.,-1.])
        position=pos_local[0].detach().numpy()*sign
        velocity=ground_vel_local[0].detach().numpy()*sign
        rates=ang_vel_body[0].detach().numpy()*sign
        r=float(roll[0]);p=-float(pitch[0]);speed=float(np.linalg.norm(velocity[:2]))
        outer=self.outer.step(position_n=position,velocity_n=velocity,roll=r,pitch=p)
        rate_sp=self.inner.attitude.step(r,p,outer['roll_sp'],outer['pitch_sp'],speed)
        for _ in range(8):
            allocation,outputs=self.inner.step_rates(rates,rate_sp,np.zeros(3),speed,outer['motor'])
        motor,left,right,rudder=allocation.command
        actions=torch.tensor([[0.,-rudder,-(left+right)/2,-(left-right)/2]],dtype=pos_local.dtype)
        diag={'tecs_throttle_sp':pos_local.new_tensor([motor]),
              'allocation_raw':pos_local.new_tensor([[motor,*allocation.raw_surfaces]]),
              'flight_chain':dict(**outer,rate_sp=rate_sp.tolist(),torque=[o.torque for o in outputs],
                 integral=[a.state.integral for a in self.inner.axes],gain=[a.state.cached_gain for a in self.inner.axes],
                 unallocated_torque=allocation.unallocated_torque.tolist())}
        return actions,diag
