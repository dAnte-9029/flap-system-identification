import json
from pathlib import Path
import numpy as np
import pytest
from system_identification.control.px4_flight_chain import FlightAllocator,FlightAttitudeController,FlightInnerLoop
from system_identification.control.px4_native_outer import NativeOuter
ROOT=Path(__file__).resolve().parents[1]
PARAMS=ROOT/'docs/analysis/results/flight_chain_alignment/flight_parameters.json'

def parameters():return json.loads(PARAMS.read_text())

def test_allocator_preserves_axes_and_reports_lost_pitch_authority():
    a=FlightAllocator(parameters())
    x=a.allocate([0,1,0],.7)
    np.testing.assert_allclose(x.command,[.7,.5,.5,0],atol=1e-7)
    np.testing.assert_allclose(x.unallocated_torque,0,atol=1e-7)
    x=a.allocate([1,1,0],.7)
    assert x.raw_surfaces[1]>1 and x.unallocated_torque[1]>0
    assert x.unallocated_torque[0]>0

def test_allocation_feedback_blocks_pitch_windup_then_allows_release():
    p=parameters();p.update(FW_USE_AIRSPD=0,FW_ARSP_SCALE_EN=0,FW_GC_EN=0)
    loop=FlightInnerLoop(p)
    for _ in range(10):loop.step_rates([0,0,0],[2,2,0],[0,0,0],8,.7)
    before=loop.axes[1].state.integral
    for _ in range(100):loop.step_rates([0,0,0],[2,2,0],[0,0,0],8,.7)
    assert loop.axes[1].state.integral==before
    loop.step_rates([0,0,0],[0,-.3,0],[0,0,0],8,.7)
    assert loop.axes[1].state.integral<before

def test_attitude_coordinates_and_previous_yaw_state():
    c=FlightAttitudeController(parameters())
    assert c.step(0,0,0,.1,8)[1]==pytest.approx(.25)
    first=c.step(.1,.2,.2,.2,8)
    second=c.step(.1,.2,.2,.2,8)
    assert second[0]<first[0] and second[1]>first[1]

def test_unsupported_b2b_fails_instead_of_silently_changing_mode():
    p=parameters();p['FLAP_SLOW_EN']=1
    with pytest.raises(ValueError,match='B2b'):FlightInnerLoop(p)

def test_native_outer_determinism_limits_and_heading_correction():
    build=ROOT/'artifacts/px4_e624_outer'
    if not (build/'libpx4_outer.so').exists():pytest.skip('Build native library with scripts/build_px4_native_outer.py')
    args=dict(origin_ne=[0,0],heading=0,altitude=20,speed=8)
    def run():
        with NativeOuter(build,PARAMS,**args) as c:
            return [c.step(position_n=[0,5,-20],velocity_n=[8,0,0],roll=0,pitch=np.deg2rad(15)) for _ in range(150)]
    a,b=run(),run();assert a==b
    assert all(-np.deg2rad(30)<=x['roll_sp']<0 for x in a)
    assert all(np.deg2rad(-5)-1e-6<=x['pitch_sp']<=np.deg2rad(30)+1e-6 for x in a)
    assert all(.1-1e-6<=x['motor']<=.98+1e-6 for x in a)

def test_flight_adapter_native_command_and_reset_are_deterministic():
    import torch
    from system_identification.integration.px4_flight_straight import FlightStraightAdapter
    build=ROOT/'artifacts/px4_e624_outer'
    if not (build/'libpx4_outer.so').exists():pytest.skip('Native build required')
    def run():
        c=FlightStraightAdapter(build,PARAMS,position_n=np.array([0.,0.,-20.]),velocity_n=np.array([8.,0.,0.]))
        try:
            result=[]
            for _ in range(20):
                action,diag=c.compute_actions(pos_local=torch.tensor([[0.,-2.,20.]]),ground_vel_local=torch.tensor([[8.,0.,0.]]),
                    roll=torch.tensor([.05]),pitch=torch.tensor([-.2]),yaw=torch.tensor([0.]),ang_vel_body=torch.zeros((1,3)))
                # The legacy action transport must preserve all four native commands.
                common=-action[0,2].item();diff=-action[0,3].item()
                command=np.array([diag['tecs_throttle_sp'].item(),common+diff,common-diff,-action[0,1].item()])
                expected=FlightAllocator(parameters()).allocate(diag['flight_chain']['torque'],diag['tecs_throttle_sp'].item()).command
                np.testing.assert_allclose(command,expected,atol=1e-7)
                result.append((action.tolist(),diag['flight_chain']))
            return result
        finally:c.close()
    assert run()==run()
