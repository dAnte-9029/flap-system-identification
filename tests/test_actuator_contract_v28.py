import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from actuator_contract_v28 import held_commands,output_pwm,allocation


def test_causal_hold_uses_both_timestamps_and_does_not_read_later_commands():
    motor={'timestamp':np.array([10,20,40]),'timestamp_sample':np.array([8,25,39]),'control[0]':np.array([.1,.2,.9])}
    servo={'timestamp':np.array([9,21,45]),'timestamp_sample':np.array([8,20,44]),**{f'control[{i}]':np.array([0.,.3,.9]) for i in range(3)}}
    q=np.array([20,30]);a,*_=held_commands(motor,servo,q)
    np.testing.assert_allclose(a,[[.1,0,0,0],[.2,.3,.3,.3]])
    motor['control[0]'][2]=999
    np.testing.assert_array_equal(a,held_commands(motor,servo,q)[0])
    with pytest.raises(ValueError):held_commands(motor,servo,np.array([0]))


def params():
    p=dict(THR_MDL_FAC=0.,PWM_MAIN_REV=17)
    for port,lo,hi in [(1,1000,2200),(2,1000,2200),(3,1100,1900),(5,800,1800)]:
        p[f'PWM_MAIN_MIN{port}']=lo;p[f'PWM_MAIN_MAX{port}']=hi
    for j,values in enumerate([[-.55,1,0],[.55,1,0],[0,0,1]]):
        for axis,value in zip(['R','P','Y'],values):p[f'CA_SV_CS{j}_TRQ_{axis}']=value
    return p


def test_pwm_mapping_motor_unsigned_servo_reverse_and_optional_center():
    p=params();a=np.array([[0.,-1,-1,-1],[.5,0,0,0],[1.,1,1,1]])
    np.testing.assert_array_equal(output_pwm(a,p),[[1100,2200,1000,1800],[1500,1600,1600,1300],[1900,1000,2200,800]])
    p['PWM_MAIN_CENT2']=1400
    assert output_pwm(a,p)[1,2]==1400


def test_allocation_common_differential_and_clipping():
    p=params();a=allocation(np.array([[.55,.4,.3],[10,10,10]]),np.array([.6,.7]),p)
    np.testing.assert_allclose(a[0],[.6,-.3,.7,.3])
    assert np.all(abs(a[:,1:])<=1)
