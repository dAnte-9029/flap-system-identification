"""Physical sign and throttle contracts for the straight-line adapter."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_isaac_surrogate_straight import native_command,euler


def test_nose_up_flu_command_increases_both_elevons():
    u,_=native_command(torch.tensor([[0.,0.,-.2,0.]]),torch.tensor([.6]))
    torch.testing.assert_close(u,torch.tensor([[.6,.2,.2,0.]]))


def test_roll_and_yaw_frd_effectiveness_signs():
    u,_=native_command(torch.tensor([[0.,-.3,0.,.2]]),torch.tensor([.5]))
    assert float(-.55*u[0,1]+.55*u[0,2])>0
    assert float(u[0,3])>0  # negative FLU yaw -> positive FRD yaw


def test_throttle_is_tecs_effort_and_mix_is_limited():
    u,raw=native_command(torch.tensor([[-1.,0.,-1.,1.]]),torch.tensor([.7]))
    assert abs(float(u[0,0])-.7)<1e-6
    assert float(raw[0,2])==2 and float(u[0,2])==1


def test_flu_nose_up_quaternion_has_negative_pitch():
    q=torch.tensor([[.995004165,0.,-.099833417,0.]])
    _,pitch,_=euler(q)
    torch.testing.assert_close(pitch,torch.tensor([-.2]))
