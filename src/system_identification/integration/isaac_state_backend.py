"""Main V2 state backend for an Isaac world aligned North/West/Up (NWU).

No PhysX stepping, force addition, extra actuator filtering or frequency-to-motor
conversion. Commands are model-native [motor,left,right,rudder], not the existing
Isaac physical plant's [frequency,rudder,pitch,roll] action space.
"""
from dataclasses import dataclass
import torch
from system_identification.models.main_v2_simulator import MainV2Simulator, SimulatorState
from system_identification.models.trajectory_main_v1 import _rotation_body_to_ned


@dataclass(frozen=True)
class IsaacObservation:
    position_w: torch.Tensor
    velocity_w: torch.Tensor
    quaternion_wb: torch.Tensor
    angular_velocity_b: torch.Tensor
    angular_velocity_w: torch.Tensor


def observation(state: SimulatorState, origins_w: torch.Tensor) -> IsaacObservation:
    """NED/FRD -> NWU/FLU, quaternion wxyz, world x explicitly North.

    R_new = diag(1,-1,-1) R_old diag(1,-1,-1).
    This basis change maps q to (w,x,-y,-z), NOT a one-sided rotation.
    """
    sign=state.position_n.new_tensor([1.,-1.,-1.])
    qsign=state.quaternion_nb.new_tensor([1.,1.,-1.,-1.])
    q=state.quaternion_nb*qsign
    omega=state.angular_velocity_b*sign
    omega_w=torch.bmm(_rotation_body_to_ned(q),omega.unsqueeze(-1)).squeeze(-1)
    return IsaacObservation(state.position_n*sign+origins_w,state.velocity_n*sign,q,omega,omega_w)


class IsaacLearnedStateBackend:
    """Explicit model-step clock; display/render clocks cannot advance dynamics.

    State is warm-reset from a complete model state or validated real history.
    Checkpoint identity must be supplied by its verified loading workflow.
    """
    def __init__(self, simulator: MainV2Simulator, *, checkpoint_id: str, model_dt_s: float=.02):
        if not checkpoint_id or not .015<=model_dt_s<=.025:
            raise ValueError('Explicit checkpoint identity and supported ~50 Hz model clock required')
        self.simulator=simulator
        self.checkpoint_id=checkpoint_id
        self.model_dt_s=model_dt_s
        self.state=None
        self.origins_w=None
        self.steps=0

    def reset(self, *, state: SimulatorState, origins_w: torch.Tensor | None=None):
        clone=self.simulator.reset(state=state)
        origins=torch.zeros_like(clone.position_n) if origins_w is None else origins_w.clone()
        if origins.shape!=clone.position_n.shape or origins.device!=clone.position_n.device or origins.dtype!=clone.position_n.dtype or not torch.isfinite(origins).all():
            raise ValueError('Origins must match state batch/device/dtype and be finite')
        self.state=clone;self.origins_w=origins;self.steps=0
        return self.observe()

    def observe(self):
        if self.state is None:raise RuntimeError('Warm reset required')
        return observation(self.state,self.origins_w)

    @torch.inference_mode()
    def step(self, command: torch.Tensor):
        if self.state is None:raise RuntimeError('Warm reset required')
        if command.device!=self.state.position_n.device or command.dtype!=self.state.position_n.dtype:
            raise ValueError('Command device/dtype must match model state')
        if command.shape!=(len(self.state.position_n),4) or not torch.isfinite(command).all():
            raise ValueError('Expected finite [motor,left,right,rudder] commands')
        if torch.any((command[:,0]<0)|(command[:,0]>1)) or torch.any(command[:,1:].abs()>1):
            raise ValueError('Motor [0,1], surface commands [-1,1]; no silent clipping')
        dt=command.new_full((len(command),),self.model_dt_s)
        self.state,diag=self.simulator.step(self.state,command,dt)
        self.steps+=1
        return self.observe(),diag

    def snapshot(self):
        self.observe()
        return dict(version=1,checkpoint_id=self.checkpoint_id,model_dt_s=self.model_dt_s,steps=self.steps,
                    state=self.state.snapshot(),origins_w=self.origins_w.detach().cpu().clone())

    def restore(self, saved, *, device='cpu'):
        if saved.get('version')!=1 or saved.get('checkpoint_id')!=self.checkpoint_id or saved.get('model_dt_s')!=self.model_dt_s:
            raise ValueError('Backend snapshot contract mismatch')
        if not isinstance(saved['steps'],int) or saved['steps']<0:raise ValueError('Invalid step counter')
        self.reset(state=SimulatorState.from_snapshot(saved['state'],device=device),origins_w=saved['origins_w'].to(device))
        self.steps=saved['steps']
        return self.observe()
