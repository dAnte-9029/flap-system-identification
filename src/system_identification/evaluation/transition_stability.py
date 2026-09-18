"""Local state-chart diagnostics; neither physical safety nor asymptotic proofs.

State chart: p_NED, v_NED, body-right SO(3) rotation, omega_body, f, phi.
Hidden, actuator proxies and immutable phase anchor are held fixed in J_x.
"""
from dataclasses import replace
import numpy as np
import torch
from system_identification.models.trajectory_main_v1 import _quaternion_multiply, _normalize_quaternion

CHART_SCALES=(1.,1.,1.,2.,2.,2.,.35,.35,.35,2.,2.,2.,3.,1.)


def rotation_exp(vector):
    angle=torch.linalg.vector_norm(vector,dim=-1,keepdim=True)
    return torch.cat((torch.cos(angle/2),.5*torch.sinc(angle/(2*torch.pi))*vector),-1)


def rotation_log(quaternion):
    q=_normalize_quaternion(quaternion)
    q=torch.where(q[:,:1]<0,-q,q)
    norm=torch.linalg.vector_norm(q[:,1:],dim=-1,keepdim=True)
    factor=2*torch.atan2(norm,q[:,:1])/norm.clamp_min(1e-12)
    return q[:,1:]*torch.where(norm<1e-7,2*torch.ones_like(norm),factor)


def perturb_state(state,normalized_delta):
    d=normalized_delta*normalized_delta.new_tensor(CHART_SCALES)
    return replace(state,position_n=state.position_n+d[:,:3],velocity_n=state.velocity_n+d[:,3:6],
        quaternion_nb=_normalize_quaternion(_quaternion_multiply(state.quaternion_nb,rotation_exp(d[:,6:9]))),
        angular_velocity_b=state.angular_velocity_b+d[:,9:12],flap_frequency_hz=state.flap_frequency_hz+d[:,12],
        relative_phase_rad=torch.remainder(state.relative_phase_rad+d[:,13],2*torch.pi))


def state_difference(state,reference):
    conjugate=reference.quaternion_nb*reference.quaternion_nb.new_tensor([1.,-1.,-1.,-1.])
    rotation=rotation_log(_quaternion_multiply(conjugate,state.quaternion_nb))
    phase=state.relative_phase_rad-reference.relative_phase_rad
    values=torch.cat((state.position_n-reference.position_n,state.velocity_n-reference.velocity_n,rotation,
        state.angular_velocity_b-reference.angular_velocity_b,(state.flap_frequency_hz-reference.flap_frequency_hz)[:,None],
        torch.atan2(torch.sin(phase),torch.cos(phase))[:,None]),-1)
    return values/values.new_tensor(CHART_SCALES)


def physical_jacobian(simulator,state,command,dt,epsilon=1e-3):
    """Central differences in dimensionless tangent coordinates, native dt.

    Full 14x14 J contains position neutral modes. Also report the 11x11 block
    without position. This partial Jacobian omits hidden/proxy feedback.
    """
    with torch.no_grad():
        reference,_=simulator.step(state,command,dt)
        columns=[]
        for j in range(14):
            delta=state.position_n.new_zeros((len(command),14));delta[:,j]=epsilon
            plus,_=simulator.step(perturb_state(state,delta),command,dt)
            minus,_=simulator.step(perturb_state(state,-delta),command,dt)
            columns.append((state_difference(plus,reference)-state_difference(minus,reference))/(2*epsilon))
    return torch.stack(columns,-1)


def jacobian_spectra(matrix):
    a=np.asarray(matrix,dtype=np.float64)
    return dict(largest_singular=float(np.linalg.svd(a,compute_uv=False)[0]),
                spectral_radius=float(np.abs(np.linalg.eigvals(a)).max()),
                dynamic_largest_singular=float(np.linalg.svd(a[3:,3:],compute_uv=False)[0]),
                dynamic_spectral_radius=float(np.abs(np.linalg.eigvals(a[3:,3:])).max()),
                rigid_largest_singular=float(np.linalg.svd(a[3:12,3:12],compute_uv=False)[0]),
                rigid_spectral_radius=float(np.abs(np.linalg.eigvals(a[3:12,3:12])).max()))
