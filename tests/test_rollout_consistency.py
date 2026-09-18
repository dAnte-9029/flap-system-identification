from dataclasses import fields,replace
import numpy as np
import torch
from test_main_v2_simulator import make_simulator,make_inputs,initial
from system_identification.evaluation.transition_stability import perturb_state,state_difference,physical_jacobian,jacobian_spectra


def test_state_chart_rotation_phase_and_nonphysical_state_are_preserved():
    sim,x=make_simulator(),make_inputs();s=initial(sim,x)
    s=replace(s,relative_phase_rad=torch.tensor([2*torch.pi-1e-4,1e-4]))
    d=torch.zeros((2,14));d[:,6:9]=torch.tensor([.02,-.03,.01]);d[:,13]=torch.tensor([.02,-.02]);d[:,3]=.05
    p=perturb_state(s,d)
    torch.testing.assert_close(state_difference(p,s),d,atol=2e-6,rtol=1e-4)
    for name in ('gru_hidden','drive_state','tail_state','phase_anchor'):
        torch.testing.assert_close(getattr(p,name),getattr(s,name),rtol=0,atol=0)
    q=replace(p,quaternion_nb=-p.quaternion_nb)
    torch.testing.assert_close(state_difference(q,s),d,atol=2e-6,rtol=1e-4)


def test_jacobian_preserves_translation_neutral_modes_and_is_finite():
    sim,x=make_simulator(),make_inputs();s=initial(sim,x)
    jac=physical_jacobian(sim,s,x['future_controls'][:,0],x['dt_s'][:,0])
    assert jac.shape==(2,14,14) and torch.isfinite(jac).all()
    torch.testing.assert_close(jac[:,:3,:3],torch.eye(3).expand(2,3,3),atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(jac[:,3:,:3],torch.zeros((2,11,3)),atol=1e-6,rtol=0)
    for item in jac.numpy(): assert jacobian_spectra(item)['spectral_radius']>=.999


def test_short_prefix_loss_ignores_labels_after_horizon_and_quaternion_sign():
    from system_identification.models.trajectory_main_v1 import TorchTrajectoryPrediction
    from system_identification.training.rollout_consistency import short_prefix_loss
    sim,x=make_simulator(),make_inputs()
    with torch.no_grad():p=sim.model(**x)
    values={f.name:getattr(p,f.name).clone() for f in fields(p)}
    values['velocity_n'][:,1:21]+=.1;values['angular_velocity_b'][:,1:21]+=.1
    truth=TorchTrajectoryPrediction(**values)
    for h in (5,10,20):
        expected=short_prefix_loss(p,truth,h)
        poison={k:v.clone() for k,v in values.items()}
        for k in poison:poison[k][:,h+1:]=float('nan')
        poison['quaternion_nb']=-poison['quaternion_nb']
        actual=short_prefix_loss(p,TorchTrajectoryPrediction(**poison),h)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)


def test_zero_added_objective_is_exact_legacy_and_closed_loop_gradient_reaches_earlier_steps():
    from system_identification.training.rollout_consistency import consistency_objective,short_prefix_loss
    from system_identification.training.trajectory_main_v1 import trajectory_rollout_loss
    sim,x=make_simulator(),make_inputs()
    p=sim.model(**x)
    truth=replace(p,velocity_n=p.velocity_n.detach()+.1,angular_velocity_b=p.angular_velocity_b.detach()+.1,
                  quaternion_nb=p.quaternion_nb.detach())
    torch.testing.assert_close(consistency_objective(p,truth),trajectory_rollout_loss(p,truth,objective_steps=50),rtol=0,atol=0)
    # Actuator wrapper freezes the base by contract: test recurrent gradients on base alone.
    base=sim.model.base_model
    for parameter in base.parameters():parameter.requires_grad_(True)
    with torch.no_grad():base.derivative_head[-1].weight.normal_(0,.01)
    p=base(**x);p.velocity_n.retain_grad()
    target=replace(p,velocity_n=p.velocity_n.detach()+.1,angular_velocity_b=p.angular_velocity_b.detach()+.1,quaternion_nb=p.quaternion_nb.detach())
    loss=short_prefix_loss(p,target,20);loss.backward()
    assert torch.isfinite(base.recurrent_cell.weight_hh.grad).all()
    assert base.recurrent_cell.weight_hh.grad.abs().sum()>0
    assert p.velocity_n.grad[:,1:21].abs().sum()>0
