import hashlib
import numpy as np
import pytest
import torch

from system_identification.integration.control_dynamics_candidate import JointSupport,load_candidate,support_vector
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel


def test_support_checks_joint_distance_not_only_box():
    # Training joint support lies near the diagonal despite wide marginal bounds.
    ref=np.array([[-1.]*18,[1.]*18])
    s=JointSupport(ref,np.zeros(18),np.ones(18),-np.ones(18),np.ones(18),1.)
    query=np.array([[1.]*18,[-1.,1.]*9,[np.nan]*18])
    out=s.query(query)
    np.testing.assert_array_equal(out['accepted'],[True,False,False])
    np.testing.assert_array_equal(out['box'],[True,True,False])


def test_support_ignores_arbitrary_phase_anchor_and_keeps_command_change():
    f=np.zeros((2,12));f[:,9:11]=[[0,1],[1,0]]
    command=np.array([[.5,.2,.2,0],[.5,.2,.2,0]])
    old=command.copy();old[1,1:3]=.1
    x=support_vector(f,command,old)
    np.testing.assert_array_equal(x[0,:14],x[1,:14])
    assert x[0,15]==0 and np.isclose(x[1,15],.1)


def test_loader_checks_hash_and_preserves_weights(tmp_path):
    m=CausalHistoryTrajectoryModel(hidden_size=8,use_controls=True,
        feature_mean=np.zeros(12),feature_std=np.ones(12),control_mean=np.zeros(4),control_std=np.ones(4),
        derivative_mean=np.zeros(7),derivative_std=np.ones(7))
    path=tmp_path/'model.pt'
    torch.save(dict(format='control_derivative_ensemble_v1',hidden_size=8,use_controls=True,
        member_seeds=[17],member_state_dicts=[m.state_dict()]),path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    loaded=load_candidate(path,sha)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,loaded.members[0].state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='checksum'):load_candidate(path,'0'*64)
