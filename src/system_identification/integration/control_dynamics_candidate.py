"""Load frozen candidate and query a train-fitted joint support reference.

This is an experimental prediction interface, not a flight controller API.
Support acceptance is necessary coverage evidence, never causal validation.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

from system_identification.models.control_ensemble import ControlDerivativeEnsemble
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel


STATE_INDICES = (0,1,2,3,4,5,6,7,8,11)
FEATURE_NAMES = ('vb_x','vb_y','vb_z','p','q','r','gravity_b_x','gravity_b_y','gravity_b_z','flap_frequency_hz',
                 'drive','common','differential','rudder','drive_delta5','common_delta5','differential_delta5','rudder_delta5')


def command_coordinates(u):
    return np.stack([u[...,0],(u[...,1]+u[...,2])/2,(u[...,1]-u[...,2])/2,u[...,3]],-1)


def support_vector(features,command,past_command5):
    c=command_coordinates(np.asarray(command))
    return np.concatenate([np.asarray(features)[...,STATE_INDICES],c,c-command_coordinates(np.asarray(past_command5))],-1)


def load_candidate(path,expected_sha256):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('candidate checksum mismatch')
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    if bundle['format']!='control_derivative_ensemble_v1' or len(bundle['member_seeds'])!=len(bundle['member_state_dicts']):
        raise ValueError('candidate format mismatch')
    members=[]
    for state in bundle['member_state_dicts']:
        stats={k:state[k].numpy() for k in ('feature_mean','feature_std','control_mean','control_std','derivative_mean','derivative_std')}
        model=CausalHistoryTrajectoryModel(hidden_size=bundle['hidden_size'],use_controls=bundle['use_controls'],**stats)
        model.load_state_dict(state,strict=True)
        members.append(model)
    return ControlDerivativeEnsemble(members).eval()


class JointSupport:
    def __init__(self,reference,mean,scale,lower,upper,distance_limit):
        self.mean=np.asarray(mean);self.scale=np.asarray(scale)
        self.lower=np.asarray(lower);self.upper=np.asarray(upper)
        self.distance_limit=float(distance_limit)
        reference=np.asarray(reference)
        if reference.ndim!=2 or reference.shape[1]!=len(FEATURE_NAMES) or self.mean.shape!=(len(FEATURE_NAMES),):
            raise ValueError('support reference shape mismatch')
        if not np.isfinite(reference).all() or not np.isfinite(self.scale).all() or np.any(self.scale<=0) or self.distance_limit<=0:
            raise ValueError('invalid support reference')
        self.tree=cKDTree((reference-self.mean)/self.scale)

    def query(self,values):
        x=np.asarray(values)
        if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES):raise ValueError('support query shape mismatch')
        finite=np.isfinite(x).all(1)
        distance=np.full(len(x),np.inf)
        distance[finite]=self.tree.query((x[finite]-self.mean)/self.scale,k=1)[0]
        box=finite&((x>=self.lower)&(x<=self.upper)).all(1)
        return dict(accepted=box&(distance<=self.distance_limit),box=box,distance=distance)
