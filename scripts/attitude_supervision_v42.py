"""Yaw-invariant gravity-direction supervision, added to unchanged v35 loss."""
import math
import numpy as np
import torch
import train_control_windows_v2 as base
from joint_increment_v34 import objective as legacy_objective, local_increment_loss
from system_identification.models.trajectory_main_v1 import _rotation_body_to_ned

ANGLE_SCALE_RAD = .1


def gravity_direction(quaternion):
    # R_nb.T @ [0,0,1] equals the third row of R_nb.
    return _rotation_body_to_ned(quaternion)[...,2,:]


def tilt_loss(predicted, target):
    delta=gravity_direction(predicted)-gravity_direction(target)
    # Sum the3vector coordinates; mean over batch/time. Locally tilt_angle^2/.1^2.
    return delta.square().sum(-1).mean()/(ANGLE_SCALE_RAD**2)


def components(model,full_x,full_y,local_x,local_y):
    pred=model(**full_x)
    full=base.loss_fn(pred,full_y,model)
    local=local_increment_loss(model(**local_x),local_y,model.rate_scale)
    tilt=tilt_loss(pred['quaternion'],full_y['quaternion'])
    dot=(pred['quaternion']*full_y['quaternion']).sum(-1).clamp(-1,1)
    old_attitude=.2*(4*(1-dot.square()).clamp_min(0)/.01).mean()
    return full+local,tilt,old_attitude


def objective(model,full_x,full_y,local_x,local_y,weight):
    if not math.isfinite(weight) or weight<0:raise ValueError('Invalid tilt weight')
    if weight==0:
        loss,_,_=legacy_objective(model,full_x,full_y,local_x,local_y,weight=1.)
        return loss,loss,None
    legacy,tilt,_=components(model,full_x,full_y,local_x,local_y)
    return legacy+weight*tilt,legacy,tilt


def calibrated_weight(legacy_norms,tilt_norms):
    a=np.asarray(legacy_norms,dtype=float);b=np.asarray(tilt_norms,dtype=float)
    if a.shape!=b.shape or a.size!=12 or not np.isfinite([a,b]).all() or (b<=1e-12).any():
        raise ValueError('Twelve finite nonzero fit-only calibration pairs required')
    raw=.5*float(np.median(a/b))
    return float(np.clip(raw,.01,10.)),raw
