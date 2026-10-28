"""Hamilton wxyz, active FRD -> NED rotations; right/body tangent increments."""
import torch
import torch.nn.functional as F


def unit(q):
    return F.normalize(q, dim=-1)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a, b):
    return torch.cat((a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True),
                      a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]
                      +torch.cross(a[..., 1:], b[..., 1:], dim=-1)), -1)


def exp(rot):
    theta = torch.linalg.vector_norm(rot, dim=-1, keepdim=True)
    return torch.cat((torch.cos(theta/2), .5*torch.sinc(theta/(2*torch.pi))*rot), -1)


def log(q):
    q = unit(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    n = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    scale = 2*torch.atan2(n, q[..., :1])/n.clamp_min(1e-8)
    return q[..., 1:]*torch.where(n < 1e-8, 2*torch.ones_like(n), scale)


def residual(q, observed):
    return log(multiply(conjugate(unit(q)), unit(observed)))


def rotate(q, v):
    q = unit(q)
    t = 2*torch.cross(q[..., 1:], v, dim=-1)
    return v+q[..., :1]*t+torch.cross(q[..., 1:], t, dim=-1)


def state_residual(x, y):
    return torch.cat((y[..., :6]-x[..., :6], residual(x[..., 6:10], y[..., 6:10]),
                      y[..., 10:13]-x[..., 10:13]), -1)


def retract(x, delta):
    return torch.cat((x[..., :6]+delta[..., :6],
                      unit(multiply(x[..., 6:10], exp(delta[..., 6:9]))),
                      x[..., 10:13]+delta[..., 9:12]), -1)
