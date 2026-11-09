"""Diagnostic-only dual command midpoint flow, copied from frozen batch.tensor_step.
Only uF replaces vector-field command; u retains phase motor and actuator updates.
Tail commands enter vector_field only via coupling @ ((uF_tail-actuator)/tau).
"""
import torch
from system_identification.models.pacld.batch import BatchedPACLD, phase, covariance, select
from system_identification.models.pahbd import geometry as so3

class DualPACLD(BatchedPACLD):
    def dual_step(self, s, u, uF, dt):
        x = s.x
        acc, alpha, dz, dc = self.vector_field(s, uF)
        fm, pm = phase(self.flow.phase.coefficients, s.f, s.phi, u[..., :1], dt/2)
        am = u[..., 1:] + (s.actuator-u[..., 1:])*torch.exp(-dt/(2*self.config.actuator_tau))
        midx = torch.cat((x[..., :3]+x[..., 3:6]*dt/2, x[..., 3:6]+acc*dt/2,
                          so3.unit(so3.multiply(x[..., 6:10], so3.exp(x[..., 10:13]*dt/2))),
                          x[..., 10:13]+alpha*dt/2), -1)
        mid = s.with_(x=midx, z_s=s.z_s+dz*dt/2, c=s.c+dc*dt/2,
                      f=fm, phi=pm, actuator=am)
        acc, alpha, dz, dc = self.vector_field(mid, uF)
        newx = torch.cat((x[..., :3]+midx[..., 3:6]*dt, x[..., 3:6]+acc*dt,
                          so3.unit(so3.multiply(x[..., 6:10], so3.exp(midx[..., 10:13]*dt))),
                          x[..., 10:13]+alpha*dt), -1)
        f, phi = phase(self.flow.phase.coefficients, s.f, s.phi, u[..., :1], dt)
        actuator = u[..., 1:] + (s.actuator-u[..., 1:])*torch.exp(-dt/self.config.actuator_tau)
        cov = torch.diag_embed(s.variance[..., 12:14]) if s.phase_covariance is None else s.phase_covariance
        cov = covariance(cov, dt, self.flow.phase.coefficients[2], self.config.frequency_noise_psd, self.config.phase_noise_psd)
        variance = (s.variance[..., :12]+torch.sigmoid(self.flow.process_logit)*dt).clamp(self.config.variance_min, self.config.variance_max)
        result = s.with_(x=newx, z_s=s.z_s+dz*dt, c=s.c+dc*dt, f=f, phi=phi,
                       actuator=actuator, variance=torch.cat((variance, cov.diagonal(dim1=-2, dim2=-1)), -1),
                       phase_covariance=cov)

        return select(dt>0,result,s)
