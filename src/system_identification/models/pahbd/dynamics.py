"""PA-HBD vector field. No physical mass/inertia or measured acceleration labels."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from . import geometry as so3


@dataclass
class Config:
    latent: int = 12
    width: int = 128
    forcing_harmonics: int = 3
    control_harmonics: int = 3
    rank: int = 2
    mode: str = 'A'
    max_step: float = .005
    phase_step: float = .15  # maximum highest-harmonic phase advance per substep
    latent_rate: float = .8
    latent_bound: float = 2.
    actuator_tau: float = .06  # modeling prior, NOT measured servo time constant
    variance_min: float = 1e-5
    variance_max: float = 25.
    phase_mode: str = 'relative'  # Fourier relative coordinate or disabled; native data anchors it to Hall
    replay_seconds: float = .6  # bounded decision-time replay; older readings rejected
    uncertainty_mode: str = 'gaussian'  # deterministic or Gaussian harmonic expectation
    frequency_noise_psd: float = .04  # Hz^2/s, uncalibrated modeling assumption
    phase_noise_psd: float = .01  # rad^2/s, independent continuous diffusion
    frequency_coefficients: tuple = (3.6354511983566393,23.393570068972757,-5.126725395181777)

    def __post_init__(self):
        if self.uncertainty_mode not in ('deterministic','gaussian') or min(self.frequency_noise_psd,self.phase_noise_psd)<0 or self.replay_seconds<=0:
            raise ValueError('invalid uncertainty/replay contract')
        if self.mode not in ('A','B') or self.phase_mode not in ('relative','disabled'):
            raise ValueError('unsupported model/reference mode')
        if min(self.latent,self.width,self.rank,self.forcing_harmonics,self.control_harmonics) < 1:
            raise ValueError('network sizes must be positive')
        if min(self.max_step,self.phase_step,self.latent_rate,self.latent_bound,self.actuator_tau,self.variance_min) <= 0:
            raise ValueError('positive numerical bounds required')
        if self.variance_max <= self.variance_min:
            raise ValueError('invalid variance interval')


class PhasePropagator(nn.Module):
    """Exact ZOH solution inherited from phase_forecast.py; coefficients frozen train fit."""
    def __init__(self, coefficients):
        super().__init__()
        self.register_buffer('coefficients', torch.tensor(coefficients, dtype=torch.float64))

    def forward(self, f, phi, motor, dt):
        b0,bu,bf = self.coefficients.to(f).unbind()
        c = b0+bu*motor
        if abs(float(bf)) < 1e-9:
            return f+c*dt, phi+2*math.pi*(f*dt+.5*c*dt*dt)
        steady = -c/bf
        integral = steady*dt+(f-steady)*torch.expm1(bf*dt)/bf
        return steady+(f-steady)*torch.exp(bf*dt), phi+2*math.pi*integral

    def covariance(self, covariance, dt, process_noise=None):
        """Optional full 2x2 (f,phi) covariance interface; no calibrated uncertainty claim."""
        # Exact continuous linear SDE: df=b_f f dt+sqrt(q_f)dW_f,
        # dphi=2*pi*f dt+sqrt(q_phi)dW_phi. Van Loan integral obeys semigroup.
        if dt < 0:raise ValueError('negative covariance duration')
        A=covariance.new_tensor([[float(self.coefficients[2]),0.],[2*math.pi,0.]])
        Q=torch.zeros_like(A) if process_noise is None else process_noise.to(covariance)
        if Q.shape != (2,2) or not torch.allclose(Q,Q.T) or torch.linalg.eigvalsh(Q).min() < 0:
            raise ValueError('process_noise must be PSD spectral density')
        block=torch.cat((torch.cat((A,Q),1),torch.cat((torch.zeros_like(A),-A.T),1)),0)
        E=torch.matrix_exp(block*dt);F=E[:2,:2];discrete=E[:2,2:]@F.T
        result=F@covariance@F.T+discrete
        return (result+result.transpose(-1,-2))/2


class SlowBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('input_mean', torch.zeros(10))
        self.register_buffer('input_scale', torch.tensor([10.]*3+[3.]*3+[9.80665]*3+[5.]))
        self.encoder = nn.Sequential(nn.Linear(10+cfg.latent,cfg.width),nn.SiLU(),
                                     nn.Linear(cfg.width,cfg.width),nn.SiLU())
        self.memory = nn.Linear(cfg.width+3,cfg.latent)
        self.retention = nn.Parameter(torch.zeros(cfg.latent))

    def features(self, s):
        q=s.x[...,6:10];g=torch.zeros_like(s.x[...,:3]);g[...,2]=9.80665
        return torch.cat((so3.rotate(so3.conjugate(q),s.x[...,3:6]),s.x[...,10:13],
                          so3.rotate(so3.conjugate(q),g),s.f),-1)

    def forward(self,s):
        c=self.encoder(torch.cat(((self.features(s)-self.input_mean)/self.input_scale,
                                  s.h/self.cfg.latent_bound),-1))
        # Bounded attracting target can be state dependent and persist under forcing.
        # Rate cap prevents h from being a free high-frequency phase oscillator.
        target=self.cfg.latent_bound*torch.tanh(self.memory(torch.cat((c,s.actuator),-1)))
        rate=self.cfg.latent_rate*torch.sigmoid(self.retention)
        dh=rate*(target-s.h)/(2*self.cfg.latent_bound)
        return c,dh


class ControlMixer(nn.Module):
    """Conditional coefficient heads; low-rank phase control maps preserve three channels."""
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.mean=nn.Linear(cfg.width,6+6*3)
        self.forcing=nn.Linear(cfg.width,2*cfg.forcing_harmonics*6)
        self.control=nn.Linear(cfg.width,2*cfg.control_harmonics*6*cfg.rank)
        self.right=nn.Parameter(torch.randn(cfg.control_harmonics,cfg.rank,3)*.1)
        self.register_buffer('reference_control',torch.zeros(3))
        self.register_buffer('gauge_offset',torch.tensor(0.))
        for layer in (self.mean,self.forcing,self.control):
            nn.init.normal_(layer.weight,std=.01);nn.init.zeros_(layer.bias)
        with torch.no_grad():self.mean.bias[2]=-9.80665

    @staticmethod
    def rotate_coefficients(coefficients, offset):
        """phi'=phi+offset: c'=c*cos(kd)-s*sin(kd), s'=c*sin(kd)+s*cos(kd)."""
        # coefficient shape (...,H,2,6[,rank]); harmonic axis chosen by caller via flatten.
        H=coefficients.shape[-3]
        k=torch.arange(1,H+1,device=coefficients.device,dtype=coefficients.dtype)
        c,s=coefficients.unbind(-2);cd=torch.cos(k*offset)[...,None];sd=torch.sin(k*offset)[...,None]
        return torch.stack((c*cd-s*sd,c*sd+s*cd),-2)

    def coefficients(self,c):
        lead=c.shape[:-1];cfg=self.cfg
        raw=self.mean(c);mean=raw[...,:6];B=raw[...,6:].reshape(*lead,6,3)
        forcing=self.forcing(c).reshape(*lead,cfg.forcing_harmonics,2,6)
        left=self.control(c).reshape(*lead,cfg.control_harmonics,2,6*cfg.rank)
        forcing=self.rotate_coefficients(forcing,self.gauge_offset)
        left=self.rotate_coefficients(left,self.gauge_offset).reshape(*lead,cfg.control_harmonics,2,6,cfg.rank)
        G=torch.einsum('...hkor,hrj->...hkoj',left,self.right)
        return mean,B,forcing,G

    def forward(self,c,actuator,phi,quality,phase_variance=None):
        mean,B,F,G=self.coefficients(c);da=actuator-self.reference_control
        avg=mean+torch.einsum('...ij,...j->...i',B,da)
        def basis(H):
            k=torch.arange(1,H+1,device=phi.device,dtype=phi.dtype)
            result=torch.stack((torch.cos(phi*k),torch.sin(phi*k)),-1)
            if phase_variance is not None and self.cfg.uncertainty_mode=='gaussian':
                result=result*torch.exp(-.5*k.square()*phase_variance)[...,None]
            return result
        enabled=quality if self.cfg.phase_mode=='relative' else torch.zeros_like(quality)
        forcing=torch.einsum('...hk,...hko->...o',basis(self.cfg.forcing_harmonics),F)*enabled
        interaction=torch.einsum('...hk,...hkoj,...j->...o',basis(self.cfg.control_harmonics),G,da)*enabled
        if self.cfg.mode=='B':interaction=interaction*0
        return avg+forcing+interaction,dict(mean=avg,forcing=forcing,interaction=interaction)
