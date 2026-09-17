"""Past-only statistical phase reference; no dataset access or mechanical-zero claim."""
from dataclasses import replace
import numpy as np
import torch
from .main_v2_simulator import MainV2Simulator


def circular(x):
    return np.arctan2(np.sin(x), np.cos(x))


def harmonic_design(phi, order=3):
    phi=np.asarray(phi)
    return np.stack([f(k*phi) for k in range(1,order+1) for f in (np.sin,np.cos)],axis=-1)


def interval_design(phase):
    """Average harmonics over each observed phase interval, no future derivative.

    Exact for linear phase within an interval. This accounts for the backward
    difference's interval timing and sinc response rather than ignoring delay.
    """
    phase=np.unwrap(np.asarray(phase));d=np.diff(phase);mid=phase[:-1]+d/2
    return harmonic_design(mid)*np.repeat(np.stack([np.sinc(k*d/(2*np.pi)) for k in (1,2,3)],axis=-1),2,axis=-1)


def rotate_coefficients(coeff, offset):
    """Coefficients of T(phi+offset), with coeff shape [6, channels]."""
    coeff=np.asarray(coeff);offset=np.asarray(offset)
    out=np.empty(offset.shape+coeff.shape)
    for k in range(1,4):
        a,b=coeff[2*k-2],coeff[2*k-1];c=np.cos(k*offset)[...,None];s=np.sin(k*offset)[...,None]
        out[...,2*k-2,:]=c*a-s*b;out[...,2*k-1,:]=s*a+c*b
    return out


def estimate_windows(time,phase,rates,velocity_n,ends,length,coeff,scales,channels=(0,1),grid_size=360):
    """Independent reset estimates; every sufficient statistic ends at `ends`.

    E1 uses backward q increments only. E2 adds backward NED vertical velocity
    increments (observable navigation state, not IMU specific force). No filter,
    center difference, or sample beyond the reset enters an estimate. Each
    channel allows a nonnegative gain and DC offset as matching nuisance terms.
    """
    ends=np.asarray(ends,dtype=int)
    # Cut before any computation, including unwrap, to protect the past-only API.
    stop=int(ends.max())+1;time=np.asarray(time)[:stop];phase=np.unwrap(np.asarray(phase)[:stop])
    if np.any(np.diff(time)<=0):raise ValueError('timestamps must increase within one segment')
    dt=np.diff(time);X=interval_design(phase)
    y=np.column_stack([np.diff(np.asarray(rates)[:stop,1])/dt,np.diff(np.asarray(velocity_n)[:stop,2])/dt])
    def prefix(a):return np.concatenate([np.zeros((1,)+a.shape[1:]),np.cumsum(a,axis=0)])
    sx=prefix(X);sy=prefix(y);xx=prefix(X[:,:,None]*X[:,None,:]);xy=prefix(X[:,:,None]*y[:,None,:]);yy=prefix(y*y)
    starts=np.maximum(ends-length,0);n=np.maximum(ends-starts,1)
    xs=sx[ends]-sx[starts];ys=sy[ends]-sy[starts]
    G=xx[ends]-xx[starts]-xs[:,:,None]*xs[:,None,:]/n[:,None,None]
    B=xy[ends]-xy[starts]-xs[:,:,None]*ys[:,None,:]/n[:,None,None]
    Y=yy[ends]-yy[starts]-ys*ys/n[:,None]
    grid=np.arange(grid_size)*2*np.pi/grid_size;C=rotate_coefficients(coeff,grid)
    scores=np.zeros((len(ends),grid_size))
    for j in channels:
        c=C[:,:,j];cross=B[:,:,j]@c.T;den=G.reshape(-1,36)@(c[:,:,None]*c[:,None,:]).reshape(grid_size,36).T
        scores+=(Y[:,j,None]-np.maximum(cross,0)**2/np.maximum(den,1e-10))/(n[:,None]*float(scales[j])**2)
    best=scores.argmin(1);offset=grid[best];offset=np.where(ends-starts>=7,offset,0.)
    # Distinct secondary peak at least 30 degrees away, exposes harmonic aliases.
    distance=np.abs(circular(grid[None,:]-grid[best,None]));second=np.min(np.where(distance>=np.pi/6,scores,np.inf),axis=1)
    return dict(offset_rad=offset,score=scores[np.arange(len(ends)),best],score_gap=second-scores[np.arange(len(ends)),best],
                duration_s=time[ends]-time[starts],observed_cycles=(phase[ends]-phase[starts])/(2*np.pi),
                history_intervals=ends-starts,fallback=(ends-starts<7))


class PhaseReferenceSimulator(MainV2Simulator):
    """Raw phase is preserved; immutable anchor=-offset encodes canonical phase.

    canonical_phase=wrap(relative_phase_rad-phase_anchor). The existing snapshot
    already stores the anchor; restoring never re-estimates offset or history.
    """
    def reset_phase(self, *, offset_rad, **warm_inputs):
        angle=warm_inputs['relative_phase_rad']+offset_rad
        h=warm_inputs['history_state_features'].clone();s=h[:,:,9].clone();c=h[:,:,10].clone()
        h[:,:,9]=s*torch.cos(angle[:,None])+c*torch.sin(angle[:,None])
        h[:,:,10]=c*torch.cos(angle[:,None])-s*torch.sin(angle[:,None])
        state=self.reset(**{**warm_inputs,'history_state_features':h})
        state=replace(state,phase_anchor=-offset_rad.clone());self._validate_state(state)
        return state
