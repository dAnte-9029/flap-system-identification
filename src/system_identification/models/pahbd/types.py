"""Explicit runtime contracts. Times are seconds relative to each window decision."""
from dataclasses import dataclass, replace
import torch


@dataclass
class State:
    x: torch.Tensor  # (...,13): p_N, v_N, q_NB(wxyz), omega_B
    h: torch.Tensor  # (...,latent)
    f: torch.Tensor  # (...,1), Hz
    phi: torch.Tensor  # (...,1), unwrapped rad
    actuator: torch.Tensor  # (...,3), filtered normalized left/right elevon, rudder
    variance: torch.Tensor  # (...,14): p,v,body rotation,omega,f,phi diagonal proxy
    phase_quality: torch.Tensor  # (...,1) fixed causal reference confidence [0,1]

    phase_covariance: torch.Tensor | None = None  # joint unwrapped (f, phi), never modulo covariance

    def with_(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class Observation:
    measurement: float
    availability: float
    value: torch.Tensor  # (...,15): x(13),f,phi; unused quaternion must be identity
    mask: torch.Tensor  # (...,14): tangent p,v,R,omega,f,phi
    variance: torch.Tensor  # (...,14), source noise floor, not learned per window
    quality: torch.Tensor  # (...,14), [0,1]
    source: str
    frame: str = 'NED/FRD/Hamilton-wxyz'


@dataclass
class ControlTape:
    times: list[float]
    values: torch.Tensor  # (events,4) or (events,batch,4); motor,left,right,rudder

    def intervals(self,start,end):
        """Exact ZOH breakpoints, cached per native event/query interval."""
        import bisect
        if not hasattr(self,'_interval_cache'):self._interval_cache={}
        key=(start,end)
        if key not in self._interval_cache:
            i=bisect.bisect_right(self.times,start+1e-12)-1
            if i<0:raise ValueError('no causal control at integration start')
            t=start;segments=[]
            while t<end-1e-12:
                boundary=min(end,self.times[i+1] if i+1<len(self.times) else end)
                if boundary-t>1e-12:segments.append((i,boundary-t))
                t=boundary
                if i+1<len(self.times) and t>=self.times[i+1]-1e-12:i+=1
            self._interval_cache[key]=segments
        return self._interval_cache[key]

    def canonical(self):
        key=(id(self.values),self.values._version,tuple(self.times),self.values.requires_grad)
        if getattr(self,'_prepared_key',None)==key:return self
        if getattr(self,'_canonical_key',None)==key:return self._canonical_tape
        if not self.times or len(self.times)!=len(self.values):raise ValueError('empty/mismatched control tape')
        if any(b<=a for a,b in zip(self.times,self.times[1:])):raise ValueError('control times must strictly increase')
        cpu=self.values.detach().cpu()
        if not torch.isfinite(cpu).all():raise ValueError('nonfinite controls')
        indices=[0]
        for i in range(1,len(self.times)):
            if self.values.requires_grad or not torch.equal(cpu[i],cpu[indices[-1]]):indices.append(i)
        tape=ControlTape([self.times[i] for i in indices],self.values[indices])
        tape._cpu_values=cpu[indices]
        tape._prepared_key=(id(tape.values),tape.values._version,tuple(tape.times),tape.values.requires_grad)
        # Never cache a differentiable indexing graph across backwards.
        if not self.values.requires_grad:self._canonical_key=key;self._canonical_tape=tape
        return tape


@dataclass
class Window:
    initial: State
    start: float
    history: list[Observation]
    controls: ControlTape
    targets: list[Observation]
    metadata: dict
