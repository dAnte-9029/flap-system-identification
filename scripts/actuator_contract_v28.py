"""Deterministic command reconstruction and configured output maps, no fitting."""
import numpy as np
from reference_timing_v14 import effective_indices

CHANNELS=['motor0','servo0','servo1','servo2']
PORTS=[3,1,2,5]


def held_commands(motor,servo,query):
    ix=[effective_indices(d['timestamp'],d['timestamp_sample'],query) for d in [motor,servo]]
    a=np.column_stack([motor['control[0]'][ix[0]]]+[servo[f'control[{i}]'][ix[1]] for i in range(3)])
    if not np.isfinite(a).all():raise ValueError('nonfinite used command')
    pub=np.column_stack([d['timestamp'][j] for d,j in zip([motor,servo],ix)]).astype(np.int64)
    sample=np.column_stack([d['timestamp_sample'][j] for d,j in zip([motor,servo],ix)]).astype(np.int64)
    return a.astype(np.float32),pub,sample,ix


def output_pwm(command,params):
    """Active normal-operation map; no motor stop, disarm, failsafe or spool state."""
    a=np.asarray(command,dtype=float).copy()
    factor=params['THR_MDL_FAC']
    if factor>np.finfo(np.float32).eps and factor<=1:
        b=1-factor;a[:,0]=-(b/(2*factor))+np.sqrt(b*b/(4*factor*factor)+a[:,0]/factor)
    a[:,0]=2*a[:,0]-1  # checked nonreversible motor
    result=[]
    for j,port in enumerate(PORTS):
        v=a[:,j]*(-1 if int(params['PWM_MAIN_REV'])&(1<<(port-1)) else 1)
        lo,hi=[params[f'PWM_MAIN_{key}{port}'] for key in ['MIN','MAX']]
        center=params.get(f'PWM_MAIN_CENT{port}')
        if j>0 and center is not None and 800<=center<=2200:
            pwm=np.where(v<0,lo+(v+1)*(center-lo),center+v*(hi-center))
        else:pwm=lo+(v+1)*.5*(hi-lo)
        result.append(np.clip(np.floor(pwm+.5),0,65535))
    return np.column_stack(result)


def allocation(torque,thrust,params):
    """Fixedwing pseudo-inverse, normalized thrust, zero trims/flaps/slew asserted by caller."""
    matrix=np.array([[params[f'CA_SV_CS{j}_TRQ_{axis}'] for j in range(3)] for axis in ['R','P','Y']],float)
    servo=np.asarray(torque)@np.linalg.inv(matrix).T
    return np.column_stack([np.clip(thrust,0,1),np.clip(servo,-1,1)])


def stats(a):
    a=np.asarray(a,dtype=float)
    if not a.size:return dict(n=0)
    if not np.isfinite(a).all():raise ValueError('nonfinite audited value')
    return dict(n=int(a.size),min=float(a.min()),p50=float(np.median(a)),p95=float(np.quantile(a,.95)),
                max=float(a.max()),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))))
