"""One-tick offline oracle probes; never mutate factual controller histories."""
import copy
from dataclasses import fields
import numpy as np
from system_identification.control.px4_c9af import ControllerState

BRANCHES=['mm','mt','tm','tt','p','q','r','roll','pitch','rates','attitude',
          'integral','compression','yaw_memory']
GROUPS={'integral':['integral'],
        'compression':['gain','cached_gain','hpf','energy','previous_effort'],
        'yaw_memory':['yaw_euler','yaw_body','last_rate_sp']}
SCHEMA=[(f.name,3 if isinstance(getattr(ControllerState(),f.name),list) else 1)
        for f in fields(ControllerState)]


def vector(c):
    return np.concatenate([np.atleast_1d(getattr(c.state,name)).astype(float) for name,_ in SCHEMA])


def copy_controller(c):
    clone=copy.copy(c)
    clone.state=copy.deepcopy(c.state)
    return clone


def from_vector(template,value):
    c=copy_controller(template);offset=0
    if len(value)!=sum(n for _,n in SCHEMA) or not np.isfinite(value).all():raise ValueError('state vector')
    for name,width in SCHEMA:
        a=value[offset:offset+width];offset+=width
        v=a.tolist() if width>1 else float(a[0])
        if name in ['positive','negative']:v=[bool(x) for x in v]
        setattr(c.state,name,v)
    return c


def one_tick(c,rp,rates,reference,air,manual,tick_us,feedback):
    """Called at the model interval left endpoint, which updates attitude first."""
    roll_sp,pitch_sp,motor,yaw=reference
    c.attitude(*rp,roll_sp,pitch_sp,float(air),0.,manual=bool(manual),yaw_stick=yaw)
    result=c.rate(rates,c.state.last_rate_sp,[0.,0.,0.],float(air),0.,motor,tick_us*1e-6,
                  landed=False,reset_integral=False,feedback=feedback)
    return np.asarray(result['commands']),np.abs(result['raw_surfaces'])>1.


def probes(model_controller,teacher_controller,model_rp,model_rates,true_rp,true_rates,
           reference,air,manual,tick_us,feedback):
    commands=[];clipped=[]
    for name in BRANCHES:
        base=teacher_controller if name in ['tm','tt'] else model_controller
        c=copy_controller(base)
        use_true=name in ['mt','tt']
        rp=np.array(true_rp if use_true else model_rp,copy=True)
        rates=np.array(true_rates if use_true else model_rates,copy=True)
        if name in ['p','q','r']:rates[['p','q','r'].index(name)]=true_rates[['p','q','r'].index(name)]
        if name in ['roll','pitch']:rp[['roll','pitch'].index(name)]=true_rp[['roll','pitch'].index(name)]
        if name=='rates':rates=np.array(true_rates,copy=True)
        if name=='attitude':rp=np.array(true_rp,copy=True)
        if name in GROUPS:
            for key in GROUPS[name]:setattr(c.state,key,copy.deepcopy(getattr(teacher_controller.state,key)))
        command,sat=one_tick(c,rp,rates,reference,air,manual,tick_us,feedback)
        commands.append(command);clipped.append(sat)
    return np.stack(commands),np.stack(clipped)
