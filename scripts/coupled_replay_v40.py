"""Explicit causal multirate coupling. No log I/O or target states in model loop."""
import numpy as np
import torch
from system_identification.control.px4_c9af import C9Controller
from system_identification.models.actuator_command_step import ActuatorCommandStepper
from system_identification.models.reference_gru_unit_gravity import unit_state_features

VARIANTS = [('warm2_plus',2500,'source_no_interleave'),
            ('warm2_minus',2500,'source_no_interleave'),
            ('warm2_20ms',20000,'source_no_interleave'),
            ('warm2_delayed_feedback',2500,'delayed_allocation')]


def angles(q):
    w,x,y,z=np.asarray(q,dtype=float).T
    return np.column_stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                            np.arcsin(np.clip(2*(w*y-z*x),-1,1))])


def controllers(snapshots):
    result=[]
    for snapshot in snapshots:
        c=C9Controller(snapshot['parameters']);c.restore(snapshot)
        assert all(g[2]==0 for g in c.gains), 'Acceleration proxy zero requires D=0'
        result.append(c)
    return result


def interval(cs, quaternion, rates, tape, air, manual, tick_us, feedback):
    """One20ms interval; left endpoint action to plant, remaining actions diagnostic.

    tape[B,8,4] holds roll_sp,pitch_sp,thrust,yaw_stick on2.5ms grid.
    Controller rate2.5ms and attitude10ms;20ms alternative calls both once.
    Initial air proxy is treated as a fresh constant synthetic observation.
    """
    if tick_us not in (2500,20000):raise ValueError('Unsupported clock')
    rp=angles(quaternion);b=len(cs)
    if tape.shape!=(b,8,4) or rates.shape!=(b,3):raise ValueError('Coupling shape')
    actions=[];torques=[];refs=[];clipped=[]
    for micro in range(0,8,tick_us//2500):
        aa=[];tt=[];rr=[];ss=[]
        for j,c in enumerate(cs):
            roll,pitch,motor,yaw=tape[j,micro]
            if micro%4==0:
                c.attitude(*rp[j],roll,pitch,float(air[j]),0.,manual=bool(manual[j]),yaw_stick=yaw)
            sp=c.state.last_rate_sp
            out=c.rate(rates[j],sp,[0.,0.,0.],float(air[j]),0.,motor,tick_us*1e-6,
                       landed=False,reset_integral=False,feedback=feedback)
            aa.append(out['commands']);tt.append(out['torque']);rr.append(sp.copy())
            ss.append(np.abs(out['raw_surfaces'])>1.)
        actions.append(aa);torques.append(tt);refs.append(rr);clipped.append(ss)
    actions=np.asarray(actions).transpose(1,0,2)
    return dict(command=actions[:,0],torque=np.asarray(torques)[0],rates_sp=np.asarray(refs)[0],
                micro_commands=actions, surface_clipped=np.asarray(clipped).transpose(1,0,2))


def rollout(model, initial_inputs, snapshots, tape, air, manual, variant, support):
    """Future references allowed; future measured plant/actuator state unavailable."""
    name,tick,feedback=variant
    adapter=ActuatorCommandStepper(model);state=adapter.initialize(**initial_inputs)
    cs=controllers(snapshots);b=len(cs);out={k:[] for k in ['quaternion','velocity_n','rates_b',
        'commands','torque','rates_sp','micro_commands','surface_clipped','command_outside','state_outside']}
    assert tape.shape==(b,400,4)
    dt=torch.full((b,),.02,dtype=torch.float32)
    for k in range(50):
        # All support flags at action time use generated/current state only.
        features=torch.cat([unit_state_features(state.velocity_n,state.quaternion,state.rates_b,
            state.phase,state.anchor,state.frequency),state.air],-1).numpy()
        d=interval(cs,state.quaternion.numpy(),state.rates_b.numpy(),tape[:,k*8:(k+1)*8],air,manual,tick,feedback)
        command=d['command']
        out['state_outside'].append((features<support['state_min'])|(features>support['state_max']))
        out['command_outside'].append((command<support['command_min'])|(command>support['command_max']))
        state,pred=adapter.step(state,torch.as_tensor(command,dtype=torch.float32),dt)
        for key in ['quaternion','velocity_n','rates_b']:out[key].append(pred[key].numpy())
        for key in ['torque','rates_sp','micro_commands','surface_clipped']:out[key].append(d[key])
        out['commands'].append(command)
    return {k:np.stack(v,1) for k,v in out.items()}


def teacher(quaternion, rates, snapshots, tape, air, manual, variant):
    """Matched held20ms measured-state controller diagnostic, kept separate."""
    _,tick,feedback=variant;cs=controllers(snapshots);out={}
    for k in range(50):
        d=interval(cs,quaternion[:,k],rates[:,k],tape[:,k*8:(k+1)*8],air,manual,tick,feedback)
        for key,value in d.items():out.setdefault('commands' if key=='command' else key,[]).append(value)
    return {k:np.stack(v,1) for k,v in out.items()}
