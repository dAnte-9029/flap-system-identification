"""Independent DOP853 controlled periodic truth for engineering verification only."""
import math
import numpy as np
import torch
from scipy.integrate import solve_ivp
from system_identification.models.pahbd import State,Observation,ControlTape,Window


def controlled_periodic_window(config,horizon=.5,dtype=torch.float64):
    start=-.25
    motor=(5*5.126725395181777-3.6354511983566393)/23.393570068972757
    control_times=[start,.075,.31]
    controls=np.array([[motor,.1,-.1,.05],[motor,-.2,.15,-.1],[motor,.15,.05,.12]])
    control_times=[t for t in control_times if t<horizon]
    controls=controls[:len(control_times)]
    y0=np.r_[np.zeros(6),1.,np.zeros(6),controls[0,1:]] # x13 + actuator3
    def rhs(t,y,u):
        phi=2*math.pi*5*t
        a=y[13:16]
        acc=np.array([.2+.5*a[0]+.4*np.cos(phi),.3*a[1]+.1*a[2]*np.sin(phi),
                      -9.80665+.25*np.sin(phi)+.3*a[2]])
        alpha=np.array([.2*np.sin(phi)+.6*a[0]*np.cos(phi),.1*a[1],.2*a[2]])
        q=y[6:10]/np.linalg.norm(y[6:10]);w=y[10:13]
        cross=2*np.cross(q[1:],acc);an=acc+q[0]*cross+np.cross(q[1:],cross)+[0,0,9.80665]
        qdot=.5*np.r_[-q[1:]@w,q[0]*w+np.cross(q[1:],w)]
        return np.r_[y[3:6],an,qdot,alpha,(u[1:]-a)/config.actuator_tau]
    solutions=[];y=y0.copy()
    for i,t in enumerate(control_times):
        end=control_times[i+1] if i+1<len(control_times) else horizon
        sol=solve_ivp(lambda t,y:rhs(t,y,controls[i]),(t,end),y,method='DOP853',rtol=1e-11,atol=1e-12,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        solutions.append((t,end,sol));y=sol.y[:,-1]
    def truth(t):
        for a,b,sol in solutions:
            if a-1e-12<=t<=b+1e-12:return sol.sol(t)
        raise ValueError(t)
    T=lambda v:torch.as_tensor(v,dtype=dtype)
    initial=State(T(y0[:13]),T(np.zeros(config.latent)),T([5.]),T([2*math.pi*5*start]),
                  T(controls[0,1:]),T([.1]*14),T([1.]))
    events=[];targets=[]
    for name,step,offset,sl in [('vehicle_local_position',.021,.003,slice(0,6)),
                                ('vehicle_attitude',.019,.005,slice(6,9)),
                                ('vehicle_angular_velocity',.013,.002,slice(9,12))]:
        for t in np.arange(start+.01,horizon-1e-8,step):
            y=truth(float(t));value=np.r_[y[:13],5.,2*math.pi*5*t]
            mask=np.zeros(14);mask[sl]=1
            e=Observation(float(t),float(t+offset),T(value),T(mask),T([.04]*14),T([1.]*14),name)
            if e.availability<=0:events.append(e)
            if t>0:targets.append(e)
    targets.sort(key=lambda e:e.measurement)
    cycles=[]
    for i,e in enumerate(targets):
        candidates=[(j,abs(t.measurement-e.measurement-.2)) for j,t in enumerate(targets) if t.source==e.source and t.measurement>e.measurement]
        if candidates:
            j,error=min(candidates,key=lambda p:p[1])
            if error<.006:cycles.append((i,j))
    window=Window(initial,start,sorted(events,key=lambda e:e.availability),ControlTape(control_times,T(controls)),targets,
                  dict(split='train',flight_id='synthetic_periodic',window_id='synthetic_periodic:0',
                       phase_reference='known_synthetic',phase_valid=True,cycle_pairs=cycles,
                       interpretation='engineering_only_independent_DOP853_truth'))
    return window,truth
