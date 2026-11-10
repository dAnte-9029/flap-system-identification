"""Finite MIMO ARX / scalar-innovation ARMAX with causal segment resets.
Rows of theta map lagged outputs/inputs (and intercept) into p,q,r.
C is a shared scalar monic MA polynomial, not per-sample parameters.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize

BURN=15

def design(s,cfg,no_input=False):
    na,nb,d,reg=cfg;y=s['y'];u=s['u'];n=len(y);start=max(na,nb+d-1,BURN)
    X=np.column_stack([np.ones(n-start)]+[y[start-j:n-j] for j in range(1,na+1)]+([] if no_input else [u[start-j:n-j] if j else u[start:] for j in range(d,d+nb)]))
    return X,y[start:],start

def radius(theta,na):
    a=theta[1:1+3*na].reshape(na,3,3).transpose(0,2,1)
    companion=np.zeros((na*3,na*3));companion[:3]=np.concatenate(a,axis=1)
    companion[3:,:-3]=np.eye(3*(na-1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))

def noise(z):
    # Schur recursion for monic denominator of inverse innovation filter.
    c=np.array([1.])
    for k in z:
        c=np.r_[c,0]+k*np.r_[0,c[::-1]]
    return c

def stats(blocks,c):
    p=blocks[0][0].shape[1];G=np.zeros((p,p));R=np.zeros((p,3));YY=np.zeros(3);n=0
    for X,Y in blocks:
        Z=lfilter([1.],c,np.column_stack([X,Y]),axis=0)
        # Conditional likelihood with zero innovations at each segment start.
        x=Z[:,:p];y=Z[:,p:];G+=x.T@x;R+=x.T@y;YY+=(y*y).sum(0);n+=len(y)
    return G,R,YY,n

def solve_stats(st,reg):
    G,R,YY,n=st;pen=np.maximum(np.diag(G),1e-9)*reg;pen[0]=0
    theta=np.linalg.solve(G+np.diag(pen),R)
    return theta,pen

def fit(blocks,cfg,nc=0):
    def profile(z,ret=False):
        c=noise(z);st=stats(blocks,c);theta,pen=solve_stats(st,cfg[3]);G,R,YY,n=st
        # Direct residual norm avoids cancellation in YY-2*theta'R+theta'Gtheta.
        sse=sum(np.sum(lfilter([1.],c,Y-X@theta,axis=0)**2,axis=0) for X,Y in blocks)
        obj=float((sse.sum()+np.sum(pen[:,None]*theta**2))/n)
        return (theta,c,st,pen,obj) if ret else obj
    def gradient(z):
        theta,c,st,pen,obj=profile(z,True);n=st[3];grad=np.zeros(len(z))
        dc=[]
        for j in range(len(z)):
            zp=np.array(z).copy();zm=zp.copy();zp[j]+=1e-5;zm[j]-=1e-5
            dc.append((noise(zp)-noise(zm))/2e-5)
        weights=theta**2;weights[0]=0
        for X,Y in blocks:
            xf=lfilter([1.],c,X,axis=0);e=lfilter([1.],c,Y-X@theta,axis=0)
            for j,der in enumerate(dc):
                de=-lfilter(der,c,e,axis=0);dx=-lfilter(der,c,xf,axis=0)
                grad[j]+=2*np.sum(e*de)+2*cfg[3]*np.sum((xf*dx).sum(0)[:,None]*weights)
        return grad/n
    opt=None;z=np.zeros(nc)
    if nc:
        objective_scale=max(profile(z),1e-12)
        opt=minimize(lambda zz:profile(zz)/objective_scale,z,jac=lambda zz:gradient(zz)/objective_scale,method='L-BFGS-B',bounds=[(-.95,.95)]*nc,options={'maxiter':80,'ftol':1e-11,'gtol':1e-7})
        z=opt.x
        if not opt.success:
            opt=minimize(lambda zz:profile(zz)/objective_scale,opt.x,method='Powell',bounds=[(-.95,.95)]*nc,options={'maxiter':80,'ftol':1e-12,'xtol':1e-7})
        if not opt.success:raise RuntimeError('innovation optimization: '+opt.message)
    theta,c,st,pen,obj=profile(z,True)
    rho=radius(theta,cfg[0])
    if rho>=.9999:
        G,R,YY,n=st
        def fun(v):
            t=v.reshape(theta.shape);return float((np.sum(t*(G@t))-2*np.sum(t*R)+np.sum(pen[:,None]*t*t))/n)
        def jac(v):
            t=v.reshape(theta.shape);return (2*(G@t-R+pen[:,None]*t)/n).ravel()
        op=minimize(fun,theta.ravel(),jac=jac,method='SLSQP',constraints=[{'type':'ineq','fun':lambda v:.9999-radius(v.reshape(theta.shape),cfg[0])}],options={'maxiter':300,'ftol':1e-10})
        if not op.success:raise RuntimeError('stable AR fit: '+op.message)
        theta=op.x.reshape(theta.shape);rho=radius(theta,cfg[0])
    return dict(theta=theta,c=c,cfg=cfg,rho=rho,objective=obj,noise_reflection=z,iterations=0 if opt is None else int(opt.nit),optimizer='analytic_profile_gradient_v1')

def innovations(model,s):
    X,Y,start=design(s,model['cfg'],model['theta'].shape[0]==1+3*model['cfg'][0])
    e=lfilter([1.],model['c'],Y-X@model['theta'],axis=0)
    return e,start

def rollout(model,s,stride=10):
    na,nb,d,_=model['cfg'];theta=model['theta'];c=model['c'];e,start=innovations(model,s)
    origins=np.arange(start+10,len(s['y'])-10,stride)
    hist=[s['y'][origins-j].copy() for j in range(na)]
    preds=[]
    for h in range(1,11):
        x=[np.ones((len(origins),1))]+hist[:na]
        if len(theta)>1+3*na:x += [s['u'][origins+h-j] for j in range(d,d+nb)]
        y=np.column_stack(x)@theta
        for j in range(1,len(c)):
            if h-j<=0:y+=c[j]*e[origins+h-j-start]
        preds.append(y);hist=[y]+hist[:-1]
    return origins,np.stack(preds,axis=1)

def pulse(model):
    na,nb,d,_=model['cfg'];theta=model['theta'];hist=[np.zeros(3) for _ in range(na)];out=[]
    for h in range(1,11):
        x=[np.zeros(1)]+hist
        if len(theta)>1+na*3:
            for j in range(d,d+nb):
                u=np.zeros(4);u[1]=float(1<=h-j<=model.get('pulse_steps',5));x.append(u)
        y=np.concatenate(x)@theta;out.append(y);hist=[y]+hist[:-1]
    return np.asarray(out)

def sign_refit(model,blocks,sign,horizon_index=9):
    """Refit ALL dynamic coefficients; C fixed at training estimate.
    Match absolute training-fit 200ms gain, equalities at +/- magnitude.
    This is a conditional profile, not a global profile over noise parameters.
    """
    import copy
    out=copy.deepcopy(model);shape=model['theta'].shape;G,R,YY,n=stats(blocks,model['c'])
    _,pen=solve_stats((G,R,YY,n),model['cfg'][3]);target=sign*abs(pulse(model)[horizon_index,0])
    scale=max(abs(target),.1)
    def gain(v):
        mm=dict(model,theta=v.reshape(shape));return pulse(mm)[horizon_index,0]
    def fun(v):
        t=v.reshape(shape);return float((np.sum(t*(G@t))-2*np.sum(t*R)+np.sum(pen[:,None]*t*t))/n)
    def jac(v):
        t=v.reshape(shape);return (2*(G@t-R+pen[:,None]*t)/n).ravel()
    opt=minimize(fun,model['theta'].ravel(),jac=jac,method='SLSQP',constraints=[{'type':'eq','fun':lambda v:(gain(v)-target)/scale},{'type':'ineq','fun':lambda v:.9999-radius(v.reshape(shape),model['cfg'][0])}],options={'maxiter':400,'ftol':1e-10})
    if not opt.success:raise RuntimeError('sign profile: '+opt.message)
    out.update(theta=opt.x.reshape(shape),rho=radius(opt.x.reshape(shape),model['cfg'][0]),profile_target=target,profile_horizon_index=horizon_index,profile_iterations=int(opt.nit))
    assert abs(pulse(out)[horizon_index,0]-target)<1e-5
    return out

def blocks_for(seq,cfg,no_input=False):return [design(s,cfg,no_input)[:2] for s in seq]
