"""Offline rate-tape attribution and independent quaternion integration."""
import math
import numpy as np

BLOCKS={'20_40ms':(0,2),'60_100ms':(2,5),'120_200ms':(5,10),
        '220_500ms':(10,25),'520_1000ms':(25,50),'full1s':(0,50)}


def normalize(q):
    q=np.asarray(q,dtype=float)
    return q/np.linalg.norm(q,axis=-1,keepdims=True)


def angle(a,b):
    a=normalize(a);b=normalize(b)
    sign=np.where(np.sum(a*b,axis=-1,keepdims=True)<0,-1.,1.)
    return 4*np.arcsin(np.clip(np.linalg.norm(a-sign*b,axis=-1)/2,0,1))


def integrate(initial_quaternion,initial_rates,rates,dt):
    """Body-rate right-product exponential, midpoint endpoints, numpy float64."""
    q=normalize(initial_quaternion);previous=np.asarray(initial_rates,float);result=[]
    for k in range(rates.shape[1]):
        rotation=(previous+rates[:,k])*.5*dt[:,k,None]
        magnitude=np.linalg.norm(rotation,axis=-1,keepdims=True)
        dq=np.concatenate([np.cos(magnitude*.5),rotation*(.5*np.sinc(magnitude/(2*np.pi)))],axis=-1)
        scalar=q[:,:1]*dq[:,:1]-np.sum(q[:,1:]*dq[:,1:],axis=-1,keepdims=True)
        vector=q[:,:1]*dq[:,1:]+dq[:,:1]*q[:,1:]+np.cross(q[:,1:],dq[:,1:])
        q=normalize(np.concatenate([scalar,vector],axis=-1));result.append(q.copy());previous=rates[:,k]
    return np.stack(result,axis=1)


def shapley(values,players):
    """Exact additive allocation of nonlinear tape substitutions; no causal claim."""
    output=[]
    for j in range(players):
        contribution=np.zeros_like(values[0],dtype=float)
        for mask in range(1<<players):
            if mask&(1<<j):continue
            size=mask.bit_count();weight=math.factorial(size)*math.factorial(players-size-1)/math.factorial(players)
            contribution+=weight*(values[mask|(1<<j)]-values[mask])
        output.append(contribution)
    return np.stack(output,axis=-1)


def reference_activity(reference,scale):
    """Known planned-reference variation only, no future observed response."""
    delta=np.diff(np.asarray(reference,float),axis=1)/np.asarray(scale,float)
    return np.sqrt(np.mean(delta**2,axis=(1,2)))


def rate_tape_attribution(initial_q,initial_w,baseline,aux,truth_q,dt,kind):
    players=3 if kind=='axis' else 2
    if kind not in ['axis','time']:raise ValueError(kind)
    values={}
    for mask in range(1<<players):
        tape=baseline.copy()
        for j in range(players):
            if not mask&(1<<j):continue
            if kind=='axis':tape[:,:,j]=aux[:,:,j]
            elif j==0:tape[:,:5]=aux[:,:5]
            else:tape[:,5:]=aux[:,5:]
        values[mask]=angle(integrate(initial_q,initial_w,tape,dt),truth_q)**2
    contributions=shapley(values,players)
    np.testing.assert_allclose(contributions.sum(-1),values[(1<<players)-1]-values[0],atol=1e-12)
    return contributions,values[0],values[(1<<players)-1]
