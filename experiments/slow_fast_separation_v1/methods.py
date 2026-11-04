"""Fixed synchronous separation operators. No data loading or parameter selection."""
import numpy as np
GRID=2*np.pi*(np.arange(8)+.5)/8

def weights(t,lo,hi):
    edges=np.r_[lo,(t[:-1]+t[1:])/2,hi]
    w=np.diff(edges)/(hi-lo)
    assert np.all(w>=0) and np.isclose(w.sum(),1)
    return w

def linear(x,t,y):
    j=np.clip(np.searchsorted(t,x)-1,0,len(t)-2)
    return y[j]+(x-t[j])*(y[j+1]-y[j])/(t[j+1]-t[j])

def cell(t,p,y,lo,hi):
    w=weights(t,lo,hi);g=np.full(8,np.nan);gt=np.full(8,np.nan)
    for j,q in enumerate(GRID):
        k=np.searchsorted(p,q)
        if 0<k<len(p) and p[k]>p[k-1]:
            z=(q-p[k-1])/(p[k]-p[k-1]);g[j]=y[k-1]+z*(y[k]-y[k-1]);gt[j]=t[k-1]+z*(t[k]-t[k-1])
    return dict(t=t,p=p,y=y,w=w,mu=w@y,tau=w@t,g=g,gt=gt,lo=lo,hi=hi)

def design(t,p,ref):
    return np.column_stack([np.ones(len(t)),t-ref]+[f(h*p) for h in (1,2,3) for f in (np.cos,np.sin)])

def fit(cycles,n,holdout=False):
    ids=np.arange(n-25,n+26);ids=ids[ids%5!=2] if holdout else ids
    cs=[cycles[i] for i in ids];target=cycles[n];ref=(target['lo']+target['hi'])/2
    t=np.concatenate([c['t'] for c in cs]);p=np.concatenate([c['p'] for c in cs]);y=np.concatenate([c['y'] for c in cs]);w=np.concatenate([np.full(len(c['t']),1/np.sqrt(len(c['t']))) for c in cs])
    X=design(t,p,ref);coef=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
    trendcoef=np.linalg.lstsq(X[:,:2]*w[:,None],y*w,rcond=None)[0]
    qt=target['t'];qp=target['p'];Z=design(qt,qp,ref)
    a=Z[:,2:]@coef[2:];ap=Z@coef;baseline=Z[:,:2]@trendcoef
    tau=np.array([c['tau'] for c in cs]);mu=np.array([c['mu'] for c in cs]);d=np.full(8,np.nan)
    for j in range(8):
        good=np.array([np.isfinite(c['g'][j]) for c in cs]);offset=ids[good]-n
        if good.sum()<8 or (offset<0).sum()<2 or (offset>0).sum()<2:continue
        tg=np.array([c['gt'][j] for c in cs])[good];g=np.array([c['g'][j] for c in cs])[good]
        residual=g-linear(tg,tau,mu);xx=np.vander(offset/25,4,increasing=True)
        d[j]=np.linalg.lstsq(xx,residual,rcond=None)[0][0]
    if np.isfinite(d).all():
        b=np.interp(qp,GRID,d,period=2*np.pi);b-=target['w']@b;bp=linear(qt,tau,mu)+b
    else:b=np.full(len(qt),np.nan);bp=b.copy()
    return baseline,ap,bp,a,b,d

def causal_template(cs, lead=1):
    assert len(cs)==51
    tau=np.array([c['tau'] for c in cs]);mu=np.array([c['mu'] for c in cs]);d=np.full(8,np.nan)
    for j in range(8):
        good=np.array([np.isfinite(c['g'][j]) for c in cs]);m=np.arange(-51,0)[good]
        if good.sum()<8:continue
        t=np.array([c['gt'][j] for c in cs])[good];y=np.array([c['g'][j] for c in cs])[good]-linear(t,tau,mu)
        coef=np.linalg.lstsq(np.column_stack([np.ones(len(m)),m/51]),y,rcond=None)[0]
        d[j]=coef[0]+coef[1]*(lead-1)/51
    # A past-only temporal measure for zero-mean centering; never current-cycle future times.
    if np.isfinite(d).all():d-=cs[-1]['w']@np.interp(cs[-1]['p'],GRID,d,period=2*np.pi)
    slope=np.linalg.lstsq(np.column_stack([np.ones(3),tau[-3:]-tau[-1]]),mu[-3:],rcond=None)[0]
    return d,slope,tau[-1]
