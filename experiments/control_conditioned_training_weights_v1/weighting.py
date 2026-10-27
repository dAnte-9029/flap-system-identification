"""Pure command/time functions. No state targets, model output, fit or file I/O."""
import numpy as np

def coordinates(u):
 u=np.asarray(u,dtype=np.float64)
 return np.stack((u[...,0],(u[...,1]+u[...,2])/2,(u[...,2]-u[...,1])/2,u[...,3]),axis=-1)

def canonical(t,u):
 t=np.asarray(t,np.int64);u=np.asarray(u,np.float64)
 if len(t)!=len(u)+1 or np.any(np.diff(t)<=0):raise ValueError('strict half-open tape required')
 keep=np.r_[True,np.any(u[1:]!=u[:-1],axis=1)]
 return np.r_[t[:-1][keep],t[-1]],u[keep]

def integral(t,v,at):
 """Exact piecewise-constant integral from t[0]; no extrapolation."""
 at=np.asarray(at,np.int64);v=np.asarray(v,float)
 if np.any(at<t[0]) or np.any(at>t[-1]):raise ValueError('unsupported query')
 cum=np.r_[0.,np.cumsum(np.diff(t)*v)]
 ix=np.searchsorted(t,at,side='right')-1;ix=np.minimum(ix,len(v)-1)
 return cum[ix]+(at-t[ix])*v[ix]

def compute_weights(t,u,outputs,mask,*,auto_stable=True,supported_until=None):
 """t uses relative integer us from decision; t[0]=0, last boundary explicit.

 supported_until is the end of a valid control prefix, determined from causal
 command ages and registered segment support. Earlier output weights do not
 depend on command values/support after that output. Padding weights are 1.
 """
 outputs=np.asarray(outputs,np.int64);mask=np.asarray(mask,bool)
 result=dict(w0=1.,w1=1.,w2=np.ones(len(outputs)),w1_supported=False,
             w2_supported=np.zeros(len(outputs),bool),delta_d=np.nan,
             mean_change=np.full(4,np.nan),rms_change=np.full(4,np.nan),
             exposure_us=np.zeros(len(outputs)),duration300_us=np.nan)
 try:
  t,u=canonical(t,u)
 except (ValueError,IndexError):return result
 if t[0]!=0 or len(u)==0:return result
 # Invalid later values cut the usable prefix; never retrospectively disable early steps.
 good=np.isfinite(u).all(1)&(u[:,0]>=0)&(u[:,0]<=1)&(abs(u[:,1:])<=1).all(1)
 invalid=np.flatnonzero(~good);end=int(t[-1] if not len(invalid) else t[invalid[0]])
 if supported_until is not None:end=min(end,int(supported_until))
 if end<=0:return result
 ix=np.searchsorted(t,end,side='left');tt=np.r_[t[:ix],end];uu=u[:ix]
 cu=coordinates(uu);du=cu-cu[0]
 if end>=300000:
  means=np.array([integral(tt,du[:,j],300000)/300000 for j in range(4)])
  rms=np.sqrt([integral(tt,du[:,j]**2,300000)/300000 for j in range(4)])
  result.update(delta_d=float(means[2]),mean_change=means,rms_change=np.asarray(rms),w1_supported=True,
                duration300_us=float(integral(tt,(abs(du[:,2])>=.05).astype(float),300000)))
  if auto_stable:result['w1']=1+float(np.clip((abs(means[2])-.02)/.08,0,1))
 active=mask&(outputs>=0)&(outputs<=end)
 result['w2_supported']=active
 ok=np.flatnonzero(active);hi=outputs[ok];lo=np.maximum(0,hi-300000)
 exposure=integral(tt,(abs(du[:,2])>=.05).astype(float),hi)-integral(tt,(abs(du[:,2])>=.05).astype(float),lo)
 exposure=np.clip(exposure,0,300000);result['exposure_us'][ok]=exposure
 if auto_stable:result['w2'][ok]=1+np.clip(exposure/100000,0,1)
 return result

def ess(w,axis=None):
 w=np.asarray(w,np.float64);den=np.sum(w*w,axis=axis)
 return np.divide(np.sum(w,axis=axis)**2,den,out=np.zeros_like(den,dtype=float),where=den>0)
