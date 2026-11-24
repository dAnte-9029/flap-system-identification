"""Stage-one candidates; validation always uses the archived shared selection score."""
import argparse
import json
from types import SimpleNamespace
from common import *
from train import run
from harmonic import candidate_model
def drift_units(data):
    """Training-only flight-pooled window RMS error of a constant-v, constant-omega rollout."""
    initial,y,t,ends=data[1],data[2],data[3],data[6]
    total=torch.zeros(2,2,device='cuda',dtype=torch.float64)
    count=torch.zeros_like(total)
    for j in range(0,len(initial),2048):
        s,obs,times,ix=[x[j:j+2048] for x in (initial,y,t,ends)]
        for k,comp in enumerate([0,2]):
            i=ix[:,2:,comp].clamp_min(0)
            rows=torch.arange(len(s),device=s.device)[:,None]
            dt=times[rows,i,0 if comp==0 else 1,None]
            if comp==0:
                pred=s[:,None,:3]+dt*s[:,None,3:6]
                sq=(pred-obs[rows,i,:3]).square().sum(-1)
            else:
                pred=qmul(s[:,None,6:10],qexp(s[:,None,10:]*dt))
                sq=geodesic(pred,obs[rows,i,6:10]).square()
            ok=ix[:,2:,comp]>=0
            total[:,k]+=(sq*ok).sum(0)
            count[:,k]+=ok.sum(0)
    return (total/count).float().clamp_min(1e-8)


def main(candidate,gpu):
    state={}
    def data(cache,split):
        values,ids=conditional_data(cache,split,TAPES)
        if split=='train':
            state['units']=drift_units(values)
        return values,ids
    def objective(model,b,units):
        base,score,errors,valid=conditional_loss(model,b,units)
        # Keep the shared score and per-component MSE exactly as defined for all baselines.
        late=torch.stack([errors[:,2:,0],errors[:,2:,2]],-1)
        mask=torch.stack([valid[:,2:,0],valid[:,2:,2]],-1)
        drift=(late/state['units']*mask).sum()/mask.sum().clamp_min(1)
        return base+.5*drift,score,errors,valid
    def build(mean,scale):
        return candidate_model(candidate,mean,scale)
    cfg=SimpleNamespace(cache=HISTORY,out=OUT/candidate,gpu=gpu,seed=17,model=candidate,epochs=24,batch=1024)
    extra=dict(macro_dt=.1,harmonic_integration='exact segment increment' if candidate=='c2' else 'left derivative',control_tapes=str(TAPES),
               late_pose_loss_weight=.5,late_horizons_s=[1.,2.],drift_scaling='train constant-v/omega endpoint MSE',
               selection='unchanged shared flight-equal validation score')
    run(cfg,load_data=data,build_model=build,loss_fn=objective,extra_config=extra)
    (cfg.out/'drift_scales_squared.json').write_text(json.dumps(state['units'].tolist()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('candidate',choices=['c1','c2'])
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    main(a.candidate,a.gpu)
