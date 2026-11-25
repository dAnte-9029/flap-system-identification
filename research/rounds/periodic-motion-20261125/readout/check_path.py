import json
import torch
from fit import load,get_base,HERE,FlightPredictor
from native_path import query

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    b,_=load('train','cuda')
    state=torch.load(get_base('incumbent17'),weights_only=True,map_location='cpu')
    c=FlightPredictor(state['mean'],state['std']).cuda().core
    c.load_state_dict(state)
    i=torch.arange(0,len(b['hist']),31,device='cuda')
    vals=[]
    for sel in i.split(512):
        h=b['hist'][sel,:180];u=b['u0'][sel];t=b['time'][sel];p=b['pooled'][sel]
        a=b['time_native'][sel];q=b['pooled_native'][sel];y=b['y_native'][sel]
        reference=c(h,p,u,t)
        same=query(c,h,u,t,p,t,p)
        fixed=query(c,h,u,t,p,a,q)
        altered=c(h,q,u,a)
        vals.append(torch.stack([(reference-same).abs().amax().expand(2),(fixed-altered).square().sum(-1).mean((0,1)),
             (fixed-y).square().sum(-1).mean((0,1)),(altered-y).square().sum(-1).mean((0,1))]).cpu())
    z=torch.stack(vals)
    out=dict(train_windows=len(i),canonical_endpoint_max_error=z[:,0].max().item(),
             native_offgrid_path_disagreement_rms=z[:,1].mean(0).sqrt().tolist(),
             canonical_prefix_native_rms=z[:,2].mean(0).sqrt().tolist(),jittered_prefix_native_rms=z[:,3].mean(0).sqrt().tolist())
    assert out['canonical_endpoint_max_error']<1e-5
    print(out)
    (HERE/'path_check.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
