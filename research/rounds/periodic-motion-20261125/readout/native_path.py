"""Read a frozen recurrent state at off-grid points while preserving its canonical integration path."""
import torch


@torch.inference_mode()
def query(core,hist,u0,canonical,canonical_pooled,t,pooled):
    b=len(hist)
    result=t.new_empty((b,t.shape[1],2,3))
    for ch in (0,1):
        n=(canonical[:,None,:,ch] < t[:,:,None,ch]-1e-7).sum(-1)
        for k in range(canonical.shape[1]+1):
            rows,cols=torch.where(n==k)
            if rows.numel()==0:continue
            old=canonical[rows,:k,ch]
            end=t[rows,cols,ch,None]
            tt=torch.cat((old,end),1)[:,:,None].expand(-1,-1,2)
            cp=canonical_pooled[rows,:k,ch]
            ep=pooled[rows,cols,ch,None,:]
            pp=torch.cat((cp,ep),1).unsqueeze(2).expand(-1,-1,2,-1)
            result[rows,cols,ch]=core(hist[rows],pp,u0[rows],tt)[:,-1,ch]
    return result
