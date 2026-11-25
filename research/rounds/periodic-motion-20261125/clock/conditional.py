"""Causal command-conditioned phase increment for an already trained carrier."""
import json
import math
from pathlib import Path
import torch


def corrected_model(parent, coefficients):
    """All parameters and the origin component are identical to the original checkpoint."""
    class Prediction(parent):
        def components(self,hist,pooled,u0,time):
            standardized=(hist-self.mean)/self.std
            initial=self.encoder(standardized)
            slow=self.slow_motion(initial[:,None,None],pooled,u0,time)
            context=torch.cat((initial,standardized[:,180:]),dim=-1)
            co=self.carrier_coefficients(context).reshape(-1,2,3,2)
            co=co*(hist[:,171]*hist[:,172])[:,None,None,None]
            f=hist[:,165,None,None]
            history=hist[:,:180].reshape(-1,6,30)
            avail=history[...,22]
            fav=(history[...,15]*avail).sum(1)/avail.sum(1).clamp(min=1)
            m=u0[:,0,None,None]
            mass=-torch.expm1(-time)
            avg=pooled[...,8]/torch.where(time>0,mass,1.)
            ramp=time+.2*torch.expm1(-time/.2)
            b=coefficients
            extra=ramp*(b[0]+b[1]*(f-3.8)+b[2]*(m-.5)+b[3]*(f-fav[:,None,None])+b[4]*(avg-m))
            angle=2*math.pi*(time*f+extra)
            pair=torch.stack((torch.cos(angle),torch.sin(angle)),-1)
            wave=(co[:,None]*pair.unsqueeze(-2)).sum(-1)
            env=torch.exp(-torch.nn.functional.softplus(self.decay_logits)*time.unsqueeze(-1))
            return slow,env*wave,co[:,None,:,:,0]
    return Prediction


def load_coefficients():
    return json.loads((Path(__file__).resolve().parent/"clock_fit.json").read_text())["coefficients"]
