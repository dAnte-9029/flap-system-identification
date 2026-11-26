"""Original shared DLinear and PS loss, plus an optional causal linear command readout."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from data import H, DT

AUTHOR = Path("/home/zn/reproduction/PS_Loss/etth1_reproduction")
sys.path[:0] = [str(AUTHOR/".deps"),str(AUTHOR/"official/DLinear")]
from models.DLinear import Model as AuthorDLinear
from exp.exp_main import Exp_Main


class FlightPS(Exp_Main):
    def __init__(self, core):
        self.model=core
        self.kl_loss=nn.KLDivLoss(reduction="none")
        self.patch_len_threshold=24
        self.clamped=0
        self.calls=0

    def fouriour_based_adaptive_patching(self,true,pred):
        # The original stride becomes zero when the dominant period is <4 samples.
        power=torch.fft.rfft(true,dim=1).abs().mean((0,2))
        index=int(power[1:].argmax())+1
        length=min((true.shape[1]//index)//2,self.patch_len_threshold)
        self.clamped+=int(length<2)
        self.calls+=1
        length=max(2,length)
        return self.create_patches(true,length,length//2),self.create_patches(pred,length,length//2)


class Forecast(nn.Module):
    def __init__(self, controlled=False):
        super().__init__()
        self.core=AuthorDLinear(SimpleNamespace(seq_len=32,pred_len=H,enc_in=6,individual=False))
        self.command=nn.Linear(4,6,bias=False) if controlled else None
        if controlled:
            nn.init.zeros_(self.command.weight)

    def grid(self,x):
        # Input is standardized absolute motion. Return standardized increments.
        increment=self.core(x)-x[:,-1:]
        return torch.cat((torch.zeros_like(x[:,:1]),increment),1)

    def at(self,grid,t,u=None):
        b,k,_=t.shape
        idx=(t/DT).clamp(0,H)
        lo=idx.long().clamp_max(H-1)
        alpha=(idx-lo)[...,None]
        values=grid.reshape(b,H+1,2,3)
        def take(i):
            return torch.gather(values,1,i[...,None].expand(b,k,2,3))
        result=torch.lerp(take(lo),take(lo+1),alpha)
        if self.command is not None:
            result=result+torch.einsum("bkcf,cjf->bkcj",u,self.command.weight.reshape(2,3,4))
        return result

    def regular(self,grid,u):
        k=u.shape[1]
        result=grid[:,1:k+1]
        return result if self.command is None else result+self.command(u)
