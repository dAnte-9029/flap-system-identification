import copy
import torch
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import window_loss
import pytest

@pytest.mark.parametrize('mode',['A','B'])
def test_exact_updates(mode):
    torch.set_num_threads(1);torch.manual_seed(17)
    c=Config(width=16,latent=4,mode=mode)
    a=PAHBD(c).double();b=copy.deepcopy(a)
    # Original forward, including repeated canonicalization.
    def original(w,times=None):
        belief=b.encode(w)
        return belief,b.flow.rollout(belief,times,w.controls)
    b.forward=original
    w,_=controlled_periodic_window(c,.05)
    oa=torch.optim.Adam(a.parameters());ob=torch.optim.Adam(b.parameters())
    for _ in range(3):
        for m,o in ((a,oa),(b,ob)):
            o.zero_grad(set_to_none=True)
            loss,*_=window_loss(m,w);loss.backward()
            if m is a:expected=loss.detach().clone()
            else:assert torch.equal(expected,loss)
        for x,y in zip(a.parameters(),b.parameters()):
            assert (x.grad is None)==(y.grad is None)
            if x.grad is not None:assert torch.equal(x.grad,y.grad)
        for m,o in ((a,oa),(b,ob)):
            torch.nn.utils.clip_grad_norm_(m.parameters(),10.);o.step()
        for x,y in zip(a.parameters(),b.parameters()):
            assert torch.equal(x,y)
            for k,v in oa.state[x].items():
                assert torch.equal(v,ob.state[y][k]) if torch.is_tensor(v) else v==ob.state[y][k]
