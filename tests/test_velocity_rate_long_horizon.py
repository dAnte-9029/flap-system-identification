"""Long-rollout causal prefix invariance with the unchanged GRU inference API."""
import torch
import pytest
from system_identification.models.velocity_rate_gru import VelocityRateGRU


def inputs():
    h=torch.zeros(2,27,20,dtype=torch.float64);h[...,6]=1;h[...,18]=1;h[...,19]=4
    events=torch.randn(2,100,3,5,dtype=torch.float64)
    events[...,4]=torch.tensor([.005,.007,.008],dtype=torch.float64)
    return dict(history=h,history_mask=torch.ones_like(h,dtype=torch.bool),history_dt=torch.full((2,27),.02,dtype=torch.float64),
        history_length=torch.tensor([27,26]),initial=h[:,-1,:13].clone(),dt=torch.full((2,100),.02,dtype=torch.float64),
        events=events,event_length=torch.full((2,100),3,dtype=torch.long))


@pytest.mark.parametrize('control',[False,True])
def test_hundred_step_prefix_never_uses_later_commands_or_future_truth(control):
    torch.set_num_threads(2);torch.manual_seed(1)
    model=VelocityRateGRU(dict(mean=[0]*11,scale=[1]*11,output_scale=[.1]*6),control=control).double().eval()
    with torch.no_grad():torch.nn.init.normal_(model.head.weight,std=.01)
    x=inputs()
    with torch.no_grad():
        y=model(x)
        short={k:v.clone() for k,v in x.items()}
        for k in ['dt','events','event_length']:short[k]=short[k][:,:25]
        torch.testing.assert_close(y[:,:25],model(short),rtol=1e-10,atol=1e-10)
        altered={k:v.clone() for k,v in x.items()}
        altered['events'][:,25:,:,:4]*=100
        altered['target']=torch.full((2,100,13),float('nan'))
        altered['target_phase_frequency']=torch.full((2,100,2),float('nan'))
        torch.testing.assert_close(y[:,:25],model(altered)[:,:25],rtol=0,atol=0)
        if not control:
            altered['history'][...,13:17]=float('nan');altered['events'].fill_(float('nan'))
            altered['event_length'].fill_(99999)
            torch.testing.assert_close(y,model(altered),rtol=0,atol=0)
        assert torch.isfinite(y).all()
        torch.testing.assert_close(torch.linalg.vector_norm(y[...,6:10],dim=-1),torch.ones(2,100,dtype=torch.float64))
