"""Numerical and information-boundary tests for the first velocity/rate experiment."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from system_identification.models.velocity_rate_gru import (VelocityRateGRU,integrate,geodesic,rotation6,baseline,errors)
from system_identification.data.velocity_rate_dataset import VelocityRateDataset,action_replay
from prepare_velocity_rate_learning import interval_events,CACHE
from train_velocity_rate_learning import INPUT_KEYS,batch,window_loss


def norm():
    return dict(mean=[0]*11,scale=[1]*11,output_scale=[.1]*6)


def fixture():
    h=torch.zeros(3,27,20); h[...,6]=1; h[...,18]=1; h[...,19]=5
    return dict(history=h,history_mask=torch.ones_like(h,dtype=torch.bool),history_dt=torch.full((3,27),.02),
                history_length=torch.tensor([26,27,26]),initial=h[:,-1,:13].clone(),dt=torch.full((3,10),.02),
                events=torch.zeros(3,10,2,5),event_length=torch.ones(3,10,dtype=torch.long))


def test_kinematic_frames_trapezoid_and_quaternion_sign():
    s=torch.zeros(1,13,dtype=torch.float64); s[:,6]=1; s[:,3]=2; s[:,12]=1
    out=integrate(s,torch.tensor([[4.,0,0]]),s[:,10:13],torch.tensor([.1]))
    torch.testing.assert_close(out[:,:3],torch.tensor([[.3,0,0]],dtype=torch.float64))
    assert abs(float(geodesic(out[:,6:10],s[:,6:10]))-.1)<1e-7
    torch.testing.assert_close(rotation6(out[:,6:10]),rotation6(-out[:,6:10]))
    assert geodesic(out[:,6:10],-out[:,6:10]).item()<1e-12


def test_event_boundaries_order_and_duration():
    motor=pd.DataFrame({'timestamp_us':[80,150,200,250],'timestamp_sample_us':[-1]*4,'control[0]':[.1,.9,.4,.5]})
    servo=pd.DataFrame({'timestamp_us':[70,160,260],'timestamp_sample_us':[-1]*3,'control[0]':[.2,.8,.5],'control[1]':[-.3,-.4,-.5],'control[2]':[.6,.7,.8]})
    ev,le=interval_events(motor,servo,np.array([100,200]),np.array([200,300]),1000)
    assert le.tolist()==[3,3]
    np.testing.assert_allclose(ev[0,:3,0],[.1,.9,.9])
    np.testing.assert_allclose(ev[0,:3,4],[50e-6,10e-6,40e-6])
    assert ev[1,0,0]==np.float32(.4)  # event exactly at next boundary belongs there
    replay=action_replay(motor,servo,np.array([100,200,300]),1000)
    np.testing.assert_allclose(ev[:,0,:4],replay['step_control'])


@pytest.mark.parametrize('control',[False,True])
def test_autonomy_sign_mask_and_no_position_feature(control):
    torch.manual_seed(3); m=VelocityRateGRU(norm(),control=control)
    torch.nn.init.normal_(m.head.weight,std=.02)
    x=fixture(); y=m(x)
    z={k:v.clone() for k,v in x.items()}
    z['history'][...,:3]=123456
    z['history'][...,6:10]*=-1
    torch.testing.assert_close(y,m(z),rtol=0,atol=0)
    # Arbitrary future truth does not enter inference; poisoning it is immaterial.
    z['target']=torch.full((3,10,13),float('nan'))
    z['target_phase_frequency']=torch.full((3,10,2),999.)
    torch.testing.assert_close(y,m(z),rtol=0,atol=0)
    if not control:
        z['events'].fill_(float('nan')); z['event_length'].fill_(99999)
        z['history'][...,13:17].fill_(999)
        torch.testing.assert_close(y,m(z),rtol=0,atol=0)
    z=fixture(); z['history_mask'][...,17:20]=False
    a=m(z); z['history'][...,17:20]=999
    torch.testing.assert_close(a,m(z),rtol=0,atol=0)
    loss=errors(y,baseline(x['initial'],x['dt'])).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    torch.testing.assert_close(torch.linalg.vector_norm(y[...,6:10],dim=-1),torch.ones(3,10))


def test_geodesic_identity_gradient_and_padding_loss():
    s=torch.zeros(2,3,13); s[...,6]=1; p=s.clone().requires_grad_()
    mask=torch.tensor([[True,True,False],[True,False,False]])
    loss=window_loss(p,s,mask,torch.ones(4),torch.ones(4)).mean()
    loss.backward(); assert torch.isfinite(p.grad).all()
    q=s.clone(); q[:,2,3]=1000
    assert window_loss(q,s,mask,torch.ones(4),torch.ones(4)).sum()==0


@pytest.mark.parametrize('split',['train','validation'])
def test_packed_samples_match_authoritative_loader(split):
    if not (CACHE/f'{split}.npz').exists(): pytest.skip('experiment cache not materialized')
    a=np.load(CACHE/f'{split}.npz')
    ds=VelocityRateDataset(ROOT/'dataset/velocity_rate_multistep_v1',split=split,verify_hashes=False)
    candidates=[0,len(ds)//2,len(ds)-1]
    longer=np.flatnonzero(ds.windows.future_count.to_numpy()==11)
    candidates.append(int(longer[0]))
    for i in candidates:
        sample=ds[i]; inp=sample['inputs']; h=int(a['history_length'][i]); k=len(inp['future_dt_s'])
        np.testing.assert_allclose(a['history'][i,:h,3:],inp['history'][:,3:],rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(a['history_mask'][i,:h],inp['history_mask'])
        np.testing.assert_allclose(a['dt'][i,:k],inp['future_dt_s'],rtol=1e-6)
        expected=sample['targets']['state'].copy(); expected[:,:3]-=inp['initial_state'][:3]
        np.testing.assert_allclose(a['target'][i,:k],expected,rtol=1e-5,atol=3e-5)
        tape=inp['future_control']; grid=np.r_[0,np.cumsum(inp['future_dt_s'])]+sample['metadata']['initial_measurement_us']*1e-6
        for j in range(k):
            count=a['event_length'][i,j]
            ev=a['events'][i,j,:count]
            np.testing.assert_allclose(ev[:,4].sum(),a['dt'][i,j],atol=1e-8)
            np.testing.assert_allclose(ev[0,:4],tape['step_control'][j],atol=1e-7)
            inside=tape['event_step_index']==j
            controls=tape['event_control'][inside]
            # A step boundary is inserted when no native event is exactly at the boundary.
            assert count in (len(controls),len(controls)+1)
            if len(controls): np.testing.assert_allclose(ev[-len(controls):,:4],controls,atol=1e-7)
