import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_localization_v19 import integrate,angle,shapley,rate_tape_attribution,reference_activity


def test_constant_rate_and_quaternion_sign():
    q=np.array([[1.,0,0,0]]);r=np.zeros((1,50,3));r[:,:,2]=1
    pred=integrate(q,r[:,0],r,np.full((1,50),.02))
    np.testing.assert_allclose(angle(pred,q[:,None]),np.arange(1,51)[None]*.02,atol=1e-12)
    np.testing.assert_allclose(angle(pred,-pred),0,atol=1e-12)


def test_midpoint_uses_observed_origin_once():
    q=np.array([[1.,0,0,0]]);r=np.zeros((1,2,3));r[:,:,1]=1
    pred=integrate(q,np.zeros((1,3)),r,np.full((1,2),.02))
    np.testing.assert_allclose(angle(pred,q[:,None]),[[.01,.03]],atol=1e-12)


def test_shapley_interaction_split_and_efficiency():
    # value=a+2b+6ab: interaction is shared equally, not double-counted.
    values={0:np.array([0.]),1:np.array([1.]),2:np.array([2.]),3:np.array([9.])}
    np.testing.assert_allclose(shapley(values,2),[[4.,5.]])


def test_axis_and_time_tape_attribution():
    q=np.array([[1.,0,0,0]]);initial=np.zeros((1,3));b=np.zeros((1,10,3));a=b.copy();a[:,5:,1]=.5
    truth=np.repeat(q[:,None],10,axis=1);dt=np.full((1,10),.02)
    axes,v0,v1=rate_tape_attribution(q,initial,b,a,truth,dt,'axis')
    np.testing.assert_allclose(axes[:,:,[0,2]],0);np.testing.assert_allclose(axes[:,:,1],v1-v0)
    times,_,_=rate_tape_attribution(q,initial,b,a,truth,dt,'time')
    np.testing.assert_allclose(times[:,:,0],0);np.testing.assert_allclose(times[:,:,1],v1-v0)


def test_reference_activity_uses_scale_and_changes_only():
    x=np.zeros((2,50,5));x[1,:,0]=np.arange(50)
    np.testing.assert_allclose(reference_activity(x,np.ones(5)),[0,1/np.sqrt(5)])
    np.testing.assert_allclose(reference_activity(x+15,np.ones(5)),reference_activity(x,np.ones(5)))
