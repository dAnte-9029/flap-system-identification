import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from run_joint_command_response import patterns,flight_macro,roll

def test_joint_patterns_allow_other_channels():
 u=np.zeros((3,50,4));u[0,1:,0]=1;u[0,1:,3]=1;u[1,1:25,0]=1;u[1,25:,0]=-1
 labels,active,_=patterns(u,np.ones(4),np.ones(4)*.1)
 assert labels.tolist()==['sustained','reversing','lower_change'];assert active[0].sum()==2

def test_macro_flight_not_pooled():
 v=np.array([[1.],[1.],[9.]]);ids=np.array(['a','a','b']);mask=np.ones(3,bool)
 np.testing.assert_allclose(flight_macro(v,ids,mask),[2.])

def test_wrapped_roll_sign_invariant():
 q=np.array([[.5,.5,.5,.5]])
 np.testing.assert_allclose(roll(q),roll(-q))
