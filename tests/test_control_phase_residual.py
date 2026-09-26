import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from screen_control_phase_residual import harmonic_increment


def history():
    phi=np.linspace(-4*np.pi,0,26)
    x=np.zeros((2,26,12));x[:,:,9]=np.sin(phi);x[:,:,10]=np.cos(phi)
    x[:,:,3]=2*np.sin(phi)+.5
    x[:,:,4]=np.cos(phi)-.2
    x[:,:,5]=.7
    return x


def test_zero_initial_increment_and_constant_rate():
    x=history();y=harmonic_increment(x,np.zeros((2,5)),2)
    np.testing.assert_array_equal(y,0)
    z=harmonic_increment(x,np.tile(np.linspace(0,2,5),(2,1)),2)
    np.testing.assert_allclose(z[:,:,2],0,atol=1e-12)


def test_known_periodic_change_not_mean_or_trend():
    x=history();delta=np.array([[0,np.pi/2,np.pi]]*2)
    y=harmonic_increment(x,delta,1)
    np.testing.assert_allclose(y[:,:,0],2*np.sin(delta),atol=.06)
    np.testing.assert_allclose(y[:,:,1],np.cos(delta)-1,atol=.06)


def test_future_phase_tail_cannot_change_prefix_and_input_is_unchanged():
    x=history();before=x.copy();phase=np.tile(np.linspace(0,2,10),(2,1));poison=phase.copy();poison[:,5:]=100
    a=harmonic_increment(x,phase,2);b=harmonic_increment(x,poison,2)
    np.testing.assert_array_equal(a[:,:5],b[:,:5]);np.testing.assert_array_equal(x,before)
