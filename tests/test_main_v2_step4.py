"""Physical contracts for the offline audit, using synthetic signals only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from main_v2_step4_tools import offline_derivatives,integrate_intervals,harmonic_design,nearest_indices


def test_raw_derivative_integrates_exact_native_increment():
    t=np.cumsum(.02+.001*np.sin(np.arange(400)))
    y=np.column_stack([np.sin(t),t*t])
    d=offline_derivatives(t,y)
    for k in [2,5,10,25]:
        np.testing.assert_allclose(integrate_intervals(d['D0_raw'],np.diff(t),k),y[k:]-y[:-k],atol=1e-12)


def test_lowpass_removes_high_frequency_without_deleting_fundamental():
    t=np.arange(1000)*.02
    y=(np.sin(2*np.pi*3*t)+.3*np.sin(2*np.pi*19*t))[:,None]
    a=offline_derivatives(t,y)['D1_lp6'][25:-25,0]
    expected=np.diff(np.sin(2*np.pi*3*t))/.02
    assert np.sqrt(np.mean((a-expected[25:-25])**2))<1


def test_reanchoring_erases_physical_position_but_log_phase_retains_it():
    phi=np.array([.3,2.1,4.])
    relative=harmonic_design(phi-phi,1)
    assert np.unique(relative,axis=0).shape[0]==1
    assert np.unique(harmonic_design(phi,1),axis=0).shape[0]==3


def test_neighbors_respect_exclusion_and_identity():
    q=np.array([[0.],[4.]])
    ref=np.arange(6)[:,None]
    ban=np.zeros((2,6),bool);ban[0,:2]=True;ban[1,4]=True
    idx,d=nearest_indices(q,ref,ban,k=2,device='cpu')
    assert idx[0,0]==2
    assert 4 not in idx[1]
    assert np.isfinite(d).all()


def test_corrected_history_preserves_log_phase_across_origins():
    from train_main_v2_step4_probes import phase_history
    physical=np.linspace(.1,5,26)
    x=np.zeros((2,420))
    for row,anchor in enumerate([1.,2.]):
        h=x[row,:312].reshape(26,12)
        h[:,9]=np.sin(physical-anchor);h[:,10]=np.cos(physical-anchor)
    y=phase_history(x,np.array([1.,2.]))
    np.testing.assert_allclose(y[0,:312],y[1,:312],atol=1e-14)
