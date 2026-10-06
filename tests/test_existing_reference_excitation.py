import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mine_existing_reference_excitation import pulse_candidates, runs, euler


def test_signed_returning_pulses_and_no_overlap():
    ref=np.zeros((1200,2));ref[100:140,1]=2;ref[700:750,1]=-2
    found=pulse_candidates(ref,np.ones(1200,bool),1)
    assert len(found)==2
    assert [f['amplitude_deg'] for f in found]==[2,-2]
    assert all(f['single_axis'] for f in found)
    assert found[1]['index']-found[0]['index']>=350


def test_mode_gap_and_nonreturning_step_not_admitted():
    ref=np.zeros((800,2));ref[100:,1]=2
    assert not pulse_candidates(ref,np.ones(800,bool),1)
    ref[140:,1]=0
    valid=np.ones(800,bool);valid[120]=False
    assert not pulse_candidates(ref,valid,1)


def test_other_axis_tag_and_threshold_sensitivity():
    ref=np.zeros((800,2));ref[100:140,1]=.8;ref[100:140,0]=2
    found=pulse_candidates(ref,np.ones(800,bool),.5)
    pitch=[f for f in found if f['axis']==1]
    assert len(pitch)==1 and not pitch[0]['single_axis']
    assert not [f for f in pulse_candidates(ref,np.ones(800,bool),1) if f['axis']==1]


def test_runs_and_normalized_quaternion():
    assert runs(np.array([True,True,False,True]))==[(0,2),(3,4)]
    theta=.2
    q=np.array([[np.cos(theta/2),0,np.sin(theta/2),0]])*2
    np.testing.assert_allclose(euler(q),[[0,np.rad2deg(theta)]],atol=1e-12)
    assert np.isnan(euler(np.zeros((1,4)))).all()


def test_ramp_screen_accepts_finite_transition_and_rejects_broken_mode():
    from mine_existing_reference_ramps import pulse_candidates as ramps
    ref=np.zeros((800,2))
    ref[100:115,1]=np.linspace(0,2,15)
    ref[115:150,1]=2
    ref[150:165,1]=np.linspace(2,0,15)
    valid=np.ones(800,bool)
    found=ramps(ref,valid,1)
    assert len(found)==1 and found[0]['amplitude_deg']==2
    valid[120]=False
    assert not ramps(ref,valid,1)
