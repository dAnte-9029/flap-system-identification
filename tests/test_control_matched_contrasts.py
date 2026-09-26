import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_control_matched_contrasts import choose_pairs,nonoverlapping_origins,current_modes


def inputs():
    z=np.zeros((4,6));u=np.zeros((4,5,4));u[:, :,1]=np.arange(4)[:,None]
    dose=u.mean(1)
    return [z,u,dose,np.ones(4)*.1,np.array(['a','b','c','d']),np.ones(4),.5,np.ones(4),1,.25]


def test_pairs_are_disjoint_and_ordered_by_treatment():
    args=inputs();pairs=choose_pairs(*args)
    assert len(pairs)==2
    ids=[]
    for _,hi,lo,delta,_ in pairs:
        ids.extend([hi,lo]);assert args[2][hi,1]>args[2][lo,1] and delta>=.5
    assert len(set(ids))==len(ids)


def test_same_flight_or_different_mode_not_matched():
    args=inputs();args[4][:]='a'
    assert choose_pairs(*args)==[]
    args=inputs();args[5]=np.arange(4)
    assert choose_pairs(*args)==[]


def test_other_input_and_elapsed_calipers():
    args=inputs();args[1][:,:,0]=np.arange(4)[:,None]
    assert choose_pairs(*args)==[]
    args=inputs();args[3]=np.array([.1,.2,.3,.4])
    assert choose_pairs(*args)==[]


def test_nonoverlap_includes_past_history():
    o=pd.DataFrame(dict(log_id=['a']*3,segment_id=[0]*3,start_sample_in_segment=[25,30,61]))
    np.testing.assert_array_equal(nonoverlapping_origins(o,10),[0,2])


def test_current_mode_ignores_duplicate_invalid_rows():
    s=pd.DataFrame(dict(log_id=['a']*3,segment_id=[-1,-1,0],sample_in_segment=[-1,-1,25],nav_state=[0,0,3],valid_core=[False,False,True]))
    o=pd.DataFrame(dict(log_id=['a'],segment_id=[0],start_sample_in_segment=[25]))
    np.testing.assert_array_equal(current_modes(s,o),[3])
