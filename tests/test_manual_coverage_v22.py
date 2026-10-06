import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_manual_coverage_v22 import metrics,activity_bin,rc_pairs


def test_message_updates_are_not_value_changes_and_age_uses_older_time():
    d={'timestamp':np.array([10,20,30]),'timestamp_sample':np.array([8,19,28])}
    ix,age,m=metrics(d,np.array([10,15,20,25,30]),np.array([0.,0.,1.]))
    assert m['unique_messages']==3 and m['updates']==2 and m['value_changes']==1
    np.testing.assert_allclose(age,[.002,.007,.001,.006,.002])


def test_activity_bins_allow_quantization_roundoff_not_large_changes():
    assert activity_bin(.0023809525,1/420)=='at_most_1_count'
    assert activity_bin(2/420,1/420)=='1_to_2_counts'
    assert activity_bin(3/420,1/420)=='over_2_counts'


def test_rc_pairs_reject_stale_samples_and_invalid_channels():
    manual={'timestamp':np.array([10,20]),'timestamp_sample':np.array([9,19]),
            'yaw':np.array([0.,0.]),'valid':np.array([1,1]),'data_source':np.array([1,1])}
    rc={'timestamp':np.array([15,25,30]),'timestamp_last_signal':np.array([14,18,29]),
        'values[3]':np.array([1210,1420,1922]),'rc_lost':np.zeros(3),
        'rc_failsafe':np.zeros(3),'channel_count':np.array([4,4,3])}
    p=rc_pairs(rc,manual,15,30,(3,1000,1420,1922,1))
    assert p.valid_pair.tolist()==[True,False,False]
    np.testing.assert_allclose(p.difference,[-.5,0,1])
