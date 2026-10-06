import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_conditional_excitation import build_rows
from system_identification.data.trajectory_dataset import STATE_COLUMNS,CONTROL_COLUMNS


def samples():
    rows=[]
    for log in ['a','b']:
        for i in range(40):
            r=dict(log_id=log,segment_id=0,sample_in_segment=i,timestamp_us=1000000+i*20000,valid_core=True,relative_flap_phase_rad=.2*i,flap_frequency_hz=4.)
            r.update({k:0. for k in STATE_COLUMNS});r['attitude_q_w']=1.;r['velocity_ned_m_s_x']=8.
            r.update({k:.001*i for k in CONTROL_COLUMNS});rows.append(r)
    return pd.DataFrame(rows)


def test_current_target_action_does_not_leak_into_features():
    s=samples();xs,xh,y,ids=build_rows(s);changed=s.copy()
    changed.loc[(changed.log_id=='a')&(changed.sample_in_segment==20),list(CONTROL_COLUMNS)]=.9
    xs2,xh2,y2,ids2=build_rows(changed);i=ids.index[(ids.log_id=='a')&(ids.sample_in_segment==20)][0]
    np.testing.assert_array_equal(xs[i],xs2[i]);np.testing.assert_array_equal(xh[i],xh2[i]);assert not np.array_equal(y[i],y2[i])
    assert (ids2==ids).all().all()


def test_histories_do_not_cross_log_starts_or_invalid_gaps():
    s=samples();s.loc[(s.log_id=='a')&(s.sample_in_segment==12),'valid_core']=False
    _,_,_,ids=build_rows(s)
    assert (ids.sample_in_segment>=10).all()
    assert not ((ids.log_id=='a')&ids.sample_in_segment.between(12,22)).any()
