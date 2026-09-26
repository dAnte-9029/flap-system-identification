import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
from audit_september_data_sufficiency import nonoverlap

def test_nonoverlap_earliest_and_boundary():
    f=pd.DataFrame(dict(window_id=['b','a','c','d'],origin_timestamp_us=[.2,0.,1.,1.1],end_us=[1.2,1.,2.,2.1]))
    assert nonoverlap(f)==['a','c']
def test_empty_and_no_mutation():
    f=pd.DataFrame(dict(window_id=['b','a'],origin_timestamp_us=[1.,0.],end_us=[2.,1.]));before=f.copy()
    assert nonoverlap(f.iloc[:0])==[]
    assert nonoverlap(f)==['a','b'];pd.testing.assert_frame_equal(f,before)
def test_saved_coverage_invariants():
    p=Path(__file__).resolve().parents[1]/'docs/analysis/results/september_data_sufficiency_v1'
    f=pd.read_csv(p/'excitation_coverage.csv')
    assert (f.n_nonoverlap<=f.n_windows).all()
    assert (f.positive+f.negative==f.n_windows).all()
    assert (f.nonoverlap_positive+f.nonoverlap_negative==f.n_nonoverlap).all()
    assert f.groupby('partition').total_origins.first().to_dict()=={'Sep19':3202,'Sep8':871,'train':28293,'validation':2582}
