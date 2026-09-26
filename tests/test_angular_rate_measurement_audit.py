import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_angular_rate_measurement import past_index,sample_aligned_index

def test_past_only_and_duplicate_latest():
    np.testing.assert_array_equal(past_index([10,20,20,30],[5,10,20,29]),[-1,0,2,2])
def test_sample_and_publication_both_constrained():
    np.testing.assert_array_equal(sample_aligned_index([15,25,35],[10,20,30],[22,35,35],[21,21,9]),[0,1,-1])
def test_reject_unsorted():
    with pytest.raises(ValueError):past_index([20,10],[20])
def test_no_array_mutation():
    p=np.array([15,25]);s=np.array([10,20]);before=(p.copy(),s.copy())
    sample_aligned_index(p,s,[25],[12]);np.testing.assert_array_equal(p,before[0]);np.testing.assert_array_equal(s,before[1])
