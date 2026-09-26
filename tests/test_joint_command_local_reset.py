import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from run_joint_command_local_reset import endpoint_indices,paired_rmse

def test_matched_endpoints():
 np.testing.assert_array_equal(endpoint_indices(range(0,50,5)),np.arange(5,51,5))
def test_paired_error_not_mean_predictions():
 a,b=paired_rmse(np.array([[1.],[9.]]),np.array([[4.],[4.]]));np.testing.assert_allclose(a,[np.sqrt(5)]);np.testing.assert_allclose(b,[2])
def test_reject_unpaired():
 with pytest.raises(ValueError):paired_rmse(np.zeros((2,1)),np.zeros((1,1)))
