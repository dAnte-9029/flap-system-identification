import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_rate_reference_v23 import errors


def test_full_prefix_and_endpoint_differ_and_ignore_later_errors():
    true={'rates_b':np.zeros((1,3,3)),'velocity_n':np.zeros((1,3,3)),
          'quaternion':np.array([[[1.,0,0,0]]*3])}
    pred={k:v.copy() for k,v in true.items()}
    pred['rates_b'][0,:,1]=[0,2,99]
    pred['quaternion']*=-1
    wave,end=errors(pred,true,2)
    np.testing.assert_allclose(wave['q'],[2]);np.testing.assert_allclose(end['q'],[4])
    np.testing.assert_allclose(wave['attitude'],[0])
