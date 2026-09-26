import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from audit_stabilized_roll_chain import states,align,euler

def test_hysteresis_and_reversal():
 np.testing.assert_array_equal(states([0,.11,.08,.04,-.12,-.07,0]),[0,1,1,0,-1,-1,0])
def test_past_only_freshness():
 d={'timestamp':np.array([100,200]),'x':np.array([1.,2.])}
 x,ok=align(d,'x',np.array([50,100,150,300]),.000075)
 np.testing.assert_array_equal(ok,[False,True,True,False]);assert x[2]==1

def test_roll_quaternion():
 a=np.deg2rad(20);q=np.array([[np.cos(a/2),np.sin(a/2),0,0]])
 np.testing.assert_allclose(euler(q),[[20,0]],atol=1e-12)

def test_exact_firmware_stabilized_mode_constant():
 import json
 root=Path(__file__).resolve().parents[1]/'docs/analysis/results/stabilized_roll_chain_v1'
 evidence=json.loads((root/'firmware_evidence.json').read_text())
 messages=[r for r in evidence if r['path'].endswith('VehicleStatus.msg')]
 assert messages and all(r['available'] for r in messages)
 assert all(any('NAVIGATION_STATE_STAB=15' in ''.join(line.split()) for line in r['lines']) for r in messages)
 assert 'nav_state==15' in json.loads((root/'protocol.json').read_text())['selection']
