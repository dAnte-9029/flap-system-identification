"""Evaluation BASE is independent of Macro admission and supports factual clocks."""
from dataclasses import replace
import pytest
import torch
from system_identification.models.pacld import PACLD,Config
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.models.pahbd import Config as TruthConfig
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.evaluation.pacld_gpu import predict

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_explicit_base_and_arbitrary_queries(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 torch.set_num_threads(1);torch.manual_seed(17);cfg=Config(width=8)
 ref=PACLD(cfg).double();batch=BatchedPACLD(cfg).to(device);batch.load_state_dict(ref.state_dict())
 w=controlled_periodic_window(TruthConfig(),.2)[0]
 w=replace(w,metadata=dict(w.metadata,split='validation',actual_horizon_s=.2,cycle_pairs=[]))
 times=[0.,.037,.101,.199]
 result=predict(batch,[w,w],[times,times])
 with torch.no_grad():
  belief,modes=ref.predict_pair(w,times)
 for r in result:
  torch.testing.assert_close(r['belief'].x,belief.x,rtol=1e-8,atol=2e-10)
  for mode in ('FULL','BASE'):
   torch.testing.assert_close(torch.stack([s.x for s in r['predictions'][mode]]),torch.stack([s.x for s in modes[mode]]),rtol=1e-8,atol=2e-10)
 with pytest.raises(ValueError):predict(batch,[w],[[.201]])
