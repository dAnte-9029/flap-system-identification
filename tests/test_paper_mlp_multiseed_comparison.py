"""Frozen MLP factory and paired model-family comparison contracts."""
import copy
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_paper_mlp_multiseed_comparison import build_mlp,parameter_hash,stages
from run_paper_baseline_comparison import build_model as old_factory
from test_paper_baselines import stats,inputs


def test_seed_configurable_factory_matches_original17_and_changes_random_parts():
    s=stats();old=old_factory('B1_MLP',s);fresh=build_mlp(17,s)
    for k,v in old.state_dict().items():assert torch.equal(v,fresh.state_dict()[k])
    a=build_mlp(23,s);b=build_mlp(23,s);c=build_mlp(42,s)
    assert parameter_hash(a)==parameter_hash(b)!=parameter_hash(c)
    for model in (a,b,c,fresh):
        assert sum(p.numel() for p in model.parameters())==5703
        assert all(p.requires_grad for p in model.parameters())
        assert torch.count_nonzero(model.derivative_head.mlp.network[-1].weight)==0
        assert torch.count_nonzero(model.derivative_head.mlp.network[-1].bias)==0
    assert not torch.equal(a.derivative_head.mlp.network[0].weight,c.derivative_head.mlp.network[0].weight)


def test_no_history_use_and_fixed_normalization():
    model=build_mlp(23,stats());x=inputs();state=copy.deepcopy(model.state_dict())
    with torch.no_grad():before=model(**x)
    x['history_state_features'].fill_(float('nan'));x['history_controls'].fill_(float('nan'))
    with torch.no_grad():after=model(**x)
    for a,b in zip(before,after):assert torch.equal(a,b)
    for k,v in model.state_dict().items():assert torch.equal(v,state[k])
