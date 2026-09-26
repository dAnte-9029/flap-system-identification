import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
import numpy as np
from system_identification.training.trajectory_main_v1 import trajectory_rollout_loss
from system_identification.training.trajectory_main_v1 import TorchTrajectoryPrediction
import run_paper_rollout_horizon_ablation as r

def test_loss_mean_and_tail():
    n=51
    zero=torch.zeros(2,n,3);scalar=torch.zeros(2,n);q=torch.zeros(2,n,4);q[:,:,0]=1
    truth=TorchTrajectoryPrediction(zero,zero,q,zero,scalar,scalar)
    p=TorchTrajectoryPrediction(torch.ones_like(zero),2*torch.ones_like(zero),q,2*torch.ones_like(zero),scalar,3*torch.ones_like(scalar))
    # 3 position + 3 velocity + 3 rate + .1 frequency, independent of K.
    for k in (25,50): torch.testing.assert_close(trajectory_rollout_loss(p,truth,objective_steps=k),torch.tensor(9.1))
    source={'increment_scales':[1.,1.],'increment_weights':[.2,.3]}
    # Constant offsets cancel in lag2. Final-stage frequency term: .2*9.
    torch.testing.assert_close(r.objective(p,truth,25,source),torch.tensor(10.9))
    p.position_n[:,26:]+=100
    torch.testing.assert_close(r.objective(p,truth,25,source),torch.tensor(10.9))

def test_budget():
    for s in r.SEEDS:
        stages=r.m.stages(s)
        assert [v[2] for v in stages]==[s,s+12]
        assert sum(v[1] for v in stages)*int(np.ceil(28293/256))==7215

def test_flight_equal_seed_sd_and_gain():
    import pandas as pd
    import report_paper_rollout_horizon_ablation as report
    from system_identification.evaluation.paper_baselines import aggregate_flights, METRIC_MAP
    rows=[]
    for model,offset in [('K25',0),('K50',1)]:
        for seed,shift in [(17,0),(23,1),(42,2)]:
            for flight,count,error in [('a',1,1),('b',9,3)]:
                for _ in range(count):rows.append(dict(model=model,seed=seed,cohort='Sep7',log_id=flight,horizon_s=.5,**{k:error+shift+offset for k in METRIC_MAP}))
    per,summary,_=aggregate_flights(pd.DataFrame(rows))
    multi,pair,fp,gain=report.summarize(per,summary)
    g=multi.query('model == "K25" and cohort == "ALL"')
    np.testing.assert_allclose(g['mean'],3);np.testing.assert_allclose(g['std'],1)
    np.testing.assert_allclose(pair.paired_error_delta,-1)
    np.testing.assert_allclose(fp.paired_error_delta,-1)
    np.testing.assert_allclose(gain.relative_gain_pct,25)
