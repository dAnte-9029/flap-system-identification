import json
import subprocess
import pandas as pd
from workspace import *

names=('c2','s2a','s2b','s2c','s3a','s3b','s3c')


def main():
    rankings=json.loads((RUN/'2s_scores.json').read_text())
    scores=rankings['shared_scores']
    assert set(scores)==set(names) and not (RUN/'extrapolated3_comparison.csv').exists()
    rep=min(names,key=scores.get)
    assert rep==rankings['representative']
    checked=json.loads((RUN/'2s_pipeline.json').read_text())
    assert set(checked['checks'])==set(names[1:])
    selected=(STAGE1 if rep=='c2' else RUN)/rep
    df=pd.read_csv(RUN/'2s_comparison.csv')
    rule=dict(representative=rep,checkpoint=str(selected/'best.pt'),
              four_baseline_checkpoints={n:str((OLD if n=='gru' else SUITE)/n/'best.pt') for n in BASELINES},
              git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
              selection='minimum archived shared development score, earlier candidate on tie',
              shared_scores=scores,development_wins=int((df.gap<0).sum()),seed=17,batch=1024,updates=2976,
              comparison_windows='all admitted 3 s train/validation windows created by data3.py',
              training_recipe_3s='No pretraining; 24 shuffled cycles of 124 batches; AdamW and original cosine schedule. Each model uses identical samples.',
              score_3s='1/4 * (native point normalized MSE of v and omega + normalized mean endpoint MSE of p and SO3 angle); five endpoints 0.2/0.5/1/2/3 s, equal valid weights; flight-equal mean. All four scaling units fit on the 3 s training partition.',
              baseline_loss='shared trajectory loss only',
              main_loss='shared + 0.25 first-complete-beat slow target auxiliary + 0.5 mean of position/attitude endpoint MSE at 1/2/3 s scaled separately by train-only constant-velocity/rate endpoint MSE',
              extra_endpoint_weight={'s3b':.16,'s3c':.05}.get(rep,0.),
              extra_endpoint_formula='if weight positive: horizon-proportional (.5,1,2,3) weighted mean across four components of log1p(batch endpoint MSE/(1e-4*train scale squared))')
    with (RUN/'freeze.json').open('x') as f: json.dump(rule,f,indent=2)
    (RUN/'stage3.md').write_text(f"第三阶段完成三次正式训练。六个新增候选与 C2 按原共享验证分数选定 {rep}，分数 {scores[rep]:.6f}，12 项领先 {rule['development_wins']} 项。所有 3 s 性能评估均在 freeze.json 写出后进行。\n")
    print(rep)


if __name__=='__main__': main()
