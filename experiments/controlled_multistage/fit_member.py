import argparse
import json
from types import SimpleNamespace
import torch
from workspace import *
from candidates import build
from fit_stage import drift_units
from train import run
from objectives import training_loss


def main(name,gpu):
    scales={}
    def load_data(cache,split):
        data,ids=conditional_data(cache,split,TAPES)
        if split=='train':
            scales['drift']=drift_units(data)
        return data,ids
    def loss(model,b,units):
        return training_loss(name,model,b,units,scales['drift'])
    cfg=SimpleNamespace(cache=HISTORY,out=RUN/name,gpu=gpu,seed=17,model=name,epochs=24,batch=1024)
    run(cfg,load_data=load_data,build_model=lambda mean,scale:build(name,mean,scale),loss_fn=loss,
        extra_config=dict(control_tapes=str(TAPES),late_pose_loss_weight=.5,macro_dt=.02 if name=='s3a' else (.04 if name=='s2c' else .1),
                          late_horizons_s=[1,2],selection='archived shared score'))
    (cfg.out/'drift_scales_squared.json').write_text(json.dumps(scales['drift'].tolist()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    main(a.name,a.gpu)
