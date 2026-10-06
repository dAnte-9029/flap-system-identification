"""Repeat train-gap and reference-dependence diagnostics after the unit correction."""
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_reference_unit_gravity import OUT,ART,SEEDS,model_from
from control_windows_v3 import predict,OLD
from evaluate_control_windows_v2 import error_rows,summarize


def main():
    out=OUT/'diagnosis';out.mkdir(exist_ok=False)
    x,y,stats=base.load();w=pd.read_csv(OLD/'windows_v2.csv');rows=[];averaged=[];components=[]
    for arm in ['A','A_long']:
        for seed in SEEDS:
            model=model_from(ART/f'{arm}_{seed}/final.pt')
            for partition in ['train','validation']:
                meta=w[w.partition==partition];ix=meta.array_index.to_numpy();truth={k:v[ix].numpy() for k,v in y.items()}
                for intervention in ['observed','reference_held']:
                    pred=predict(model,x,ix,hold=(intervention=='reference_held'))
                    frame=summarize(error_rows(pred,truth,meta,arm,seed));frame['partition']=partition;frame['intervention']=intervention;rows.append(frame)
                    if intervention!='observed':continue
                    delta=pred['rates_b']-truth['rates_b']
                    for j,index in enumerate(ix):
                        for axis,name in enumerate(['p','q','r']):
                            mean=delta[j,:,axis].mean();centered=delta[j,:,axis]-mean
                            components.append(dict(arm=arm,seed=seed,partition=partition,array_index=int(index),log_id=meta.iloc[j].log_id,
                                axis=name,mean_bias_mse=float(mean**2),centered_waveform_mse=float(np.mean(centered**2))))
                    # Purely supplemental; no replacement of preregistered raw-rate gate.
                    for step in [10,25,50]:
                        error=delta[:,step-10:step].mean(1)
                        frame=meta[['log_id','window_id']].copy();frame['arm']=arm;frame['seed']=seed
                        frame['partition']=partition;frame['horizon_s']=step*.02
                        for j,name in enumerate(['p','q','r']):frame[name]=error[:,j]**2
                        averaged.append(frame)
    df=pd.concat(rows);df.to_csv(out/'per_flight.csv',index=False)
    df.groupby(['arm','partition','intervention','subset','horizon_s'])[['p','q','r','velocity','attitude']].mean().to_csv(out/'summary.csv')
    avg=pd.concat(averaged);np.sqrt(avg.groupby(['arm','seed','partition','horizon_s','log_id'])[['p','q','r']].mean()).groupby(['arm','partition','horizon_s']).mean().to_csv(out/'causal_200ms_mean_rate_rmse.csv')
    pd.DataFrame(components).to_csv(out/'rate_error_components.csv',index=False)
    base.write(out/'scope.json',dict(averaged_rate='diagnostic0.2s causal averaging adds filtering delay; not substituted for frozen raw-state gate',
        reference_hold='model dependence diagnostic, not a measured counterfactual',sealed_test_accessed=False))


if __name__=='__main__':
    torch.set_num_threads(1);main()
