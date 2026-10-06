"""Bounded reference sensitivity versus observed error; not controller optimization."""
import json
import numpy as np
import pandas as pd
import torch
from control_windows_v3 import OUT,OLD,ART,SEEDS,base,model_from,predict
from control_windows_v3_ridge import features


def main():
    dest=OUT/'action_probe';dest.mkdir(exist_ok=False)
    x,y,stats=base.load();w=pd.read_csv(OLD/'windows_v2.csv')
    train=w[w.partition=='train'].array_index.to_numpy();val=w[w.partition=='validation'].array_index.to_numpy()
    ref=x['future_reference'].numpy();low,high=np.quantile(ref[train].reshape(-1,5),[.01,.99],axis=0)
    protocol=dict(candidate='A_long all3seeds',increment_deg=1.,ramp_s=.2,horizon_s=.5,
        axes=['roll_reference','pitch_reference'],bounds='train1percent/99percent marginal reference envelope; not joint counterfactual support certification',
        statistic='predicted change of p/q compared with original recorded-response prediction error',
        interpretation='sensitivity diagnostic, not causal-effect truth or optimizer benefit',
        no_controller_optimization=True,low=low.tolist(),high=high.tolist(),sealed_test_accessed=False)
    base.write(dest/'protocol.json',protocol)
    with np.load(OUT/'ridge/k25.npz') as z:
        ridge={k:z[k] for k in ['mean','scale','coefficient']}
    original_features=features(x,25)
    rows=[]
    for axis in [0,1]:
        ramp=np.minimum(np.arange(1,51)/10,1)*np.deg2rad(1)
        valid=(ref[val,:,axis].min(1)-ramp.max()>=low[axis])&(ref[val,:,axis].max(1)+ramp.max()<=high[axis])
        indices=val[valid]
        if not len(indices):continue
        for seed in SEEDS:
            model=model_from(ART/f'A_long_{seed}/final.pt');original=predict(model,x,indices)
            for sign in [-1,1]:
                modified={**x,'future_reference':x['future_reference'].clone()}
                modified['future_reference'][:,:,axis]+=torch.tensor(sign*ramp,dtype=torch.float32)
                changed=predict(model,modified,indices)
                response=changed['rates_b'][:,24,axis]-original['rates_b'][:,24,axis]
                feature_delta=(features(modified,25)[indices]-original_features[indices])/ridge['scale']
                ridge_response=(feature_delta@ridge['coefficient'][1:])[:,axis]
                error=original['rates_b'][:,24,axis]-y['rates_b'][indices,24,axis].numpy()
                for j,index in enumerate(indices):
                    rows.append(dict(array_index=int(index),log_id=w.iloc[index].log_id,seed=seed,axis=['p','q'][axis],sign=sign,
                        predicted_change=float(response[j]),ridge_predicted_change=float(ridge_response[j]),
                        observed_prediction_error=float(error[j])))
    r=pd.DataFrame(rows);r.to_csv(dest/'per_window.csv',index=False)
    report=[]
    for axis,g in r.groupby('axis'):
        # Flight-equal RMS, then seed average; no invented noise-free control signal.
        by=g.groupby(['seed','log_id']).agg(effect_mse=('predicted_change',lambda v:np.mean(v**2)),
            error_mse=('observed_prediction_error',lambda v:np.mean(v**2)))
        means=np.sqrt(by).mean()
        report.append(dict(axis=axis,windows=int(g.array_index.nunique()),flights=int(g.log_id.nunique()),
            predicted_effect_rms=float(means.effect_mse),observed_error_rms=float(means.error_mse),
            error_to_effect_ratio=float(means.error_mse/max(means.effect_mse,1e-12)),
            ridge_sign_agreement_fraction=float(((g.predicted_change*g.ridge_predicted_change)>0).mean()),
            interpretation='different model family is a sensitivity check, not causal ground truth; common-mode error could cancel in action differences'))
    base.write(dest/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    torch.set_num_threads(1);main()
