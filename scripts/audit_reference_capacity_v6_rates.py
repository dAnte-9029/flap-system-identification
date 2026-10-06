"""Check output-parametrization bounds on factual predictions, not control safety."""
import json
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT

def main():
    x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');rows=[]
    for scope in ['tiny','formal']:
        if not (OUT/scope).exists():continue
        for path in sorted((OUT/scope).glob('*')):
            if not path.is_dir():continue
            run=json.loads((path/'run_protocol.json').read_text())
            for step in [1600,6400]:
                for part in (['fit8'] if scope=='tiny' else ['train','validation']):
                    file=path/f'{step}_{part}_predictions.npz'
                    with np.load(file) as z:ix=z['array_index'];pred=z['rates_b']
                    initial=x['initial_rates_b'][ix].numpy();truth=y['rates_b'][ix].numpy()
                    for label,rates in [('predicted',pred),('logged',truth)]:
                        acc=np.diff(np.concatenate([initial[:,None,:],rates],axis=1),axis=1)/.02
                        residual=(rates-initial[:,None,:])/stats['rate_scale']
                        for j,axis in enumerate(['p','q','r']):
                            rows.append(dict(scope=scope,arm=run['arm'],seed=run['seed'],steps=step,partition=part,signal=label,axis=axis,
                                absolute_increment_acceleration_p99=float(np.quantile(abs(acc[:,:,j]),.99)),
                                fraction_above_120=float(np.mean(abs(acc[:,:,j])>120.001)),fraction_at_120=float(np.mean(abs(abs(acc[:,:,j])-120)<.001)),
                                fraction_at_origin_6sigma=float(np.mean(abs(abs(residual[:,:,j])-6)<.0001))))
    a=pd.DataFrame(rows);a.to_csv(OUT/'rate_bound_audit.csv',index=False)
    base.write(OUT/'rate_bound_scope.json',dict(script_sha256=base.sha(__file__),purpose='factual output bounds only; no unseen-reference safety or physical validity claim',
        interpretation='direct mode removes the angular-acceleration cap; absence of bound violations on logged references cannot certify counterfactual behavior',sealed_test_accessed=False))
if __name__=='__main__':main()
