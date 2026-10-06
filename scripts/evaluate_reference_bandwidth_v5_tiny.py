"""Training-only launch gate, comparing identical targets and original fit8 IDs."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT,load,causal_expand_50hz
from train_reference_unit_gravity import model_from
from train_reference_bandwidth_v5 import metrics
from control_windows_v3 import predict


def main():
    torch.set_num_threads(1);x,y,stats,w=load();protocol=json.loads((OUT/'protocol.json').read_text())
    meta=w.set_index('window_id').loc[protocol['tiny_cohort']].reset_index();assert (meta.partition=='train').all();ix=meta.array_index.to_numpy()
    parts=[];waves=[];checkpoints={}
    for arm in ['raw100','aa50']:
        path=OUT/'tiny'/f'{arm}_17';status=json.loads((path/'status.json').read_text());assert status['status']=='complete'
        assert base.sha(path/'final.pt')==status['checkpoint_sha256'];checkpoints[arm]=status['checkpoint_sha256']
        parts.append(pd.read_csv(path/'per_flight.csv'));waves.append(pd.read_csv(path/'waveform_errors.csv'))
    original=base.ROOT/'docs/analysis/results/control_windows_v4/tiny_fit.pt';manifest=json.loads((original.parent/'delivery_manifest.json').read_text())
    assert base.sha(original)==manifest['artifacts'][str(original.relative_to(base.ROOT))]
    oldx,_,_=base.load();model=model_from(original);p=predict(model,oldx,ix)
    initial={'rates_b':oldx['initial_rates_b'][ix],'quaternion':oldx['initial_quaternion'][ix],'velocity_n':oldx['initial_velocity_n'][ix]}
    pp={k:v.numpy() for k,v in causal_expand_50hz({k:torch.from_numpy(v) for k,v in p.items()},initial).items()}
    f,wa=metrics(pp,{k:v[ix] for k,v in y.items()},x['history_state'][ix,:,3:6],meta,'tiny50',17);f['cohort']='fit8';wa['cohort']='fit8';parts.append(f);waves.append(wa)
    frames=pd.concat(parts);wave=pd.concat(waves);frames.to_csv(OUT/'tiny_per_flight.csv',index=False);wave.to_csv(OUT/'tiny_waveform_errors.csv',index=False)
    # Equal flights, then equal rate axes; endpoints are supplemental, not launch criterion.
    summary=wave.groupby(['arm','readout','log_id'])[['p','q','r','attitude','velocity']].mean().groupby(['arm','readout']).mean();summary.to_csv(OUT/'tiny_summary.csv')
    decisions=[]
    for candidate,comparator,readout in [('raw100','tiny50','raw50'),('aa50','raw100','aa50')]:
        c=summary.loc[(candidate,readout)];b=summary.loc[(comparator,readout)];gain=float(1-c[['p','q']].mean()/b[['p','q']].mean());ratios=(c[['attitude','velocity']]/b[['attitude','velocity']]-1)
        decisions.append(dict(candidate=candidate,comparator=comparator,readout=readout,waveform_gain_fraction=gain,
            attitude_regression_fraction=float(ratios.attitude),velocity_regression_fraction=float(ratios.velocity),passed=bool(gain>=.15 and (ratios<=.1).all())))
    arms=[a['candidate'] for a in decisions if a['passed']]
    if 'aa50' in arms and 'raw100' not in arms:arms.insert(0,'raw100')
    base.write(OUT/'tiny_decision.json',dict(decisions=decisions,formal_arms=arms,formal_seeds=[17,23,42],
        comparator_reason='raw100 required as matched internal-grid baseline whenever aa50 advances',
        train_only_gate=True,validation_used_for_gate=False,checkpoint_hashes=checkpoints,
        tiny50_upsampling='causal previous-prediction hold at odd10ms; raw50 gate uses original20ms targets and predictions exactly',
        control_usable=False,reason='small-cohort capacity gate only; no controller or causal control evidence',sealed_test_accessed=False))
    print((OUT/'tiny_decision.json').read_text())
if __name__=='__main__':main()
