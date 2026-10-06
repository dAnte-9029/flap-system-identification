"""Training-only launch gates for compute budget and output parametrization."""
import json
import pandas as pd
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT
METRICS=['p','q','r','attitude','velocity']

def main():
    waves=[];frames=[]
    for arm in ['integrated','direct']:
        path=OUT/'tiny'/f'{arm}_17';s=json.loads((path/'status.json').read_text());assert s['status']=='complete'
        for ck in s['completed']:assert base.sha(path/f"step_{ck['step']}.pt")==ck['checkpoint_sha256']
        for step in [1600,6400]:
            waves.append(pd.read_csv(path/f'{step}_waveform.csv'));frames.append(pd.read_csv(path/f'{step}_per_flight.csv'))
    a=pd.concat(waves);a.to_csv(OUT/'tiny_waveform.csv',index=False);pd.concat(frames).to_csv(OUT/'tiny_per_flight.csv',index=False)
    summary=a.groupby(['arm','steps','log_id'])[METRICS].mean().groupby(['arm','steps']).mean();summary.to_csv(OUT/'tiny_summary.csv')
    decisions=[]
    for name,carm,cstep,barm,bstep in [('budget','integrated',6400,'integrated',1600),('output1600','direct',1600,'integrated',1600),('output6400','direct',6400,'integrated',6400)]:
        c=summary.loc[(carm,cstep)];b=summary.loc[(barm,bstep)];gain=float(1-c[['p','q']].mean()/b[['p','q']].mean());regression=c[['attitude','velocity']]/b[['attitude','velocity']]-1
        decisions.append(dict(comparison=name,candidate=f'{carm}{cstep}',baseline=f'{barm}{bstep}',gain_fraction=gain,
            attitude_regression_fraction=float(regression.attitude),velocity_regression_fraction=float(regression.velocity),passed=bool(gain>=.15 and (regression<=.1).all())))
    if any(d['passed'] for d in decisions[1:]):arms=['integrated','direct']
    elif decisions[0]['passed']:arms=['integrated']
    else:arms=[]
    base.write(OUT/'tiny_decision.json',dict(decisions=decisions,formal_arms=arms,steps=6400,snapshots=[1600,6400],seeds=[17,23,42],
        gate_uses_train_only=True,validation_used_for_selection=False,control_validated=False,sealed_test_accessed=False))
    print((OUT/'tiny_decision.json').read_text())
if __name__=='__main__':main()
