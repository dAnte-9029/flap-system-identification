"""Compute common-flight gains for removing A and retain the established reference systems."""
import importlib.util
import json
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('flight_contrasts',HERE.parent/'coupled-slow-latent/compare.py')
tools=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
ROOT=tools.ROOT
STUDY=ROOT/'research/experiments/study-091d89f322fa'
OLD=ROOT/'research/experiments/study-817e3b3225fa'

def extract(parent,stage,tag):
    out=parent/stage/'attempt-1/seed-17/output'
    return tools.scores(out/tag),json.loads((out/'validation.json').read_text())[tag]

def run():
    state=json.loads((STUDY/'state.json').read_text());eid=next(iter(state['experiments']))
    outputs={'ABC':extract(STUDY,'baseline','state'),'BC':extract(STUDY,eid,'state'),
             'old_AC':extract(OLD,'baseline','baseline'),'old_ABC':extract(OLD,'exp-36f22ad1bdf8','full')}
    meta=pd.read_parquet(ROOT/'outputs/history_slow_fast_20261119_cache/validation_windows.parquet',columns=['flight_id','sampling_group']).drop_duplicates().sort_values('flight_id')
    check=set(meta.loc[meta.groupby('sampling_group').cumcount()%3==0,'flight_id'])
    assert len(check)==13
    results={}
    for mod,ref in [('BC','ABC'),('BC','old_AC'),('ABC','old_AC'),('BC','old_ABC')]:
        a,b=outputs[ref][0],outputs[mod][0]
        cols=['v05','w05','v1','w1','v2','w2','full_all','full_high','future_first','beat_mean','native','centered','2.0_minus125','2.0_minus20']
        assert all(a[k].index.equals(b[k].index) for k in cols)
        result={k:tools.bootstrap(a[k],b[k],False) for k in cols}
        result['subset13']={k:tools.bootstrap(a[k],b[k],False,check) for k in cols}
        results[mod+'_vs_'+ref]=result
    (HERE/'paired.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    for k,r in outputs.items():
        m=r[1];print(k,{name:round(x,5) for name,x in m['endpoint'].items()},'slow,centered,future-change,offgrid',
                      *[round(v,5) for v in (m['beats']['full_all']['rmse'],m['beats']['measured']['centered'],m['beats']['future_first']['rmse'],m['shifted']['2.0']['minus125'])])
    for k,r in results['BC_vs_ABC'].items():
        if k=='subset13':continue
        print(k,round(r['gain'],5),[round(i,5) for i in r['ci95']],r['improved'],r['flights'])
if __name__=='__main__':run()
