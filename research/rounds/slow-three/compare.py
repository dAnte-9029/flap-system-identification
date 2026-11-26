"""Flight-paired development contrasts. Positive gain means the first model reduces RMSE."""
import importlib.util
import json
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
source=HERE.parent/'coupled-slow-latent/compare.py'
spec=importlib.util.spec_from_file_location('paired_tools',source)
tools=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
STUDY=tools.ROOT/'research/experiments/study-817e3b3225fa'

def run():
    state=json.loads((STUDY/'state.json').read_text())
    snapshots={}
    for label,ident in [('A+C','baseline')]+[(f'R{k}',e) for k,e in enumerate(state['experiments'],1)]:
        path=STUDY/ident/'attempt-1/seed-17/output'
        if not (path/'validation.json').exists():continue
        for name,values in json.loads((path/'validation.json').read_text()).items():
            if 'beats' not in values:continue
            tag=label if label=='A+C' else label+'/'+name
            snapshots[tag]=tools.scores(path/name)
    meta=pd.read_parquet(tools.ROOT/'outputs/history_slow_fast_20261119_cache/validation_windows.parquet',columns=['flight_id','sampling_group']).drop_duplicates().sort_values('flight_id')
    check=set(meta.loc[meta.groupby('sampling_group').cumcount()%3==0,'flight_id'])
    assert len(check)==13
    comparisons={}
    for r in (1,2,3):
        full=f'R{r}/full'
        if full not in snapshots:continue
        pairs=[(full,'A+C'),(full,f'R{r}/ablated'),(full,f'R{r}/full_held'),(f'R{r}/ablated','A+C')]
        if r>1:pairs.append((full,f'R{r-1}/full'))
        if r==3:pairs.extend([(f'R{r}/ablated','R2/ablated'),(full,'R1/full'),(full,'R2/ablated')])
        for model,ref in pairs:
            if model not in snapshots or ref not in snapshots:continue
            a,b=snapshots[ref],snapshots[model]
            scores={k:tools.bootstrap(a[k],b[k],k.endswith('_align')) for k in a.keys() & b.keys()}
            scores['subset13']={k:tools.bootstrap(a[k],b[k],False,check) for k in ('w05','w1','w2','v05','v1','v2','full_all','centered','2.0_minus125')}
            comparisons[model+' vs '+ref]=scores
    (HERE/'paired.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False)+'\n')
    for pair,s in comparisons.items():
        print(pair,' '.join(f"{k} {s[k]['gain']:+.5f} [{s[k]['ci95'][0]:+.5f},{s[k]['ci95'][1]:+.5f}]" for k in ('full_all','w2','centered','2.0_minus125')))
if __name__=='__main__':run()
