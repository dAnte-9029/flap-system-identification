#!/usr/bin/env python3
"""One-command GPU experiment workflow; immutable baseline, isolated worker runs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
NAMES=['A0_baseline_retrain','A1_longer_rollout','A2_multi_horizon','A3_dynamic_delta']


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--results',type=Path,default=ROOT/'docs/analysis/results/main_v2_training_objective_ablation');pa.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/main_v2_training_objective_ablation');pa.add_argument('--reuse-diagnostics',action='store_true');args=pa.parse_args()
    if args.artifacts.exists() and any(args.artifacts.iterdir()):raise FileExistsError(args.artifacts)
    args.artifacts.mkdir(parents=True,exist_ok=True);args.results.mkdir(parents=True,exist_ok=True)
    def run(script,*arguments):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)
    if not args.reuse_diagnostics:run('diagnose_main_v2_objectives.py','--output',args.results/'pretraining')
    if not (args.results/'pretraining/summary.json').exists():raise RuntimeError('missing completed pretraining diagnosis')
    jobs=[]
    for i,name in enumerate(NAMES):
        folder=args.artifacts/'workers'/name;folder.mkdir(parents=True)
        results=folder/'results';results.mkdir();(results/'pretraining').symlink_to((args.results/'pretraining').resolve(),target_is_directory=True)
        device='cuda:1' if i%2==0 else 'cuda:0'
        cmd=[sys.executable,str(ROOT/'scripts/run_main_v2_objective_ablation.py'),'--output',str(folder/'models'),'--results',str(results),'--device',device,'--experiment',name]
        log=(folder/'run.log').open('w');p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=log)
        jobs.append(dict(name=name,device=device,folder=folder,process=p,log=log))
    while any(j['process'].poll() is None for j in jobs):
        state=[]
        for j in jobs:
            path=j['folder']/'models/status.json'
            try:progress=json.loads(path.read_text()) if path.exists() else {}
            except json.JSONDecodeError:progress={}
            state.append(dict(experiment=j['name'],device=j['device'],pid=j['process'].pid,exit_code=j['process'].poll(),progress=progress))
        (args.artifacts/'status.json').write_text(json.dumps(dict(status='training',jobs=state),indent=2))
        # Child process sleep, not an assistant tool wait; status remains durable.
        time.sleep(5)
    for j in jobs:
        j['log'].close()
        if j['process'].returncode:raise RuntimeError(f"worker failed: {j['name']}; see {j['folder']/'run.log'}")
    summaries=[]
    for j in jobs:
        model_dir=j['folder']/'models'/j['name'];(args.artifacts/j['name']).symlink_to(model_dir.resolve(),target_is_directory=True)
        frame=pd.read_csv(j['folder']/'results/training_summary.csv');frame['device']=j['device'];frame['concurrent_jobs']=4;summaries.append(frame)
    pd.concat(summaries).to_csv(args.results/'training_summary.csv',index=False)
    for name in ['protocol.json','experiment_manifest.csv']:
        (args.results/name).write_bytes((jobs[0]['folder']/'results'/name).read_bytes())
    (args.artifacts/'status.json').write_text(json.dumps(dict(status='evaluating',experiments=NAMES),indent=2))
    run('evaluate_main_v2_objectives.py','--artifacts',args.artifacts,'--results',args.results)
    run('report_main_v2_objectives.py','--artifacts',args.artifacts,'--results',args.results)
    (args.artifacts/'status.json').write_text(json.dumps(dict(status='completed',experiments=NAMES),indent=2))

if __name__=='__main__':main()
