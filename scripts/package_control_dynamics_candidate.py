"""Create a checksum-verified standalone inference package and verify it in /tmp."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import numpy as np
import torch
import scipy

import run_paper_rollout_horizon_ablation as source
from system_identification.integration.control_dynamics_candidate import load_candidate
from system_identification.models.control_ensemble import PHYSICAL

m=source.m
BASE=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=BASE/'delivery'
EXPERIMENTS=('baseline','response_probe','local_identification','ensemble_candidate','px4_guarded_closed_loop',
             'px4_guarded_summary','credibility','planning_stress','past_residual_information','planning_error_margin')


def run():
    m.configure()
    index={}
    for name in EXPERIMENTS:
        path=BASE/name/'completion.json';c=m.read_json(path)
        assert c['status']=='complete' and c['heldout_accessed'] is False
        m.verify_pins(c['outputs'])
        index[name]=dict(completion_path=m.rel(path),completion_sha256=m.file_hash(path),outputs=c['outputs'])
    OUT.mkdir(parents=True,exist_ok=False);bundle=OUT/'candidate';bundle.mkdir()
    def copy(src,relative):
        target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    for src,dest in [(BASE/'ensemble_candidate/model.pt','model.pt'),(BASE/'px4_guarded_closed_loop/support.npz','support.npz'),
        (BASE/'px4_guarded_closed_loop/feature_ranges.csv','feature_ranges.csv'),(BASE/'px4_guarded_closed_loop/guard.json','guard.json'),
        (BASE/'past_residual_information/empirical_envelope.csv','empirical_envelope.csv'),
        (BASE/'model_card_zh.md','MODEL_CARD_ZH.md'),(BASE/'report.md','REPORT_ZH.md'),
        (m.ROOT/'scripts/use_control_dynamics_candidate.py','run_example.py')]:copy(src,dest)
    # Source snapshot avoids dependence on an installed project package or repository cwd.
    for path in sorted((m.ROOT/'src/system_identification').rglob('*.py')):
        copy(path,path.relative_to(m.ROOT))
    sp,batches,_=source.inputs();b=batches['validation'];identity=b.trajectory.window_ids.astype(str)
    i=int(np.argsort(identity)[len(identity)//2])
    # No future true state or future flight commands included in the portable example.
    inputs={k:getattr(b.trajectory.truth,k)[i:i+1,0] for k in PHYSICAL}
    inputs.update(history_state_features=b.history_state_features[i:i+1],history_controls=b.history_controls[i:i+1],history_mask=b.history_mask[i:i+1])
    inputs['future_controls']=np.repeat(b.trajectory.controls[i:i+1,:1],10,axis=1)
    inputs['dt_s']=np.full((1,10),.02)
    np.savez_compressed(bundle/'example_inputs.npz',**inputs)
    model=load_candidate(bundle/'model.pt',m.file_hash(bundle/'model.pt'))
    with torch.inference_mode():
        out=model(**{k:torch.as_tensor(v,dtype=torch.bool if k=='history_mask' else torch.float32) for k,v in inputs.items()})
    np.savez_compressed(bundle/'expected_predictions.npz',**{k:getattr(out,k).numpy() for k in PHYSICAL})
    m.write_json(bundle/'example_provenance.json',dict(window_id=identity[i],log_id=str(b.trajectory.log_ids[i]),
        selection='middle lexicographic original validation identity, independent of error',future_truth_included=False,
        command='synthesized current-command hold for200ms',expected='model predictions only'))
    m.write_json(bundle/'runtime.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,
        device='cpu',environment='/home/zn/anaconda3/envs/flap-train-gpu',dependencies_changed=False))
    m.write_json(bundle/'manifest.json',dict(format='control_dynamics_research_bundle_v1',
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),base_commit='e64a1ba',
        scope='frozen short-horizon prediction candidate plus coverage/error diagnostics; NOT control/RL validated',
        dataset_id=sp['dataset_id'],dataset_manifest_sha256=sp['manifest_sha256'],sealed_test_accessed=False,
        files={str(p.relative_to(bundle)):m.file_hash(p) for p in sorted(bundle.rglob('*')) if p.is_file()}))
    # Fresh process outside repository, PYTHONPATH removed. Imports resolve from copied src.
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env['MPLCONFIGDIR']='/tmp/flap-paper-mpl'
    result=subprocess.run([sys.executable,str(bundle/'run_example.py'),'--bundle',str(bundle)],cwd='/tmp',env=env,capture_output=True,text=True,check=False)
    (OUT/'smoke_stdout.txt').write_text(result.stdout);(OUT/'smoke_stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError(f'standalone smoke failed: {result.stderr}')
    smoke=json.loads(result.stdout);assert smoke['status']=='passed' and smoke['dataset_access'] is False
    archive=OUT/'control-dynamics-candidate-v1.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(bundle,arcname='control-dynamics-candidate-v1')
    m.write_json(OUT/'experiment_index.json',index)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,
        standalone_smoke=smoke,archive_sha256=m.file_hash(archive),script_sha256=m.file_hash(Path(__file__)),
        archive=str(archive.relative_to(m.ROOT)),control_ready=False,
        outputs={m.rel(p):m.file_hash(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('standalone package verified',archive,flush=True)


if __name__=='__main__':run()
