"""K25 development ablation; explicit train/validation inputs only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, time, subprocess, traceback, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_paper_standard_gru_multiseed as m
from run_paper_baseline_comparison import _FrequencyLossAdapter, causality_check, write_json
from system_identification.training.trajectory_main_v1 import _model_call, trajectory_rollout_loss, predict_history_trajectory_model
from system_identification.training.main_v2_increment import increment_terms, train_increment_stage
from system_identification.models.trajectory import TrajectoryPrediction
from system_identification.evaluation.paper_baselines import endpoint_metrics, aggregate_flights
ROOT=m.ROOT
OUT=ROOT/'docs/analysis/results/paper_rollout_horizon_ablation_v1'
ART=ROOT/'artifacts/paper_rollout_horizon_ablation_v1'
DEVICE='cuda:1'
SEEDS=(17,23,42)

def predpath(s):
    return m.OLD/f'{m.MODEL}_predictions.npz' if s==17 else m.ART/f'seed{s}/predictions.npz'

def inputs():
    source=m.read_json(m.OUT/'protocol.json')
    # Deliberately do not invoke historical all-artifact verifiers.
    assert m.file_hash(m.ART/'prepared.pt')==source['prepared_sha256']
    batches=torch.load(m.ART/'prepared.pt',map_location='cpu',weights_only=False)
    cp=torch.load(m.checkpoint_path(17),map_location='cpu',weights_only=False)
    stats={k:cp['state_dict'][k].numpy() for k in m.STATS}
    assert m.normalization_hash(stats)==source['normalization_sha256']
    for split,n in [('train',28293),('validation',2582)]:
        b=batches[split]; ids=pd.read_csv(m.BASE/f'{split}_origins.csv')
        assert len(b.trajectory.window_ids)==n
        np.testing.assert_array_equal(b.trajectory.window_ids,ids.window_id)
        assert set(b.trajectory.log_ids)==set(source[f'{split}_flights'])
        assert b.history_state_features.shape==(n,26,12)
        assert b.trajectory.controls.shape[1]==50
    return source,batches,stats

def objective(p,t,k,source,frequency=True):
    iv,iw=increment_terms(p,t,source['increment_scales'])
    loss=trajectory_rollout_loss(p,t,objective_steps=k)
    if frequency: loss=loss+.2*(p.flap_frequency_hz[:,1:k+1]-t.flap_frequency_hz[:,1:k+1]).square().mean()
    return loss+source['increment_weights'][0]*iv+source['increment_weights'][1]*iw

def precheck(source,batches,stats):
    b=m.subset(batches['train'],3)
    model=m.build_model(17,stats)
    old=m.original_build_model(m.MODEL,stats)
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in model.state_dict().items())
    other=m.build_model(23,stats); same=m.build_model(23,stats); different=m.build_model(42,stats)
    assert all(torch.equal(a,b) for a,b in zip(other.parameters(),same.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(other.parameters(),different.parameters()))
    assert sum(p.numel() for p in model.parameters())==21383 and all(p.requires_grad for p in model.parameters())
    model.to(DEVICE)
    def call(batch,k): return _model_call(model,batch,np.arange(3),use_history=True,rollout_steps=k,device=torch.device(DEVICE))
    p25,t25=call(b,25); p50,t50=call(b,50)
    for a,c in zip(p25,p50): torch.testing.assert_close(a,c[:,:26],rtol=0,atol=0)
    torch.testing.assert_close(objective(p25,t25,25,source),objective(p50,t50,25,source),rtol=0,atol=0)
    grads=torch.autograd.grad(objective(p25,t25,25,source),tuple(model.parameters()))
    poisoned=copy.deepcopy(b)
    for value in vars(poisoned.trajectory.truth).values(): value[:,26:]+=123
    poisoned.trajectory.controls[:,25:]+=17
    pp,tt=call(poisoned,25)
    torch.testing.assert_close(objective(pp,tt,25,source),objective(p25,t25,25,source),rtol=0,atol=0)
    gp=torch.autograd.grad(objective(pp,tt,25,source),tuple(model.parameters()))
    for a,c in zip(grads,gp): torch.testing.assert_close(a,c,rtol=0,atol=0)
    iv=increment_terms(p50,t50,source['increment_scales'])
    for a,c in zip(iv,increment_terms(p25,t25,source['increment_scales'])): torch.testing.assert_close(a,c,rtol=0,atol=0)
    assert p50.position_n.shape[1]==51
    for value in p50: assert torch.isfinite(value).all()
    torch.testing.assert_close(torch.linalg.vector_norm(p50.quaternion_nb,dim=-1),torch.ones_like(p50.relative_phase_rad),rtol=0,atol=1e-5)
    checks=causality_check(model.cpu(),m.subset(batches['validation'],3),DEVICE)
    return dict(status='passed',prefix25_bitwise=True,tail_label_loss_gradient_invariance=True,
        tail_control_invariance=True,initialization_parity=True,parameter_count=21383,
        two_step_increment_parity=True,eval_state_count=51,causality=checks)

def prepare():
    assert not (OUT/'protocol.json').exists(),'do not overwrite registration'
    OUT.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
    source,batches,stats=inputs()
    checks=precheck(source,batches,stats)
    pins={}
    def pin(path,expected=None):
        h=m.file_hash(path)
        assert expected is None or expected==h, str(path)
        pins[m.rel(path)]=h
    pin(m.ART/'prepared.pt',source['prepared_sha256'])
    pin(m.OUT/'protocol.json');pin(m.OUT/'training_runs.csv')
    pin(ROOT/source['manifest_path'],source['manifest_sha256'])
    for part in ('train','validation'):
        pin(m.BASE/f'{part}_origins.csv',source['train_window_sha256' if part=='train' else 'validation_origin_sha256'])
        for kind in ('samples','windows'):
            name=f'{kind}_{part}.parquet';pin((ROOT/source['manifest_path']).parent/name,source['artifact_sha256'][name])
    runs=pd.read_csv(m.OUT/'training_runs.csv')
    complete=m.read_json(m.OUT/'completion.json')
    for s in SEEDS:
        pin(m.checkpoint_path(s),runs.set_index('seed').loc[s,'checkpoint_sha256'])
        path=predpath(s)
        expected=source['frozen_input_sha256'].get(m.rel(path),complete['artifact_sha256'].get(m.rel(path)))
        assert expected is not None
        pin(path,expected)
    for path,h in source['frozen_input_sha256'].items():
        if path.startswith('src/system_identification/'):
            pin(ROOT/path,h)
    for name in ('run_paper_rollout_horizon_ablation.py','run_paper_standard_gru_multiseed.py','run_paper_baseline_comparison.py'):
        pin(ROOT/'scripts'/name)
    protocol=dict(experiment='paper_rollout_horizon_ablation_v1',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),source_protocol=m.rel(m.OUT/'protocol.json'),
        history_steps=26,train_unroll_steps=[25,50],eval_rollout_steps=50,seeds=list(SEEDS),
        architecture=source['architecture'],parameter_count=21383,training_schedule=source['training_schedule'],
        candidate_override={'rollout_steps':25},increment_scales=source['increment_scales'],increment_weights=source['increment_weights'],
        normalization_sha256=source['normalization_sha256'],train_flights=source['train_flights'],validation_flights=source['validation_flights'],
        train_origins=28293,validation_origins=2582,frozen_input_sha256=pins,
        prior_heldout_results_known=True,heldout_data_accessed_this_run=False,
        access_scope='explicit original train/validation cache, parquet, checkpoint and validation predictions only; no independent-test inputs/results',
        checkpoint_rule='last epoch65, no validation selection',stages='base seed s; continuation s+12; AdamW reset',
        budget='65 epochs,111 updates/epoch,7215 updates; not equal transitions/FLOPs/wall time',
        loss_scope='trajectory and frequency means over states1..K; lag2 increments unchanged; no detach or teacher forcing',
        metrics=source['metrics'],horizons_s=source['horizons_s'],primary_horizon_s=.5,
        comparison='K25-K50 negative favors K25; relative gain=(K50-K25)/K50; zero denominator undefined',
        execution={'device':DEVICE,'concurrent_workers':1,'automatic_retry':False},
        main_model_remains='H26/K50',runtime=m.runtime(17))
    write_json(OUT/'protocol.json',protocol);write_json(OUT/'sanity_checks.json',checks)
    (OUT/'loss_definition.md').write_text('''# Frozen loss definition\nFor k=25 candidate (50 baseline), average over states 1..k, excluding t0.\nPosition: mean ||dp||²; velocity: mean ||dv/2||²; attitude: mean 4(1-dot(qhat,q)²)/0.35² with unit quaternions; rate: mean ||dw/2||²; phase: 0.1 mean(2-2cos(dphase)); frequency: 0.1 mean(df/3)². Continuation adds 0.2 mean(df)² in Hz² on the SAME 1..k interval.\nBoth stages retain frozen weighted lag2 velocity/rate increment errors from t0 to t2, with original scales/weights. Zero actuator regularization. No teacher forcing, intermediate detach, or loss reweighting.\nValidation L25 and L50 both include the final-stage frequency term and lag2 terms. They are window-weighted auxiliary objectives; physical endpoint metrics retain equal-flight aggregation. Different original training losses are not directly comparable.\n''')
    print('PRECHECK PASSED / PROTOCOL FROZEN',flush=True)

def worker(s):
    folder=ART/f'seed{s}';folder.mkdir(exist_ok=False)
    try:
        protocol=m.read_json(OUT/'protocol.json');m.verify_pins(protocol['frozen_input_sha256'])
        source,batches,stats=inputs();model=m.build_model(s,stats)
        write_json(folder/'runtime.json',m.runtime(s));started=time.monotonic()
        for stage,epochs,seed,lr,freq in m.stages(s):
            m.seed_all(seed)
            def callback(_,hist):
                hist.to_csv(folder/f'{stage}_history.csv',index=False)
                write_json(folder/'status.json',dict(status='training',stage=stage,epoch=int(hist.epoch.iloc[-1]),seed=s))
            trained,hist=train_increment_stage(_FrequencyLossAdapter(model) if freq else model,batches['train'],
                scales=source['increment_scales'],weights=source['increment_weights'],device=DEVICE,epochs=epochs,
                seed=seed,learning_rate=lr,actuator=freq,steps=25,batch_size=256,callback=callback)
            model=trained.model if freq else trained;m.validate_history(hist,epochs)
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            assert m.normalization_hash({k:model.state_dict()[k].numpy() for k in m.STATS})==source['normalization_sha256']
            torch.save(dict(state_dict=model.state_dict(),seed=s,train_unroll_steps=25,stage=stage,final_epoch=40 if not freq else 65,
                protocol_sha256=m.file_hash(OUT/'protocol.json')),folder/('base.pt' if not freq else 'model.pt'))
        write_json(folder/'status.json',dict(status='complete',training_time_s=time.monotonic()-started,seed=s,final_epoch=65))
    except BaseException:
        write_json(folder/'status.json',dict(status='failed',seed=s,error=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    m.configure()
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='worker': worker(int(sys.argv[2]))
