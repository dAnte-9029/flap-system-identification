"""Freeze tilt-loss preparation, audit kinematics, calibrate on fit only, reproduce v35."""
import copy
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_joint_increment_v34 import load_inputs,read,DATA,LOCAL,PREP as V34,ActuatorCommandGRU,fingerprint
from joint_increment_v34 import select_local,objective as legacy_objective
from attitude_supervision_v42 import components,objective,calibrated_weight
from actuator_training_v30 import model_from
from audit_controller_support_v26 import verify_parent

OUT=base.ROOT/'docs/analysis/results/control_windows_v42'
V35=OUT.parent/'control_windows_v35'
PARENTS=[DATA,LOCAL,V34,V35,OUT.parent/'control_windows_v40',OUT.parent/'control_windows_v41']


def gradients(model,bx,by,lx,ly):
    old,tilt,att=components(model,bx,by,lx,ly);parameters=tuple(model.parameters())
    a=torch.cat([g.reshape(-1) for g in torch.autograd.grad(old,parameters,retain_graph=True)])
    b=torch.cat([g.reshape(-1) for g in torch.autograd.grad(tilt,parameters,retain_graph=True)])
    c=torch.cat([g.reshape(-1) for g in torch.autograd.grad(att,parameters)])
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(c).all()
    return dict(v35_loss=float(old.detach()),tilt_loss=float(tilt.detach()),old_weighted_attitude_loss=float(att.detach()),
        v35_grad_norm=float(a.norm()),tilt_grad_norm=float(b.norm()),old_attitude_grad_norm=float(c.norm()),
        cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20)))


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for parent in PARENTS:verify_parent(parent)
    OUT.mkdir(exist_ok=False)
    names=['scripts/attitude_supervision_v42.py','scripts/audit_attitude_kinematics_v42.py','scripts/prepare_attitude_v42.py',
           'scripts/train_attitude_v42.py','scripts/run_attitude_training_v42.py','tests/test_attitude_supervision_v42.py']
    protocol=dict(stage='v42 attitude-supervision preparation',status='frozen_before_audit_and_calibration',
        code={name:base.sha(base.ROOT/name) for name in names},parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in PARENTS},
        scope='Original407train parents, fixed5flightfolds3seeds, H26K50 and40070parameters; no developmentvalidation/sealed/reserved, no outcome-selected windows.',
        kinematics=dict(native='Reintegrate everyv40modelrate tape with original midpoint quaternion update; require exact stored quaternion equality.',
            oracles=['native','true_p','true_q','true_r','true_pq','true_pr','true_qr','true_pqr','remove_window_rate_bias'],
            gate='Primarywarm2_plus/v35 full1s true-rate tiltRMSE <=50%native; >=4/5flights improve. Otherwise stop before calibration/training preparation. This is a diagnostic suitability check, not a physical noise bound.'),
        loss=dict(formula='L_new = unchanged_v35_full_plus_local_loss + lambda_fold * mean_batch_time(sum_xyz((g(qpred)-g(qtrue))^2))/.1^2',
            gravity='g(q)=R_nb(q).T*[0,0,1], normalizeq; yaw and quaternion-sign invariant. No Euler singularity. Localmetric=d_pitch^2+cos(pitch)^2*d_roll^2, not equal Euler weights away from small pitch.',
            angle_scale_rad=.1,old_terms='All v35 global attitude/rate/velocity plus localrate terms retained unchanged.',
            zero='Exact call to v35objective withlocalweight1, skip extra loss graph.'),
        coefficient=dict(rule='lambda_fold=clip(.5*median(norm(gradL_v35)/norm(gradL_tilt)),.01,10).',
            samples='Exactly first4frozen32-parent draws for each of3seeds in thisfold; freshoriginalinitialization; pairedv34localoffsets;12gradientpairs. Commonweight across3seeds per fold.',
            exclusions='No OOF gradients, no coefficient search, no recalibration after seeing final-model gradients.',
            final_fit_audit='Same fixed fit-only batches at v35final checkpoints, report only. Does not set weight.'),
        training=dict(updates=1600,batch=32,local_batch=32,seeds=[17,23,42],folds=5,optimizer='AdamW lr3e-4 weight_decay1e-4 clip_grad_norm1',
            initialization='Same original fresh40070parameter initialization asv35, not finetuning.',
            samples='Exact v34parentdraws and localoffsets; no extra recurrent branch for tilt; extra quaternion operations/backprop still add cost.',
            control='First full1600step zero-tilt fold0seed17 exactly reproduce frozenv35 weights/buffers.',
            candidate_training_this_preparation=False),
        evaluation=dict(primary='OOF complete1s tiltRMSE improves>=5% vs v35, all3seeds improve and>=4/5flights improve.',
            protections='Original18: p/r1s waveform andvelocity/attitude.5/1s endpoints perseed regress<=5%; addq1swave perseed androll/pitch1swave perseed,27total.',
            legacy='Report originalq>=5%improvement condition separately; do not relabel tilt success as a q gate pass.',
            absolute='Original active .5s p/q<=.15rad/s,attitude<=3deg,velocity<=.5m/s,12cells unchanged.',
            coupled='All4v40variants andall407parents withgeneratedactions. Surface0/1 meanRMSE improves>=5%, allseeds and>=4/5flights improve; all4actionchannels perseed regress<=5%. Keep state protections and absolute gates on coupled trajectories.',
            promotion='No automatic control-readiness promotion. Five flights repeatedly used for development; exactfirmware andindependentcontrolresponse remain unproven.'),
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'protocol.json',protocol)
    base.write(OUT/'dataset_manifest.json',dict(full_authority=str((DATA/'dataset_manifest.json').relative_to(base.ROOT)),
        full_authority_sha256=base.sha(DATA/'dataset_manifest.json'),full_contract=read(DATA/'dataset_manifest.json'),
        rolling_authority=str((LOCAL/'dataset_manifest.json').relative_to(base.ROOT)),rolling_authority_sha256=base.sha(LOCAL/'dataset_manifest.json'),
        rolling_arrays_sha256=base.sha(LOCAL/'rolling_inputs.npz'),partitions=['original_train_only'],normalization_refit=False))
    base.write(OUT/'status.json',dict(status='kinematic_audit'))
    subprocess.run([sys.executable,'scripts/audit_attitude_kinematics_v42.py'],cwd=base.ROOT,check=True)
    assert read(OUT/'kinematics_verification.json')['preparation_gate']
    base.write(OUT/'status.json',dict(status='fit_only_calibration'))
    x,y,lx,ly=load_inputs();rows=[];batches=[];runs=[];weights=[]
    oldruns=read(V34/'training_plan.json')['runs']
    for fold in range(5):
        info=read(DATA/f'fold{fold}/manifest.json');fit=info['fit_indices']
        assert not set(fit)&set(info['query_indices'])
        with np.load(DATA/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
        for old in [r for r in oldruns if r['fold']==fold]:
            seed=old['seed'];torch.manual_seed(seed);fresh=ActuatorCommandGRU(stats)
            assert fingerprint(fresh)==old['initial_parameter_sha256']
            plan=V34/old['sample_plan'];assert base.sha(plan)==old['sample_plan_sha256']
            with np.load(plan) as z:draws=z['array_index'];offsets=z['offset']
            baseline=V35/f'fold{fold}/joint_increment/seed{seed}/final.pt';trained=model_from(baseline)
            for k in range(4):
                ids=draws[k];assert set(ids)<=set(fit)
                bx={name:v[ids] for name,v in x.items()};by={name:v[ids] for name,v in y.items()}
                local_x,local_y=select_local(lx,ly,ids,offsets[k],fit)
                batches.append(dict(fold=fold,seed=seed,batch=k,parents=ids.tolist(),offsets=offsets[k].tolist()))
                for stage,model in [('fresh',fresh),('v35_final_fit_only',trained)]:
                    rows.append(dict(fold=fold,seed=seed,batch=k,stage=stage,**gradients(model,bx,by,local_x,local_y)))
            runs.append(dict(fold=fold,seed=seed,sample_plan=str(plan.relative_to(base.ROOT)),sample_plan_sha256=base.sha(plan),
                initial_parameter_sha256=old['initial_parameter_sha256'],normalization_sha256=base.sha(DATA/f'fold{fold}/normalization.npz'),
                baseline_checkpoint=str(baseline.relative_to(base.ROOT)),baseline_checkpoint_sha256=base.sha(baseline),steps=1600,batch=32,local_weight=1.))
        selected=[r for r in rows if r['fold']==fold and r['stage']=='fresh']
        weight,raw=calibrated_weight([r['v35_grad_norm'] for r in selected],[r['tilt_grad_norm'] for r in selected])
        weights.append(dict(fold=fold,weight=weight,unclipped_weight=raw,clipped=weight!=raw,gradient_pairs=12))
        for run in runs:
            if run['fold']==fold:run['tilt_weight']=weight
        print(f'fold{fold}: fit-only tilt weight frozen at {weight:.9g}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'gradient_audit.csv',index=False)
    base.write(OUT/'calibration_batches.json',batches);base.write(OUT/'weights.json',weights)
    base.write(OUT/'training_plan.json',dict(status='prepared_pending_preflight',runs=runs,candidate_runs=15,zero_control=dict(fold=0,seed=17,steps=1600),
        preparation_protocol_sha256=base.sha(OUT/'protocol.json'),weights_sha256=base.sha(OUT/'weights.json')))
    checks=[]
    for run in runs:
        fold,seed=run['fold'],run['seed'];fit=read(DATA/f'fold{fold}/manifest.json')['fit_indices']
        with np.load(DATA/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
        with np.load(base.ROOT/run['sample_plan']) as z:draws=z['array_index'];offsets=z['offset']
        torch.manual_seed(seed);fresh=ActuatorCommandGRU(stats)
        models=[copy.deepcopy(fresh) for _ in range(3)]
        opts=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) for m in models]
        for k in range(3):
            ids=draws[k];bx={n:v[ids] for n,v in x.items()};by={n:v[ids] for n,v in y.items()}
            local_x,local_y=select_local(lx,ly,ids,offsets[k],fit)
            for j,(m,opt) in enumerate(zip(models,opts)):
                opt.zero_grad(set_to_none=True)
                loss=legacy_objective(m,bx,by,local_x,local_y,weight=1.)[0] if j==0 else objective(m,bx,by,local_x,local_y,0. if j==1 else run['tilt_weight'])[0]
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            for key,value in models[0].state_dict().items():torch.testing.assert_close(value,models[1].state_dict()[key],rtol=0,atol=0)
            aa,bb=[o.state_dict() for o in opts[:2]];assert aa['param_groups']==bb['param_groups']
            for key,vv in aa['state'].items():
                for key2,v in vv.items():torch.testing.assert_close(v,bb['state'][key][key2],rtol=0,atol=0)
            for key,value in models[2].named_buffers():torch.testing.assert_close(value,dict(fresh.named_buffers())[key],rtol=0,atol=0)
        assert fingerprint(models[2])!=fingerprint(models[1])
        checks.append(dict(fold=fold,seed=seed,disposable_steps=3,zero_parameters_optimizer_exact=True,candidate_finite_nonzero_effect=True,buffers_unchanged=True))
    base.write(OUT/'preflight.json',dict(status='passed',checks=checks,temporary_optimizer_updates=135,temporary_checkpoints_retained=0))
    base.write(OUT/'status.json',dict(status='zero_weight_1600_reproduction'))
    subprocess.run([sys.executable,'scripts/train_attitude_v42.py','--fold','0','--seed','17','--zero-control'],cwd=base.ROOT,check=True)
    assert read(OUT/'zero_weight_reproduction.json')['status']=='passed'
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_attitude_supervision_v42.py','tests/test_joint_increment_v34.py'],cwd=base.ROOT,text=True,capture_output=True)
    base.write(OUT/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0,test.stdout+test.stderr
    for parent in PARENTS:verify_parent(parent)
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    base.write(OUT/'verification.json',dict(status='passed',kinematics_gate=True,fit_only_calibration=True,preflight_runs=15,
        complete1600_zero_reproduction=True,candidate_training_started=False,model_promoted=False,
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    plan=read(OUT/'training_plan.json');plan['status']='ready_not_started';base.write(OUT/'training_plan.json',plan)
    base.write(OUT/'status.json',dict(status='preparation_complete_report_pending'))


if __name__=='__main__':main()
