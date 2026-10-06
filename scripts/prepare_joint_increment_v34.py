"""Freeze local-supervision sampling/loss/budget and validate disposable updates."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_joint_increment_v34 import PREP,OUT,DATA,LOCAL,CONTROL,load_inputs,read,verify_parent,ActuatorCommandGRU,fingerprint
from joint_increment_v34 import balanced_offsets,select_local,objective,LOCAL_WEIGHT


def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    parents=[DATA,CONTROL,LOCAL]
    for root in parents:verify_parent(root)
    names=['scripts/joint_increment_v34.py','scripts/train_joint_increment_v34.py','scripts/run_joint_increment_v34.py',
        'scripts/prepare_joint_increment_v34.py','tests/test_joint_increment_v34.py','tests/test_rollout_trace_v32.py',
        'scripts/train_control_windows_v2.py','scripts/prepare_actuator_inputs_v29.py','scripts/actuator_training_v30.py',
        'scripts/reference_groupcv_v8.py','src/system_identification/models/actuator_command_gru.py',
        'src/system_identification/models/reference_gru_unit_gravity.py','src/system_identification/models/trajectory_main_v1.py']
    pins={name:base.sha(base.ROOT/name) for name in names}
    PREP.mkdir(exist_ok=False)
    protocol=dict(stage='v34 joint rolling-increment supervision preparation',status='frozen_before_formal_training',code=pins,
        parents={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in parents},
        scope='Original407train parents and5flight folds. ActuatorCommandGRU only, unchanged40070parameters, H26/K50 andnormalization. Original nonoverlapping parents retained; rolling correlated observations do not become independent training windows.',
        objective=dict(formula='L = unchanged_full50_loss + 1.0 * mean_batch_axes[((rate_local_pred-rate_local_true)/rate_scale)^2]',
            local_weight=LOCAL_WEIGHT,axes=['p','q','r'],normalization='Same existing per-fold rate_scale, no new variance fit and no dt division.',
            increment_identity='Predicted andtrue increments have identical observed initialrate, which cancels. Compute endpoint difference directly to avoid redundant subtraction rounding.',
            weighting='Full loss keeps rates+velocity+0.2attitude exactly. Local term weight1 equals coefficient of full rate loss; average over3axes, not an axis sum. This adds a second equally weighted rate objective; no total-loss renormalization, q-only term or weight search.',
            limits='Rolling true histories are used only in the extra FIT-flight training branch. Inference architecture remains original; future state labels never enter the full1s rollout or a claimed forecast.'),
        sampling='Reuse exact v30 1600x32 parent draws for eachfold/seed. One localoffset0..49 per parent occurrence. Independent per-parent RNG SeedSequence([34,fold,seed,array_index]) drives repeated shuffled50-offset cycles. Every fullcycle coversall50; count difference <=1 perparent. Row-major occurrence order, independent of model RNG/outcomes. Reject queryparents; no filtering or changing parent weights.',
        data='Full inputsv29; rolling observed H26 inputsv33 with latest causalphase/frequency/airspeed. Use only corresponding fit indices for bothlossbranches. Preserve existing September phase/frequencycontracts; no April dataset or prior.',
        training=dict(candidate_runs=15,folds=5,seeds=[17,23,42],steps=1600,batch=32,local_batch=32,
            optimizer='AdamW lr3e-4 weight_decay1e-4, clip total gradient norm1 AFTER summed loss backward.',
            checkpoint='Fixed final1600; no early stopping or checkpoint selection.',
            pairing='Alltrainable initialparameters andparentdraws match v30. Reuse15v30 controls only after fullzero-weight reproduction fold0/seed17 exactly matches final parameters/buffers. Zero failure blocks allcandidates.',
            zero_weight_reproduction_runs=1,zero_weight_steps=1600,planned_total_training_runs=16),
        budget=dict(parent_exposures_each=51200,local_exposures_each=51200,
            reference_recurrent_positions_each=1600*32*(26+50),candidate_recurrent_positions_each=1600*32*(26+50+26+1),
            relative_recurrent_position_increase=27/76,
            disclaimer='Matched optimizerupdates andparentdraws, NOT matched compute. GRUposition count is workload accounting, not FLOPs or walltime; fused encoder anddecoder costs differ. Record actual duration. Zero reproduction is additional verification cost.'),
        comparison='Primarycontrol v30 actuator model, same input boundary. v23 rate model is secondary descriptive context only. Evaluate same407OOF parent targets, perflight then perseedequal weighting; these5flights are repeatedly used development, not unopenedfinaltest.',
        gates=dict(primary='Full1s q waveform RMSE improves>=5% vs v30, all3seedspositive, >=4/5flightspositive.',
            protection='Perseed p/r full1swaveform andvelocity/attitude .5/1s endpoints regress<=5% (18checks).',
            absolute='Originalactive-window .5s p/q<=.15rad/s, attitude<=3deg, velocity<=.5m/s for eachseed (12checks).',
            promotion='Keep relative/protection andabsolute gates distinct; no causal/control promotion even if allpass.',
            secondary='All50 rolling local p/q/r andincrement accumulation on sameOOF parents; no local-result substitute for full1s gate. Held-command diagnostic retains its no-counterfactual-truth limit.'),
        evaluation='Save completefit/OOF free predictions andOOF commandhold. Report20ms/.1s/.5s/1s endpoints andwaveforms, completeoriginalactive masks, localrolling secondarymetrics, perflight/perseed andwindow97 context. No querylabel use in training or normalization.',
        preflight='3disposableupdates per15fold/seed; paired legacy vszero-weight exact parameters/optimizerstate, candidatefinite andunchangedbuffers. No disposablecheckpoint retained, no outcome-based coefficient selection. Nextlaunch still requires full1600step zero reproduction.',
        next_launch='OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_joint_increment_v34.py',
        next_output=str(OUT.relative_to(base.ROOT)),formal_training_started=False,model_promoted=False,baseline_retained='A1600',
        phase_contract=read(DATA/'dataset_manifest.json')['phase_contract'],frequency_contract=read(DATA/'dataset_manifest.json')['frequency_contract'],
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(PREP/'protocol.json',protocol)
    x,y,lx,ly=load_inputs();windows=pd.read_csv(DATA/'windows.csv')
    assert windows.partition.eq('train').all() and len(windows)==407
    runs=[];checks=[];exposure_rows=[]
    for oldrun in read(DATA/'training_plan.json')['runs']:
        fold,seed=oldrun['fold'],oldrun['seed'];info=read(DATA/f'fold{fold}/manifest.json');fit=info['fit_indices']
        assert set(windows.iloc[fit].log_id).isdisjoint(windows.iloc[info['query_indices']].log_id)
        with np.load(base.ROOT/oldrun['draws_path']) as z:draws=z['array_index']
        assert base.sha(base.ROOT/oldrun['draws_path'])==oldrun['draws_sha256']
        offsets=balanced_offsets(draws,fit,fold,seed)
        dest=PREP/f'fold{fold}/seed{seed}';dest.mkdir(parents=True)
        np.savez_compressed(dest/'sample_plan.npz',array_index=draws,offset=offsets)
        # Verify eachparent's exact offset exposure independently with bincount.
        for parent in fit:
            used=offsets[draws==parent];counts=np.bincount(used,minlength=50)
            assert counts.max()-counts.min()<=1
            for start in range(0,len(used)-49,50):np.testing.assert_array_equal(np.sort(used[start:start+50]),np.arange(50))
            exposure_rows.append(dict(fold=fold,seed=seed,array_index=parent,parent_exposures=len(used),
                local_min_count=int(counts.min()),local_max_count=int(counts.max()),local_anchors_covered=int((counts>0).sum())))
        with np.load(DATA/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
        assert base.sha(DATA/f'fold{fold}/normalization.npz')==oldrun['normalization_sha256']
        torch.manual_seed(seed);initial=ActuatorCommandGRU(stats)
        assert fingerprint(initial)==oldrun['initial_parameter_sha256']
        models=[copy.deepcopy(initial) for _ in range(3)]
        optimizers=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) for m in models]
        for step in range(3):
            ids=draws[step];offs=offsets[step]
            bx={k:v[ids] for k,v in x.items()};by={k:v[ids] for k,v in y.items()}
            local_x,local_y=select_local(lx,ly,ids,offs,fit)
            for j,(model,opt) in enumerate(zip(models,optimizers)):
                opt.zero_grad(set_to_none=True)
                if j==0:loss=base.loss_fn(model(**bx),by,model)
                else:loss,_,_=objective(model,bx,by,local_x,local_y,weight=0 if j==1 else LOCAL_WEIGHT)
                assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert torch.isfinite(norm)
                opt.step()
            for k,v in models[0].state_dict().items():torch.testing.assert_close(v,models[1].state_dict()[k],rtol=0,atol=0)
            a,b=[opt.state_dict() for opt in optimizers[:2]]
            assert a['param_groups']==b['param_groups']
            for parameter,values in a['state'].items():
                for k,value in values.items():torch.testing.assert_close(value,b['state'][parameter][k],rtol=0,atol=0)
            for k,buffer in models[2].named_buffers():torch.testing.assert_close(buffer,dict(initial.named_buffers())[k],rtol=0,atol=0)
        assert fingerprint(models[2])!=fingerprint(models[1]),'local gradient made no change'
        baseline=CONTROL/f'fold{fold}/seed{seed}/final.pt'
        baseline_run=read(CONTROL/f'fold{fold}/seed{seed}/run_protocol.json')
        assert baseline_run['initial_parameter_sha256']==oldrun['initial_parameter_sha256']
        assert baseline_run['draws_sha256']==oldrun['draws_sha256']
        runs.append(dict(fold=fold,seed=seed,sample_plan=str((dest/'sample_plan.npz').relative_to(PREP)),sample_plan_sha256=base.sha(dest/'sample_plan.npz'),
            parent_draws_path=oldrun['draws_path'],parent_draws_sha256=oldrun['draws_sha256'],initial_parameter_sha256=oldrun['initial_parameter_sha256'],
            normalization_sha256=oldrun['normalization_sha256'],control_checkpoint=str(baseline.relative_to(base.ROOT)),control_checkpoint_sha256=base.sha(baseline),
            steps=1600,batch=32,local_batch=32,local_weight=LOCAL_WEIGHT))
        checks.append(dict(fold=fold,seed=seed,disposable_steps=3,zero_parameters_and_optimizer_exact=True,
            candidate_finite_and_nonzero_effect=True,buffers_unchanged=True,fit_only_parent_and_local_sampling=True))
        print(f'fold{fold} seed{seed}: offsets frozen; disposable3-step checks passed',flush=True)
    exposure=pd.DataFrame(exposure_rows);exposure.to_csv(PREP/'sampling_exposure.csv',index=False)
    base.write(PREP/'training_plan.json',dict(status='ready_not_started',runs=runs,zero_weight_reproduction=dict(fold=0,seed=17,steps=1600),
        expected_candidate_runs=15,expected_total_training_runs=16,protocol_sha256=base.sha(PREP/'protocol.json')))
    base.write(PREP/'dataset_manifest.json',dict(dataset_id='joint_increment_preparation_v34',partition='original_train_only',windows=407,
        full_dataset_manifest=str((DATA/'dataset_manifest.json').relative_to(base.ROOT)),full_dataset_manifest_sha256=base.sha(DATA/'dataset_manifest.json'),
        full_arrays_sha256=base.sha(DATA/'windows.npz'),rolling_dataset_manifest=str((LOCAL/'dataset_manifest.json').relative_to(base.ROOT)),
        rolling_dataset_manifest_sha256=base.sha(LOCAL/'dataset_manifest.json'),rolling_arrays_sha256=base.sha(LOCAL/'rolling_inputs.npz'),
        protocol_sha256=base.sha(PREP/'protocol.json'),sources=read(DATA/'dataset_manifest.json')['sources'],
        phase_contract=protocol['phase_contract'],frequency_contract=protocol['frequency_contract'],normalization_refit=False))
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_joint_increment_v34.py','tests/test_rolling_history_v33.py'],cwd=base.ROOT,text=True,capture_output=True)
    base.write(PREP/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0,test.stdout+test.stderr
    base.write(PREP/'verification.json',dict(status='passed',checks=checks,tests_passed=8,formal_training_started=False,
        temporary_optimizer_updates=135,temporary_checkpoints_retained=0,
        exposure_min=int(exposure.parent_exposures.min()),exposure_max=int(exposure.parent_exposures.max()),
        min_local_anchors_covered=int(exposure.local_anchors_covered.min()),all_parent_offset_counts_balanced=True,
        local_data_fit_only=True,normalization_refit=False,validation_accessed=False,sealed_test_accessed=False))
    for root in parents:verify_parent(root)
    for name,digest in pins.items():assert base.sha(base.ROOT/name)==digest
    text=f'''# v34：局部增量＋完整递推联合监督准备

**数据引用、局部抽样、损失、预算、训练脚本和验收协议已冻结；正式训练尚未启动。**保留A1600，没有新性能或控制可用性结论。

## 固定训练目标

L = 原完整50步损失 + 1.0 × mean[((局部角速度预测−局部真值)/原rate_scale)²]。

原完整损失仍是角速度＋速度＋0.2姿态，全部50步保留。新增项对p/q/r三轴及batch取均值，系数固定1，不搜索、不对总损失重新归一化。这等于增加一个与原角速度项同权的目标，并非不改变原权重分配；其他轴及状态的保护仍不可放宽。

局部预测与真值从同一观测角速度出发，所以增量之差等于端点之差；直接计算端点差避免多余减法舍入。沿用原rate_scale，不除dt，不把20ms有限差分当作瞬时加速度真值。模型结构、40070参数、H26/K50、输入定义和归一化均不改。

## 局部起点与数据边界

15组分别复用v30全部1600×32父窗抽样。每次父窗出现，额外抽其50个滚动起点之一；每个父窗使用独立随机序列反复打乱0..49，每完成50次出现就均匀覆盖一次，剩余次数各位置相差不超过1。随机源由[34,fold,seed,array_index]固定，不影响Torch初始化，也不读取误差选择位置。

原407个不重叠父窗和5飞行划分保留。训练时两个分支都只准访问当前折4个fit飞行；query父窗传给局部采样器会报错。滚动历史仍在原支持内、使用当时可用的26点记录；真实状态只进入独立的局部训练分支，不注入完整1秒自由递推。没有新归一化、窗口删除或独立样本数膨胀。

本次15份采样计划的父窗曝光最少/最多为{exposure.parent_exposures.min()}/{exposure.parent_exposures.max()}次；每个父窗至少覆盖{exposure.local_anchors_covered.min()}个局部位置。详细曝光见[sampling_exposure.csv](sampling_exposure.csv)，每组完整计划位于fold*/seed*/sample_plan.npz。

## 预算和对照

候选仍为5折×seed17/23/42，每组1600更新、batch32，AdamW lr3e−4/weight_decay1e−4，联合loss一次backward后总梯度裁剪1，固定最终检查点。主对照是同执行命令边界的v30；v23仅作描述性背景。

优化更新数与父窗曝光相同，但计算量不同：每个batch元素原26编码＋50递推，共76个循环位置；新增局部26编码＋1递推，合计103，循环位置增加35.53%。每组从3891200增到5273600个位置。该计数不是FLOPs或实测时间，正式训练须记录耗时，不能称计算量完全匹配。

未来启动先做fold0/seed17的系数0完整1600步复现，要求所有参数和buffer与v30逐值相同，失败则阻止候选启动。随后15个候选复用冻结的15个v30对照。共计划16次正式训练执行，其中1次仅用于零权重复现，不是候选或参数搜索。

## 验收和边界

- 完整1秒q波形误差相对v30至少改善5%，三个种子均改善、至少4/5飞行改善。
- 每种子p/r完整1秒及速度/姿态0.5/1秒端点退化不超过5%，共18项保护。
- 原活动窗口0.5秒p/q≤0.15rad/s、姿态≤3度、速度≤0.5m/s，共12项绝对门槛，单独报告。
- 原407个OOF父窗全部评价；滚动单步和离线增量累计只作次要诊断，不能替代完整递推门槛或控制证据。

即使数值门槛通过，也不能自动认证因果舵效或新控制器安全。五飞行OOF已反复用于开发，不是未开封最终测试；本轮不访问开发验证、sealed或reserved。

## 预检与执行入口

8项测试通过，覆盖均衡抽样与随机源隔离、禁止query进入局部训练、增量损失量纲/三轴平均、零权重梯度等价、局部分支梯度，以及滚动时序/因果性。15组各做3步临时更新预检：原v30路径与新零权重路径参数及AdamW状态逐值一致，联合分支梯度/更新有限且确实改变权重，统计buffer不变。共135次临时optimizer调用，没有保存临时检查点，**不等于已经完成正式1600步训练**。

可执行命令（本轮未运行）：

```bash
OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_joint_increment_v34.py
```

输出计划写入control_windows_v35。启动器先验证零权重复现，再并行执行15候选；训练结束状态明确为training_complete_evaluation_pending，后续须按本协议完成指标复算、完整检查点重放与报告，不能以训练完成替代验收。

入口：[协议](protocol.json)、[训练计划](training_plan.json)、[数据引用](dataset_manifest.json)、[预检](verification.json)。此对照不同于旧v18只强化原起点首步q；这里额外训练滚动H26下三轴局部增量，但收益尚未验证。
'''
    (PREP/'report_zh.md').write_text(text)
    base.write(PREP/'decision.json',dict(status='preparation_complete',formal_training_started=False,model_promoted=False,
        planned_candidates=15,planned_zero_controls=1,baseline_retained='A1600',
        next_step='Execute frozen zero-weight full1600 reproduction then15 joint-loss runs, evaluate all original gates and rolling secondary diagnostics.'))
    base.write(PREP/'delivery_manifest.json',dict(artifacts={str(p.relative_to(PREP)):base.sha(p) for p in sorted(PREP.rglob('*')) if p.is_file()},
        code=pins,formal_training_started=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))
    assert not OUT.exists(),'formal execution directory should not be created during preparation'
    print('v34 prepared:15 candidate plans plus1 required zero control; no formal training started',flush=True)


if __name__=='__main__':main()
