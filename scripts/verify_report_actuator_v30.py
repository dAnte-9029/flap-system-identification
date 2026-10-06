"""Independently replay checkpoints, recompute gates, then seal v30 results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from actuator_training_v30 import OUT, DATA, CONTROL, load, verify_protocol, verify_parent, model_from
from reference_groupcv_v8 import predict
from prepare_actuator_inputs_v29 import fingerprint
from system_identification.models.actuator_command_gru import ActuatorCommandGRU


def read(path): return json.loads(path.read_text())


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    verify_protocol()
    for root in [DATA, CONTROL]: verify_parent(root)
    assert read(OUT / 'launch/status.json')['status'] == 'complete'
    x, y = load(); w = pd.read_csv(DATA / 'windows.csv')
    metric_rows = []; checks = []; seen = {s: [] for s in [17,23,42]}
    for run in read(DATA / 'training_plan.json')['runs']:
        fold, seed = run['fold'], run['seed']; root = OUT / f'fold{fold}/seed{seed}'
        info = read(DATA / f'fold{fold}/manifest.json'); ids = info['query_indices']; seen[seed].extend(ids)
        status = read(root / 'status.json')
        assert status['status'] == 'complete' and status['step'] == 1600
        assert base.sha(root / 'final.pt') == status['checkpoint_sha256']
        rp = read(root / 'run_protocol.json')
        for k,v in run.items(): assert rp[k] == v, k
        assert rp['protocol_sha256'] == base.sha(OUT / 'protocol.json')
        model = model_from(root / 'final.pt')
        ck = torch.load(root / 'final.pt', map_location='cpu', weights_only=False)
        assert (ck['fold'],ck['seed'],ck['steps']) == (fold,seed,1600)
        assert ck['protocol_sha256'] == base.sha(OUT / 'protocol.json')
        with np.load(DATA / f'fold{fold}/normalization.npz') as z:
            stats = {k:z[k] for k in z.files}
            for k,v in stats.items(): np.testing.assert_array_equal(getattr(model,k).numpy(), v.astype(np.float32))
        torch.manual_seed(seed); init = ActuatorCommandGRU(stats)
        assert fingerprint(init) == run['initial_parameter_sha256']
        assert sum(p.numel() for p in model.parameters()) == 40070
        assert model.reference_mean[4] == 0 and model.reference_scale[4] == 1
        held = dict(x); held['future_command'] = x['future_command'][:, :1].expand(-1,50,-1)
        for name, data, indices in [('fit',x,info['fit_indices']),('oof',x,ids),('oof_held',held,ids)]:
            pred = predict(model, data, indices)
            path = root / (name+'.npz' if name=='oof_held' else name+'_predictions.npz')
            with np.load(path) as z:
                np.testing.assert_array_equal(z['array_index'],indices)
                for k,v in pred.items(): np.testing.assert_array_equal(v,z[k])
        changed = dict(x); changed['future_command'] = x['future_command'].clone()
        changed['future_command'][:,10:] += .1
        a = predict(model,x,ids[:2]); b = predict(model,changed,ids[:2])
        for k in a: np.testing.assert_array_equal(a[k][:,:10],b[k][:,:10])
        # Independent OOF metric computation, without evaluate_actuator_v30.errors.
        truth = {k:v[ids].numpy().astype(float) for k,v in y.items()}
        for arm, directory in [('baseline',CONTROL/f'fold{fold}/seed{seed}'),('actuator',root)]:
            with np.load(directory/'oof_predictions.npz') as z:
                np.testing.assert_array_equal(z['array_index'],ids)
                rates = (z['rates_b'].astype(float)-truth['rates_b'])**2
                velocity = ((z['velocity_n'].astype(float)-truth['velocity_n'])**2).mean(-1)
                angle = 2*np.arccos(np.clip(np.abs((z['quaternion'].astype(float)*truth['quaternion']).sum(-1)),0,1))
            quantities = dict(p=rates[:,:,0],q=rates[:,:,1],r=rates[:,:,2],velocity=velocity,attitude=angle**2)
            active = w.iloc[ids].active_reference.to_numpy(bool)
            for key, squared in quantities.items():
                for horizon in [1,5,25,50]:
                    for kind, value in [('wave',squared[:,:horizon].mean()),('endpoint',squared[:,horizon-1].mean())]:
                        metric_rows.append(dict(seed=seed,fold=fold,arm=arm,metric=key,horizon=horizon,kind=kind,rmse=float(np.sqrt(value))))
                if key != 'r':
                    metric_rows.append(dict(seed=seed,fold=fold,arm=arm,metric=key,horizon=25,kind='active',
                                            rmse=float(np.sqrt(squared[active,24].mean()))))
        checks.append(dict(fold=fold,seed=seed,all_predictions_exactly_replayed=True,normalization_buffers_exact=True,
                           initial_parameters_match=True,prefix_causality=True,parameters=40070))
    for values in seen.values(): np.testing.assert_array_equal(sorted(values),np.arange(407))
    independent = pd.DataFrame(metric_rows)
    comp = independent[independent.kind!='active'].groupby(['seed','metric','horizon','kind','arm']).rmse.mean().unstack()
    actual = pd.read_csv(OUT/'comparison.csv'); actual = actual[actual.cohort=='oof'].set_index(['seed','metric','horizon','kind'])
    np.testing.assert_allclose(comp[['baseline','actuator']],actual.loc[comp.index,['baseline','actuator']],rtol=1e-12,atol=1e-12)
    q = comp.reset_index(); q = q[(q.metric=='q')&(q.horizon==50)&(q.kind=='wave')]
    improvement = float(100*(1-q.actuator.mean()/q.baseline.mean()))
    f = independent[(independent.metric=='q')&(independent.horizon==50)&(independent.kind=='wave')]
    flights = f.groupby(['fold','arm']).rmse.mean().unstack(); positive = int((flights.actuator<flights.baseline).sum())
    c = comp.reset_index()
    protect = c[((c.kind=='wave')&(c.horizon==50)&c.metric.isin(['p','r']))|
                ((c.kind=='endpoint')&c.horizon.isin([25,50])&c.metric.isin(['velocity','attitude']))].copy()
    protect['improvement_percent'] = 100*(1-protect.actuator/protect.baseline)
    protection_count = int((protect.improvement_percent>=-5).sum())
    active = independent[(independent.kind=='active')&(independent.arm=='actuator')].groupby(['seed','metric']).rmse.mean()
    recorded = pd.read_csv(OUT/'absolute_active_gate.csv').set_index(['seed','metric'])
    np.testing.assert_allclose(active,recorded.loc[active.index,'rmse'],rtol=1e-12,atol=1e-12)
    limits = {'p':.15,'q':.15,'velocity':.5,'attitude':np.deg2rad(3)}
    absolute_count = sum(v<=limits[key] for (seed,key),v in active.items())
    e = read(OUT/'evaluation.json')
    np.testing.assert_allclose(improvement,e['primary_q_wave_improvement_percent'],rtol=0,atol=1e-10)
    assert positive == e['positive_flights'] and protection_count == e['protection_passed']
    assert len(protect) == e['protection_total'] == 18 and len(active) == e['absolute_active_total'] == 12
    assert absolute_count == e['absolute_active_passed']
    allseeds = bool((q.actuator<q.baseline).all())
    assert allseeds == e['all_seeds_q_positive']
    assert (improvement>=5 and allseeds and positive>=4 and protection_count==18) == e['descriptive_gate']
    independent.to_csv(OUT/'independent_oof_metrics.csv',index=False)
    flights['improvement_percent'] = 100*(1-flights.actuator/flights.baseline)
    flights.to_csv(OUT/'q_by_flight.csv')
    base.write(OUT/'verification.json',dict(status='passed',checks=checks,unique407_oof_per_seed=True,
        independent_all_oof_metrics_and_gates_match=True,parent_hashes_verified=True,
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    full = c[(c.kind=='wave')&(c.horizon==50)].groupby('metric')[['baseline','actuator']].mean()
    table = '\n'.join(f'| {k} | {full.loc[k,"baseline"]:.6f} | {full.loc[k,"actuator"]:.6f} | {100*(1-full.loc[k,"actuator"]/full.loc[k,"baseline"]):+.2f}% |' for k in ['p','q','r','velocity','attitude'])
    seed_table = '\n'.join(f'| {r.seed} | {r.baseline:.6f} | {r.actuator:.6f} | {100*(1-r.actuator/r.baseline):+.2f}% |' for r in q.itertuples())
    flight_table = '\n'.join(f'| {i} | {r.baseline:.6f} | {r.actuator:.6f} | {r.improvement_percent:+.2f}% |' for i,r in flights.iterrows())
    failures = '\n'.join(f'- seed{r.seed} {r.metric} {r.horizon*.02:g}s {r.kind}：退化{-r.improvement_percent:.2f}%。' for r in protect[protect.improvement_percent < -5].itertuples()) or '无保护项失败。'
    hold = pd.read_csv(OUT/'command_hold_diagnostic.csv').groupby(['seed','log_id','metric'])[['factual_mse','held_mse','effect_mse']].mean().pow(.5).groupby('metric').mean()
    hold_table = '\n'.join(f'| {i} | {r.factual_mse:.6f} | {r.held_mse:.6f} | {r.effect_mse:.6f} |' for i,r in hold.iterrows())
    fit = pd.read_csv(OUT/'comparison.csv'); fit = fit[(fit.cohort=='fit')&(fit.kind=='wave')&(fit.horizon==50)&(fit.metric=='q')]
    fit_gain = 100*(1-fit.actuator.mean()/fit.baseline.mean())
    next_step = ('冻结15模型，先对真实OOF操纵片段审计方向、延迟、幅值、持续响应与耦合，并分离输入支持范围；暂不接入新控制器。'
                 if e['descriptive_gate'] else '冻结15模型，先按既有飞行/模式/扑频层定位q误差与输入支持，比较训练内和OOF差异；不继续盲目增加网络、删窗或改权重。')
    report = f'''# v30：执行命令条件模型15组训练与评价

完成v29冻结的5折×3种子×1600步训练，保留原407窗、H26/K50、40070参数和同一批次抽样。对照复用15个已逐值重放的v23角速度参考模型。**条件预测描述门槛{'通过' if e['descriptive_gate'] else '未通过'}，控制用途未晋升；保留A1600。**

## 条件预测结果

完整1秒q波形RMSE改善{improvement:.4f}%，{positive}/5飞行改善，3个种子均改善={allseeds}；保护{protection_count}/18，原活动窗口绝对门槛{absolute_count}/12。训练内q改善{fit_gain:.4f}%。这些OOF飞行已经反复用于开发，不能称为未开封最终测试。

先在每飞行内汇总平方误差并开方，再对飞行、种子等权。p/q/r单位rad/s，速度m/s，姿态rad；正百分比为改善。

| 量 | v23角速度参考 | v30执行命令 | 改善 |
|---|---:|---:|---:|
{table}

| seed | q基线 | q候选 | 改善 |
|---|---:|---:|---:|
{seed_table}

| fold | q基线 | q候选 | 改善 |
|---|---:|---:|---:|
{flight_table}

保护失败项：

{failures}

全部时域及逐种子比较见[comparison.csv](comparison.csv)，逐窗口见[metrics_per_window.csv](metrics_per_window.csv)，原活动标记0.5秒p/q≤0.15rad/s、姿态≤3度、速度≤0.5m/s见[absolute_active_gate.csv](absolute_active_gate.csv)。未缩小评价范围。

## 输入边界与同预算控制

四输入是分配后的motor0/servo0/servo1/servo2归一化命令，不是实测舵角、推力或气动力。四命令补恒零第五槽，保持GRU64结构与40070参数；历史参考和未来参考同时替换，不是只改变未来信息的单因素试验。旧滞后执行历史保留，不加入控制器8列。

命令历史origin−25..0、旧执行历史origin−26..−1、未来origin..+49、标签origin+1..+50；不读取最后审计命令行。各折归一化只用其4个fit飞行的唯一命令行，其他统计继承。状态、标签、相位和9月扑频合同均未改。5飞行全部407不重叠窗口包含Stabilized274/Mission133，共618.64秒。

15组从对应v23相同可训练参数初始化，使用同一1600×32抽样，AdamW lr3e−4/weight_decay1e−4/clip1、原完整1秒损失与CPU确定性设置；固定最后1600步检查点，没有挑选最佳中途结果。未全407重新拟合候选。

## 命令保持诊断

只将未来四命令固定在起点，历史不变。该人工干预没有真实反事实标签；以下是对原日志真值的诊断差异，不能当作真实恒定命令下的精度或舵效。

| 轴 | 原命令误差 | 保持命令误差 | 两预测之差RMS |
|---|---:|---:|---:|
{hold_table}

未来日志命令由反馈产生，可能携带未来状态/扰动信息；即使误差改善也只是不同信息边界的条件预测收益。以后闭环必须让模型状态驱动实际控制/分配链生成命令，不能使用未来日志替代。方向、延迟、幅值和持续响应等独立控制证据仍缺失；当前没有新控制器闭环或实飞安全结论。

## 验证与下一步

4项针对性测试和15组初值/抽样预检通过。15个检查点的全部fit/OOF/命令保持共45组预测逐值重放；模型统计buffer、前缀因果性及每种子407唯一OOF检查通过。独立实现复算全部OOF时域指标及主/保护/绝对门槛，结果一致。源数据、父产物与冻结代码哈希通过；未读取开发验证、sealed或reserved数据。

{next_step} 该下一阶段尚未执行；本轮不把条件预测门槛当作因果控制认证。

交付：[冻结执行协议](protocol.json)、[上游数据](../control_windows_v29/dataset_manifest.json)、[训练状态](launch/status.json)、[评价](evaluation.json)、[独立核验](verification.json)。模型位于fold0–4/seed17、23、42/final.pt，输入接口为ActuatorCommandGRU，不能直接替换参考模型。
'''
    (OUT/'report_zh.md').write_text(report)
    base.write(OUT/'decision.json',dict(status='training_and_evaluation_complete',model_promoted=False,control_ready=False,
        baseline_retained='A1600',descriptive_gate=e['descriptive_gate'],next_step=next_step))
    verify_protocol()
    for root in [DATA,CONTROL]: verify_parent(root)
    code = dict(read(OUT/'protocol.json')['source_pins'])
    code[str(Path(__file__).resolve().relative_to(base.ROOT))] = base.sha(__file__)
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        trained_models=15,reused_controls=15,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))
    print(json.dumps(e,ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
