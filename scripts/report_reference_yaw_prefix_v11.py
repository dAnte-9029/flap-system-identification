"""Audit saved prefix-aware candidates and deliver the bounded v11 result."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,STAT_KEYS
from reference_yaw_prefix_v11 import prefix_features
from reference_error_structure_v9 import history_features
from run_reference_yaw_prefix_v11 import OUT,V8,V10
from run_reference_axis_diagnostics_v10 import KEYS
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table
from system_identification.models.reference_gru_yaw_reference import ReferenceGRUYawReference,reference_prefix_features
from system_identification.models.reference_gru_yaw_mean import ReferenceGRUYawMean
from system_identification.models.trajectory_main_v1 import _normalize_quaternion,_quaternion_multiply,_delta_quaternion


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    for name in ['control_windows_v8','control_windows_v9','control_windows_v10']:
        verify_manifest(OUT.parent/name/'delivery_manifest.json',['artifacts','code'])
    decision=json.loads((OUT/'oof_decision.json').read_text());dev=json.loads((OUT/'development_status.json').read_text())
    x,y,w=load_train();features=prefix_features(x);hf=history_features(x,'full');meanrows=[];reloads=[]
    verification=json.loads((OUT/'verification.json').read_text())
    for fold in range(5):
        with np.load(OUT/f'fold{fold}/mean_predictor.npz') as z:
            ids=z['query_indices'];fit=z['fit_indices'];params={k:z[k] for k in ['mean','scale','coefficient','target_mean']};expected=z['query_mean_prediction']
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());assert fit.tolist()==info['fit_indices'] and ids.tolist()==info['query_indices']
        assert not set(w.iloc[fit].log_id)&set(w.iloc[ids].log_id)
        direct=(((features[ids]-params['mean'])/params['scale'])*params['coefficient']).sum(-1)+params['target_mean']
        np.testing.assert_allclose(direct,expected,rtol=1e-10,atol=1e-10)
        with np.load(V10/f'fold{fold}/mean_predictor.npz') as z:hp={k:z[k] for k in params}
        history=(hf[ids]-hp['mean'])/hp['scale']@hp['coefficient']+hp['target_mean']
        for arm,pred in [('history',history),('prefix',direct)]:
            for k in [25,50]:
                mse=(pred[:,k-1]-y['rates_b'][ids,:k,2].numpy().mean(1))**2
                df=w.iloc[ids][['array_index','log_id']].copy();df['arm']=arm;df['horizon_s']=k*.02;df['mean_mse']=mse;meanrows.append(df)
        for seed in [17,23,42]:
            path=OUT/f'fold{fold}/seed{seed}';ck=torch.load(path/'candidate.pt',map_location='cpu',weights_only=False);sd=ck['state_dict']
            assert ck['protocol_sha256']==base.sha(OUT/'protocol.json')
            model=ReferenceGRUYawReference({k:sd[k] for k in STAT_KEYS},params);model.load_state_dict(sd);model.eval()
            with np.load(path/'oof_predictions.npz') as z:
                np.testing.assert_array_equal(z['array_index'],ids);saved={k:z[k][:4] for k in KEYS}
            ix=ids[:4];batch={k:v[ix] for k,v in x.items()}
            with torch.no_grad():pred=model(**batch)
            err=max(float(np.max(abs(pred[k].numpy()-saved[k]))) for k in KEYS);assert err<1e-5
            quat=_normalize_quaternion(x['initial_quaternion'][ix]);previous=x['initial_rates_b'][ix];poses=[]
            for k in range(50):
                rate=torch.tensor(saved['rates_b'][:,k]);dt=x['dt_s'][ix,k,None]
                quat=_normalize_quaternion(_quaternion_multiply(quat,_delta_quaternion(.5*(previous+rate)*dt)));poses.append(quat);previous=rate
            pose=float(np.max(abs(torch.stack(poses,1).numpy()-saved['quaternion'])));assert pose<1e-5
            allbatch={k:v[ids] for k,v in x.items()}
            ff=reference_prefix_features(allbatch['history_state'],allbatch['history_reference'],allbatch['history_actuator'],allbatch['current_mode'],allbatch['future_reference']).numpy()
            forecast=(((ff-params['mean'])/params['scale'])*params['coefficient']).sum(-1)+params['target_mean']
            forecast_error=float(np.max(abs(forecast-expected)));assert forecast_error<1e-5
            # With actual nonzero historical regressors, the copied recurrence must still match v10.
            hpadded=dict(mean=np.column_stack([np.tile(hp['mean'],(50,1)),np.zeros((50,10))]),
                scale=np.column_stack([np.tile(hp['scale'],(50,1)),np.ones((50,10))]),
                coefficient=np.column_stack([hp['coefficient'].T,np.zeros((50,10))]),target_mean=hp['target_mean'])
            hm=ReferenceGRUYawMean({k:sd[k] for k in STAT_KEYS},hp)
            pm=ReferenceGRUYawReference({k:sd[k] for k in STAT_KEYS},hpadded)
            weights={k:v for k,v in sd.items() if not k.startswith('yaw_')}
            hm.load_state_dict(weights,strict=False);pm.load_state_dict(weights,strict=False);hm.eval();pm.eval()
            with torch.no_grad():a=hm(**batch,diagnostics=True);b=pm(**batch,diagnostics=True)
            history_error=max(float((a[k]-b[k]).abs().max()) for k in a);assert history_error<1e-5
            reloads.append(dict(fold=fold,seed=seed,reload_max_abs=err,quaternion_reintegration_max_abs=pose,
                numpy_torch_forecast_max_abs=forecast_error,history_only_equivalence_max_abs=history_error))
    rate=pd.read_csv(OUT/'rate_per_window.csv');rr=rate[rate.cohort=='oof'].groupby(['arm','seed','log_id','horizon_s'])[['r_mean_mse','r_wave_mse']].mean().pow(.5).groupby(['arm','horizon_s']).mean().reset_index().rename(columns={'r_mean_mse':'r_mean_rmse','r_wave_mse':'r_wave_rmse'})
    rr.to_csv(OUT/'yaw_summary.csv',index=False)
    mr=pd.concat(meanrows);mr.to_csv(OUT/'direct_mean_per_window.csv',index=False)
    ms=mr.groupby(['arm','log_id','horizon_s']).mean_mse.mean().pow(.5).groupby(['arm','horizon_s']).mean().reset_index(name='direct_mean_rmse');ms.to_csv(OUT/'direct_mean_summary.csv',index=False)
    response=pd.read_csv(OUT/'reference_sensitivity.csv');sr=response.groupby(['arm','seed','log_id','horizon_s','axis']).reference_effect_mse.mean().pow(.5).groupby(['arm','horizon_s','axis']).mean().reset_index(name='effect_rms')
    sr.to_csv(OUT/'reference_sensitivity_summary.csv',index=False)
    # Recomputed old counterfactual branch must reproduce immutable v10 evidence.
    old=pd.read_csv(V10/'reference_sensitivity.csv').replace({'arm':{'yaw':'history'}})
    keys=['array_index','log_id','arm','seed','horizon_s','axis'];paired=old.merge(response[response.arm!='yaw'],on=keys,suffixes=('_old','_new'),validate='one_to_one')
    assert len(paired)==len(old)
    contrast_error=float(abs(paired.reference_effect_mse_old-paired.reference_effect_mse_new).max());assert contrast_error<1e-10
    testlog=Path('/tmp/reference-v11-tests.log').read_text();assert '27 passed' in testlog and 'failed' not in testlog
    (OUT/'tests.txt').write_text(testlog)
    verification.update(checkpoint_reloads=reloads,focused_tests_passed=27,old_reference_contrast_max_abs=contrast_error,
        source_manifests_verified=['v8','v9','v10'],development_evaluation=dev['status'])
    base.write(OUT/'verification.json',verification)
    ss=pd.read_csv(OUT/'state_summary.csv');ss=ss[(ss.cohort=='oof')&(ss.subset=='all')&ss.horizon_s.isin([.5,1.])]
    response_r=sr[sr.axis=='r'].pivot(index='horizon_s',columns='arm',values='effect_rms');response_r['prefix_vs_baseline']=response_r.yaw/response_r.baseline;response_r['history_vs_baseline']=response_r.history/response_r.baseline
    shown=rr.replace({'arm':{'yaw':'v11 prefix','history':'v10 history'}})
    gain=decision['gain_vs_history_mean_waveform']
    report=['# 参考模型 v11：计划参考前缀进入 r 均值预测','',
        '**完成5折×3种子的同折对照。参考响应的削弱有所缓解，但预测增益几乎不变，未通过预设收益门槛。保留A1600研究基线，不晋升控制用途。**','',
        '## 本轮实际改动','',
        '仅将当前计划参考增量5维、截至当前步的计划参考平均增量5维加入原94维历史特征。固定alpha100、逐预测步独立岭回归，预测前1至50步的真实累计平均r。每折仅用原4个训练飞行拟合参数及归一化，第5个飞行同时排除于GRU、岭回归和归一化；没有使用其他折GRU残差作为训练标签。',
        '407个原训练窗口、原5折、seed17/23/42保持不变，未重新筛选、重训GRU或调整限制。模型只读已观测历史与计划参考前缀，不读未来实际执行器、状态、频率或相位。September相位契约为日志相对相位，物理零点未确认，扑频比7.909090995788574，与April canonical ratio8不同。',
        '修正环完全保持v10：100ms smoothstep、幅值限制为训练r标准差、变化率为每秒2倍该标准差；修正角速度积分到姿态并参与后续GRU递推。这些限制是研究约束。该模型包含固定PX4内环，不能与同一内环重复串联。','',
        '## 同折预测结果','',
        '先计算每飞行RMSE，再对飞行和种子等权平均；单位rad/s。baseline为同折冻结hold模型，history为v10，prefix为v11。','',table(shown),'',
        f'相对原基线，1秒均值误差改善{decision["mean_gain"]:.2%}，完整r波形改善{decision["waveform_gain"]:.2%}；相对v10仅改善{gain[0]:.2%}和{gain[1]:.2%}。3/3种子、5/5飞行相对基线均改善，但预设的10%和5%门槛仍未满足。',
        f'配对整飞行bootstrap的95%探索性区间：均值收益[{decision["bootstrap95_mean_waveform"][0][0]:.2%}, {decision["bootstrap95_mean_waveform"][1][0]:.2%}]，波形收益[{decision["bootstrap95_mean_waveform"][0][1]:.2%}, {decision["bootstrap95_mean_waveform"][1][1]:.2%}]。这5个飞行已多轮用于开发，不是独立盲测；区间没有校正研究选择过程。','',
        table(pd.DataFrame(decision['flight_gains'])),'',
        '岭回归本身的累计均值读出如下，与完整动力学滚动的指标分开：','',table(ms),'',
        '新增参考特征并非在所有时域都改善：直接均值预测器在1秒误差下降约4.76%，在0.5秒反而增加约7.01%；完整修正模型0.5秒均值/波形误差相对v10分别增加约0.15%/0.15%。因此本轮的非退化判断仅指预设1秒主指标，不代表各时域全面改善。',
        '状态端点误差（姿态为rad）：','',table(ss[['arm','horizon_s','p','q','r','attitude','velocity']]),'',
        f'相对基线的其他受保护端点指标最大退化{decision["maximum_protected_regression"]:.2%}，低于5%限制。绝对门槛仍失败，0.5秒p约0.50–0.51、q约0.34–0.35rad/s，姿态约5.4–5.7度。r修正不足以解决p/q主误差。','',
        '## 参考响应比较','',
        '分别输入日志计划参考，以及保持起点参考的计划；保持时同时更新均值预测器与GRU的参考输入。下表是两次模型预测差值的RMS，不是真实飞机控制增益或因果响应真值。','',table(response_r.reset_index()),'',
        '1秒响应保留比例按seed分别为：'+', '.join(f'{seed}: {value:.2%}' for seed,value in decision['reference_effect_ratio_per_seed'].items())+'。均落在预设[90%,110%]研究比较带内；这个带仅防止候选相对旧模型大幅改变参考敏感性，不能认证旧模型本身正确。',
        '加入前缀确实缓解了历史均值修正对参考响应的压制，但仍存在残余削弱；也没有证据说明真实飞机应当具有与基线相同的响应幅度。当前结果不能把参考响应保留等同于可用于控制。','',
        '## 验收、交付与下一步','',
        f'完整OOF门槛：{decision["full_gate"]}；参考响应保护：{decision["reference_effect_guard"]}；不弱于v10：{decision["history_nonregression"]}。开发验证状态：{dev["status"]}。由于收益门槛失败，本轮没有评价739个开发验证窗口，sealed/reserved未访问。',
        '27项针对性测试通过；15个checkpoint重载、前缀因果性、关闭修正复现、姿态独立再积分、numpy/torch均值预测一致性通过。使用真实非零历史回归参数，v11禁用新增特征后与v10推理一致；旧参考保持对照可复现。v8/v9/v10交付哈希及本轮协议源文件已校验。',
        '协议见protocol.json，候选见fold*/seed*/candidate.pt，逐窗口/飞行结果及校验见CSV和verification.json。候选供研究复现，不替换A1600。',
        '下一步应回到占主要误差的p/q完整波形：先按仅由历史状态与输入定义的条件，对同折OOF的波形相关、幅值及参考响应误差分层，寻找重复出现的失配条件。保留所有有效弱响应样本；只有出现可重复线索后，才冻结一个针对性的重训对照。不继续放宽r门槛或经验限制，也不依据本轮结果认定必须补飞。','']
    (OUT/'report_zh.md').write_text('\n'.join(report))
    fig,axes=plt.subplots(1,2,figsize=(11,4.4));xx=np.arange(3);labels=['Baseline','History mean','Prefix mean']
    vals=rr[rr.horizon_s==1.].set_index('arm').loc[['baseline','history','yaw']]
    axes[0].bar(xx-.18,vals.r_mean_rmse,.36,label='Window mean');axes[0].bar(xx+.18,vals.r_wave_rmse,.36,label='Full waveform');axes[0].set(ylabel='Yaw RMSE (rad/s)',title='OOF 1 s, flight-equal');axes[0].legend()
    vals=sr[(sr.axis=='r')&(sr.horizon_s==1.)].set_index('arm').loc[['baseline','history','yaw']]
    axes[1].bar(xx,vals.effect_rms);axes[1].set(ylabel='Factual - held reference RMS (rad/s)',title='Model sensitivity; no flight counterfactual')
    for ax in axes:ax.set_xticks(xx,labels);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'yaw_prefix_comparison.png',dpi=160);plt.close(fig)
    code=list((base.ROOT/'scripts').glob('*yaw_prefix_v11.py'))+[base.ROOT/'src/system_identification/models/reference_gru_yaw_reference.py',base.ROOT/'tests/test_reference_yaw_prefix_v11.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',code={str(p.relative_to(base.ROOT)):base.sha(p) for p in code},
        artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        focused_tests_passed=27,development_status=dev['status'],control_validated=False,sealed_test_accessed=False))
    verify_manifest(OUT/'delivery_manifest.json',['code','artifacts'])
    print(json.dumps(dict(status='complete',artifact_count=len(json.loads((OUT/'delivery_manifest.json').read_text())['artifacts']),max_reload=max(r['reload_max_abs'] for r in reloads),max_pose=max(r['quaternion_reintegration_max_abs'] for r in reloads),max_forecast=max(r['numpy_torch_forecast_max_abs'] for r in reloads)),indent=2))

if __name__=='__main__':main()
