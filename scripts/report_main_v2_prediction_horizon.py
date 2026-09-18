"""Render frozen short-horizon evidence, keeping oracle information budgets explicit."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def native_prefix_analysis(out):
    """Labels only, no model call: check intervening errors, not just endpoints."""
    from system_identification.evaluation.trajectory import assemble_trajectory_windows
    from system_identification.evaluation.main_v2_free_running import endpoint_errors
    from system_identification.evaluation.prediction_horizon import error_statistics,STEPS
    samples=pd.read_parquet(ROOT/'dataset/trajectory_v1_august_f5_c4/samples_validation.parquet')
    windows=pd.read_csv(out/'origins.csv');windows['state_sample_count']=26
    truth=assemble_trajectory_windows(samples,windows).truth
    native=[];prefix=[];joint=[]
    for model in ['S0','Step5_S1']:
        with np.load(out/f'{model}_B_autonomous.npz') as z:
            pred={k:z[k] for k in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','flap_frequency_hz','relative_phase_rad']};dt=z['dt_s']
        error=[endpoint_errors(pred,truth,k) for k in range(1,26)]
        for key in ['velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','position_m','phase_rad']:
            values=np.stack([e[key] for e in error],1)
            for j in range(25):native.append(dict(model=model,metric=key,steps=j+1,actual_ms_mean=1000*dt[:,:j+1].sum(1).mean(),**error_statistics(values[:,j],windows.log_id)))
            for k in STEPS:
                maximum=values[:,:k].max(1)
                prefix.append(dict(model=model,metric=key,steps=k,actual_ms_mean=1000*dt[:,:k].sum(1).mean(),**error_statistics(maximum,windows.log_id)))
        for k in STEPS:
            ev=np.stack([e['velocity_m_s'] for e in error[:k]],1);eq=np.stack([e['attitude_deg'] for e in error[:k]],1)
            for velocity,attitude in [(.5,5.),(1.,10.)]:
                mask=((ev<velocity)&(eq<attitude)).all(1);endpoint=(ev[:,-1]<velocity)&(eq[:,-1]<attitude)
                joint.append(dict(model=model,steps=k,velocity_threshold=velocity,attitude_threshold=attitude,prefix_joint_fraction=mask.mean(),endpoint_joint_fraction=endpoint.mean(),n=len(mask)))
    for filename,records in [('native_error_curve.csv',native),('prefix_max_error.csv',prefix),('joint_threshold_summary.csv',joint)]:pd.DataFrame(records).to_csv(out/filename,index=False)
    return pd.DataFrame(native),pd.DataFrame(prefix),pd.DataFrame(joint)


def table(df):
    def fmt(x):return f'{x:.5f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None))+'\n'


def report(out):
    native,prefix,joint=native_prefix_analysis(out)
    manifest=json.loads((out/'manifest.json').read_text());m=pd.read_csv(out/'prediction_horizon_metrics.csv');v=pd.read_csv(out/'variation_summary.csv');threshold=pd.read_csv(out/'control_threshold_summary.csv')
    allm=m[m.log_id=='ALL'];core=['velocity_m_s','attitude_deg','body_rate_rad_s'];models=['S0','Step5_S1']
    colors={'S0':'tab:blue','Step5_S1':'tab:orange'}
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,core):
        for model in models:
            for mode,style,label in [('B_autonomous','-','autonomous'),('A_teacher_refresh','--','refreshed 1-step')]:
                g=allm[(allm.model==model)&(allm['mode']==mode)&(allm.metric==key)].sort_values('steps')
                if mode=='B_autonomous':
                    curve=native[(native.model==model)&(native.metric==key)].sort_values('steps')
                    ax.plot(curve.actual_ms_mean,curve.equal_flight_rmse,style,color=colors[model],label=f'{model}: {label}')
                    ax.scatter(g.actual_ms_mean,g.equal_flight_rmse,s=16,color=colors[model])
                else:ax.plot(g.actual_ms_mean,g.equal_flight_rmse,style,marker='o',color=colors[model],label=f'{model}: {label}')
        hold=allm[(allm.model=='kinematic_hold')&(allm.metric==key)].sort_values('steps');ax.plot(hold.actual_ms_mean,hold.equal_flight_rmse,':',color='gray',label='kinematic hold')
        ax.set(title=key,xlabel='Elapsed native time (ms)',ylabel='Equal-flight RMSE');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'rmse_vs_horizon.png',dpi=160);plt.close(fig)
    gaps=[]
    for model in models:
        for key in core:
            a=allm[(allm.model==model)&(allm['mode']=='A_teacher_refresh')&(allm.metric==key)].set_index('steps')
            b=allm[(allm.model==model)&(allm['mode']=='B_autonomous')&(allm.metric==key)].set_index('steps')
            for k in a.index:gaps.append(dict(model=model,steps=k,metric=key,actual_ms_mean=b.loc[k,'actual_ms_mean'],teacher_one_step=a.loc[k,'equal_flight_rmse'],autonomous_k_step=b.loc[k,'equal_flight_rmse'],gap=b.loc[k,'equal_flight_rmse']-a.loc[k,'equal_flight_rmse'],ratio=b.loc[k,'equal_flight_rmse']/a.loc[k,'equal_flight_rmse']))
    gap=pd.DataFrame(gaps);gap.to_csv(out/'teacher_autonomous_gap.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,core):
        for model in models:
            g=gap[(gap.model==model)&(gap.metric==key)].sort_values('steps');ax.plot(g.actual_ms_mean,g.gap,'-o',label=model)
        ax.axhline(0,color='gray',ls=':');ax.set(title=key,xlabel='Elapsed native time (ms)',ylabel='Autonomous k-step RMSE minus refreshed 1-step RMSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'teacher_vs_autonomous_gap.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for model in models:
        for mode,style in [('B_autonomous','-'),('A_teacher_refresh','--')]:
            g=v[(v.model==model)&(v['mode']==mode)&(v.log_id=='ALL')].sort_values('steps');ax.plot(g.actual_ms_mean,g['median'],style,marker='o',color=colors[model],label=f'{model}: {mode}')
    ax.axhline(1,color='gray',ls=':');ax.set(xlabel='Elapsed native time (ms)',ylabel='Median body-rate variation ratio',title='At 1 step this is increment amplitude, not oscillation fidelity');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'variation_vs_horizon.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,core):
        for model in models:
            for mode,style in [('B_autonomous','-'),('A_teacher_refresh','--')]:
                g=allm[(allm.model==model)&(allm['mode']==mode)&(allm.metric==key)].sort_values('steps');ax.plot(g.actual_ms_mean,g.relative_error_growth,style,marker='o',color=colors[model],label=f'{model}: {mode}')
        ax.set(title=key,xlabel='Elapsed native time (ms)',ylabel='RMSE(k) / RMSE(1)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'state_error_growth.png',dpi=160);plt.close(fig)
    hold=allm[allm.model=='kinematic_hold'].set_index(['steps','metric']);skill=[]
    for row in allm[allm['mode']=='B_autonomous'].itertuples():
        reference=hold.loc[(row.steps,row.metric),'equal_flight_rmse']
        skill.append(dict(model=row.model,steps=row.steps,metric=row.metric,hold_rmse=reference,model_rmse=row.equal_flight_rmse,change_vs_hold_pct=100*(row.equal_flight_rmse/reference-1)))
    skill=pd.DataFrame(skill);skill.to_csv(out/'kinematic_reference_comparison.csv',index=False)
    wide=allm.pivot(index=['model','mode','steps'],columns='metric',values='equal_flight_rmse').reset_index()
    b=allm[allm['mode']=='B_autonomous'].set_index(['model','steps','metric'])
    teacher=allm[allm['mode']=='A_teacher_refresh'].set_index(['model','steps','metric'])
    variations=v[(v.log_id=='ALL')&(v['mode']=='B_autonomous')].set_index(['model','steps'])
    coverage=joint[(joint.velocity_threshold==.5)&(joint.attitude_threshold==5.)].set_index(['model','steps'])
    def rm(model,k,key):return float(b.loc[(model,k,key),'equal_flight_rmse'])
    def fraction(model,k):return 100*float(coverage.loc[(model,k),'prefix_joint_fraction'])
    recommendation=dict(preferred_control_validation_model='Step5_S1',primary_prediction_steps=5,primary_nominal_ms=100,
        exploratory_upper_steps=10,exploratory_upper_nominal_ms=200,baseline_local_steps=[1,2],
        safe_short_label='error-budget language only: S0 20ms; S1 20-40ms; no certified physical safety or all-channel high fidelity',
        long_horizon_nominal_ms=500,long_horizon_recommended=False,MPC_control_validation='CONDITIONAL YES: measurement-refreshed, constrained short-horizon validation only',
        RL_long_model_only_training_ready=False,controller_executed=False,models_trained=False,sealed_test_opened=False,
        scope='accuracy-supported validation windows, not optimal controller horizon or closed-loop stability guarantee')
    if manifest['smoke']:recommendation=dict(smoke=True,recommendation_valid=False)
    (out/'summary.json').write_text(json.dumps(recommendation,indent=2))
    text='''# Step 9.1 — Control-oriented Prediction Horizon Identification

本轮只对冻结S0和Step5 S1做GPU evaluation；未训练、未改变权重/actuator/phase/integration/split，sealed test未打开。S0没有替换为Step8未晋级候选。所有正式指标使用同722起点、5条validation flight、warm26、相同command tape/native dt。

## Mode contract — avoid false k-step claims

- **A_teacher_refresh：NOT DEPLOYABLE diagnostic**。在第k个endpoint，使用真实x_(k-1)、截至t_(k-1)的26点真实history重新编码，预测一步。沿用Step7 A_native的当前时刻phase anchor；actuator proxies由t0初始化后沿commands更新。A的一列k表示时间位置，不表示k步forecast lead。
- **B_autonomous：真正从t0发出的k-step forecast**。只在t0使用truth/history，随后全部使用自身physical/hidden/proxy状态，读取同一future command tape和native dt。没有future frequency/phase/state输入。
- 若“teacher-state k-step”指从相同真实t0发出连续k步预测，它与B数学上相同，不能另造一个更好的Mode A。这里按“每一步输入真实状态、history保持真实”的文字采用teacher-refresh诊断，明确不同信息预算。A/B gap包含truth refresh/history重编码及anchor策略差异，不是严格因果百分比。
- A拼接序列的variation很容易因truth refresh接近1，不是autonomous dynamics恢复证据。k=1只有两个点，variation等价于单次增量幅值比；20–100ms也可能不足一个扑翼周期。

## Metrics

主要标量为五条flight各自vector/geodesic RMSE的等权平均；同时保存pooled RMSE、mean/median/p90/p95/max和每flight/每origin。姿态使用quaternion geodesic角度，不计算quaternion分量RMSE。phase使用circular error。

'''
    if not manifest['smoke']:
        answers=f'''## 五个问题与窗口建议

**本轮建议：使用Step5 S1进入100ms、measurement-refreshed的受限MPC/control验证；200ms作为需要额外验证的上界，500ms不作为默认窗口。S0仅优先用于20ms局部调用，40ms需放宽误差预算，不能把它的200ms endpoint偶然下降解释为整个前缀都准确。** 这是基于本数据的控制验证建议，不是最优控制lookahead的理论鉴定，也不是已经控制成功。

### Q1：多少ms以内保持高质量？

必须按状态区分。S0在20ms的v/attitude RMSE为 {rm('S0',1,'velocity_m_s'):.3f}m/s / {rm('S0',1,'attitude_deg'):.3f}°；40ms v增至 {rm('S0',2,'velocity_m_s'):.3f}m/s，p90为 {b.loc[('S0',2,'velocity_m_s'),'p90']:.3f}m/s。S1在20/40ms v为 {rm('Step5_S1',1,'velocity_m_s'):.3f}/{rm('Step5_S1',2,'velocity_m_s'):.3f}m/s，attitude为 {rm('Step5_S1',1,'attitude_deg'):.3f}/{rm('Step5_S1',2,'attitude_deg'):.3f}°，是更好的短期候选。

但**没有证明任何窗口的所有刚体动态都已高保真**：20ms S0/S1 body-rate RMSE已为 {rm('S0',1,'body_rate_rad_s'):.3f}/{rm('Step5_S1',1,'body_rate_rad_s'):.3f}rad/s（约 {np.rad2deg(rm('S0',1,'body_rate_rad_s')):.1f}/{np.rad2deg(rm('Step5_S1',1,'body_rate_rad_s')):.1f}deg/s 的向量误差RMS）；同点variation是增量幅值比，不是频谱/周期保真。raw recorded rate变化包含快速成分，本轮也没有把它全部解释为可控物理响应。

| Window category | S0 | Step5 S1 | Interpretation |
| --- | --- | --- | --- |
| Safe short horizon | 20ms | 20–40ms | 仅表示较小的v/attitude误差，不是安全认证；body-rate局部误差仍存在 |
| Practical control-validation horizon | 20ms优先；40ms需明确容许误差 | 100ms起步；200ms探索上界 | 每次规划用新测量/合法history初始化；不能用刷新后的teacher误差代替B |
| Long / degraded horizon in this study | 100ms velocity已明显偏差；500ms整体不推荐 | 500ms不推荐默认使用 | 本轮只到500ms，不用5秒结果替代短时域证据 |

### Q2：teacher/autonomous gap何时明显扩大？

1步完全一致，验证逐数组相等。2步（实际约40ms）已经出现清楚差距：S0的autonomous/refreshed teacher v比值={rm('S0',2,'velocity_m_s')/float(teacher.loc[('S0',2,'velocity_m_s'),'equal_flight_rmse']):.2f}、attitude比值={rm('S0',2,'attitude_deg')/float(teacher.loc[('S0',2,'attitude_deg'),'equal_flight_rmse']):.2f}；S1对应 {rm('Step5_S1',2,'velocity_m_s')/float(teacher.loc[('Step5_S1',2,'velocity_m_s'),'equal_flight_rmse']):.2f}/{rm('Step5_S1',2,'attitude_deg')/float(teacher.loc[('Step5_S1',2,'attitude_deg'),'equal_flight_rmse']):.2f}。100ms差距进一步扩大。不是到秒级才开始；但A不断注入truth、B没有，差距本身不能归为某一个隐藏机制。

### Q3：推荐控制prediction horizon？

**S1推荐100ms作为第一轮控制验证窗口，200ms作为探索上界。** 100ms v/attitude/rate RMSE={rm('Step5_S1',5,'velocity_m_s'):.3f}m/s / {rm('Step5_S1',5,'attitude_deg'):.3f}° / {rm('Step5_S1',5,'body_rate_rad_s'):.3f}rad/s，variation={variations.loc[('Step5_S1',5),'median']:.3f}。200ms对应 {rm('Step5_S1',10,'velocity_m_s'):.3f}m/s / {rm('Step5_S1',10,'attitude_deg'):.3f}° / {rm('Step5_S1',10,'body_rate_rad_s'):.3f}rad/s，variation={variations.loc[('Step5_S1',10),'median']:.3f}。

不是只按RMSE选：整个100ms prefix内同时v<0.5m/s、attitude<5°的origin比例是 {fraction('Step5_S1',5):.2f}%，到200ms降为 {fraction('Step5_S1',10):.2f}%，到500ms仅 {fraction('Step5_S1',25):.2f}%。这些是描述性覆盖率，未人为设定成功门槛。S0对应100/200ms仅 {fraction('S0',5):.2f}%/{fraction('S0',10):.2f}%；其200ms endpoint联合比例却为 {100*coverage.loc[('S0',10),'endpoint_joint_fraction']:.2f}%，说明单一endpoint会掩盖中间误差。

相对S0，S1的100ms v误差改变 {100*(rm('Step5_S1',5,'velocity_m_s')/rm('S0',5,'velocity_m_s')-1):+.2f}%，200ms改变 {100*(rm('Step5_S1',10,'velocity_m_s')/rm('S0',10,'velocity_m_s')-1):+.2f}%；100ms rate误差改变 {100*(rm('Step5_S1',5,'body_rate_rad_s')/rm('S0',5,'body_rate_rad_s')-1):+.2f}%。因此increment监督确实扩展了本轮可用于控制验证的精度窗口，尽管此前未改善5秒trajectory。

### Q4：短transition还是长期递推累积？

**两者都有，不能把A排除后只选B。** 一步angular误差在两个模型中都不小；S1相同结构的一步velocity误差大幅改善，也说明局部transition精度本身重要。Teacher误差随后维持同一量级，而自主v/attitude在40–100ms已经积累，限制更长窗口。数据支持“局部误差经过多步传播”，不支持“只有长时间才出错”或“原one-step已足够好”的泛化说法。E(k)/E(1)需同时看绝对误差：S1因为E(1)更小，归一化增长可能更大，不能据此判断它绝对预测更差。

### Q5：能否进入MPC/RL control validation？

**受限MPC/短时控制模型验证：CONDITIONAL YES。** S1存在合理的100ms试验窗口，可从20ms原生更新、warm26真实过去history、每次规划测量刷新开始验证；200ms需单独检查约束和误差尾部。需要在下一阶段验证候选控制动作、动作排序/局部响应和闭环约束；当前recorded-command benchmark没有证明这些。

**长episode纯模型RL训练或直接真机闭环就绪：NO。** 短时模型调用可进入离线验证，但不能把每周期teacher reset的优势当作无真实状态反馈的RL simulator能力。本轮没有运行MPC/RL，也没有证明所有控制任务只需100ms；真实所需lookahead仍取决于任务、代价、约束和控制带宽。

'''
        text=text.replace('## Metrics\n',answers+'## Metrics\n')
    text+=f"Seed=17，GPU={manifest['gpu']}，native dt min/mean/max={manifest['dt_s']['min']:.6f}/{manifest['dt_s']['mean']:.6f}/{manifest['dt_s']['max']:.6f}s。1/2/5/10/25步只是名义20/40/100/200/500ms；actual_ms_mean/min/max保存在指标表，未用固定20ms替代积分。\n\n"
    text+=table(allm[(allm.model=='S0')&(allm['mode']=='B_autonomous')&(allm.metric=='velocity_m_s')][['steps','nominal_ms','actual_ms_mean','actual_ms_min','actual_ms_max']])+'\n'
    text+=table(wide)+'\n![RMSE](rmse_vs_horizon.png)\n\n'
    text+='## Threshold descriptions, not success gates\n\nthreshold使用严格<，不据此颁发安全/控制成功判断。RMSE通过不代表每个rollout通过；同时列出p90和逐origin fraction。\n\n'+table(threshold[(threshold['mode']=='B_autonomous')][['model','steps','metric','threshold','equal_flight_rmse','rmse_below','p90','p90_below','fraction_below']])
    text+='\n## Growth and teacher gap\n\n'+table(gap)+'\n![Gap](teacher_vs_autonomous_gap.png)\n\n![Growth](state_error_growth.png)\n\n'
    text+='每个native step另算E(t)，避免只看5个endpoint遗漏中间误差峰值。下面报告同一origin在整个prefix内同时满足velocity/attitude阈值的比例；仍是描述性覆盖率，不是成功/安全gate。\n\n'+table(joint)+'\n'
    text+='## Dynamic variation and kinematic hold reference\n\n简单reference固定t0的NED velocity、body rate和frequency，按native dt积分position/quaternion/phase；不拟合参数、不加入第三个NN。它检查复杂模型是否优于局部保持/运动学外推，不代表真实控制器。\n\n'+table(v[(v.log_id=='ALL')&(v['mode']=='B_autonomous')][['model','steps','median','p10','p90','valid_count']])+'\n'+table(skill[skill.metric.isin(core)])+'\n![Variation](variation_vs_horizon.png)\n\n'
    text+='## Stability support\n\n'+table(pd.read_csv(out/'stability_summary.csv'))+'\n训练域越界不等同物理不安全；finite也不等同模型可用于闭环控制。\n\n'
    text+='## Control interpretation\n\n结果只识别该validation command-tape分布下的模型误差窗口，不能从离线预测误差唯一确定MPC代价/约束/频率所需的最佳lookahead，也没有验证优化器提出的新commands或闭环动作因果响应。进入下一步受限控制验证不等于已证明MPC稳定或允许长时间model-only RL训练。本轮不运行控制器或RL。\n\n'
    text+='## Reproduction and verification\n\n```bash\n/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_prediction_horizon.py --output /tmp/main-v2-prediction-horizon --device cuda:1\n```\n\n使用空目录。manifest记录所有冻结input/checkpoint/source hashes；verification.json记录旧结果保护、checkpoint不变、GPU、native prefix与历史trace parity及相关pytest/diff检查。全仓上轮633 passed/1 failed（外部PX4缺少rpm_pid_params.c）的限制仍保留，本轮不修改外部PX4。\n'
    (out/'report.md').write_text(text)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);report(a.parse_args().output)
