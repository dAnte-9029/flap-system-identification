"""Step 8 paired-flight comparisons, gates and bounded-scope conclusions."""
from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(df):
    def f(v):return f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(f,r))+' |' for r in df.itertuples(index=False,name=None))+'\n'


def main(results,artifacts):
    protocol=json.loads((results/'protocol.json').read_text());names=[e['name'] for e in protocol['experiments']]+['Step5_S1']
    def concat(file):return pd.concat([pd.read_csv(results/n/file) for n in names],ignore_index=True)
    metrics=concat('per_horizon.csv');flights=concat('per_flight.csv');variation=concat('variation.csv');growth=concat('error_growth.csv');teacher=concat('teacher_one_step.csv')
    metrics.to_csv(results/'free_running_metrics.csv',index=False);variation.to_csv(results/'variation_summary.csv',index=False);growth.to_csv(results/'error_growth.csv',index=False);teacher.to_csv(results/'teacher_one_step.csv',index=False)
    pert=pd.concat([pd.read_csv(results/n/f'perturbation_{s}.csv') for n in names for s in ['train','validation']],ignore_index=True)
    pert['kind']=pert.perturbation.str.split('_').str[0]
    pk=pert.groupby(['experiment','partition','kind','step']).agg(amplification_median=('amplification','median'),amplification_p90=('amplification',lambda x:x.quantile(.9)),contraction_fraction=('amplification',lambda x:(x<1).mean()),velocity_rmse=('velocity_error',lambda x:np.sqrt(np.mean(x*x))),attitude_rmse=('attitude_error',lambda x:np.sqrt(np.mean(x*x))),body_rate_rmse=('body_rate_error',lambda x:np.sqrt(np.mean(x*x)))).reset_index()
    pk.to_csv(results/'perturbation_by_kind.csv',index=False)
    ps=pert.groupby(['experiment','partition','perturbation','step']).agg(actual_duration_s=('actual_duration_s','mean'),amplification_median=('amplification','median'),amplification_p90=('amplification',lambda x:x.quantile(.9)),velocity_rmse=('velocity_error',lambda x:np.sqrt(np.mean(x*x))),attitude_rmse=('attitude_error',lambda x:np.sqrt(np.mean(x*x))),body_rate_rmse=('body_rate_error',lambda x:np.sqrt(np.mean(x*x))),nominal_velocity_rmse=('nominal_velocity_error',lambda x:np.sqrt(np.mean(x*x)))).reset_index()
    ps.to_csv(results/'perturbation_recovery.csv',index=False)
    jac=[]
    for n in names:
        for s in ['train','validation']:
            frame=pd.read_csv(results/n/f'jacobian_{s}.csv')
            matrices=np.load(results/n/f'jacobian_{s}.npz')['jacobians'].reshape(-1,14,14).astype(np.float64)
            frame['rigid_largest_singular']=np.linalg.svd(matrices[:,3:12,3:12],compute_uv=False)[:,0]
            frame['rigid_spectral_radius']=np.abs(np.linalg.eigvals(matrices[:,3:12,3:12])).max(1)
            jac.append(frame)
    j=pd.concat(jac,ignore_index=True)
    js=j.groupby(['experiment','partition','step']).agg(largest_singular_median=('largest_singular','median'),largest_singular_p95=('largest_singular',lambda x:x.quantile(.95)),spectral_radius_median=('spectral_radius','median'),dynamic_singular_median=('dynamic_largest_singular','median'),dynamic_radius_median=('dynamic_spectral_radius','median'),rigid_singular_median=('rigid_largest_singular','median'),rigid_radius_median=('rigid_spectral_radius','median'),epsilon_max_singular_difference=('singular_epsilon_difference','max'),epsilon_max_radius_difference=('radius_epsilon_difference','max')).reset_index();js.to_csv(results/'jacobian_summary.csv',index=False)
    baseline=metrics[metrics.experiment=='S0'].set_index('horizon_s');changes=[]
    logids=sorted(flights.log_id.unique());draw=np.array(list(itertools.product(range(len(logids)),repeat=len(logids))))
    keys=['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad']
    for name in names[1:]:
        for h in [.2,.5,1,2,3,5]:
            a=flights[(flights.experiment==name)&(flights.horizon_s==h)].set_index('log_id').loc[logids]
            b=flights[(flights.experiment=='S0')&(flights.horizon_s==h)].set_index('log_id').loc[logids]
            for metric in keys:
                x=a[metric+'_rmse'].to_numpy();y=b[metric+'_rmse'].to_numpy();boot=100*(x[draw].mean(1)/y[draw].mean(1)-1)
                changes.append(dict(experiment=name,horizon_s=h,metric=metric,change_pct=100*(x.mean()/y.mean()-1),ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975),improved_flights=int((x<y).sum())))
    changes=pd.DataFrame(changes);changes.to_csv(results/'paired_changes.csv',index=False)
    tm=teacher.groupby(['experiment','metric']).rmse.mean();gates=[]
    vv=variation.set_index(['experiment','horizon_s','scope'])
    for name in names[1:]:
        ch=changes[changes.experiment==name];short=ch[ch.horizon_s.isin([.2,.5])&ch.metric.isin(['velocity_m_s','attitude_deg','body_rate_rad_s'])]
        mid=ch[ch.horizon_s.isin([.5,1,2])&ch.metric.isin(['velocity_m_s','attitude_deg'])]
        long=ch[ch.horizon_s.isin([3,5])&ch.metric.isin(['velocity_m_s','attitude_deg','body_rate_rad_s'])]
        g1=all(tm[name,k]/tm['S0',k]<=1.10 for k in ['velocity_m_s','attitude_deg','body_rate_rad_s'])
        g2=bool((mid.change_pct<0).all() and (mid.ci95_high<0).any())
        g3=any((g.change_pct<0).all() and (g.ci95_high<0).any() for _,g in long.groupby('metric')) and (long.change_pct<=10).all()
        g4=all(vv.loc[(name,5.,scope),'median']>=.95*vv.loc[('S0',5.,scope),'median'] for scope in ['prefix','last1s'])
        failure=json.loads((results/name/'summary.json').read_text());ref=json.loads((results/'S0/summary.json').read_text())
        g5=all(failure[k]<=ref[k] for k in ['support_failures','numeric_failures','clipping_failures'])
        local=bool((short.change_pct<=10).all());freq=bool((ch[ch.metric=='frequency_hz'].change_pct<=10).all())
        score=float(np.exp(np.mean(np.log(1+long[long.metric.isin(['velocity_m_s','attitude_deg'])].change_pct/100))))
        gates.append(dict(experiment=name,gate1_teacher=bool(g1),gate2_mid=bool(g2),gate3_long=bool(g3),gate4_variation=bool(g4),gate5_support=bool(g5),short_guard=local,frequency_guard=freq,promoted=bool(g1 and g2 and g3 and g4 and g5 and local and freq),selection_score=score))
    gates=pd.DataFrame(gates);gates.to_csv(results/'rollout_loss_ablation.csv',index=False)
    eligible=gates[(gates.experiment!='Step5_S1')&gates.promoted]
    pool=eligible if len(eligible) else gates[gates.experiment!='Step5_S1']
    candidate=pool.sort_values('selection_score').iloc[0];best=candidate.experiment
    # E(t) fit is descriptive over 0.5-5s, never an asymptotic stability proof.
    fits=[]
    for (name,logid,metric),g in growth[growth.horizon_s>=.5].groupby(['experiment','log_id','metric']):
        x=g.actual_time_s.to_numpy();y=g.rmse.to_numpy();linear=np.polyfit(x,y,1);expo=np.polyfit(x,np.log(np.maximum(y,1e-12)),1)
        late=g[g.horizon_s>=3]
        fits.append(dict(experiment=name,log_id=logid,metric=metric,linear_slope=linear[0],exponential_log_slope=expo[0],linear_fit_rmse=np.sqrt(np.mean((np.polyval(linear,x)-y)**2)),exponential_fit_rmse=np.sqrt(np.mean((np.exp(np.polyval(expo,x))-y)**2)),late_3to5_slope=np.polyfit(late.actual_time_s,late.rmse,1)[0]))
    fits=pd.DataFrame(fits);fits.to_csv(results/'growth_fits.csv',index=False)
    plotnames=['S0','Step5_S1',best]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,['velocity_m_s','attitude_deg','body_rate_rad_s']):
        for name in plotnames:
            g=growth[(growth.experiment==name)&(growth.metric==key)].groupby('step').agg(t=('actual_time_s','mean'),e=('rmse','mean'));ax.plot(g.t,g.e,label=name)
        ax.set(title=key,xlabel='Native elapsed time (s)');ax.legend()
    fig.tight_layout();fig.savefig(results/'error_growth_curve.png',dpi=150);plt.close(fig)
    for filename,data,xcol,ycol in [('variation_vs_time',variation[variation.scope=='prefix'],'horizon_s','median'),('horizon_comparison',metrics,'horizon_s','velocity_m_s_equal_log_rmse')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for name,g in data.groupby('experiment'):ax.plot(g[xcol],g[ycol],marker='o',label=name)
        ax.set(xlabel=xcol,ylabel=ycol);ax.legend();fig.tight_layout();fig.savefig(results/(filename+'.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,kind in zip(axes,['velocity','rate','attitude']):
        for name in plotnames:
            g=pk[(pk.experiment==name)&(pk.partition=='validation')&(pk.kind==kind)].sort_values('step')
            native=pert[(pert.experiment==name)&(pert.partition=='validation')&(pert.kind==kind)].groupby('step').actual_duration_s.mean()
            ax.plot(native.loc[g.step],g.amplification_median,label=name)
        ax.axhline(1,color='gray',linestyle='--');ax.set(title=kind,xlabel='Native elapsed time (s)',ylabel='Median perturbation amplification');ax.legend()
    fig.tight_layout();fig.savefig(results/'perturbation_recovery.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));data=[j[(j.experiment==name)&(j.partition=='validation')&(j.step==0)].largest_singular for name in names]
    ax.boxplot(data,tick_labels=names);ax.set(ylabel='Partial physical-state J largest singular (scaled chart)');fig.tight_layout();fig.savefig(results/'jacobian_distribution.png',dpi=150);plt.close(fig)
    selected_experiment=next(e for e in protocol['experiments'] if e['name']==best)
    selection=dict(best_candidate=best,promoted=bool(candidate.promoted),ranking='rank eligible candidates by geometric mean of 3/5s velocity/attitude RMSE ratio; if none eligible, report best observed but do not promote',added_horizon=selected_experiment['added_horizon'],lambda_roll=selected_experiment['lambda_roll'],bounded_growth_proven=False,sealed_test_opened=False)
    (results/'summary.json').write_text(json.dumps(selection,indent=2))
    core=['velocity_m_s','attitude_deg','body_rate_rad_s']
    best_changes=changes[changes.experiment==best].pivot(index='horizon_s',columns='metric',values='change_pct')
    teacher_pct={k:100*(tm[best,k]/tm['S0',k]-1) for k in core}
    final_variation={n:{s:float(vv.loc[(n,5.,s),'median']) for s in ['prefix','last1s']} for n in ['S0',best,'Step5_S1']}
    support=pd.DataFrame([json.loads((results/n/'summary.json').read_text()) for n in names]);support.to_csv(results/'stability_summary.csv',index=False)
    late=fits.groupby(['experiment','metric']).late_3to5_slope.mean()
    native_pert=pk[(pk.partition=='validation')&(pk.step==20)].set_index(['experiment','kind'])
    local_j=js[(js.partition=='validation')&(js.step==0)].set_index('experiment')
    direct=metrics[metrics.horizon_s==5].set_index('experiment')
    s1pct={k:100*(direct.loc[best,k+'_equal_log_rmse']/direct.loc['Step5_S1',k+'_equal_log_rmse']-1) for k in core}
    text=f'''# Step 8 — Short-Horizon Closed-Loop Consistency Training

最佳候选（不等于晋级）：**{best}**。全部gate通过：**{bool(candidate.promoted)}**。
测试组合为 H={selected_experiment['added_horizon']} native steps、lambda={selected_experiment['lambda_roll']}。

本轮实际结果：短前缀加权带来局部和长期的小幅调整，但**没有形成联合精度、动态保真和稳定性均达标的候选**。不晋级，不启动RL。以下结论限于该六模型、固定种子和预算的实验，不能推广成所有loss设计都无效。

## 七个问题的直接回答

1. **是否改善autonomous simulator？局部、小幅改善，未达到本轮成功标准。** {best} 的5s velocity/attitude变化为 {best_changes.loc[5.,'velocity_m_s']:+.2f}% / {best_changes.loc[5.,'attitude_deg']:+.2f}%，但0.5s为 {best_changes.loc[.5,'velocity_m_s']:+.2f}% / {best_changes.loc[.5,'attitude_deg']:+.2f}%，1s为 {best_changes.loc[1.,'velocity_m_s']:+.2f}% / {best_changes.loc[1.,'attitude_deg']:+.2f}%。所有候选都未通过中期联合改善gate；并非因短期超过10%严重退化被剔除。
2. **最佳测试组合为H={selected_experiment['added_horizon']}、lambda={selected_experiment['lambda_roll']}**，实际prefix平均 {selected_experiment['duration_mean_s']:.6f}s，范围 {selected_experiment['duration_min_s']:.6f}–{selected_experiment['duration_max_s']:.6f}s。这里的最佳仅指已测试组合的描述性排名，不是最优horizon证明。增加lambda到1未产生持续长期收益。
3. **更有证据的是局部velocity误差改善，未证实长期稳定性机制被修复。** {best} teacher一步v/attitude/rate变化为 {teacher_pct['velocity_m_s']:+.2f}% / {teacher_pct['attitude_deg']:+.2f}% / {teacher_pct['body_rate_rad_s']:+.2f}%。5s variation prefix从 {final_variation['S0']['prefix']:.5f} 到 {final_variation[best]['prefix']:.5f}，末1s从 {final_variation['S0']['last1s']:.5f} 到 {final_variation[best]['last1s']:.5f}，衰减仍然明显。
4. **没有从指数发散转为有界增长的证据。** baseline本就不是已证实的指数发散；{best} 3–5s velocity误差斜率仍为 {late[best,'velocity_m_s']:.5f}m/s²（S0 {late['S0','velocity_m_s']:.5f}），attitude误差斜率仍为 {late[best,'attitude_deg']:.5f}deg/s（S0 {late['S0','attitude_deg']:.5f}），并未形成平台。
5. **扰动恢复没有一致改善。** validation 20步后的velocity/rate/attitude扰动中位放大率：S0={native_pert.loc[('S0','velocity'),'amplification_median']:.5f}/{native_pert.loc[('S0','rate'),'amplification_median']:.5f}/{native_pert.loc[('S0','attitude'),'amplification_median']:.5f}，{best}={native_pert.loc[(best,'velocity'),'amplification_median']:.5f}/{native_pert.loc[(best,'rate'),'amplification_median']:.5f}/{native_pert.loc[(best,'attitude'),'amplification_median']:.5f}。这是同command tape下的模型内敏感性，不是扰动后真实飞机的恢复试验。
6. **Jacobian没有一致、足以说明更稳定的变化。** validation teacher operating points的完整J最大singular中位数 S0={local_j.loc['S0','largest_singular_median']:.6f}，{best}={local_j.loc[best,'largest_singular_median']:.6f}；刚体子块最大singular为 {local_j.loc['S0','rigid_singular_median']:.6f}→{local_j.loc[best,'rigid_singular_median']:.6f}，刚体spectral radius为 {local_j.loc['S0','rigid_radius_median']:.6f}→{local_j.loc[best,'rigid_radius_median']:.6f}。变化方向并不一致，接近1的谱还受差分精度限制。
7. **没有全面超过Step5 S1。** {best} 相对S1的5s velocity/attitude/rate误差变化为 {s1pct['velocity_m_s']:+.2f}% / {s1pct['attitude_deg']:+.2f}% / {s1pct['body_rate_rad_s']:+.2f}%，轨迹更好；但prefix/last1s variation={final_variation[best]['prefix']:.5f}/{final_variation[best]['last1s']:.5f}，明显低于S1的 {final_variation['Step5_S1']['prefix']:.5f}/{final_variation['Step5_S1']['last1s']:.5f}。它没有把S1的动态恢复与baseline以上的轨迹精度合并起来。

## 科学问题与实际objective

代码审计纠正：原Main V2已有50步自主递推state loss，不是teacher-forced one-step objective。Step7证明真实physical输入下hidden稳定，并未证明局部transition误差已经足够小。因此本实验检验的是**保留原50步objective、增加短前缀权重**，不能声称第一次引入closed-loop training，或已证明distribution mismatch为唯一原因。

经训练契约确认：L=L_legacy50+lambda/H sum(k=1..H)[L_velocity+L_attitude+L_body_rate]，H=5/10/20，lambda=.1/.5/1。原phase/frequency/position项及actuator正则仍保留。所有state normalization沿用原合同（v/2、omega/2、rotation/.35）；A/B/C分量诊断不是额外训练的state-group ablation。6模型矩阵只改变短前缀H/lambda，optimizer、seeds17/29、40+25epochs、680+425updates及最后epoch选择保持一致。所有步来自原forward自递推，future truth只进入loss，无teacher forcing。

具体duration、数据/来源hash、选择门槛见protocol.json与experiment_manifest.csv；5/10/20是实现步数，积分仍读取native dt。S0参数与state tensor hash以及722起点复现见baseline_reproduction.json。checkpoint容器hash不要求相同，因为metadata不同。

优先通道在前H步的系数由1/50变成1/50+lambda/H，后续仍为1/50，总系数由1增为1+lambda。这既增加早期时间权重，也增加这三类状态相对其他通道的总权重；本矩阵不能把两者隔离，不能由性能变化直接证明temporal credit assignment的特定机制。

## Q1/Q2：是否改善，最佳horizon？

'''
    text+=('有候选通过预声明晋级门槛。\n\n' if len(eligible) else '**没有候选通过全部预声明晋级门槛。** 最佳排名只是描述性比较，不作为最终模型晋级。\n\n')
    text+=table(gates)+'\n'+table(pd.DataFrame(protocol['experiments']))
    text+='\n'+table(metrics[['experiment','horizon_s']+[k+'_equal_log_rmse' for k in keys]])
    text+='\n模型排名不能代替gate。优先从通过全部gate的模型中按3/5s v/att比值几何平均选取；若无通过者，仅报告排名最佳。短期/teacher/frequency>10%为退化警戒，variation允许最多5%相对下降；support/numeric/clipping不得增加。中期要求0.5/1/2s v/att均改善且至少一项paired-flight CI不跨零；长期3/5s至少一个核心状态持续改善且其余不超过10%退化。五flight cluster bootstrap，不把重叠722起点作为独立样本；只有一个配对训练seed政策，不宣称跨seed统计稳定。区间是这五条validation flight的描述性重采样，并非新飞行分布或多重模型选择后的确认性显著性保证。\n\n'
    text+='## Q3：local transition还是long-term stability？\n\n分别报告one-step、轨迹和扰动/Jacobian，不能由单一RMSE归因。\n'+table(teacher.groupby(['experiment','metric']).rmse.mean().unstack().reset_index()[['experiment','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz']])
    text+='\n'+table(changes[(changes.experiment==best)&changes.metric.isin(['velocity_m_s','attitude_deg','body_rate_rad_s'])])
    text+='\n## Q4：error growth从unstable变为bounded吗？\n\n**不能证明渐近bounded。** 只观察0–5秒；角度geodesic天然上限180°，不能把角误差饱和当系统稳定。原baseline也未被证明指数发散。下面保留线性、指数拟合及3–5秒斜率；正的末段斜率意味着该窗口仍增长，有限不等于有界稳定。\n'+table(fits[fits.metric.isin(['velocity_m_s','attitude_deg'])].groupby(['experiment','metric'])[['linear_fit_rmse','exponential_fit_rmse','late_3to5_slope']].mean().reset_index())
    text+='\n![Growth](error_growth_curve.png)\n\n## Q5：扰动恢复？\n\n固定train每flight10点、validation每flight10个共同origin：velocity整体±5/10%；三轴omega±5deg/s；三轴SO(3)右扰动±2deg。h/proxies保持原初始化以隔离physical扰动。运行20个native步，同一recorded command tape，不重新计算controller。真实未来并不是扰动后的反事实真值，因此同时报告相对真实轨迹误差和相对**未扰动模型轨迹**的放大率；后者是模型内数值敏感性，不是真实闭环稳定证明。距离只含缩放后的v/rotation/omega（不混入translation中性漂移），全部方向完整保存。\n'+table(ps[ps.step==20].groupby(['experiment','partition'])[['amplification_median','velocity_rmse','attitude_rmse']].mean().reset_index())
    text+='\n![Perturbations](perturbation_recovery.png)\n\n## Q6：state Jacobian更稳定？\n\nJ_x是native-step map的14维physical切空间partial Jacobian，h/proxies/phase anchor固定；不是完整recurrent系统Jacobian。坐标为p,v,body-right SO(3),omega,f,phase；固定尺度1m/2m/s/.35rad/2rad/s/3Hz/1rad。最大singular依赖尺度，spectral radius在相似缩放下不变。central差分epsilon=.001，并以.002检查数值敏感性。位置平移必有中性unit modes，因此另报去掉position的11维block。不得用J的单步谱单独断言长期稳定。\n'+table(js[js.step.isin([0,249])])
    text+='\n![Jacobian](jacobian_distribution.png)\n\n## Q7：是否超过Step5 increment？\n\n同722起点直接比较S0、Step5_S1、最佳short-prefix候选。Step5只读旧checkpoint/trace，未重训或覆盖。\n'+table(metrics[(metrics.experiment.isin(plotnames))&metrics.horizon_s.isin([2,3,5])][['experiment','horizon_s','velocity_m_s_equal_log_rmse','attitude_deg_equal_log_rmse','body_rate_rad_s_equal_log_rmse']])
    text+='\n'+table(variation[variation.experiment.isin(plotnames)])+'\n![Variation](variation_vs_time.png)\n\n'
    text+='所有模型数值/clip事件及训练域支持检查：\n\n'+table(support)+'\n支持域越界减少不是物理安全认证，也不能替代trajectory与variation门槛。\n\n'
    text+='## Loss components and training budget\n\n'
    if (results/'loss_component_analysis.csv').exists():
        components=pd.read_csv(results/'loss_component_analysis.csv')
        text+=table(components[['experiment','horizon_steps','group_A','group_B','group_C','group_priority']])
    text+=table(pd.read_csv(results/'training_summary.csv')[['experiment','base_epochs','actuator_epochs','wall_time_s','base_samples_per_s','actuator_samples_per_s','peak_gpu_bytes','parameter_hash']])
    text+='\nWall-clock/throughput记录实际共享机器环境，不是独占GPU的效率比较。\n\n'
    text+='## Verification limits\n\n'
    if (results/'full_suite_result.json').exists():
        full=json.loads((results/'full_suite_result.json').read_text())
        text+=f"全仓pytest：{full['passed']} passed，{full['failed']} failed。失败为 `{full['failing_test']}`：{full['reason']}。没有修改外部PX4或隐藏该失败；Main V2相关回归另见pytest.log。\n\n"
    text+='刚体9维子块（v/rotation/omega）另报，以排除frequency→phase积分对完整Jacobian范数的影响。接近1的特征值必须与epsilon控制的数值误差对照。S0本来就可能强烈收缩rate扰动；收缩可能反映过度阻尼，只有与trajectory和variation联合改善才有意义。\n\n'
    text+='## 复现\n\n```bash\n/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_rollout_consistency.py --results /tmp/main-v2-step8/results --artifacts /tmp/main-v2-step8/models --device cuda:1 --contract legacy50_plus_short_prefix\n```\n\n使用空目录。sealed test未打开，不训练RL，不更改架构、integration、phase或actuator constants。原文件hash与pytest/git diff验收见verification.json。\n'
    (results/'report.md').write_text(text)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--artifacts',type=Path,required=True);a=ap.parse_args();main(a.results,a.artifacts)
