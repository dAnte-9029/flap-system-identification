"""Seal the prepared single-loss comparison; no candidate performance claims."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-v42-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from prepare_attitude_v42 import OUT,PARENTS,read
from audit_controller_support_v26 import verify_parent


def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(rows)


def main():
    assert read(OUT/'verification.json')['status']==read(OUT/'independent_verification.json')['status']=='passed'
    assert read(OUT/'zero_weight_reproduction.json')['status']=='passed'
    protocol=read(OUT/'protocol.json');weights=pd.DataFrame(read(OUT/'weights.json'))
    k=pd.read_csv(OUT/'kinematics_summary.csv')
    k=k[(k.variant=='warm2_plus')&(k.stage==35)&(k.horizon==50)&(k.kind=='wave')].pivot(index='oracle',columns='metric',values='rmse')
    order=['native','true_p','true_q','true_r','true_pq','true_pr','true_qr','true_pqr','remove_window_rate_bias'];k=k.loc[order,['roll','pitch','tilt']]
    gradients=pd.read_csv(OUT/'gradient_audit.csv').merge(weights[['fold','weight']],on='fold')
    gradients['weighted_tilt_grad_ratio']=gradients.weight*gradients.tilt_grad_norm/gradients.v35_grad_norm
    summary=gradients.groupby('stage')[['v35_loss','tilt_loss','old_weighted_attitude_loss','v35_grad_norm','tilt_grad_norm','old_attitude_grad_norm','cosine','weighted_tilt_grad_ratio']].median()
    summary.to_csv(OUT/'gradient_summary_median.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,metric in zip(axes,['roll','pitch','tilt']):
        ax.bar(order,k[metric],color=['#bb6040' if s=='native' else '#287eaa' for s in order])
        ax.set_title(metric);ax.set_ylabel('Full1s angle RMSE (rad)');ax.tick_params(axis='x',rotation=70);ax.grid(axis='y',alpha=.2)
    fig.suptitle('v35 warm2_plus — kinematic rate substitutions, not deployed corrections')
    fig.savefig(OUT/'kinematics.png',dpi=150);plt.close(fig)
    zero=read(OUT/'zero_control/status.json');tests=read(OUT/'test_results.json')
    next_plan=dict(status='ready_not_started',objective='Run the15 frozen tilt-supervision candidates and compare with v35',
        launch='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_attitude_training_v42.py',
        output='docs/analysis/results/control_windows_v43',runs=15,updates_each=1600,
        baseline='Frozenv35 jointincrement checkpoints; completed zero-tilt1600 control need not be repeated.',
        coefficient_source='weights.json; shared per fold across3seeds, no postlaunch tuning.',
        requirements=['Verify preparation and checkpoint/data hashes before launch.',
            'Retain all407 OOF parents and existing active masks; compare tilt/roll/pitch plus original p/q/r/velocity/geodesic attitude metrics perseed/flight.',
            'Independently replay saved checkpoints and preserve all original buffers/initialization/sampling; candidate changes remain loss-only.',
            'After conditional scoring, use the same4v40controller/model coupling variants andteacher baselines for generated action/state evaluation; do not substitute future log commands in this coupled stage.',
            'Apply protocol.json primary/protection/absolute gates exactly; report legacyq gate separately. Even success does not itself certify independent response or new-controller real-flight safety.'],
        no_controller_tuning=True,no_real_flight=True,no_developmentvalidation_sealed_reserved=True)
    base.write(OUT/'next_execution_plan.json',next_plan)
    decision=dict(status='attitude_training_preparation_complete',kinematics_preparation_gate=True,
        full1600_zero_weight_reproduction=True,fit_only_coefficients_frozen=True,candidate_runs_ready=15,
        candidate_training_started=False,model_promoted=False,control_ready=False,
        baseline='A1600 retained; v35 remains the frozen research baseline',
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'decision.json',decision)
    report=f'''# 姿态持续误差监督准备 v42

**运动学核验、fit-only梯度审计、单项新增损失及五折权重、15组训练合同已完成；1600步零权重控制逐值复现v35。**正式候选训练尚未启动，没有新的模型精度或控制可用性结论。

## 姿态误差与角速度历史

v40全部9768条模型轨迹的四元数，都由保存的角速度通过原生中点积分逐值复现。随后沿相同初态，分别替换整段p/q/r及其组合，或去掉预测角速度误差的窗口均值，再仅重做运动学积分；不重跑神经模型或控制器。这些使用未来实测量的结果只作离线诊断，不能部署或当作真实反事实。

v35 warm2_plus完整1秒误差，先飞行内RMSE再五飞行/三种子等权，单位rad：

{table(k.reset_index())}

原roll误差0.061488，用整段实测p替换后为0.031284；原pitch0.047390，用整段实测q替换后为0.018541。**过去角速度误差会通过积分形成当前姿态偏差**，因此v41“仅当前p/q替换对当前动作帮助小”并不意味着角速度学习不重要。

全部实测p/q/r积分后tilt误差0.020128，约为原0.075065的26.81%，5/5飞行下降，通过预先冻结的“总体≤原来一半、至少4/5飞行下降”准备检查。仍有日志角速度与姿态标签的不一致，不能说动力学积分没有任何测量/时间/离散化残差，也不能把0.020128当不可约物理噪声下界。去窗口均值偏差只能将tilt降到0.053415，未解释全部误差。

![运动学对照](kinematics.png)

完整4假设、v30/v35、5飞行3种子、9种角速度带均保存，未按好坏筛窗。该准备检查不是新的独立测试；仍只使用原5个train飞行407父窗，开发验证/sealed/reserved保持未访问。

## 新增什么损失

保留v35原完整50步损失（角速度＋速度＋0.2全姿态）和局部角速度增量项，额外加入：

`L_new = L_v35 + lambda_fold * mean_batch,time(||g(q_pred) − g(q_true)||²) / 0.1²`

其中`g(q)=R_nb(q).T·[0,0,1]`为机体系单位重力方向，先归一化四元数。向量三个坐标求和、batch/time求平均；该项不受全局航向和四元数符号影响，无欧拉角±π跳变或pitch90°奇异性。小误差下近似`(d_pitch² + cos²(pitch)*d_roll²)/0.1²`，不是任意姿态下roll/pitch完全等权；原全姿态项保留，航向监督没有删除。

0.1rad尺度沿用原姿态损失口径，没有重拟合数据归一化。只新增一项损失，不加输入、网络层或控制器。40070参数、H26/K50、固定扑频/相位/空速代理不变。

## 权重如何冻结

在任何本轮审计/校准运行前写入[protocol.json](protocol.json)：每折使用三个种子各自原1600×32采样计划的前4个fit batch及对应局部offset，在原始初始化上获取12对参数梯度范数。

`lambda_fold = clip(0.5 * median(||grad L_v35|| / ||grad L_tilt||), 0.01, 10)`

每折三个种子共享该系数。校准仅用该折4个fit飞行，OOF父窗不参与；不根据OOF效果搜索系数。另在既有v35最终检查点的同一批fit数据上做梯度审计，**这些结果不参与定权或再次调整**。

{table(weights)}

五折均未触及预设上下限。这里只是在初始化附近控制新增梯度的相对规模，不保证所有batch或训练后期一直是0.5倍，也不等同于最优权重。

60组初始化与60组最终fit审计的中位数：

{table(summary.reset_index())}

初始化新增加权梯度/旧联合梯度范数中位数约0.5，在v35最终fit审计约0.238；两者的梯度方向相关性也改变。因此本轮没有“旧损失完全忽略姿态”的结论：旧的全姿态项本来已提供显著梯度，本实验改变的是对倾斜持续误差的相对关注。完整值与校准父窗索引/offset均保留。

## 训练准备与零控制

15组各3步临时更新预检通过：旧v35路径和新lambda=0路径的参数及AdamW状态逐值相同；正权重候选梯度/更新有限且确实改变参数，归一化buffer保持。临时135次optimizer调用不保存候选检查点。

之后实际完成fold0/seed17的**完整1600步零倾斜权重复现**，全部最终参数和buffer与冻结v35完全相同，实测耗时{zero['elapsed_s']:.1f}秒。复现文件保留在[zero_control/final.pt](zero_control/final.pt)；metadata不同，不要求整个checkpoint文件字节相同。它是基线复现实验，不是新候选性能结果。

正式候选固定5折×seed17/23/42，每组从原初值训练1600更新，不是从v35微调。完整父窗与局部offset抽样、batch32＋local32、AdamW lr3e−4/weight_decay1e−4、总梯度裁剪1保持；原407父窗全部留存。没有新增循环分支，但新增四元数投影及反向传播仍有额外计算，不能称完全等计算量。

## 验收规则

新实验目标是完整1秒tilt误差相对v35改善≥5%、三个种子均改善、至少4/5飞行改善。原18项保护保持，另加每种子q完整1秒和roll/pitch完整1秒各≤5%退化，共27项保护。原“q改善≥5%”条件单独报告，tilt成功不冒充q门槛通过。

原活动窗0.5秒绝对门槛保持：p/q≤0.15rad/s、全姿态≤3°、速度≤0.5m/s，共12项。tilt指标不替代全姿态绝对门槛。

随后在完全相同的4种v40耦合假设下，用生成动作评价状态与控制命令：两升降副翼RMSE算术平均改善≥5%，三个seed均改善、至少4/5飞行改善；每个seed每个4通道动作RMSE退化不超过5%，并保持状态保护和绝对门槛。所有窗口、支持范围与饱和统计都报告，不只评价“容易”的片段。任何这些数值通过都不自动证明新控制器实飞可用。

## 核验与执行入口

测试覆盖重力方向的符号/航向不变性、角度量纲、极端pitch处有限梯度、lambda=0与v35梯度一致、正权重确有效果及校准规则。既有局部增量测试同时通过：

```text
{tests['stdout'].strip()}
```

独立用直接重力公式重新计算60组fit梯度和五折权重，核验fit/query隔离、全部运动学汇总及1600步复现参数。所有父产物/检查点/源码哈希保持；见[independent_verification.json](independent_verification.json)。

候选训练已经具备固定输入、系数、脚本和验收合同，启动命令如下；本轮未执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_attitude_training_v42.py
```

输出到`control_windows_v43`，先产生15组候选，再按冻结规则做条件预测和4假设闭环评价。[training_plan.json](training_plan.json)和[next_execution_plan.json](next_execution_plan.json)给出可执行计划。保留A1600和v35研究基线，不调控制器、不补飞、不宣称损失设计已改善真实控制。
'''
    (OUT/'report_zh.md').write_text(report)
    for parent in PARENTS:verify_parent(parent)
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    code=dict(protocol['code'])
    for name in ['scripts/verify_attitude_preparation_v42.py','scripts/report_attitude_v42.py']:code[name]=base.sha(base.ROOT/name)
    base.write(OUT/'status.json',dict(status='complete_verified',candidate_training_started=False))
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code=code,candidate_training_started=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    verify_parent(OUT);print(decision,flush=True)


if __name__=='__main__':main()
