"""Render the completed, fixed-protocol pilot report from audited result tables."""
from common import *

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 summary=pd.read_csv(OUT/'factual_summary.csv');main=summary[(summary.scope=='all_heldout')&(summary.cohort=='main')&(summary.variable=='p')&(summary.time_label=='300')&(summary.weighting=='flight')];paired=pd.read_csv(OUT/'paired_summary.csv');assign=pd.read_csv(OUT/'component_assignments.csv');cases=pd.read_csv(OUT/'prespecified_cases_300.csv');signs=pd.read_csv(OUT/'sign_comparison.csv');traj=pd.read_csv(OUT/'trajectory_summary.csv').set_index('model');h=pd.read_csv(OUT/'training_history.csv');completion=json.loads((OUT/'training_completion.json').read_text());verification=json.loads((OUT/'independent_verification.json').read_text())
 assert completion['completed']==12 and completion['optimizer_updates']==6384 and verification['passed']
 rows=[]
 for method in ['B','D']:
  for model in COMBOS:
   r=main[(main.model==model)&(main.method==method)].iloc[0];rows.append([method,model,f'{int(r.active_pairs)}/48',f'{r.direction_accuracy*100:.2f}%',f'{r.scale_ratio:.4f}',f'{r.signed_gain:.4f}',f'{r.mae:.4f}'])
 primary_table=table(['参考','组合','方向有效对','航次等权同向率','预测/真实 RMS','有符号增益','MAE rad/s'],rows)
 rows=[]
 for model in ['M0W1','M1W1']:
  for method in ['B','D']:
   g=paired[(paired.scope=='all_heldout')&(paired.model==model)&(paired.method==method)&(paired.variable=='p')].set_index('weighting');r=g.loc['pair'];rows.append([model[:2],method,int(r.gained_pair_seed_rows),int(r.lost_pair_seed_rows)]+[f'{g.loc[w,"direction_difference"]*100:+.2f}' for w in ['pair','flight','component']])
 transitions_table=table(['结构','参考','错→对','对→错','逐对 Δpp','航次等权 Δpp','分量等权 Δpp'],rows)
 rows=[]
 for (comp,fold),g in assign.groupby(['component_id','fold']):
  z=paired[(paired.scope==comp)&(paired.variable=='p')&(paired.weighting=='pair')]
  active=[int(z[z.method==meth].active_pairs.iloc[0]) for meth in ['B','D']];row=[comp,int(fold),len(g),'/'.join(map(str,active))]
  for model in ['M0W1','M1W1']:
   for meth in ['B','D']:
    r=z[(z.model==model)&(z.method==meth)].iloc[0];row.append('NA' if r.active_pairs==0 else f'{int(r.gained_pair_seed_rows)}/{int(r.lost_pair_seed_rows)}')
  rows.append(row)
 component_table=table(['分量','Fold','对数','B/D 有效对','M0 B 得/失','M0 D 得/失','M1 B 得/失','M1 D 得/失'],rows)
 rows=[]
 for uid in json.loads((HERE/'config.json').read_text())['examples']:
  g=cases[(cases.pair_uid==uid)&(cases.method=='D')].set_index('model');row=[uid,f'{g.iloc[0].real:+.4f}']
  for m in ['M0W1','M1W1']:
   r=g.loc[m];row.append(f'{r.pred_W0:+.4f} → {r.pred:+.4f}')
  rows.append(row)
 case_table=table(['固定病例','D 记录响应 rad/s','M0 W0→W1','M1 W0→W1'],rows)
 rows=[]
 for r in signs[(signs.weighting=='flight')&signs.method.isin(['B','D'])].itertuples():rows.append([r.model[:2],r.method,r.sign,r.pairs,f'{r.amplitude_bias_W0:+.4f} → {r.amplitude_bias_W1:+.4f}',f'{r.direction_W0*100:.2f}% → {r.direction_W1*100:.2f}%'])
 sign_table=table(['结构','参考','真实符号','有效对','幅度偏差 W0→W1（rad/s）','同向率 W0→W1'],rows)
 rows=[]
 for metric in ['position_m_path_rms','velocity_mps_path_rms','attitude_rad_path_rms','rate_radps_path_rms','p_path_rms','q_path_rms','r_path_rms']:
  row=[metric]
  for arch in ['M0','M1']:
   v0=traj.loc[arch+'W0',metric];v1=traj.loc[arch+'W1',metric];row.append(f'{v0:.5f} → {v1:.5f} ({(v1/v0-1)*100:+.2f}%)')
  rows.append(row)
 trajectory_table=table(['物理量：每窗 RMS 后航次等权','M0 W0→W1','M1 W0→W1'],rows)
 rows=[]
 for variable in ['q','r']:
  for method in ['B','D']:
   g=summary[(summary.scope=='all_heldout')&(summary.cohort=='main')&(summary.variable==variable)&(summary.method==method)&(summary.time_label=='300')&(summary.weighting=='flight')].set_index('model');row=[variable,method,int(g.iloc[0].active_pairs)]
   for arch in ['M0','M1']:
    a,b=g.loc[arch+'W0'],g.loc[arch+'W1'];row.extend([f'{a.direction_accuracy*100:.2f}% → {b.direction_accuracy*100:.2f}%',f'{a.scale_ratio:.4f} → {b.scale_ratio:.4f}'])
   rows.append(row)
 coupling_table=table(['轴','参考','有效对','M0 同向率','M0 RMS比','M1 同向率','M1 RMS比'],rows)
 subgroup=summary[(summary.scope=='all_heldout')&(summary.cohort.isin(['flight_disjoint','same_firmware_controller','strict','small_pair']))&(summary.variable=='p')&(summary.time_label=='300')&(summary.weighting=='flight')&summary.method.isin(['B','D'])]
 rows=[]
 for (cohort,method),g in subgroup.groupby(['cohort','method']):
  g=g.set_index('model');row=[cohort,method,int(g.iloc[0].pairs),int(g.iloc[0].active_pairs)]
  for arch in ['M0','M1']:row.append(f'{g.loc[arch+"W0","direction_accuracy"]*100:.2f}% → {g.loc[arch+"W1","direction_accuracy"]*100:.2f}%')
  rows.append(row)
 subgroup_table=table(['队列','参考','可评价对','方向有效对','M0 W0→W1','M1 W0→W1'],rows)
 report=f'''# 控制变化加权 W1：三折、单种子、6 epoch 动力学试训练

**结论：停止推进当前固定 W1 到 24 epoch 多种子比较。** 本轮未看到滚转响应方向和幅度跨多个依赖分量的一致改善。M0 的 B/D 方向各损失一对；M1 的 B 不变、D 只改善一对，且发生于 Fold0 的大分量，预测幅度接近零。普通速度轨迹稍好不能替代控制响应证据。该结论限于本权重、模型、6轮预算和重复使用的开发数据，不证明所有控制条件加权均无价值；本轮不追加 W2。

## 执行与公平性

基准 `exp/control-conditioned-training-weights-v1` / `73c03c73237aa0d6e6c7608827263a94f696f793`；新分支 `exp/control-conditioned-training-pilot-v1`。独立 worktree，旧数据、模型、权重和报告未修改。

已完成 **12/12 次从头训练、72个 epoch、6,384 次更新**，无提前停止、无恢复或重跑选优。每折四组合 M0W0/M0W1/M1W0/M1W1，seed17；Fold0/1/2 每次更新516/564/516（每轮86/94/86）。所有 best 均为 epoch6，同时保存 last、优化器、scheduler和RNG状态。M0参数37,590，M1参数36,534；结构直接导入冻结实现。第一轮六个 W0 运行的训练/验证损失与原相同折、种子的24轮实验第一轮逐值相同；这只是相同初始学习率下的回归检查，不是不同训练预算的性能比较。

每折87,239/95,627/87,750拟合窗口，原500ms历史、1000ms自主预测、六维速度/角速度更新和13维积分状态、V0及 `exact_zoh_merge_us_v1` 不变。AdamW lr0.001、weight_decay0.0001、batch1024、clip5、cosine T_max6。模型和batch顺序均有哈希；同折同结构W0/W1初始参数哈希完全相等，四组合每epoch顺序一致。

折内 mean/scale/output_scale/loss_scales 复用原独立核验的拟合侧结果，逐个核对cache、拟合索引、函数和normalization哈希。没有使用全训练归一化、留出航次或validation估计尺度。权重以 `registered_index/window_id/flight_id` 对齐，27个随机真实batch经独立身份检查；实际batch同一索引同时取输入、target、mask、weight。原float64权重逐值保留，原四分量Loss和网络仍为float32，W1仅标量加权归约为float64；全1时直接退化为原Loss。

四组合best仅由原27个validation航次等权的**未加权 W0 四项轨迹Loss**选择，最小改善1e−4。不同折尺度不同，不把折间归一化validation loss平均当作物理性能。各次训练配置、epoch LR、梯度、所有batch权重均值/和/ESS均已保存。W1实际batch ESS比最小为 {h[h.scheme=='W1'].batch_ess_ratio_min.min()*100:.2f}%；M1控制分支所有batch梯度非零且有限。

GPU0有其他任务，因此仅用GPU1，没有终止其他进程或缩减预算。训练加每轮验证的墙钟合计 {completion['total_epoch_gpu_seconds']:.2f} 秒（非纯kernel计时；不含缓存解压、后续评价和报告），峰值PyTorch allocated约1.69GiB。使用规定 `flap-train-gpu`，没有安装依赖。

## 留出范围与评价定义

48主对涉及46个留出航次、11个共享航次分量，每个组合每对只由所属fold best评价一次。另按冻结协议评价两端同时留出的背景；共105个train匹配对可评价（48主对+57其他/背景，队列标签可重叠），5个跨折背景不补配。`validation:44`只作为三个fold模型的已暴露补充病例保存，不计入排名。完整普通轨迹评价覆盖 **53,144个留出H1000窗口、46航次、四组合212,576行**。

主时刻300ms，FRD p；B沿用原生角速度物理时钟参考，D沿用270ms中心平均，A原始参考保留。两段各自相对起点增量之差为真实/预测事实差；不把匹配差当因果标签。阈值 `|D_real|>0.1 rad/s`，B有效40/48（近零8，16.67%），D有效36/48（近零12，25%）。预测方向零判据仍为原 `1e−8`；本次有效真值中没有预测恰零弃判，但许多预测小于0.1，另表完整保存。100/200/400ms和各对共同端点保留，精确500ms、D边界无支持处保持缺失，未外推或填零。

所有组合的A/B/D真值与支持掩码和旧表逐值一致。交换只替换四路未来控制，receiver历史、initial、dt完全相同，未来相位仍由历史因果外推。折内控制边缘范围支持直接复用同一拟合窗口/相同输入的已核验表；这不是联合状态-控制支持或真实反事实保证，不按支持或模型误差删样本。

## 主结果：方向与幅度

以下为航次等权描述，seed17；RMS比和增益为观察性响应差的统计量，不是物理控制效能。

{primary_table}

严格在同一结构内做 W1−W0 比较：

{transitions_table}

M0的B RMS比0.3808→0.3777，D0.6829→0.6855；M1的B0.4559→0.4541，D0.6916→0.6925。幅度低估基本保持。M0的B MAE略降但方向下降，M1的D方向增加但MAE略升，说明不能只按普通误差或单一指标判优。

![主指标](figures/direction_amplitude_gain.png)

阈值敏感性也未改变这一判断：0.05/0.2下B有效44/35对、D有效42/28对。M0的B同向率75.46→74.50% /79.24→78.06%，D67.09→66.02% /83.46→81.78%；M1的B保持75.99%/78.27%，D68.99→70.17% /80.41→82.25%。没有根据这些结果改变阈值。

## 分量、折与匹配质量

方向得失均来自负向真实响应：M0在Fold1/C02的 `train:363` 失去B/D方向，B预测−0.00519→+0.02022（真实−0.30642），D−0.00328→+0.00172（真实−0.46408）。M1唯一得分提升为Fold0/C03的 `train:788`，D+0.00300→−0.00473，真实−0.36812；W1幅度仅约真实的1.29%，属于接近零附近的符号跨越，不能当作充分重现控制响应。

Fold0包含25对但仅一个分量。去掉Fold0后，M1没有任何p方向新增收益；M0的损失保留。全11分量完整列出（得/失为方向计数，NA表示无有效方向对）：

{component_table}

![全部依赖分量](figures/dependence_components.png)

按分量分别看幅度绝对误差，M0的B有6/11改善、D有7/11改善，M1的B5/11、D7/11改善；变化小且不与跨分量方向收益一致。不能把多数微小幅度变化或25对大分量当作独立重复干预，也不构造显著性或置信区间。

冻结子集保留如下（航次等权；子集重叠，不相加）：

{subgroup_table}

同固件38对、其他安静背景及配置/日期/逐航次细表也全部保存在 `factual_summary.csv`、`per_group_300.csv`；严格匹配仅2对，无足够独立证据支撑模型胜负。

## 正负响应与固定病例

幅度偏差定义为 `|pred|−|real|`，负值表示偏小，但不代表方向正确。

{sign_table}

未出现正负方向同步改善。M0负向B/D方向下降；M1负向D增加一对，正向方向不变。M0 D有4对原已反向的预测绝对幅度进一步增加，M1 B有2对、D有3对，不能用RMS稍变大解释为更好的控制学习。

五个病例按预先固定编号完整展示，历史的“成功/失败”称谓不替代本轮实际结果：

{case_table}

- `train:124`：D仍严重低估，M1 B还保持反向（+0.0565→+0.0507，真实−1.0131），问题未解决。
- `train:336`：M0 D反向幅度0.2409→0.2528进一步放大；M1 D仍约+0.52对真实−0.1747，M1 B反向幅度也增加。
- `train:203`：本轮D两结构均同向；M0稍增幅，M1仍接近零。不能套用旧24轮实验的方向标签。
- `train:99`：D两结构仍反向，M1 W1接近零但未变为正确符号，不能称为成功。
- `train:412`：M0 D幅度稍接近记录，M1则稍远；并非一致结构收益。

完整B/D曲线和C/H见 `figures/case_train_*.png`；额外方向得失病例363/788的全时刻数据保留在事实曲线表，所有48对的散点和失败标记均保存，没有删除不利例。

## q/r 与普通轨迹误差

{coupling_table}

耦合轴效果混合：M0 q-B方向下降，q-D上升；M1 q-B上升而r-B下降。普通事实轨迹未见整体明显恶化，速度误差有约0.3–0.8%的改善，但局部方向和p轨迹没有随之统一改善。

{trajectory_table}

物理单位为m、m/s、rad、rad/s；各窗口先对有效原生步求RMS，再按航次等权，未把不同fold的归一化Loss混成同尺度物理误差。

## 内部控制敏感性

300ms、主队列、航次等权：M0的C RMS **0.10550→0.10653 rad/s**、H **0.21434→0.21413**；M1的C **0.16090→0.15811**、H **0.22619→0.22627**。W1没有系统性提高未来控制贡献，也没有明显削弱历史依赖。

CA/CB在双方绝对响应都超过0.1时同向率均100%，但M0仅10/48对、M1仅17/48对满足该条件，不能推广到全队列。全部CA/CB/C/H、交互量与逐窗口四组合预测已保存。C/H是模型内部差异分解，没有真实交换轨迹，未计算所谓反事实准确率；C变大也不作为成功标准。

## 验证、产物与停止判断

前置9项测试通过，最终 **14 passed**。完整预算、每epoch batch哈希、同结构初始化、权重逐值/身份、fold-fit归一化、未加权验证选best、ZOH冗余分段不变、批量/单样本推理、未来target/phase污染不影响预测均通过。输入按原八字段白名单送入模型，自主递推，无future target回灌。

独立核验5,184个事实主时刻行，插值/中心平均最大差 {verification['factual_max_error']:.2e}；2,016条事实汇总最大差 {verification['summary_max_error']:.2e}；1,008条内部汇总通过，C+H闭合误差 {verification['closure_max_error']:.2e}。独立SciPy检查768条预先等距选取的留出轨迹，物理误差最大差 {verification['trajectory_scipy_max_error']:.2e}。另独立核验1,152条同结构方向得失/配对汇总和正负幅度表。检查范围明确：全量汇总和身份核验，轨迹数值复算为抽查而非全部双实现。test始终未读取。

Q1：W1没有改善M0的主要方向指标，幅度变化很小，反向例仍在。

Q2：对M1效果不同但不足：B方向不变、D只在一个大分量的一例接近零预测中跨过符号边界，幅度低估未解决。

Q3：没有跨多个依赖分量的一致方向收益；唯一M1收益由Fold0/C03贡献。

Q4：正负方向未同步改善，M0出现新反向预测，已有反向病例部分被放大。

Q5：普通轨迹总体接近，部分速度指标改善；q/r局部方向互有得失，不能据此认定控制响应改善。

Q6：**不建议直接推进当前W1到24epoch多种子正式比较，按预先原则停止该方案。** 单种子短预算不足以断言所有加权方法无效，但当前证据没有满足“跨多个分量的方向与幅度积极迹象”。下一步先重新审视可辨识的数据条件与监督信息，而不是继续扩大这个固定权重实验；本轮不实施新Loss、W2、网络修改或MPC。

三折与匹配规则多次参与研究设计，只是重复使用开发数据的探索性证据，不是新的独立盲测。validation用于checkpoint选择，不能当完全独立测试；观察性记录响应不等于真实因果控制效应。

全部产物位于 `experiments/control_conditioned_training_pilot_v1/`。入口 `README.md`；配置 `config.json`；12组best/last及优化器记录在 `folds/`；原生预测、逐对/逐折/逐时刻/逐分量统计、来源SHA256和独立核验在 `results/`；PNG在 `figures/`；完整文件SHA256见 `archive_manifest.json`。
'''
 (HERE/'report_zh.md').write_text(report)
 print('Wrote audited Chinese report')
if __name__=='__main__':main()
