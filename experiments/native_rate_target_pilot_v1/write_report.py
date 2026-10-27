from common import *

def md(df,digits=6):
 def fmt(x):
  if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}' if np.isfinite(x) else 'NA'
  return str(x)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
 t=pd.read_csv(OUT/'trajectory_summary.csv');f=pd.read_csv(OUT/'factual_summary.csv');delta=pd.read_csv(OUT/'paired_summary.csv');cases=pd.read_csv(OUT/'prespecified_cases_300.csv');train=pd.read_csv(OUT/'training_summary.csv');cover=pd.read_csv(OUT/'target_coverage.csv');sign=pd.read_csv(OUT/'sign_comparison.csv');internal=pd.read_csv(OUT/'internal_summary.csv');logs=[]
 for p in (HERE/'folds').glob('fold*/training/*/seed17/training.jsonl'):logs.append(json.loads(p.read_text().splitlines()[-1]))
 log=pd.DataFrame(logs).sort_values(['fold','model'])
 trajectory=t[t.support=='common_B_prefix'][['model','horizon_ms','reference','position_m_rms','velocity_mps_rms','attitude_rad_rms','p_rms','q_rms','r_rms']].copy();trajectory['attitude_rad_rms']=trajectory.attitude_rad_rms*180/np.pi;trajectory=trajectory.rename(columns={'attitude_rad_rms':'attitude_deg_rms'})
 response=f[(f.scope=='all_heldout')&(f.cohort=='main')&(f.variable=='p')&(f.time_label=='300')&(f.weighting=='flight')&f.method.isin(['B','D'])]
 fold=f[(f.scope.isin(['fold0','fold1','fold2']))&(f.cohort=='main')&(f.variable=='p')&(f.time_label=='300')&(f.weighting=='pair')&f.method.isin(['B','D'])]
 axes=f[(f.scope=='all_heldout')&(f.cohort=='main')&f.variable.isin(['q','r'])&(f.time_label=='300')&(f.weighting=='flight')&f.method.isin(['B','D'])]
 cp=cases[['pair_uid','method','real','pred_A','pred','gained','lost']].rename(columns={'pred':'pred_B'})
 comps=delta[(delta.scope.str.startswith('C'))&(delta.variable=='p')&(delta.weighting=='pair')&delta.method.isin(['B','D'])]
 ins=internal[(internal.scope=='all_heldout')&(internal.cohort=='main')&(internal.variable=='p')&(internal.time_label=='300')&(internal.weighting=='flight')]
 text='''# 原生角速度B标签：6epoch动力学训练对照

## 结论

**不建议直接进入24epoch多种子的B标签替换比较。** 本轮6次训练全部完成，B训练未改善原生角速度自主预测，姿态路径RMS微幅增加；300ms滚转B/D方向完全没有逐对得失，幅度变化很小且各依赖分量不一致。上一轮使用真实未来角速度积分的优势，没有转化为本轮GRU自主预测优势。

这不是B参考无效或永远无法学习的证明：结论限于固定M0、C0/V0、共享原A尺度、原四项Loss、seed17和6epoch预算。建议停止将B作为原角速度目标的直接替代进行扩大训练；下一研究问题可转向潜在运动状态与异步观测的表示关系，先设计和审计，不在本轮实现新模型或Loss。

## 执行范围与公平性

基准 `ac5301011dd288e29643bf85f199ce86f146b58d`，分支 `exp/native-rate-target-pilot-v1`。使用历史 `velocity_rate_multistep_v1/main_50hz`，原train 107,920窗口/94航次、validation 36,332窗口/27航次。三个留出折为25/12/11主对、18/11/17航次；合并评价53,144个未见训练窗口、46航次、48主对、11共享航次依赖分量。Fold0的25对全部属于同一大分量C03，不是25次独立控制实验。

两方案：M0A表示C0+原A未来角速度监督，M0B表示C0+B未来角速度监督。每个模型37,590参数，均从头初始化，仅目标 `10:13` 不同。原位置、速度、姿态目标、历史、因果初值、相位、事件、dt、积分器均不变。B(t0)、truth_initial、未来真实姿态和未来真实相位未进入模型输入。第一步仍由C0和预测角速度共同积分，后续完全自主递推。

三折×两目标×seed17，各6epoch，无提前停止。batch1024，AdamW lr0.001/weight_decay0.0001，梯度裁剪5，Cosine T_max6，原四项轨迹Loss。每折两组共享已核验的拟合侧input normalization、output_scale和loss_scales，没有因B标签重新拟合尺度。各折训练严格排除留出flight及同源ULog/boot；test未访问。

**全部主结果统一加载第6轮last.pt**，不选择best。每轮同时记录原A/B参考下、共同掩码上的未加权validation四项归一化Loss，仅作诊断，不用于选epoch。这不是按B指标选出的有利结果。训练TF32开启、评价关闭，沿用旧接口的统一设置。

## 共同监督支持

B来自前两轮保存的原生测量时间插值节点，精确flight/sample/window身份关联；复用不跨20ms间隙、注册支持边界和首个缺口后终止的规则。没有重新解析原始日志或扫描时间偏移。

模型仍递推原完整dt网格。`loss_mask`单独控制两组全部四项Loss的连续前缀；不支持尾部以有限原A值占位，但梯度和Loss均不使用占位。所有窗口保留，没有零支持窗口。

'''+md(cover)+'''

全train监督减少27,387个原生节点，validation减少10,323个；平均时长约995.0/994.3ms。因此缺少完整B支持并不意味着缺少大部分时域。两臂都使用这一共同掩码，本轮A基线不能直接当作过去未缩短掩码的历史A训练。

## 训练完成情况

**6/6次，36epoch，3,192次优化更新**。每组Fold0/1/2更新为516/564/516。各折两组初始参数哈希、所有六轮注册窗口排列、尺度、更新数逐项一致；batch顺序与原冻结seed17协议一致。未重启训练、未挑选多次运行、无NaN/Inf或非有限梯度。记录的是裁剪前梯度范数，裁剪仍为5。

'''+md(train[['fold','model','epochs','updates','seconds','peak_allocated_bytes']])+'''

六组epoch计时合计约552.5秒（包括每轮验证，不含目标打包/模型载入/后续评价），最高PyTorch allocated约2.52GB；使用GPU1，不终止其他进程。环境与显存记录保留。第6轮validation诊断如下；不同折的归一化尺度不同，不跨折直接平均这些Loss：

'''+md(log[['fold','model','validation_A_flight_loss','validation_B_flight_loss']])+'''

## 普通自主轨迹

先对窗口有效步平方误差平均，再对航次内窗口平均、跨航次等权、最后开方。位置/速度为三维范数RMS，姿态为SO(3)测地角，p/q/r为FRD分量RMS。500/1000ms表示该名义时域内共同支持原生节点的路径误差，不外推到严格端点。留出航次平均最后节点分别约493.77/989.99ms。逐窗口CSV保存真实末端时刻和端点误差，未把不支持时刻补零。

'''+md(trajectory)+'''

1000ms下，对B参考的p/q/r RMS由0.402297/0.443712/0.188686变为0.402348/0.444048/0.188724 rad/s，分别增加约0.013%/0.076%/0.020%。姿态路径RMS由5.824038°变为5.826125°（+0.036%）；500ms由3.885862°变为3.887245°。这些变化很小，不能夸大成严重退化，但也不是改善。

1000ms位置RMS微降约0.0037%，速度增约0.0505%；500ms位置增约0.0075%，速度增约0.0919%。原A完整支持诊断得到相同趋势，结果不由共同掩码隐藏的尾部改善造成。

46个留出航次中，1000ms、B参考下p误差降低13个，q降低13个，r降低6个；姿态误差降低3个、位置降低32个、速度降低5个。不存在跨航次一致的原生角速度或姿态收益。旧离线积分使用真实未来速度/角速度，本轮模型自主预测二者，因此二者的误差数值和改善比例不能直接相除作为收益转化率。

## 300ms滚转响应

B为冻结原生时间参考；D为冻结270ms中心平均并保留瞬时起点。主门限abs(real)>0.1rad/s；48对中B有效40对、近零8对，D有效36对、近零12对。两组在有效对上均没有近零预测弃判（预测零门限1e-8）；真实近零不强行判断方向。以下为航次等权，幅度比和增益使用全部受支持对：

'''+md(response[['model','method','active_pairs','direction_accuracy','scale_ratio','signed_gain','positive_amplitude_bias','negative_amplitude_bias']])+'''

两组B同向率均79.7038%，D均78.7532%。逐对等权则B均80%（32/40）、D均83.3333%（30/36）；依赖分量等权B均81.0905%、D均80.3337%。**所有48主对，B/D均为错→对0、对→错0**，没有跨分量方向改进。不能把不同加权口径的百分比混为同一个指标。

航次等权RMS比B略降0.381875→0.381837，D略降0.684398→0.684076；有符号增益B 0.163572→0.163404，D 0.401496→0.401339。正负真实响应的幅度偏差都仍为负，低估没有解除。极小变化不足以支撑效果排名或显著性结论。

逐折、逐对等权结果：

'''+md(fold[['scope','model','method','active_pairs','direction_accuracy','scale_ratio','signed_gain']])+'''

Fold0的幅度比/增益略增，其余两折略降。11个依赖分量的配对幅度绝对偏差变化如下，单位rad/s，负值表示改善；无有效方向对的分量仍可计算幅度描述，但不据此判断方向：

'''+md(comps[['scope','method','active_pairs','gained_pair_seed_rows','lost_pair_seed_rows','mean_amplitude_error_change']])+'''

B参考下9/11分量幅度偏差微增，D下6/11微增；整体D幅度绝对误差约减少0.000004rad/s，只是极小混合变化。航次互斥17对、同配置/严格匹配及可评价背景均保存于 `factual_summary.csv`，不将重叠队列相加。跨折背景保持不可评价，不重新匹配。

## 固定病例与q/r耦合

300ms p响应差（rad/s），`pred_A/pred_B`分别指A/B训练模型：

'''+md(cp)+'''

train:124仍严重低估；train:336的D仍反向；train:99的B/D均反向；train:203和train:412保持部分正确响应但幅度偏小。没有因结果删除或替换病例。完整曲线见 `figures/fixed_cases.png`，两模型曲线几乎重合。

q/r航次等权方向和幅度：

'''+md(axes[['model','method','variable','active_pairs','direction_accuracy','scale_ratio','signed_gain']])+'''

q/r方向同样没有变化；幅度/增益的变化很小、混合，没有明显耦合轴崩坏，也没有新的稳定收益。

## 模型内部控制响应

复用固定历史四组合AA/AB/BA/BB，receiver历史、初值、相位和dt完全不动，只交换未来四路原生ZOH。沿用已冻结控制支持记录；折内边际范围检查无越界，但这不是联合状态/控制分布或真实反事实支持认证。

'''+md(ins[['model','CA_rms','CB_rms','C_rms','H_rms','C_over_H','both_active_pairs','same_direction']])+'''

C RMS从0.105593到0.105724rad/s，H从0.214585到0.214514。变化很小，且没有伴随事实方向改善；只有10对CA/CB同时超过原活动门限，不能把该子集方向一致推广到全部48对。C/H分解数值闭合。它们是冻结模型内部差异，没有真实交换轨迹标签，不能用于计算反事实准确率。

## Q1–Q5与下一步

- **Q1：未改善原生角速度自主预测。** 对B参考的三轴平均误差微增，航次收益不一致。
- **Q2：离线姿态积分优势未转化。** 自主姿态误差微增约0.036%，不能将之前约20%的真实角速度积分优势当作神经网络可取得的保证。
- **Q3：滚转方向没有改善。** B/D各有0对得失；幅度微变且分量间冲突，主要反向与低估病例保留。
- **Q4：没有明显全面退化，但多数运动量微幅变差。** 位置1000ms有极小改善，不能据此认定整体或控制响应改善。
- **Q5：不推荐直接扩大到24epoch多种子。** 固定6epoch诊断未显示积极机制信号，停止B标签直接替换这一路线。若继续研究，应优先明确潜在运动状态、异步观测、因果初值和监督之间的联合表示，而非据本轮结果追加Loss或参数搜索。

这里只能排除“此次直接替换方案已经显示值得扩大训练的收益”，不能证明所有B相关方法或更长期学习都无效。单种子、反复暴露的开发三折及观察性匹配差限制结论；没有独立盲测、真实反事实或MPC控制验证。

## 验证与交付

8项训练前测试通过：A/B仅目标角速度不同、窗口身份、共同连续掩码、padding梯度为零、折隔离/尺度、原C0积分、输入不读取未来真值、ZOH不变及48对/11分量身份。额外按精确身份抽查2,048窗口的6,144个首/中/末B节点，与归档值float32逐值一致。该CPU抽查曾因重复解压NPZ较慢而中止一次，改为一次加载后按相同样本重跑通过；没有中断、重启或增加任何训练。

训练后核验6/6、36epoch、3,192更新、初始化/顺序/尺度哈希相同，第6轮唯一评价；推理检查batch/单样本、冗余事件和真值投毒通过。主响应方向、RMS比、增益和得失独立简式复算通过，C+H数值闭合；普通轨迹两组支持和身份一致。`results/result_verification.json`、`training_verification.json`、`inference_checks.json`保存证据。

代码、6个last checkpoint、日志、目标与共同掩码、逐窗口/逐对/逐折CSV和5张PNG均在本目录。没有改变旧模型、原Loss、旧报告、AGENTS.md或数据分区。本轮到此停止，没有追加M1、加权、24epoch训练或MPC。
'''
 (HERE/'report_zh.md').write_text(text)
if __name__=='__main__':main()
