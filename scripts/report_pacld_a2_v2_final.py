"""Render A2-v2 measured results into the requested Chinese report."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_pacld_a2_v2 import OUT,SOURCE

def table(df):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}' if np.isfinite(v) else '不适用'
        return str(v)
    return '\n'.join(['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']+['|'+'|'.join(fmt(v) for v in row)+'|' for row in df.itertuples(index=False,name=None)])

def main():
    first=json.load(open(OUT/'summary.json'));second=json.load(open(OUT/'scoring_summary.json'))
    pred=pd.read_csv(OUT/'prediction_summary.csv');local=pd.read_csv(OUT/'local_summary.csv');g=pd.read_csv(OUT/'gain_summary.csv');score=pd.read_csv(OUT/'validation_summary.csv');gains=pd.read_csv(OUT/'validation_evaluation_gain_summary.csv');vdate=pd.read_csv(OUT/'validation_date_scores.csv');vg=pd.read_csv(OUT/'validation_evaluation_gains.csv');exc=pd.read_csv(OUT/'excitation.csv');lc=pd.read_csv(OUT/'local_coverage.csv');vc=pd.read_csv(OUT/'validation_local_coverage.csv')
    lc[lc.width==1].groupby(['date','mode']).agg(queries=('query_n','sum'),supported=('supported_queries','sum'),unsupported=('unsupported_queries','sum'),flights=('flight','nunique'),median_condition=('condition','median')).reset_index().to_csv(OUT/'support_by_date_mode.csv',index=False)
    states=[]
    for split,path in [('train',SOURCE),('validation',OUT/'validation_features')]:
        frame=pd.read_pickle(path/'features.pkl');x=np.stack(frame.X)
        for (date,mode),ix in frame.groupby(['date','mode']).groups.items():
            for j,name in [(0,'v_forward'),(1,'v_right'),(2,'v_down'),(3,'p'),(4,'q'),(5,'r'),(9,'flap_frequency')]:
                v=x[ix,j];states.append(dict(split=split,date=date,mode=mode,feature=name,n=len(ix),flights=frame.loc[ix].flight.nunique(),minimum=np.min(v),q25=np.quantile(v,.25),median=np.median(v),q75=np.quantile(v,.75),maximum=np.max(v)))
    pd.DataFrame(states).to_csv(OUT/'state_coverage.csv',index=False)
    direction=vg[(vg.width==1)&vg.direction_eligible].copy();direction=direction.groupby(['date','axis','channel']).agg(same_direction_rate=('direction_agreement','mean'),eligible_localizations=('flight','size')).reset_index();direction.to_csv(OUT/'validation_direction_dates.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for k,axis in enumerate('pr'):
        a=vdate[vdate.axis==axis]
        for name in ['M1','PACLD']:ax[k].plot(a.date,1-a[name+'_response_mse']/a.zero_response_mse,'o-',label=name)
        ax[k].axhline(0,color='gray');ax[k].set_title('Validation '+axis);ax[k].set_ylabel('Conditional response explanation');ax[k].tick_params(axis='x',rotation=35);ax[k].legend()
    fig.tight_layout();fig.savefig(OUT/'validation_explanation.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(9,4))
    for k,axis in enumerate('pr'):
        a=vg[(vg.width==1)&(vg.axis==axis)]
        for c in ['differential','rudder']:
            b=a[a.channel==c];ax[k].scatter(b.G_obs,b.G_PACLD,label=c)
        ax[k].axhline(0,color='gray');ax[k].axvline(0,color='gray');ax[k].set_xlabel('Validation observational residual G');ax[k].set_ylabel('Validation PACLD factual residual G');ax[k].set_title(axis);ax[k].legend()
    fig.tight_layout();fig.savefig(OUT/'validation_local_g.png',dpi=160);plt.close(fig)
    primary=gains[(gains.width==1)&(gains.axis=='p')&(gains.channel=='differential')].iloc[0]
    ps=score[(score.axis=='p')&(score.model=='PACLD')].iloc[0];rs=score[(score.axis=='r')&(score.model=='PACLD')].iloc[0]
    lines=['# PACLD A2-v2 控制有效性评价','',
    f'**核心结论：差动滚转方向一致，但 PACLD 幅值明显不足，不能据此认定控制响应预测已准确。** 主宽度下 validation 局部 G_obs/G_PACLD 中位数为 {primary.G_obs:.4f}/{primary.G_PACLD:.4f} rad/s/单位控制；局部相对偏差中位数 {primary.relative_bias:.1%}。独立 validation 条件解释分数 p={ps.response_score:.4%}、r={rs.response_score:.4%}。只有差动滚转进入主评分，其余作为辅助，不构造完整二维总分。','',
    '本报告评价**闭环 factual 条件控制响应一致性**。G_PACLD 是模型 factual 输出的条件关联，不是固定 Belief 下单独扰动控制所得的内部导数；本任务未计算该反事实导数。它不是随机动作试验，不建立真实因果控制增益，也不将观察性 G 当作无误差真值。差动滚转为主评价对象；垂尾偏航为反馈混杂明显的辅助诊断。指标分别报告，不合成总分。','',
    '## 1. 日期留出与局部 G 独立验证','',
    '基线 `73b29d4763830233233461e91710a167700478cc`，独立分支 `research/pacld-control-effectiveness-v2`。复用 A2-v1 的 11,221 个 train 决策、114 航次、17 日期及全部状态/参考特征。每航次决策间隔至少 1 秒；重复时域/采样组已去重。17 次留一日期外层验证，3 折拟合日期内部选择 Ridge alpha∈{1,10,100,1000}。均值/方差只从拟合日期计算；主响应为 FRD Δp、Δr，rad/s，0.2 秒原生角速度观测，不插值。','',
    'X 保留机体系速度、角速度、重力方向、频率、Hall sin/cos、决策前控制及 0.5 秒历史、率/姿态参考、控制模式和空速记录及缺失标记。冻结映射为 motor、(right-left)/2、(left+right)/2、rudder；保留四通道作调整变量。读数不是实测舵角或力矩。未来控制属于闭环 factual 记录。','',
    '以下 RMSE 按航次先等权、日期再等权；日期 bootstrap 4,000 次、seed=17。区间针对固定交叉拟合预测，未在 bootstrap 内重训。17 日期有限，且模型训练集合相互重叠，区间不能理解为因果检验。gain 为 baseline RMSE 减 model RMSE；删除通道时同时删除 50/100 ms 特征。','',table(pred),'',
    '局部模型修复：在拟合日期内按同模式构造跨航次最多 4 近邻差分；每个评价航次的决策前 X 仅用于选取邻近训练差分中心（每个查询最多 24 个）。ΔY、ΔX、ΔU 均来自训练日期，固定 alpha=10、带宽 0.5/1/2。评价日期的 Y 在拟合函数输入中被置为 NaN，定向毒化测试确认其任意变化不影响 G。随后在留出日期内跨航次差分上比较仅状态校正与状态+控制模型；评价响应不参与拟合。共享近邻和重复配对只在日期聚类汇总中计数，不被当作独立样本。','',table(local),'',
    '三带宽下差动→p 日期中位 G 全部为正，典型值约 2.70–2.83；差动→r 约 0.44–0.49，亦均为正；垂尾→r 约 -3.13 至 -3.18，各日期均为负。垂尾→p 跨日期翻号，不具备同等级方向依据。带宽未依据评价结果选择，1 为预先固定主宽度，其余为敏感性报告。','',table(g[g.channel.isin(['differential','rudder'])]),'',
    '## 2. 未来控制负对照与反馈混杂','',
    '负对照分别用 [250,300] ms、[250,350] ms 的 ZOH 控制均值减决策前值，与正常 [0,50]/[0,100] ms 的特征维数、模型族和样本覆盖一致。全部负对照晚于实际目标测量时刻，维持原有模式/时间支持。228 个决策的原始正常控制复算与上一轮特征最大差 2.22e-16。','',
    '滚转正常控制收益 0.023205 rad/s，负对照收益 0.015418，正常控制仍明显更强；偏航正常收益 0.007343，负对照收益 0.011322，后者反而更强。终点之后的控制不能导致已经发生的运动，但可经反馈、控制自相关和共同指令携带其信息。这是明显反馈关联证据，既不能因负对照显著就否定所有观察性预测价值，也不能因正常控制更强就宣称因果识别。','',
    '100 ms 正常/负对照同通道相关（电机/差动/对称/垂尾）：'+str(first['negative_normal_100ms_same_channel_correlation'])+'。完整 16×16 相关矩阵见 normal_negative_correlation.csv。','',
    '## 3. 可评价通道与有效覆盖','',
    '第一阶段结论：差动滚转具备独立日期条件预测与局部校正支持，可进入受限主评价；差动偏航为有支持的交叉项；垂尾偏航虽然方向稳定，但终点后控制的预测力更强，只作为辅助；垂尾滚转方向不确定。电机和对称尾舵保留调整项，不假定零作用。','',
    f"train 局部拟合记录 {len(lc)}，可辨识记录 {int(lc.identified.fillna(False).sum())}。控制残差矩阵秩不足、数值非有限或拟合对数不超过设计维数时不输出 G。状态支持使用训练跨航次状态距离的最大观测半径诊断，不调节百分比通过率；此边界较宽松，有限样本近邻并不证明高维共同支持。重叠比例中位数 {lc.overlap_fraction.median():.4f}、最小 {lc.overlap_fraction.min():.4f}。条件数中位数 {lc.condition.median():.3f}、最大 {lc.condition.max():.3f}。有效对数和不可评价原因见 local_coverage.csv。",'',
    '主宽度下不受支持的 train 留出查询主要集中在 2026-09-29（mode 3 为 60 个、mode 15 为 41 个），其余日期多数充分覆盖。support_by_date_mode.csv 按日期/模式给出支持数，state_coverage.csv 给出速度、角速度、频率范围。残差激励按日期及 nav_state 编号 3/15 输出 excitation.csv；不推断这些编号代表的完整任务语义。各日期不同模式的覆盖不均匀；本研究不外推到未覆盖模式、未观测扰动或实际执行器饱和后的真实气动增益。','',
    f"validation 已实际预测 3,657 个决策；其中 15 个 nav_state=4 窗口在 train 无对应模式，按支持合同标为不可评价（validation_unevaluable_modes.csv）。其余可评价 {second['samples']} 决策、{second['flights']} 航次、{second['dates']} 日期，局部可评价决策 {second['valid_local_samples']}。采用预先固定的 v1 Hall/模式/1 秒间隔及 0.5 秒完整支持要求，因此是 0.2 秒可用窗口的保守共同支持子集，并非穷尽所有仅有 0.2 秒支持的窗口。没有按 PACLD 误差筛选样本。",'',
    '## 4. 冻结 PACLD 方向、幅值、解释能力及耦合','',
    'checkpoint：`/home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1/runs/pacld_frozen_eval_seed17/model.pt`；SHA256 `9d82640692c0baec476e276696be4f89fd89a58854361bbcfdf3dbb84cb466c9` 已核实。模型配置与基线一致，train/validation 全 Hall 索引及 manifest 身份与 checkpoint 一致。新入口复用冻结 mixed precision 的只读 GPU inference，无优化器、无权重修改。序列化窗口 B1 与原 B32 预测差最大 1.32e-8 rad/s；执行过程中仅调整 CPU 预取与批大小，不改模型或样本合同。原生读取器提供相同窗口因果 Belief 与原始 ZOH；在真实目标测量时刻查询，不注入未来观测。Y_PACLD 为预测角速度减该 Belief 角速度，Y 为真实目标减最近因果测量角速度，两者各自建立状态条件基准。','',
    '根据用户明确补充授权，validation 残差回归允许作为 A/B/D 最终统计。所有条件基准、归一化及超参数仍只在 train 拟合：e(X)、m_obs(X)、m_PACLD(X) 分别拟合，不能用实测 m_obs 替换模型基准。训练残差使用留日期 cross-fit；最终 validation 残差使用全 train 拟合冻结的模型。第二阶段 U 固定为前 100 ms 四通道；M1 同口径使用这四维输入，第一阶段 50/100 ms 联合结果不直接替代。','',
    '**矩阵评价口径：**A/B/D 从 validation 实测及 PACLD 残差回归估计，使用同一四通道控制残差、train 残差缩放、固定 alpha=10、以及 train 局部邻域确定的带宽。同模式 validation 样本根据与查询状态的距离赋连续权重，排除 train 支持范围外的样本。该回归是用户允许的最终评价统计，不是新的条件基准；它的系数不进入指标 C。4 个 validation 日期 bootstrap 4,000 次，以固定 nuisance 模型计算矩阵区间；这些局部估计重复使用 validation 数据，不得把局部 G 数量当作独立样本。另保留 train-only 局部 G 参考 CSV，名称 validation_gains.csv 与本节实际评价统计 validation_evaluation_gains.csv 不同。','',
    'A：只把第一阶段独立日期方向稳定、且 validation 局部观察性 G 的日期 bootstrap 区间排除零的主通道差动→p 计入方向统计；差动→r 在 validation 多数局部区间跨零，仅少数局部、单个查询日期有方向依据，因此不据此报告总体同向率或相对误差。垂尾两项均不进入主同向率（偏航仅辅助、滚转不确定）。逐日期同向率如下；同日航次不能算独立日期重复试验。','',table(direction),'',
    'B/D：矩阵行 p/r、列差动/垂尾，单位 rad/s 每归一化控制。下表按日期局部中位数再汇总；absolute_error 为绝对差，relative_bias 仅对通过方向证据条件的差动项计算；区间包含零时不报告相对误差。表中区间列是局部区间端点的中位数，不是全局中位增益的置信区间。垂尾不输出相对百分比。三带宽全部保留，不能挑最好结果。','',table(gains),'',
    'C：真实残差 R=Y−m_obs(X)，PACLD 残差 R_P=Y_PACLD−m_PACLD(X)，轻量模型响应 R_1=M1(X,U)−m_obs(X)。分数 S=1−E[(R−R_model)²]/E[R²]；零响应基线严格为 0，负分表示比不解释任何条件残差更差。先航次内 MSE、再日期内航次等权、最后日期等权。该 residual score 仍包含未被 X 解释的状态/扰动误差，不能称纯因果控制贡献解释率。','',table(score),'','局部状态支持内的独立 validation 分数（排除外推查询；仅训练状态决定）：','',table(pd.read_csv(OUT/'validation_supported_summary.csv')),'',
    'validation 只有 4 日期，bootstrap 区间是有限日期描述性敏感性范围；不能把数千窗口视为数千独立泛化证据。原始响应 RMSE 与条件解释分数不同，不能互相替代。','',
    '## 5. 当前数据条件下结论','',
    'A2 的局部状态校正在差动滚转上通过了无 anchor 未来响应参与拟合的日期外检验；但未来控制负对照揭示明显反馈关联。因此成立的是受限的闭环条件响应评价框架，不是因果控制有效性评分。PACLD 的差动滚转方向虽一致，幅值仍显著偏弱；其约 1% 的条件解释能力不足以弥补增益差异。原始 Δp RMSE 约 0.7695，差于 M0 的 0.5324 rad/s；原始 Δr 约 0.1786，亦差于 M0 的 0.1547。条件基准移除系统偏差后得到的小正分不能替代原始预测准确性。最终模型数值表现以本报告指标 C 和各日期结果为独立 validation 证据；A/B/D 是授权后的 validation 条件关联统计，受 4 日期、正则化和闭环反馈限制；不能视为真实气动控制增益。没有强制构造完整二维总分。','',
    '## 6. 关键阻碍与复现','',
    '主要阻碍：偏航反馈混杂较强；实际执行器偏角/延迟与风扰未独立测量；17 个 train 日期和 4 个 validation 日期限制不确定性估计；高维状态共同支持证据有限；validation 残差回归只能描述观察性关联，无法消除未观测反馈混杂。当前任务未建立同合同的冻结 GRU 比较入口，因此不报告 GRU 对照，也未训练 GRU。sealed test 未访问，未执行 MPC 或动力学训练。','',
    '已通过定向检查：ZOH 时间积分、原始控制复算、跨航次配对、留出标签毒化不变性、合成 G 恢复、零响应/完美响应分数；GPU 逐窗口检查因果历史、原生目标值与目标时刻，输入数据身份与 checkpoint 核对。','',
    '![日期预测与负对照](control_effectiveness_a2_v2/date_prediction.png)\n\n![独立局部G](control_effectiveness_a2_v2/independent_local_g.png)\n\n![validation解释能力](control_effectiveness_a2_v2/validation_explanation.png)\n\n![局部矩阵比较](control_effectiveness_a2_v2/validation_local_g.png)','',
    '```bash','cd /home/zn/flap-system-identification/.worktrees/pacld-control-effectiveness-v2','export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/diagnose_pacld_a2_v2.py',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/report_pacld_a2_v2.py',
    '# validation 特征提取命令见 reproduce_pacld_a2_v2.sh',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_pacld_a2_v2_gpu.py --split train',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_pacld_a2_v2_gpu.py --split validation',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/score_pacld_a2_v2.py',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/validation_gain_statistics_a2_v2.py',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/report_pacld_a2_v2_final.py',
    '/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/verify_pacld_a2_v2.py','```','',
    '所有 CSV/JSON 与关键图随提交发布；features/context/residual NPZ/PKL 为本地可重建中间产物。默认复用 v1 本地特征，可通过 --source 指定复现的同合同 v1 目录。GPU 已存在结果需显式 --resume，按 flight+decision 恢复，禁止静默覆盖。']
    (OUT.parent/'control_effectiveness_a2_v2_zh.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
