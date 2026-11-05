"""Render only computed tables; no data loading."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/analysis/results/slow_fast_separation_ab_v1'

def block(df):return '\n```text\n'+df.to_string(index=False,float_format=lambda x:f'{x:.6g}')+'\n```\n'

def main(out=OUT):
    global OUT
    OUT=Path(out)
    a=pd.read_csv(OUT/'A_summary.csv');af=pd.read_csv(OUT/'A_per_flight.csv');b=pd.read_csv(OUT/'B_summary.csv');config=json.loads((OUT/'configuration.json').read_text());cov=pd.read_csv(OUT/'coverage.csv');sel=pd.read_csv(OUT/'B_selection.csv')
    primary=a[(a.split=='validation')&(a['null']==-1)&(a.harmonics==3)&(~a.orthogonal_time)]
    null=a[(a.split=='validation')&(a['null']>=0)&(a.harmonics==3)&(~a.orthogonal_time)].groupby(['channel','group']).agg(null_gain_min=('gain','min'),null_gain_max=('gain','max'),null_gain_mean=('gain','mean')).reset_index()
    p=primary.merge(null,on=['channel','group']);p.to_csv(OUT/'A_primary_vs_null.csv',index=False)
    paired=[]
    f=af[(af.split=='validation')&(af['null']==-1)&(af.harmonics==3)&(~af.orthogonal_time)]
    for ch in ('vD','q'):
        x=f[(f.channel==ch)&(f.group=='stable')].merge(f[(f.channel==ch)&(f.group=='changing')],on='flight_id',suffixes=('_stable','_changing'))
        for r in x.itertuples():paired.append(dict(channel=ch,flight_id=r.flight_id,stable_cycles=r.n_cycles_stable,changing_cycles=r.n_cycles_changing,stable_gain=r.gain_stable,changing_gain=r.gain_changing,stable_periodic_rms=r.periodic_rms_stable,changing_periodic_rms=r.periodic_rms_changing,rms_difference=r.periodic_rms_changing-r.periodic_rms_stable))
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'A_paired_conditions.csv',index=False)
    orth=a[(a.split=='validation')&(a['null']==-1)&(a.harmonics==3)&a.orthogonal_time]
    cols=['channel','group','n_flights','n_cycles','gain','periodic_rms','relative_fast_rms','positive_flights']
    text='''# 当前固定 F 的快慢分离机制 A/B 验证

只使用 velocity_rate_multistep_v1 的 train/validation，CPU轻量数值方法；没有访问 sealed test 数据、标签或统计表，没有训练动力学网络。vD为NED向下速度(m/s)，q为FRD俯仰角速度(rad/s)。沿用Hall工作零点、main_50hz质量门槛和原生双时间轴；没有将 sparse_10hz 插值使用。共同provenance注册表仅用于筛选train/validation航次元数据，不读取test分区。

## 方法和可复现边界

原有 `methods.fit_common` 未修改。A使用51拍窗口、H1–H3 F、run内索引mod5=2整拍留出；所有这些留出拍的运动观测均不参与F。r=raw−trend−Fast_F，Slow=raw−Fast_F。趋势来自训练支持拍的时间加权均值分段线性插值。B明确使用holdout=False，完整重新分离原始与注入数据；不能把B视为自主预测。

A的剩余趋势基线为拍内时间一次/二次项，以及由不含目标运动观测的F预测得到的共同趋势水平/斜率、Fast RMS、扑频；谐波模型增加H1至H3和上述四项工况与谐波的线性交互。参数只在train拟合，ridge=0.001预先固定；H1/H2/H3全部报告，不按validation择优。train列是拟合描述，只有validation列是未见航次预测证据。训练先每拍原生时间加权、再航次等权；评价先拍内、再航次、最后航次等权。

16个固定种子的对照分别对每个翼拍加独立[0,2π)相位偏移，train与validation独立，保留样本时刻、值、顺序、质量和分组；每次重新训练残留模型。对照仅扰乱残留回归的机械相位，不改原F输出。它是有限随机化参考范围，不是正式显著性p值。附加诊断将谐波基对每拍真实权重下的[1,t,t²]正交化，使用同一train→validation流程，检查趋势/采样投影；该操作也会移除部分真实H1，故不能当严格周期真值滤波器。

事件复用前次冻结train非零控制差分P90及原有2秒间隔事件身份。目标拍只要与事件[-0.5,+1]秒窗口相交就标为changing；stable要求整拍距任何所选事件超过2秒；中间buffer仅进入all。stable表示未邻近这些已筛选事件，不保证所有控制输入恒定。事件仅是观察性工况标签。周期残留强度是模型预测成分RMS，只有伴随未见数据正增量收益且超过随机对照时，才有可预测结构证据；不能把拟合输出幅值本身叫作已证实残留。

## A：真实数据结果
'''
    text+=block(p[cols+['null_gain_min','null_gain_max']])
    text+='\n增量收益定义1−模型MSE/非相位基线MSE。relative_fast_rms为残留预测成分RMS/原F Fast RMS。所有分航次结果见A_per_flight.csv，不依靠全局相位均值判断。\n\n去除拍内二次时间投影后的H3诊断：\n'+block(orth[cols])
    text+='\n同一航次内stable/changing覆盖及差值：\n'+block(paired.groupby('channel').agg(n_paired_flights=('flight_id','size'),stable_cycles=('stable_cycles','sum'),changing_cycles=('changing_cycles','sum'),mean_rms_difference=('rms_difference','mean'),changing_larger_flights=('rms_difference',lambda s:int((s>0).sum()))).reset_index())
    text+='''
每航次稳定/变化的收益和强度见A_paired_conditions.csv；稀少分组翼拍不被当成大量独立证据。全局模型无法保证捕捉每个航次独有、方向相反的残留系数，因此负收益不能证明该航次没有相位残留。本次没有用validation目标翼拍重拟合系数来宣称可预测性，也没有把同拍投影下降作为证据。

A代表图取字典序首个合格航次/run的中位合格目标拍及附近±2秒，完全不按残留大小选例。图中只连接单个目标拍内部，拍间空白是留出协议，不是伪造连续高频观测。红线始终标作train拟合的预测，负收益组不能解释为真实已验证周期成分。图及对应时序CSV包含raw、Fast、Slow、trend和r。

## B：已知扰动分配

每航次选vD/q共同有效内区间最长的一对run，固定共同物理中心；不按误差筛位置，每航次最多一次。评价区间及每一目标拍前后25拍均在同一连续有效run内。表内每拍沿用真实观测权重；事件阶段按原生观测时间选点并归一化原权重，不对截断阶段另定义零均值。

Slow为紧支撑sin²脉冲，单独持续0.15/0.30秒，耦合时0.20秒；0秒开始、持续时间后严格恢复零。Fast真值为1–3阶确定性波形，每拍按原生离散权重去均值。恒幅包络为1；变幅包络从0.5通过smoothstep变到1.5，变化时间0.5或2秒，之后保持。变化拍内去均值可能加入小的逐拍常量校正，这是当前分解约定的组成部分，不是绝对物理Fast。

尺度和时长在打开validation表之前冻结：
'''+block(pd.DataFrame(config['scales']).T.reset_index(names='channel'))
    text+=f"\nSlow峰值取train事件绝对净变化航次中位数与Fast RMS航次中位数的较大者，故是有意可辨识的信号机制探针，不代表每次真实控制事件幅值。Fast基础RMS由train估计。按train翼拍周期给定完整评价窗[-{config['half_window_s']:.6f},+{2+config['half_window_s']:.6f}]秒。\n"
    text+='\n覆盖：\n'+block(cov.groupby(['split','channel']).agg(n_flights=('flight_id',lambda s:s.nunique()),valid_cycles=('valid_cycles','sum'),heldout_cycles=('heldout_cycles','sum')).reset_index())
    text+='\n符合长窗口且双通道共同支持的B航次：\n'+block(sel.groupby('split').size().rename('n_flights').reset_index())
    text+='\n完整事件窗结果（先航次内RMSE平方，再航次等权开方）：\n'+block(b[b.stage=='full'][['split','channel','case','n_flights','error_rmse','normalized_error_mean','leak_max']])
    text+='\nvalidation扰动开始阶段[0,0.3]秒：\n'+block(b[(b.split=='validation')&(b.stage=='onset')][['channel','case','error_rmse','normalized_error_mean','leak_max']])
    text+='\nvalidation最后一秒恢复段：\n'+block(b[(b.split=='validation')&(b.stage=='late_recovery')][['channel','case','error_rmse','leak_max']])
    text+='''
归一化比例为误差RMS / sqrt(Slow注入RMS²+Fast注入RMS²)，先每航次计算再等权平均；无注入能量的阶段为NaN，绝不除零或宣称零误差。单独Slow时，误差就是非周期量进入Fast；单独Fast时，误差就是周期量留在Slow。耦合场景报告总错误分配，不能仅凭总误差唯一分解两条泄漏路径，二者可能相加或抵消。Slow与Fast误差在每个样本上互为相反数，所以两个RMSE相同，不是两份独立证据。Slow+Fast精确重构没有被用作成功判据。

逐航次、before/onset/changing/recovery/late_recovery/full全部指标见B_per_flight.csv、B_summary.csv。图选首个合格validation航次，展示已知Slow/Fast和两条分离增量，train/validation均保存该规则下的时序CSV。单次固定注入不是随机注入总体试验，且长连续段筛选限制了推广范围。

## 证据判断与方法决策

**事实结果：q仍有跨航次可预测的机械相位残留，vD证据较弱。** q全体validation的H3相位增量收益6.34%，22/24航次为正；预测成分RMS 0.1028 rad/s，为原Fast RMS的22.44%，超过16次随机化对照的最高收益0.21%。拍内二次时间投影去除后收益7.60%，预测成分0.05635 rad/s（原Fast的12.30%），23/24航次为正。两种成分RMS的差异说明未正交化幅值含有明显时间趋势/采样投影，不能把0.1028全部解释为真实物理振荡。vD全体收益−0.44%；去时间投影后仅+0.27%、RMS 0.00678 m/s（原Fast的1.64%）。这支持很小的剩余信息，但不足以宣布大幅周期结构，也不能排除跨航次系数抵消。

**工况关联不是统一增强。** q变化组/稳定组收益11.41%/5.37%，但预测成分RMS为0.1066/0.1118 rad/s；22个成对航次中13个变化组更大，成对平均差−0.00306 rad/s。去时间投影后两组收益7.92%/7.76%，差异明显缩小。vD变化组RMS较大（0.01809/0.01273 m/s），成对22航次中19个较大，但变化组收益仅0.13%，低于稳定组0.60%。因而不能把“预测输出更大”或“一个组收益更高”直接称为机动增强了真实周期残留。分组的23航次及639/511目标拍覆盖足以报告描述性结果，但不是独立事件干预。

**B明确暴露了时变幅值错误分配。** validation恒幅Fast的全窗归一化错误仅vD 0.083%、q 0.142%；0.5秒变幅升为21.66%、21.68%，2秒变幅仍为20.62%、20.64%。相应全窗RMSE为vD 0.09870/0.09040 m/s、q 0.10570/0.09681 rad/s，最大泄漏分别达到0.3660/0.3288 m/s和0.3982/0.3580 rad/s。变化后的恢复段误差逐渐下降；双侧离线模板在变化前也有泄漏，不能解读为真实系统提前响应。

**快速Slow不是完全不受影响。** 0.15/0.30秒脉冲在validation全窗归一化RMS比例约11.3%–11.4%/7.1%，RMSE vD约0.00262/0.00232 m/s、q约0.00282/0.00250 rad/s。开始[0,0.3]秒归一化错误仅约1.6%/1.1%，最大泄漏vD不超过0.00741 m/s、q不超过0.00892 rad/s，最后一秒恢复为零。这说明主体脉冲大体保留，但小幅污染分散在长模板支持范围；不能只凭局部峰值小或中心拍效果好判定完全保留。

**耦合变化没有解决上述问题。** 0.5秒/2秒耦合场景全窗错误比例约21.6%/20.6%，RMSE vD 0.09873/0.09044 m/s、q 0.10574/0.09686 rad/s，与对应仅Fast变幅接近。结果支持变幅不适应是这些固定探针中的主要风险；它不是未经观测验证的真实飞行动力学归因，也不是两条泄漏路径的独立估计。

**决策：保留F作为离线对照和研究候选，不宜直接作为已验证的正式Slow/Fast监督标签。** A/B形成部分一致证据：q确有可预测残留，B证明当前51拍固定系数对已知变幅存在明显错误分配；但A没有得到稳定的“机动时残留幅值普遍增加”结论，故尚不能将真实残留唯一归因于时变幅值。优先改进周期系数的时间/工况适应能力，同时保留原生时间加权零均值和对非周期快速运动的约束；不依据本轮结果选择具体新窗口、网络或训练设计。趋势误差、观测噪声、相位误差、航次特异性及不均匀采样仍是未排除解释。本轮未建立物理慢态真值，不把控制事件作为因果干预，不启动训练。

## 验证与复现

4项定向测试通过：留出运动观测改变后F/L预测不变、原生Fast零均值、逐拍相位循环偏移及时间基正交性、完整双通道窗口支持。测试命令：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/slow_fast_separation_ab_v1/test_ab.py
```

完整数值复现命令（目录必须不存在）：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/slow_fast_separation_ab_v1/run_ab.py --output /tmp/slow_fast_separation_ab_v1_reproduction
```

代码基于5d5066cb1fb4e66df795d341396849cd60e95597及本分支新增实现。configuration.json记录数据合同、train冻结参数、事件与采样规则；runtime.json记录实际耗时。没有修改F/LSF、原始日志、冻结划分、前次报告或候选标签。
'''
    (OUT/'report_zh.md').write_text(text)

if __name__=='__main__':main()
