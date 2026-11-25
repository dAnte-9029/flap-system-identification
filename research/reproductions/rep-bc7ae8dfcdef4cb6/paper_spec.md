# 论文复现规格

PDF SHA-256 3ccb2074989b449e3b0356202fb76fa076a9bc34301b7121686c136ed955353f

## 研究问题、方法与贡献

- [decomp] 从混合序列预测总运动并分别估计有意义的趋势、季节变化。（paper · PDF p.19 §3.1, components of known synthetic series）

## 模块与输入输出

- [additive] 可加配置下，总预测由趋势、季节及其他使能的分量直接求和。（paper · PDF p.5 §2.1 Eq.1）

## 公式、变量与计算流程

- [trend_model] 连续分段线性趋势；斜率每次改变δ，常数补偿−cδ保持连续。（paper · PDF p.6 §2.1.1 Eq.3）
- [fourier_model] 每个周期的五个正余弦谐波以独立权重线性组合。（paper · PDF p.7 §2.1.2 Eq.4 and §3.1.2 S）

## 损失与训练目标

- [loss] 原框架用可配置的总预测损失；核心模型仅凭加总后的带噪训练目标学习参数。（paper · PDF p.17 §2.3 estimation and §3.1）

## 数据、预处理与划分

- [synthetic] S-TS 包含一段带随机变点的趋势、30和365天季节量；五个独立6000天序列、每个周期5阶，系数均匀抽样。（paper · PDF p.20 §3.1.2 Generated Component Time Series）

## 训练配置、随机种子与预算

- [fit] 固定本地合成种子700到704，前4800天拟合、后1200天评价；按训练部分分别缩放。均匀随机转折日1450–3200，0.02高斯噪声；八个固定候选趋势结点不接收实际转折日。训练凸子模型用全批双精度LBFGS最多450步并对照独立最小二乘最优解。（inference · CPU operator and component validation setup; paper did not disclose numerical realizations for Fig.7）

## 评价口径与论文结果

- [joint] 在未参与拟合时间段逐条分别去除分量均值比较真实趋势、月周期和年周期RMSE。原论文分量比较也去均值，本文测试沿未来段延拓变化，与作者原图整体拟合范围不同。（paper · PDF p.20 §3.1.1 zero centering, Fig.5 and Fig.7）

## 中间行为与消融

- [core] 固定仓库实现单独运行分段线性和傅里叶两个类及来源于数据加载器的特征函数，与独立的复数单位圆和铰链基底核对。（official_code · neuralprophet/components/trend/piecewise_linear.py:15–97,195–267; components/seasonality/fourier.py:114–137; time_dataset.py:144–168）

## 实现来源

official · 作者仓库README引用同文同作者，LICENSE为Oskar Triebe署名MIT；仅以语法树抽取未经改写的两个模块、基类和数据特征函数并执行，避开未安装的前端依赖。
https://github.com/ourownstory/neural_prophet.git 596849107945533c7b94d53ad881d88a7a924875

## 原文与实现差异

- §3.1.2、附录B图7给出构成和曲线，未提供可精确复算的S-TS浮点结果或种子。 / 固定提交为当前MIT版本。time_dataset.py先排全部sin再cos；plot_utils.py交错排列并使用不同日期原点。 · resolved · 使用训练路径的特征顺序，与独立公式按实际列对应。采用五条同长度新数据、标准化的未来段分量与独立解核验核心机制。只登记核心协议。

## 预注册验收

core · 核对Eq.3–4数值、周期不变性及Fig.5/7来源可查的分量评价设计。论文未给逐项数字；训练切分、噪声、种子和容差为本地预注册。未复现现实数据基准、传统Prophet对照或原优化器。
每阶段最多 3 次。

- preflight/rows · 单条序列天数 · 6000 ± 0 days · synthetic
- preflight/rank · 两种5阶季节基、趋势截距与斜率加8变点的训练设计满秩 · 30 ± 0 columns · fit
- modules/trend_formula · 作者趋势相对独立铰链展开最大误差 · 0 ± 5e-08 normalized_y · trend_model
- modules/season_formula · 作者季节相对独立复指数最大误差 · 0 ± 2e-09 normalized_y · fourier_model
- modules/period_invariance · 单独把查询时间平移该季节一整周期的最大输出差 · 0 ± 2e-09 normalized_y · fourier_model
- small/slow · 五条序列中最差的未来慢分量去均值RMSE；容差设为归一化幅域的5%，容许未知变点近似误差 · 0 ± 0.05 normalized_y · joint
- small/season · 五条序列十个季节分量中最差的未来去均值RMSE；容差设为归一化幅域的1% · 0 ± 0.01 normalized_y · joint
- small/solved · 官方模块训练所得总预测与独立最小二乘解在五条未来段上的最差点误差；核对优化充分性 · 0 ± 0.001 normalized_y · core
