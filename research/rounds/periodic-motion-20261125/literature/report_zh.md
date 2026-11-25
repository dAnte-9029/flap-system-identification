# 显式建模周期运动与慢变运动的检索与复现

上一轮在相同的 49,152 次更新下发现，半秒运动历史能改善速度预测，低阶命令特征有利于角速度预测。新的问题是，能否让网络显式给出按拍翼相位重复的分量与慢变分量，并用匹配消融和独立参考区分预测收益及分解准确性。[上一轮交接](../../../memory/cycles/cycle-e6bd782009166d68.json)

首轮 OpenAlex 真实返回 50 篇，补检加入可加网络和周期残差检索词，并对两篇已知候选定向回查。最终快照有 49 篇，逐篇冻结初筛后入围 4 篇，均取得全文。四篇均有题名定向检索路径，只有 CycleNet 还由主题词路径命中。覆盖为 PARTIAL，季节趋势关键词仅取回 9/795 篇，语义检索取回 9/50，仓库查询有 403。此处终排是当前候选集中的判断。入围四篇均手工核对作者仓库、固定提交和实际许可。[检索快照](../../../literature/20261125T062201Z-1a04428be1f9/result.json) · [逐篇初筛](../../../literature/20261125T062201Z-1a04428be1f9/screening.json) · [冻结终排](../../../literature/20261125T062201Z-1a04428be1f9/agent-ranking.screened.json)

| 暂排 | 原文与作者程序 | 全文及实现核查后的选择 |
|---|---|---|
| 1 | [N-BEATS，ICLR 2020](https://arxiv.org/abs/1905.10437)，[作者许可](https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/LICENSE) | 趋势多项式和季节傅里叶堆栈符合结构目标。但作者公开版含 CC BY-NC 4.0 非商业限制，未满足本轮开源条件。 |
| 2 | [NeuralProphet](https://arxiv.org/abs/2111.15397)，[作者代码，MIT](https://github.com/ourownstory/neural_prophet/tree/596849107945533c7b94d53ad881d88a7a924875) | 入选。式 3–4 的连续分段线性趋势与傅里叶季节项可以直接读取；图 5、7 在已知真分量上分别评价趋势及各季节的去均值 RMSE。这同时给出可实施结构和识别分量的独立评价方法。 |
| 3 | [CycleNet，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bfe7998398779dde03cad7a73b1f81b6-Abstract-Conference.html)，[作者代码，Apache-2.0](https://github.com/ACAT-SCUT/CycleNet/blob/d807e51fc2dcd143885ee639d97965a7ab0926f4/models/CycleNet.py) | 可作为冻结周期模板控制。实现按样本序号模固定周期查表，从输入减去重复波形，由残差预测器外推，再加回未来周期表。其余项可以包含快运动，需另测。 |
| 4 | [CoST，ICLR 2022](https://arxiv.org/abs/2202.01575)，[作者代码，BSD-3-Clause](https://github.com/salesforce/CoST/tree/7bb5271c2b66cd5a1d634c27f5df308f190b50ec) | 趋势与季节用时域、频域对比目标学表示，表示拼接后作下游预测。若要获得可直接比较的速度、角速度分量，需另设计有物理量纲的解码器。 |

入围 PDF 已放入 `/home/zn/papers`，四个作者仓库在 `/home/zn/reproduction`。N-BEATS 与 CoST 的首页文字提取会拆开大写题名中的单词；批量本地 PDF 绑定曾被 CLI 拒绝。两篇改以原文 URL 和章节登记，人工核对原文标题、作者和页码；入选的 NeuralProphet 已通过本地 PDF 绑定。[核查证据](../../../literature/20261125T062201Z-1a04428be1f9/fulltext-evidence.json)

N-BEATS 的限制性许可还暴露出状态字段缺项。作者提供了明确的 CC BY-NC 4.0 文本，现有 CLI 状态包含开放许可、公开未授权和待核实等类别，却没有对应已核实的限制性许可。故其条目使用 `unverified` 状态，另在具体许可、源码位置与排除原因中登记查到的原文。初始访问两篇 OpenReview PDF 返回 403，最终使用作者 arXiv 版本。

## NeuralProphet 的核心结果

CPU 上运行了固定 Git 提交中的趋势、傅里叶预测类和训练用的时间特征函数。代码通过语法树提取原始类与函数，使本次可单独执行它们，绕开训练环境未安装的 Lightning、绘图和前端依赖。模型中组合趋势、月周期、年周期。训练特征按全部正弦接全部余弦排列，绘图辅助函数按谐波交错排列。数值对照使用训练函数的排列，各谐波用独立的复指数计算。[训练特征](https://github.com/ourownstory/neural_prophet/blob/596849107945533c7b94d53ad881d88a7a924875/neuralprophet/time_dataset.py#L138-L172) · [绘图函数](https://github.com/ourownstory/neural_prophet/blob/596849107945533c7b94d53ad881d88a7a924875/neuralprophet/plot_utils.py#L39-L45)

- 趋势与独立分段线性函数最大差 **2.78×10⁻¹⁶**，季节项与独立傅里叶公式最大差 **7.15×10⁻¹³**。将查询时间平移整整一个周期，各季节头输出的最大差为 **5.01×10⁻¹³**。
- 按论文 S-TS 类型生成五条各 6,000 天的带噪序列，真值含随机转折趋势和两种独立季节分量。每条仅用前 4,800 天的总值拟合，真分量和真实转折日不进入训练。模型在未来 1,200 天上逐分量对比真值，按论文消去常数偏移后，五组最差趋势 RMSE 为 **0.000734**，十个季节分量最差 RMSE 为 **0.000777**，单位均为缩放后的总序列。
- 五次 LBFGS 优化有四次到达 450 次迭代上限，一次在 394 次结束。使用同一批训练数据和独立最小二乘基底估计全局最优预测，对未来总预测的最大点差 **0.000242**，低于事前设定的 0.001 容差。
- 验收为 **VERIFIED_CORE**。作者原图、传统 Prophet 对照、默认训练器与现实业务基准均未在此范围内。本地预先指定切分、噪声、优化器、未知转折点的候选结点和阈值。合成实验中的日历周期固定、采样充分、信号族已知；归一化数值及分解可辨识性均不能直接移用于实飞。飞行实验的归因仍需同容量重训消融，分离需额外对照相位、跨分支泄漏及独立波形。 [冻结规格与容差](../../../reproductions/rep-bc7ae8dfcdef4cb6/paper_spec.md) · [逐序列结果](../../../reproductions/rep-bc7ae8dfcdef4cb6/work/synthetic_runs.json) · [复现报告](../../../reproductions/rep-bc7ae8dfcdef4cb6/reproduction_report.md)
