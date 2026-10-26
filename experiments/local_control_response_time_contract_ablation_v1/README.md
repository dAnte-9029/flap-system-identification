# Frozen time-contract input ablation

报告：[report_zh.md](report_zh.md)。源码与准入：[time_contract_zh.md](time_contract_zh.md)。

基准 `ae8598a94555c908500856cc415cf2119c396078`，分支 `exp/local-control-response-time-contract-ablation-v1`。只复现指定历史 `velocity_rate_multistep_v1/main_50hz`。实际版本是 V0 与 **V1_source_assumption**；后者不是已验证物理对齐。V1 物理对齐、V2 状态重建均未准入；无训练、选对、改权重、test 或 MPC。

## 环境与复现

使用现有 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，不安装依赖。源数据在 `/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1`；所有被读 Parquet 均按其 manifest 核验。仅 train/validation。checkpoint、前轮四个实验及其缓存必须保留，缺失不替换。

从新 worktree 根目录运行。已提交的预测受保护，`run.py` 拒绝覆盖已有 V1_predictions；完整重跑必须给新输出目录：

```bash
export LOCAL_ABLATION_OUTPUT=/tmp/local-time-contract-repro/results
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/audit.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/prepare.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/training_distribution.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/run.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/analyze.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/verify.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/plot.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_time_contract_ablation_v1/describe.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/local_control_response_time_contract_ablation_v1/test_contract.py
```

`audit.py` 只读 PX4 Git 对象与既有源码缓存；需要时读精确 commit HTTPS。提交包的 sources 已保存全部已用内容，不需要运行中切换 PX4 分支。Stage4 本地来源路径在 source_receipt 中。复现到外部结果目录时，新源码快照同样写到外部目录的 sources。

`feature_distribution_reference.csv` 是开发时独立“每个 receiver 航次分别建树排除”算法的 225 窗口参考。`describe.py` 在默认结果目录验证它；在新输出目录时读取本实验提交的该参考用于数值核对。它不参与修正规则、模型推理或统计选择。

只验证提交产物可不运行 GPU：在未设置 `LOCAL_ABLATION_OUTPUT` 的 shell 运行 `verify.py`、pytest；如果变量已设，先 `unset LOCAL_ABLATION_OUTPUT`。数值复核比重跑 gzip/NPZ 的字节相同比较更适合跨环境复现；提交版本的字节另由 archive_manifest 保护。

## 代码及结果

| 文件 | 含义 |
|---|---|
| `config.json` | 单一条件性相位传播公式，固定门限、时间、滤波及未准入版本说明 |
| `audit.py`, `sources/` | 精确 PX4 源码、Stage4 文本快照，保留许可证头；不修改 PX4 |
| `results/firmware_sources.csv`, `source_evidence.csv` | 固件标识、路径、来源、blob/SHA256、行号与证据文本 |
| `prepare.py` | 逐历史行可用性核对、相位变换、V2 括号准入、宽训练侧输入参考 |
| `training_distribution.py` | 107,920 个 H500/H1000 登记训练窗口历史并集的输入分布；只训练侧定尺度 |
| `run.py` | 复用原八张量接口、六个 V0 回归、唯一 V1 冻结推理；NoControl 与事件不变性检查 |
| `analyze.py` | 原 A/B/D 评价，原内部 C/H 分解，固定子集／权重、逐对版本差 |
| `verify.py` | 不调用主指标函数的插值、积分、单位圆旋转、图权重与核心汇总复核 |
| `test_contract.py` | 10 项因果性、边界、滤波、输入隔离、回归及划分测试 |
| `plot.py`, `describe.py` | PNG、逐对变化汇总、内部方向敏感性、种子一致性；无择优样本 |
| `results/admission.json` | 实际准入、因果括号数量、训练侧相位改变量；V2 无预测文件 |
| `results/input_changes.csv.gz` | 全部 6,011 条历史输入的窗口、航次、源时间／可用上界／决策时间、相位变化及角速度括号 |
| `results/V1_inputs.npz` | 与原 inputs 相同八张量与元数据；仅 history[...,17:19] 的有效历史行变化 |
| `results/V1_predictions.npz` | 六个模型键，452×K×13；V0 直接引用原 swap/results/predictions.npz |
| `results/window_points.csv.gz`, `paired_window_changes.csv.gz` | 每四组合／模型／种子／输出／时刻的增量及 V1−V0 |
| `results/factual_{curves,points}.csv.gz` | variant/pair_uid/split/model/seed/variable/method/time_label/supported/real/pred/error |
| `results/internal_{curves,points}.csv.gz` | 四个 Y、CA/CB/HA/HB/C/H/F、interaction、实际差动均值 gA/gB/g；无 swapped 真值 |
| `results/factual_summary.csv`, `factual_300_summary.json` | 全子集、模型、种子／pooled、时间和权重的方向／有效数／幅度／增益／正负偏差／辅助误差 |
| `results/internal_summary.csv` | 内部幅度、双有效方向、近零、比值与抵消；不是物理效能 |
| `results/paired_{factual,internal}_changes.csv.gz`, `paired_change_summary.csv` | 版本之间逐对改变量、方向得失、幅度误差变化，保留正反例 |
| `results/{seed_consistency,internal_seed_consistency,internal_direction_sensitivity}.csv` | 种子重复及近零／方向门限敏感性 |
| `results/{per_group,leave_one_flight_out}.csv` | 300 ms 逐航次／日期／配置／共享航次分量，以及移除航次的描述性复算 |
| `results/{feature_distribution,training_history_distribution}.csv` | 宽参考与真实登记训练历史并集两种输入距离；后者为报告主参考 |
| `results/*receipt.json`, `inference_checks.json`, `baseline_statistics.json`, `independent_verification.json` | 数据／权重／归一化／源码来源及实测复核结果 |
| `figures/*.png` | 9 张比较图，包含四个预先指定病例 |
| `*.log`, `execution.json`, `archive_manifest.json` | 执行记录、依赖和受管文件 SHA256 |

索引不重新选择：`case_id` 对应上一轮 `experiments/local_control_response_swap_sensitivity_v1/results/cases.csv` 的行；由 receiver/donor/window/flight/split 溯源。`pair_uid` 对应该轮 pairs/cohorts/windows。新目录 `results/{cases,pairs,cohorts,windows}.csv` 为同字节副本，供查看方便；计算仍使用已哈希的冻结上游索引。

## 统计与支持合同

- 每个版本只沿用 receiver 的完整受支持原生步；padding 的 dt=0 输出不是有效样本。绝不外推到 500 ms。
- 原始 A/B 在支持范围内报告 100/200/300/400 ms 和共同末端。D 为同一 270 ms 中心平均，要求完整支持；100/400 ms 及原始末端的 D 缺失保留。另标记 filter_endpoint，不能与 raw endpoint 混称。
- 速度保留原始 A；p/q/r 保留 A/B/D。各口径内的两版本真值、支持及门限相同；不把标签／滤波差异算作输入修正收益。
- 主 300 ms 提供逐对、航次、依赖分量等权；其他时刻提供航次等权。依赖图和权重函数复用原实现，独立复核另用图遍历。
- pooled 是种子等权的描述统计，不平均预测再算指标；种子及重叠队列不能相加充当独立证据。
- 方向：真实差绝对值>0.1，预测数值零阈值1e-8，敏感性0.05/0.2。内部 CA/CB 同向率只在两者均超过门限时算，近零／有效数同时报告。
- 内部经验增益沿用前轮实际施加差动均值和0.02分母保护。C/H 不是独立方差解释率；C 没有真实交换标签。NoControl 同时移除了历史和未来控制。
- 输入分布距离只检查九维瞬时特征，并非完整历史轨迹或相位条件的物理支持；保持单个日志的相位零点，不强行统一机械零点。
- source timestamp 是驱动读数返回后的软件锚点；availability 是构造上界。V1 仍是该源锚点和短期匀速假设下的诊断，不能称为已验证物理对齐。
