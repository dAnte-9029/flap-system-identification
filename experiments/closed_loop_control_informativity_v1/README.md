# 闭环控制信息量分析 v1

中文结论：[report_zh.md](report_zh.md)。方法、文献版本与控制链：[methods_and_chain_zh.md](methods_and_chain_zh.md)。基准SHA为 `321822d83f7a1f381ecc50ba0a4c23e5aa9fa4cf`。

本轮仅控制命令的条件回归和描述性统计；无动力学训练、GPU推理、IV模型、Loss或MPC。数据限定历史 `velocity_rate_multistep_v1/main_50hz` train/validation；test不读。辅助模型不是新的GRU。

## 复现

使用既有 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，不安装依赖。需要原数据目录 `/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1` 与原日志 `/home/zn/QgcLogs`；路径在config内显式指定。所有Parquet与ULog先验证registry／manifest SHA256。原实验及源数据不写。

为了不覆盖本次固定输出，建议先从本提交创建另一个临时 detached worktree，再在该worktree根运行。脚本的结果目标固定为其自身实验目录；不要在需要保留未归档修改的目录直接重跑。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/closed_loop_control_informativity_v1
"$PY" "$EXP/extract.py" > "$EXP/extract.log" 2>&1
"$PY" "$EXP/analyze.py" > "$EXP/analyze.log" 2>&1
"$PY" "$EXP/summarize.py" > "$EXP/summary.log" 2>&1
"$PY" "$EXP/supplement.py" > "$EXP/supplement.log" 2>&1
"$PY" "$EXP/reference_events.py" > "$EXP/reference_events.log" 2>&1
"$PY" "$EXP/verify.py" > "$EXP/verification.log" 2>&1
"$PY" -m pytest -q "$EXP/test_diagnostics.py" > "$EXP/tests.log" 2>&1
"$PY" "$EXP/archive.py"
git diff --check
```

`audit_sources.py` 可选重新取得14个固件的精确源码及终稿文献；源码快照与哈希已提交，数据统计不依赖重新联网。Dinkla终稿获取失败记录保留，不能把Drive中2024预印本当2026终稿。已有三份完整Drive文献文本仅读取研究，`*.local.txt`不公开；不拥有Drive访问时仍可复现数据统计及全部数值检验。`supplement.py`仅在本地文献缓存存在时更新文本哈希。

`analyze.py --resume` 只用于同一数据、配置和已核交叉拟合输出的后处理恢复。本轮第一次信息矩阵实现对注册起点误用精确100ms滞后导致空矩阵，改为连续注册起点滞后后复用原辅助拟合。`analyze.log`保留初次失败，`analysis_resume.log`为后处理完成记录。全新复现使用不带resume的命令。

辅助拟合共9组（7变量组＋2岭系数敏感性）×（5个train留出＋1个完整train→validation）=54个小型岭回归；不是54个飞行动力学模型。所有主结果使用固定0.01系数，无参数搜索或选择。

## 输出索引

| 路径 | 内容 |
|---|---|
| `results/source_receipt.json`、`raw_access.json`、`upstream_hashes.json` | 数据与旧资产哈希、121条ULog白名单、访问范围 |
| `results/topic_coverage.csv`、`field_coverage.csv`、`topic_summary.csv` | 每航次／topic／field可用性与25–200ms新鲜度 |
| `results/firmware_sources.csv`、`navigator_sources.csv`、`source_evidence.csv` | 精确提交源码、SHA256、行号；快照在sources |
| `results/windows.csv.gz`、`coverage_per_flight.csv` | 全候选稳定身份、工况、原控制事件、完整支持去重标记 |
| `results/crossfit_groups.csv`、`crossfit_receipt.json`、`auxiliary_fits/` | 按原ULog分组、每折训练／评价成员、训练侧预处理与回归系数 |
| `results/crossfit_predictions.npz` | 每变量组完整70,349窗口×12目标的留出预测；顺序严格对应windows |
| `results/conditional_summary.csv`、`conditional_per_flight.csv` | 各split、时域、四坐标、全窗口／不重叠窗口结果 |
| `results/coupling.csv`、`information_matrices.csv`、`gram_*.npz` | 四路条件耦合、原单位／标准化Gram与索引 |
| `results/weak_directions.csv`、`information_leave_flight_out.csv`、`actual_lag_spacing.csv` | 最弱方向、航次删除敏感性、实际滞后年龄 |
| `results/native_auto_controls.csv.gz`、`native_psd.csv.gz`、`conditional_psd.csv.gz` | 精确裁切ZOH控制、原生／100ms平均残差谱，不跨间断拼接 |
| `results/temporal_per_flight.csv`、`event_motion.csv.gz`、`temporal_curves.npz` | p/q/r前后关系、负对照、控制资格事件 |
| `results/chain_per_flight.csv`、`reference_relevance.csv`、`mission_*.csv` | 参考相关、控制链、任务切换与多轴变化 |
| `results/pair_information.csv`、`pair_failure_links.csv`、`component_failure_summary.csv` | 全113固定对、原模型逐种子失败链接、11个主依赖分量 |
| `results/pair_cohort_summary.csv`、`pair_dependence_summary.csv`、`observational_background.csv` | 所有重叠子集、逐对／航次／依赖分量等权、完整旧B/D背景 |
| `results/frozen_window_diagnostics.npz` | 所有固定匹配窗口的控制、p/q/r、参考、交叉拟合与真值数组 |
| `results/support_strata.csv`、`concentration.csv` | 速度／幅值／持续时间／日期／固件／配置及集中程度 |
| `figures/*.png` | 7张主图；`report_numbers.json`为紧凑主要数字 |
| `tests.log`、`verification.log`、`independent_verification.json` | 自动测试与独立增广最小二乘、积分、SVD及聚合复核 |

`diagnostic_arrays.npz`（114.5MB）与`causal_features.parquet`（140.7MB）为大型可重建本地缓存，不提交。`archive_manifest.json`仍记录它们的哈希、大小及local-only状态。精简固定窗口数组、完整命令留出预测、原生控制与所有统计已提交。缓存不使用LFS隐藏依赖；可由八份已登记开发Parquet与121个允许ULog重建。

坐标顺序 `[motor,common,differential,rudder]`；12个回归目标按100／300／500ms各4列排列。时间曲线grid为−500:20:+500ms。`pair_information.command_difference_300`是两个片段各自减起点命令后的偏离差，不等同原完整UA−UB；不要混用。`reference_relevance.lag_ms>0`仅为离线未来参考诊断，不进入拟合特征。Gram的lag_index是注册窗口序号，实际时间在NPZ的lag_age_us，不是精确100ms。

## 限制与实施记录

事件资格仅来自既有控制、起点历史／模式及支持；未按运动正负删样。全部辅助拟合train-only、按原ULog留出。validation只应用冻结规则；旧匹配规则本来在train探索过，所以不是前瞻盲测。残差非外生、Gram满秩非GRU辨识认证、背景非纯噪声、软件时钟非无延迟物理真值。

补充分箱固定为速度6/8m/s、平均命令偏离0.02/0.05/0.10、持续时间100/250ms；它们在主条件回归之后添加、在查看分箱结果前按物理量定义，不属于在所有本轮统计之前注册的规则，也未依据运动预测成绩挑选界限。主回归参数及变量组均保持不变。

保留`verification_initial_failure.log`（独立坐标运算的float32/float64顺序问题）和`tests_initial_failure.log`（合成秩亏Gram平方根舍入问题）。最终修正复核实现／断言，没有改动分析数值。所有测试通过后归档，Git提交SHA和真实推送状态由交付消息给出，不在文件中循环写入自身提交SHA。
