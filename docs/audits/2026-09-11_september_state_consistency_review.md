# 九月轨迹状态一致性诊断

日期：2026-09-11。独立运行：[september_state_consistency_20260911](../../artifacts/september_state_consistency_20260911)。状态 completed。

## 完成内容与范围

将其他 agent 在 `/tmp/run_diag.py`、`analyze_diag.py`、`finalize_step2.py` 的预测、积分和汇总逻辑整合为 [run_september_state_consistency.py](../../scripts/run_september_state_consistency.py) 及 [trajectory_consistency.py](../../src/system_identification/evaluation/trajectory_consistency.py)。旧脚本原文归档在本次独立运行的 legacy_source，不再依赖临时脚本执行。旧结果和原始日志未改写。

依据 [诊断契约](../contracts/2026-09-11_september_state_consistency.md)，复用 history_26 缓存预测，严格核对 6,669 个验证窗口的 ID、dt、真实位置/速度及原始误差数组；位置和速度误差完全一致。读取注册表指定的 15 条 train/validation ULog，验证原始文件哈希、全部位置/速度值及两个时间戳。训练窗口 3,966 个，验证窗口 6,669 个；没有训练、修改标签或打开 9 月 8 日测试。

## 指标口径已统一

主指标为每条日志内的三维向量 RMSE，再对日志等权平均；pooled-window 指标独立列出，单轴指标明确标注。旧 step2_report.json 中的 5.175 m 是分量平均 RMS，乘 sqrt(3) 才是 pooled-vector 8.964 m；原模型 equal-log 为 8.918 m。它们不是不同模型的性能改善。

下表全部为 **equal-log vector RMSE（m）**。第三列是速度误差梯形积分得到的位移误差向量大小，不是普通速度 RMSE。

| 时长 | 原模型位置误差 | 积分速度误差 | 真实位置减真实速度积分的累计残差 |
|---|---:|---:|---:|
| 1 s | 0.55704 | 0.31633 | 0.38422 |
| 2 s | 1.48006 | 1.11264 | 0.71680 |
| 3 s | 3.02236 | 2.63136 | 1.00210 |
| 5 s | 8.91795 | 8.61152 | 1.44611 |

真实累计残差的 pooled-vector 对应为 0.51094/0.95571/1.33951/1.94013 m。此前检查消息引用的是这些 pooled 数；与现在 equal-log 数的差别来自汇总权重，不是数据发生变化。

预测位置积分自身的 5 秒累计残差仅约 0.00006 m。带符号分解

`位置误差(t)-位置误差(0) = 积分速度误差 + 预测积分残差 - 真实积分残差`

在数值精度内闭合。三个向量有关联，不可以把 RMS 直接相加、相减或算作独立因果贡献率；真实累计残差也不是所有预测模型不可突破的误差下限。

## 原因排查

### 1. 不是本数据管线中的位置/速度跨 topic 对齐或插值

15 条 ULog 中，所有 `x/y/z/vx/vy/vz`、publication timestamp、sample timestamp 与数据集逐样本完全相同；sample_in_log 对应原消息索引，没有位置/速度行被重采样。二者来自同一条 vehicle_local_position instance 0 消息。其他输入 topic 的对齐不属于这个结论。

### 2. publication/sample 时钟选择影响很小

相同窗口、相同状态下，验证集 5 秒真实累计残差：publication 1.446109 m，sample 1.446129 m。两种积分结果之差的向量 RMSE 为 0.001725 m。不能用该时钟差解释米级累计残差，也没有理由据此改变既定 publication-time 因果输入契约。

固定速度 lag 网格 {-100,-50,0,50,100} ms 的共同内部区间诊断如下。所有 lag 均在每个窗口两端裁去至少 110 ms，实际区间长度见 lag_per_log.csv；它不是完整 2/5 秒指标。

| lag (s), 速度读取 t+lag | train vector RMSE (m) | validation vector RMSE (m) |
|---|---:|---:|
| -0.10 | 0.46307 | 1.66637 |
| -0.05 | 0.33913 | 1.49098 |
| 0 | 0.26920 | 1.39872 |
| +0.05 | 0.30082 | 1.42400 |
| +0.10 | 0.40857 | 1.54275 |

该有限网格上整体最好的是零位移。只排除了本次网格/范围内存在明显共同固定 lag 改善，不排除更小、时变或日志特异的误差；没有拟合或应用新的时间校准。

### 3. 选中窗口内没有离散复位/估计器切换

全部 10,635 个窗口内 xy/z/vxy/vz/heading reset counter 均无变化，estimator_selector 的 primary instance 也无切换。此结论只覆盖选中窗口，不意味着整条飞行日志从未复位；也不排除连续滤波修正。

### 4. z_deriv 与 vz 的区别确实影响垂直一致性

仅诊断性地把积分使用的 vz 换为同消息的 z_deriv：

- train 2 秒垂直累计残差：0.21750 → 0.02712 m。
- validation 5 秒垂直累计残差：0.53741 → 0.07542 m。
- validation 5 秒三维累计残差：1.44611 → 1.25073 m，水平两轴没有改变。

这支持“记录的状态估计速度并不严格等于记录位置的时间导数”，尤其是垂直通道。不能直接把 z_deriv 当作真实物理速度，也没有更换现有训练标签。

### 5. 部分日志存在较强的位置跟踪误差关联

按选中窗口覆盖的唯一原始区间统计，区间位置一致性残差除 dt 的幅值，与 active estimator 的 output_tracking_error[2] 的相关系数：验证 log_28 约 0.895，log_30 约 0.797；其他日志差异较大，不能概括为全部日志均强相关。估计器实例按 selector 选择，status 采用不超过 250 ms 的因果 asof 对齐。

验证 log_30 的真实 5 秒累计残差为 4.7545 m，log_28 为 2.7778 m；log_21 只有 0.4102 m。数据状态一致性差异具有明显的日志依赖性。

只读查看本机 PX4 源码（HEAD `be0dd9b89a49a2764e31ed981abba6218bc8504a`）：EKF2.cpp 将位置、速度、z_deriv 分别发布；output_predictor.cpp 对输出缓冲的位置、速度分别应用跟踪修正。这提供了合理机制解释，但本机缺少日志固件 `c9af571575af33d5403fbb479256848797c801e2` 对象，**不是对飞行固件实现的逐行确认**。因此“连续状态估计修正是主要来源”仍是有支持的假设，不是完成因果证明。

## 对路线的影响与下一步

1. 不把记录位置/速度的全部不一致都要求物理积分模型去拟合；先明确目标是物理运动代理，还是还要预测状态估计输出中的修正。此项语义决定需要单独冻结，不能静默改数据。
2. 目前模型的 5 秒平动误差仍远大于状态一致性残差，且积分速度误差约 8.612 m；不能用 EKF 问题解释掉长期预测不足。
3. 后续真实姿态/角速度 oracle 诊断仍有价值，但应同时报告速度误差、原日志位置误差和对真实速度积分轨迹的诊断误差，区分目标一致性与动力学误差。后者只作为辅助诊断，不替换最终真实轨迹指标。
4. 需要时继续定位 log_28/log_30 的输出修正、传感器融合与原固件机制。不能凭验证误差直接删掉这些日志或将“低残差筛选”当作同测试集上的模型提升。
5. 本轮没有执行 oracle 替换，也没有开始新模型训练。应先据此修订下一轮冻结协议，再运行最小单通道干预。

## 可复现性和验证

- 新模块与已有 rollout diagnostics 共 8 项测试通过，覆盖不均匀 dt、累计位置修正、vector/component/pooled/equal-log 区别、非有限值保留、固定 lag 的共同区间与已知偏移恢复。
- 源数据和原预测在运行前后哈希不变；同窗口误差复现通过。独立 manifest 保存注册数据完整来源、原 checkpoint/统计量/误差缓存哈希与诊断实现哈希。
- 入口默认验证缓存，不重新训练；`--replay` 可从 checkpoint 重新推理。该可选 replay 分支本轮未执行，GPU 重推不属于本轮已验证结果。
- 独立输出路径不可覆盖。重跑须指定新目录：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -u scripts/run_september_state_consistency.py --output-root artifacts/september_state_consistency_new_run
```

主要文件：prediction_aggregate.csv、prediction_per_log.csv、clock_equal_log.csv、clock_per_log.csv、raw_source_audit.csv、lag_equal_log.csv、lag_per_log.csv、window_raw_audit.parquet、manifest.json、summary.json。
