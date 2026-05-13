# Whole-Log Group K-Fold Reporting Protocol

Date: 2026-05-11

这个文件记录一个重要实验叙事：既然不同 flight log 的难度会明显影响指标，就不应该通过挑选某个好看的 split 来报告主结果。更稳妥的做法是把 log-to-log variability 作为论文结果的一部分。

## 核心判断

当前数据是连续飞行日志，不能用 random sample split。即使使用 whole-log split，不同 held-out logs 的工况、横侧向激励、风扰、label 噪声和异常段也会明显改变指标，尤其是 `fy_b`, `mx_b`, `mz_b`。

因此，单个 train/val/test split 的结果容易受到 test log composition 影响。为了避免 split cherry-picking，主结果应使用：

```text
date-stratified whole-log group K-fold cross-validation
```

也就是按 `log_id` 分组做 K-fold，同时尽量按 flight date / session 均衡分配。每条 log 必须完整属于某一个 fold 的 test set，不能把同一条 log 的样本打散进 train/test。

可用英文：

> To avoid temporal leakage and reduce sensitivity to a particular held-out log composition, we report date-stratified whole-log group K-fold cross-validation, where all samples from the same flight log are assigned to the same fold while flight dates are balanced across folds.

## Date-Stratified Fold Construction

不要简单按日志编号连续切分，例如：

```text
fold 1: logs 1-6
fold 2: logs 7-12
...
```

这种做法可能导致某个 fold 的 test set 主要来自某一天，而 train set 缺少该日期的环境和实机状态。对于实机飞行数据，日期之间可能存在：

```text
airframe condition changes
battery / actuator condition changes
wind and weather changes
logging / calibration differences
controller or mission differences
```

所以 fold construction 应尽量满足：

```text
1. group by log_id: 同一条 flight log 不拆分
2. stratify by flight date/session: 每个 fold 的 test logs 尽量覆盖所有日期
3. train coverage: 每个 fold 的 train logs 也尽量覆盖所有日期
```

例如有四个日期：

```text
2026-04-12
2026-04-14
2026-04-15
2026-04-16
```

5-fold 时更理想的 test 分配是：

```text
fold 1 test: 04-12 log A, 04-14 log A, 04-15 log A, 04-16 log A
fold 2 test: 04-12 log B, 04-14 log B, 04-15 log B, 04-16 log B
...
```

这样每个 fold 都是跨日期 held-out logs，而不是“某一天整块被拿来测试”。如果某个日期 log 数量太少，无法完美均衡，要在文中说明：

> Folds are approximately date-stratified subject to the available number of logs per flight date.

主结果建议使用 date-stratified whole-log group K-fold。另一个更难的 stress test 可以是：

```text
leave-one-date-out
```

区别是：

```text
date-stratified group K-fold:
  测试普通跨 log 泛化，train 覆盖所有日期。

leave-one-date-out:
  测试跨日期 / 跨环境泛化，难度更高，作为 robustness check。
```

## Main Result

主结果不再只报一个固定 test split，而是报所有 folds 的均值和标准差：

```text
mean ± std over folds
```

主表建议字段：

```text
Target | RMSE mean ± std | R2 mean ± std
fx_b
fy_b
fz_b
mx_b
my_b
mz_b
overall
```

这样做的好处是：

```text
1. 避免选择最好看的 test split
2. 量化 held-out log 难度差异
3. 让 fy_b / lateral-directional 的不稳定性变成被解释的现象，而不是隐藏的问题
```

可用英文：

> The main metrics are reported as mean and standard deviation across group folds, rather than from a single hand-selected split.

## Additional Table: Per-Log Performance

除了主表，还应该提供 per-log performance table。这个表可以放正文简化版，完整表放 appendix / supplementary。

建议字段：

```text
log_id
fold_id
sample_count
overall_R2
fx_b_R2
fy_b_R2
fz_b_R2
mx_b_R2
my_b_R2
mz_b_R2
overall_RMSE
notes / regime tag
```

用途：

```text
1. 解释为什么某些 fold 指标低
2. 找出 easy logs 和 hard logs
3. 支撑 discussion 里的工况差异分析
4. 防止审稿人质疑只报告平均值掩盖失败 case
```

可用英文：

> Per-log metrics are reported to expose the variability across flight conditions and to identify regimes where the lateral-directional channels are less reliable.

## Main Figure: Representative Test Log

预测曲线图不要选最好看的 log。推荐选择：

```text
overall R2 最接近 per-log median 的 held-out log
```

或者如果重点展示横侧向：

```text
fy_b / lateral mean R2 接近 per-log median 的 held-out log
```

caption 里要说明选择标准：

> The representative held-out log is selected as the test log whose overall R² is closest to the median per-log R² across all folds.

这样可以避免被认为挑了一条最好看的曲线。

图中建议展示：

```text
true vs pred
6-axis wrench
absolute flight time on x-axis
segment gaps shown as blank gaps
```

## Discussion: Easy / Hard Logs

Discussion 不要只说“某些 log 难”。要分析难在哪里。

建议比较 easy / hard logs 的这些因素：

```text
airspeed / dynamic pressure range
flapping frequency range
roll/yaw maneuver intensity
lateral excitation strength
wind or sideslip proxy
phase consistency
target noise / high-frequency energy
nav mode transitions / segment breaks
label-generation sensitivity
```

合理叙事：

```text
纵向通道 fx_b, fz_b, my_b 在大多数 log 上稳定较强。
横侧向通道 fy_b, mx_b, mz_b 对 log-level distribution shift 更敏感。
fy_b 最弱，因为 side-force label 幅值小、噪声占比高，而且由 acceleration-based inverse dynamics 间接重构。
```

可用英文：

> The model generalizes consistently in the dominant longitudinal and pitch channels, while lateral-directional performance varies more strongly across logs. Hard logs typically contain stronger lateral maneuvers, lower side-force signal-to-noise ratio, or less repeatable high-frequency target components.

## 审稿风险与防御

不要写：

```text
We selected the split with the best performance.
```

也不要只放一个漂亮 test split。

更稳的写法：

> Since held-out log composition affects lateral-channel metrics, we use whole-log group K-fold cross-validation and report fold-level variability. This prevents temporal leakage and avoids over-interpreting a single favorable train/test split.

这套协议的重点不是把指标做得最好看，而是让指标站得住。对于这个项目尤其重要，因为 `fy_b` 和横侧向通道本身存在 log-dependent difficulty 和 target reliability 问题。
