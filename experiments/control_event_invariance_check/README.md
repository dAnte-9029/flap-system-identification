# 控制事件分段不变性诊断

结论见 [report_zh.md](report_zh.md)。独立分支 `exp/control-event-invariance-check`，基于 `8455e9f`。仅使用已归档的 8 个 main_50hz 验证历史和冻结 Control seeds 17/29/43；不训练、不访问测试集、不修改正式推理。

## 复现

在本 worktree 根目录执行，输出只写本实验目录：

```bash
cd /home/zn/flap-system-identification/.worktrees/control-event-invariance-check
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_invariance_check/check.py > experiments/control_event_invariance_check/run.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_invariance_check/summarize.py > experiments/control_event_invariance_check/summary.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/control_event_invariance_check/test_check.py > experiments/control_event_invariance_check/tests.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_invariance_check/report.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_invariance_check/archive.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_invariance_check/archive.py --verify
```

GPU 推理约一分钟；无需加载原始 Parquet、未来运动标签或原始 QgcLogs。已有 `traces.npz` 可直接运行后四项进行指标核对、报告和归档；`summarize.py` 使用已存逐窗口指标，不重新推理。源 NPZ 和模型权重均已随基线分支保存，校验不符即失败。

## 文件说明

- `check.py`：严格验证整个分段常值函数等价；复用原 predict/forward，hook 记录，不修改生产源码。
- `config.json`：样本规则、种子、判据、规范网格及物理前提。
- `test_check.py`：7 项定向测试，含独立重算所有窗口/种子/变体的状态差异。
- `results/selected_pairs.csv`、`windows.csv`：样本来源、选择结果及对照关系。
- `results/equivalence.csv`：全区间等价、50 Hz 控制均值、有效事件/含 padding 计算步数。
- `results/layers.csv`：事件嵌入、主状态特征和主隐状态的首次分歧与最大差值。
- `results/metrics.csv`：逐窗口、种子、表示变体、时域和状态指标；包含截至时域的峰值。
- `results/summary.csv`：全种子/分种子的均值、最大值、RMS、受影响数及相对命令交换响应比例。
- `results/traces.npz`：实际输入、逐步预测、隐状态、事件嵌入、相位/频率/掩码/dt 及状态特征。
- `results/manifest.json`：数据来源契约、权重/源码/源文件 SHA256、GPU 环境和执行记录。
- `archive_manifest.json`：本实验全部文件哈希，排除自身和 Python 缓存。

## 对照解释

`native` 是原生表示；`split2/4` 仅插入冗余同值事件；`prior_refined` 精确复现上一轮绿线方案；`prior_common` 复现上一轮共同网格中的本方控制表示；`merged` 合并原生相邻同值事件；`pad_only` 只增加无效 padding；`repeat` 完全相同输入重复推理。

`donor` 使用真实匹配记录的另一套命令，作为合理幅度的模型敏感性参照，不是实际因果响应真值。`fixed_native/fixed_split4/fixed_donor` 使用同一 5 ms 节点规则并保留全部真实断点；前两者输入完全相同。不能仅通过普通 50 Hz ZOH 忽略真实区间内变化，也不能把规范网格结果直接替换正式输出。

本队列含主动选择的已知异常，不估计总体发生率、不做统计显著性外推。重复命令无硬件重置/恢复语义是明确前提，未独立认证固件—执行器协议。
