# 控制事件来源与精确合并验证

先读 [report_zh.md](report_zh.md)。独立分支 `exp/control-event-provenance-merge-v1`，基于 `1ea494d`。不修改正式推理、训练、数据、权重或 PX4，不读取测试集。

## 复现

在本 worktree 根目录，使用固定环境：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_provenance_merge_v1/audit_source.py > experiments/control_event_provenance_merge_v1/audit.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_provenance_merge_v1/evaluate_merge.py > experiments/control_event_provenance_merge_v1/evaluation.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_provenance_merge_v1/summarize.py > experiments/control_event_provenance_merge_v1/summary.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/control_event_provenance_merge_v1/test_merge.py > experiments/control_event_provenance_merge_v1/tests.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_provenance_merge_v1/report.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_provenance_merge_v1/archive.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_provenance_merge_v1/archive.py --verify
```

`common.py` 中记录原数据和已有训练缓存路径。缓存匹配既有 `data_audit.json` SHA256；缺失或校验不符即失败。审计只读 train/validation 控制 Parquet；训练缓存只用于编码统计，不加载模型训练或测试分区。

固件源码优先复用当前目录已归档、哈希一致的精确快照，其次只读本地 Git 对象，最后通过完整固件 SHA 的项目 raw URL 取回缺失文件，不执行 git fetch、不修改 PX4 工作区。网络返回失败会显式保留 unavailable。当前 153 项均已取得精确版本。

`evaluate_merge.py` 跑完整验证，GPU 推理无梯度；两个表示张量形状和批次顺序相同。只合并所有四路值精确相等的邻段，不设数值容差，不丢弃真实跳变，不改变状态步长。

结果包括：完整逐窗口端点预测/误差、变更窗口逐步预测、逐航次及逐种子汇总、所有真实合并区间前后输入、来源统计和 17 固件版本代码快照。统计有重叠窗口计数和唯一原生状态区间计数两套口径，不能混为独立样本量。

只复算既有输出可运行 summarize / tests / report，不需要 GPU。不要将“预测对 EKF 状态的误差”解释为物理控制反事实准确性；也不要把日志 ZOH 当作已知完整的真实硬件控制函数。
