# 控制事件规范化验收

基于 `653b60d`，分支 `fix/control-event-canonicalization-v1`。完整结果见 `report_zh.md`。

冻结 1s GRU-Control 种子 17/29/43，只评价 main_50hz validation；允许少量 train 窗口仅做梯度检查，无优化器更新。没有测试集读取、训练或旧权重覆盖。

原始数据在 `/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1`；只读 1s 缓存在 `.worktrees/velocity-rate-dataset-v1/artifacts/velocity_rate_gru64_horizon_training_v1`。运行器校验验证缓存/权重哈希，独立验证器校验训练缓存、注册 manifest 和原生 train/validation 控制哈希。`legacy_velocity_rate_gru.py` 是基提交的逐字节源码快照，专门提供冻结旧语义对照。

在本 worktree 根目录运行：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_canonicalization_v1/run.py --stage preflight
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_canonicalization_v1/run.py --stage integrated
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_canonicalization_v1/verify.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_control_event_canonicalization.py tests/test_velocity_rate_learning.py tests/test_velocity_rate_long_horizon.py tests/test_velocity_rate_horizon_training.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_canonicalization_v1/report.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_event_canonicalization_v1/archive.py --verify
```

`preflight` 使用冻结旧网络加外部规范化；`integrated` 使用正式模型默认入口，核对其与前者一致。两阶段都重新推理全部验证窗口，原生、规范化、二分、四分及随机七分各跑一次。分段固定随机种子 911+batch_start；模型种子和批次一致。病例分析另复用以前八个窗口的输入快照，不按本次输出选例。

工程容差在 `acceptance.json`，参考前次已有结果预先设定，不声称是新的统计检验。每种子及合并种子的各指标都须通过；受影响窗口与全体分别统计。两阶段预测逐元素独立对比；向量误差和姿态角用 NumPy/SciPy 全量复算。

此处归档含冻结研究结果。再次运行应在新的实验副本中执行，以免覆盖本次归档；完整历史重现还应使用相应 Git 提交。统一规则说明见 `docs/control_event_encoding.md`。
