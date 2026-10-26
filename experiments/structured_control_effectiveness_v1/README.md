# Structured control effectiveness: stage 1

[中文报告](report_zh.md) · [结构与公式](model_design_zh.md)

基准 `2c99a7fcf8495cfb1d21535e928716537aa7eb73`，分支 `exp/structured-control-effectiveness-v1`。仅 seed17 × M0/M1/M2 × 6 epoch；没有执行下一阶段 27 次训练。

## 环境与资产

仅使用 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，不安装依赖。Torch2.7.0+cu128，cuda:1，RTX4090；依赖版本见 execution.json。

- 历史数据：`/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1`，仅 train/validation。
- 原 H1000 缓存：`/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1/artifacts/velocity_rate_gru64_horizon_training_v1/{train,validation}.npz`。只读，必须与原 data_audit 哈希一致，不允许缺失时重建替代。
- 复用仓库原 `velocity_rate_gru64_horizon_training_v1`、scope、rollout、timing_slowfast、swap_sensitivity、time_contract_ablation 资产。文件哈希见 results/source_receipt.json。
- 新模型只在 model.py；原模型、积分器、损失、规范化 ZOH 和裁剪函数直接导入，旧接口不改。全部新文件在本目录。

## 已执行顺序

从本 worktree 根目录，以规定 Python 执行：

1. `preflight.py`：核验缓存、注册窗口、划分、旧结果；复制冻结 normalization；记录来源。
2. `python -m pytest -q .../test_contract.py`：训练前实现检查（当时 9 项通过）。
3. `train.py`：H1000 smoke，每模型 3 更新后丢弃；三个独立六轮训练，不提前停止。
4. `holdout_plan.py`：仅冻结下一阶段划分，未训练任何留出折。
5. `evaluate.py`：best checkpoint 的全部固定四组合，以及全部原验证轨迹。
6. `verify.py`：独立插值、积分、图权重、SciPy 轨迹指标；不调用主指标函数复算。
7. `plot.py`、`describe.py`：图表、逐对变化、病例、预算。
8. 再运行 pytest：训练后共 12 项通过；`archive.py` 核验权重、上游和归档。

`.log` 保存各步骤标准输出。三个 best 都在 epoch6，与 last 参数相同（序列化文件 SHA 可因文件名不同而异）。选择只根据验证轨迹损失，没有依据控制 C/H 或响应表现选择 epoch。

## 安全复现

`train.py` 拒绝覆盖已有 training/M*。验证现有产物直接运行：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/structured_control_effectiveness_v1/test_contract.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/structured_control_effectiveness_v1/verify.py
```

完整重新试训练请在独立 checkout 中创建新的 sibling 实验目录，只复制代码和 config（不复制结果或 checkpoint），例如：

```bash
mkdir -p experiments/structured_control_effectiveness_repro/results
cp experiments/structured_control_effectiveness_v1/*.py experiments/structured_control_effectiveness_repro/
cp experiments/structured_control_effectiveness_v1/config.json experiments/structured_control_effectiveness_repro/
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/structured_control_effectiveness_repro/preflight.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/structured_control_effectiveness_repro/test_contract.py > experiments/structured_control_effectiveness_repro/tests.log
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/structured_control_effectiveness_repro/train.py > experiments/structured_control_effectiveness_repro/train.log
```

然后按以上顺序运行该新目录的 holdout_plan/evaluate/verify/plot/describe。共同 ROOT 仍为仓库根目录，上游均只读。若设备不可用先报告或明确改变新复现配置 device，不隐式安装环境。训练本身只有约 3.5 分钟；墙钟受共享 GPU 影响。原运行配置在正式训练前已固定并哈希，跨机器不保证逐位相同，应数值核验。

## 产物索引

| 路径 | 内容 |
|---|---|
| config.json / normalization.json | 六轮公平预算、固定输入／损失及原训练尺度 |
| model.py / model_design_zh.md | 六维离散残差、F/B隔离、两时间矩、相位调制 |
| training/{M0,M1,M2}/{best,last}.pt | 本轮独立权重、config、normalization、seed、epoch；绝不替换旧权重 |
| training/*/training.jsonl / summary.json | 每轮损失、梯度、lr、顺序哈希、用时、显存、权重SHA256 |
| results/source_receipt.json / smoke.json / inference_checks.json | 原缓存、来源、结构与推理门禁 |
| results/predictions.npz | M0/M1/M2 各 452×K×13 原生四组合预测；原 inputs 引用固定 swap/results/inputs.npz |
| results/{pairs,cases,windows,cohorts}.csv | 冻结上游的同字节索引副本 |
| results/{factual,internal}_{curves,points}.csv.gz | 模型／种子／pair／变量／时刻逐行结果，支持缺失显式保留 |
| results/window_points.csv.gz | case、receiver/donor window、cell、时间、增量溯源 |
| results/{factual,internal}_summary.csv / per_group.csv | 全主队列、背景与质量子集，300ms逐对／航次／分量等权及日期配置分组 |
| results/paired_changes.csv.gz / paired_change_summary.csv | 新结构相对 M0 的逐对变化、方向得失、幅度偏差 |
| results/validation_trajectory*.csv* | 原验证全集事实轨迹误差，逐窗口、航次和总表 |
| results/trajectory_verification_samples.npz | 每模型128个分布于全验证序列的固定窗口，供独立数值复核 |
| results/historical_24epoch_reference.csv | 旧冻结模型历史参照，禁止与六轮试训练当公平最终对照 |
| results/holdout_protocol.json / holdout_window_indices.npz | 下一阶段3折拟合/留出身份，未执行；各折需训练侧重新拟合统计 |
| results/holdout_{pair_assignments,coverage}.csv / next_stage_budget.csv | 图分量分配、覆盖与时长估计 |
| figures/*.png | 训练、方向幅度、C/H、耦合、子集、轨迹和四个固定病例 |
| tests.log / verification.log / results/independent_verification.json | 12项测试及独立数值复核 |
| execution.json / archive_manifest.json | 环境、实际训练范围、受管文件SHA256 |

A/B/D、270ms平均、方向阈值、背景集合、交换四路波形全部冻结。p/q/r 与 NED 速度三分量保留。不同子集、种子、重叠窗口都不当独立干预。C/H 为模型内部差异，只有 AA−BB 有对应事实响应差；不存在 swapped 真值准确率。
