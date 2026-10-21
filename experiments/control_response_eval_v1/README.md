# Control response evaluation v1

先读 [中文报告](report_zh.md)。这是冻结 GRU-Control / GRU-NoControl 的观察性响应差异验证与控制交换敏感性诊断，不是反事实真值验证。

## 来源与边界

- 审计提交 `1745345d06749c6b127e43660a9bf1947d058bea`；模型提交 `8964b44933c7364c066d3cb09440a6d2a7f558f1`。
- 新分支 `exp/control-response-eval-v1`，本实验完整源码版本为包含本目录的 Git 提交；`archive_manifest.json` 记录文件哈希，避免自引用提交 SHA。
- 只读取注册数据 `velocity_rate_multistep_v1` 的 train/validation。禁止加载测试 Parquet/原始测试日志。
- 六个模型均冻结，种子 17/29/43，无训练、调参或模型修改。
- 原审计候选大档案位于 `config.json` 的 `audit_source`；复现匹配选择需要该目录，读取前核对其归档 SHA256。缺失时应恢复同一已归档文件，不重新匹配或放宽条件。
- 数据、模型及审计档案的路径/哈希分别记录于 `run_manifest.json`、`selection_manifest.json`。原始 QgcLogs 不写入。

## 完整执行

从本 worktree 根目录，使用指定环境（不安装依赖）：

```bash
cd /home/zn/flap-system-identification/.worktrees/control-response-eval-v1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/evaluate.py > experiments/control_response_eval_v1/run.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/analyze.py > experiments/control_response_eval_v1/analysis.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/diagnostics.py > experiments/control_response_eval_v1/diagnostics.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/plot_results.py > experiments/control_response_eval_v1/plots.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/control_response_eval_v1/test_response.py > experiments/control_response_eval_v1/tests.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/verify.py > experiments/control_response_eval_v1/verification.log 2>&1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/build_report.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/archive.py
```

这些命令只覆写本实验的输出。保留当前结果时，请先复制本实验目录或在新分支运行；不要覆盖既有审计、模型或数据。

`evaluate.py` 要求 CUDA。只复算已存预测可从 `analyze.py` 开始；`verify.py` 还需要授权原始数据以独立核对标签。该复算逐航次聚合较慢，属于正常 CPU 分析，不是训练。

归档完整性复核：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/control_response_eval_v1/archive.py --verify
```

## 关键文件

| 文件 | 内容 |
| --- | --- |
| `config.json` | 固定样本/种子/时域/交换策略/方向参照/依赖分组规则 |
| `evaluate.py` | 先冻结队列，随后打包真实历史与目标；六模型 GPU 自主推理及双向控制交换 |
| `analyze.py` | A/B 指标、逐种子及航次等权汇总、依赖分组和阈值敏感性 |
| `diagnostics.py` | 匹配覆盖和初态对齐敏感性；不改变样本队列 |
| `verify.py` | 从授权 Parquet、独立四元数算法及逐航次直接均值复算 |
| `results/cohorts.parquet` | 每个队列、匹配对、端点航次、权重及依赖组；同一对可出现在多个队列 |
| `results/predictions_*.npz` | 所有输入、真实标签、事实/交换/等价细分/共同网格预测；原生步与报告点索引 |
| `results/response_metrics.parquet` | A 逐对/种子/时刻/状态指标；`pair_uid` 为时域加原审计候选行号 |
| `results/swap_metrics.parquet` | B 的固定历史响应、等价细分检查、共同网格检查 |
| `results/response_summary.csv` | 航次等权主指标与逐对等权补充，`seed=all` 平均误差平方而非集成轨迹 |
| `results/paired_comparisons.csv` | C/N 改善、依赖组置信区间可用性、航次互斥子集 |
| `results/*sensitivity.csv` | 方向门限、依赖组及初态对齐敏感性 |
| `results/condition_comparisons.csv` | 原历史匹配质量、控制强度、固件、REVIEW、高动态分组 |
| `results/figures/` | 17 张图；病例按事后改善排序，仅展示，不影响评价 |
| `results/cases.json` | 成功/失败病例 ID、选择依据及图路径 |
| `results/archive_manifest.json` | 所有新实验文件的 SHA256 和大小；不包含自身及缓存 |

## 解释约束

主验证仅 main_50hz validation–validation。train–train 是诊断，跨划分和 sparse_10hz 只统计。既有验证集曾用于 checkpoint 选择，不能称作新盲测。两时域各仅 3 个共享航次依赖组，不给主结果置信区间。实验 A 仍有初态与闭环反馈混杂；实验 B 没有反事实真值。等价命令分段会改变模型输出，这一异常已保留并单独分析。
