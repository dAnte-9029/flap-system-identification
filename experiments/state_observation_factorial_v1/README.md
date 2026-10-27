# State / observation factorial

离线2×2运动学消融，训练次数0。结论见 [report_zh.md](report_zh.md)。

从仓库根目录执行（不安装依赖）：

```bash
P=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
E=experiments/state_observation_factorial_v1
$P $E/run.py > $E/results/run.log 2>&1
$P $E/summarize.py > $E/results/summary.log 2>&1
$P $E/present.py
$P -m pytest -q $E/test_factorial.py > $E/results/tests.log 2>&1
```

`common.py`指定历史train/validation数据和H1000缓存绝对路径。依赖上游已登记缓存与 `causal_state_target_contract_v1` 保存结果；不需要checkpoint、GPU或原始ULog。不得替换为新版canonical数据或test分区。

- `run.py`：复用C0/C1、T00/T11和连续支持，补算T01/T10；仅恢复历史B表未含的支持内尾节点。
- `results/factorial_windows.csv.gz`：144,252窗口四组结果、初值误差、首步跳变、MSE因素。
- `results/summary.csv`：共同前缀/完整H1000，窗口等权/航次等权。
- `results/per_flight.csv`：121航次分支持子集结果。
- `results/factor_differences.csv.gz`、`factor_summary.csv`：因素和交互。
- `results/run_receipt.json`：复用来源哈希、恢复节点数和回归检查。
- `results/restored_B_tail.csv.gz`：精确flight/sample身份的恢复节点。
- `figures/`：3张PNG；`results/tests.log`：针对性测试。

单位：原始MSE为rad²，汇总显式带deg/deg²；角速度为rad/s。所有未来A/B均为离线记录，仅用于规定积分输入，不是模型自主预测。终点是每窗口共同有效前缀末端。复现不改变旧结果和原模型。
