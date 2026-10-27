# 因果状态与监督目标合同

[中文报告](report_zh.md)。本轮只有C0/C1状态估计及T0/T1离线运动学比较，训练0次，无GRU推理、test、Loss或模型修改。

基准 `3ee850af770631f6be3d5a9326987aa443307394`；分支 `exp/causal-state-target-contract-v1`。历史数据 `velocity_rate_multistep_v1/main_50hz`，来源复用上一轮 `control_target_alignment_v1/results/source_receipt.json`。缓存和原生Parquet路径在common.py，不重建数据、不解析ULog。

在新复现目录运行（复制本目录到同一仓库experiments下新名称，避免覆盖归档）：

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
E=experiments/causal_state_target_contract_v1
"$PY" "$E/analyze.py"
"$PY" "$E/summarize.py"
"$PY" -m pytest -q "$E/test_contract.py"
"$PY" "$E/report.py"
"$PY" "$E/archive.py"
```

不安装依赖，全部CPU。`state.py::causal_rate`不接受B或未来标签；`apply_rates`同时替换有效历史角速度与历史末端initial，其他输入和padding不变。参数已在config.json固定；实际起点无后向外推，但相同限制的后向情况有测试。

`common.py::kinematics`直接调用原模型模块的integrate/geodesic，不实例化网络。T0/T1使用相同原因果姿态、速度、dt和位置/速度目标，仅改变initial角速度和未来角速度。前方目标首次缺失后停止积分评分，不补零或恢复。所有输出为离线目标驱动的运动学一致性，不是动力学预测结果。

核心结果：

- `results/summary.csv`：窗口/航次等权初值误差、共同支持前缀及完整H1000姿态积分误差。
- `results/per_flight.csv`、`strata.csv`：逐航次、年龄和两观测变化量分组。
- `results/coverage.csv`、`history_coverage.csv`：起点准入/回退、唯一历史行覆盖。
- `results/windows.csv.gz`：144,252个注册窗口，精确window_id、支持数量、初值误差、积分均方及端点误差。
- `results/history_states.parquet`：863,457个唯一使用历史行的来源索引、测量/发布戳、C0/C1/B；B列仅为离线参考，未送入状态构建函数。
- `results/cases.csv`：train:124、train:336两端；姿态误差来自对应H1000窗口，未重算原500ms响应匹配。
- `results/tests.log`、`receipt.json`、`execution_record.json`：针对性测试及来源/范围记录。
- `figures/`：3张核心PNG；`archive_manifest.json`：新代码与结果SHA256。

同一窗口多个历史行回退不导致删窗；全量窗口相互重叠，不是独立实飞实验。B和记录姿态也不是无滤波延迟的绝对物理真值。报告建议情况C：停止当前两点外推，不直接训练C1/T1。
