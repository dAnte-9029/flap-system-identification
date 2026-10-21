# 控制可辨识性与控制响应覆盖审计

基线提交 `8964b44933c7364c066d3cb09440a6d2a7f558f1`；仅使用 `velocity_rate_multistep_v1` 的 train/validation。无需 GPU，不导入神经网络，不读取 `target_*`，不读取测试数据或测试原始日志。

## 复现

在包含已登记数据文件的本 worktree 根目录运行：

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/control_identifiability_audit_v1
"$PY" "$EXP/audit.py" > "$EXP/run.log" 2>&1
"$PY" "$EXP/summarize.py" > "$EXP/summary.log" 2>&1
"$PY" "$EXP/verify_results.py" > "$EXP/verification.log" 2>&1
"$PY" -m pytest -q "$EXP/test_audit.py" > "$EXP/tests.log" 2>&1
"$PY" "$EXP/make_report.py"
"$PY" "$EXP/archive.py"
```

依赖现有环境的 NumPy、Pandas、PyArrow、SciPy、Matplotlib、PyYAML、pyulog、pytest，不安装或修改依赖。新 worktree 中仅需把原工作区下 dataset/velocity_rate_multistep_v1 的 samples、windows、native_actuator_motors、native_actuator_servos 的 train/validation 共八个 Parquet 链接到同名位置。不能链接/读取 test 文件。脚本会校验登记的清单和所有实际读取的数据哈希；原始日志只抽查四段获准飞行。

`audit.py` 产生覆盖统计、全量跨日志匹配、去重和敏感性结果；`summarize.py` 复算事件明细、增量相关性、分组对照与样例图。只重建本实验 results 目录中的派生文件，不改数据源。

## 产物

- `report_zh.md`：方法、数值结果和证据边界。
- `config.json`：阈值、时域、匹配规则。
- `results/run_manifest.json`：来源、数据哈希、版本、原始控制抽查。
- `results/match_sensitivity.csv`：完整历史阈值 × 控制阈值 × 通道 × 采样组统计。
- `results/selected_pairs.csv.gz`：主设置无重叠匹配对及时间边界、历史误差、控制差异。
- `results/effective_events.csv.gz`：主阈值事件时间、幅度与持续时间。
- `results/pairs_{500,1000}ms.parquet`：本机保留的全量宽松历史合格候选对、各阈值判定输入。
- `results/windows_{500,1000}ms.parquet`：上述 pair 索引对应的获准窗口元数据。
- `results/archive_manifest.json`：所有归档文件哈希；大型可复现候选对与窗口文件留在本机、不加入 Git。
- `results/figures/`：分布、匹配质量与数量取舍、航次覆盖、典型匹配图。

`event_counts_per_flight.csv` 的行粒度实际为 flight × 有效连续区间 × channel × threshold；航次总数应按 flight_id、channel、scale 求和，不能把行数当航次数。`selected_*ms.csv` 是各通道独立执行无重叠贪心选取的 pair_index，因此不同通道集合不保证彼此不重叠。分组统计也各自重新选取，不应相加。
