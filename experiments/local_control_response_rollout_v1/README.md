# Frozen local factual-response replay v1

入口：[中文报告](report_zh.md)。基准 `4fecb63ad65b1314227d168ea5457676b46752d7`；目标分支 `exp/local-control-response-rollout-v1`。

固定 AUTO_stable/differential 队列，500 ms 历史、500 ms 名义未来；H1000 Control/NoControl seeds 17/29/43，外加 Hold 和过去 200 ms OLS 趋势。无训练、换控或 test 读取。所有选择复用第一步；背景和敏感性是已保存队列的成员关系。验证主对只有一个。

## 资产与依赖

使用仓库规定的 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`。实际 Python/包版本见 `dependencies.json`；本轮未安装或改动依赖。当前运行配置要求 CUDA，设备 `cuda:0`，记录为 RTX 4090。

数据根目录本次为 `/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1`。仅打开 `samples/windows/native_actuator_motors/native_actuator_servos` 的 train/validation 文件；共享 manifest/flights/provenance/split 元数据不等于 test 轨迹。8 份开发 Parquet 的字节哈希必须符合历史注册表。缺少或不匹配则失败，不能重建、训练或回退。

还需基准分支上的以下原资产（不复制或覆盖）：

- `experiments/local_control_response_scope_v1/` 的固定索引、配置、来源、`archive_manifest.json`。
- `experiments/velocity_rate_gru64_horizon_training_v1/train_1000ms/{GRU-Control,GRU-NoControl}/seed_{17,29,43}/best.pt`。
- `experiments/control_response_eval_v1/results/{run_manifest.json,archive_manifest.json,predictions_500ms.npz,windows_500ms.parquet}`。
- `experiments/control_event_canonicalization_v1/archive_manifest.json` 及 `results/integrated/{windows.parquet,predictions.npz}`。

推理使用仓库既有模型、事件规范化、数据字段和旧响应实验打包器。所有 checkpoints、数据、旧参考和关键代码 SHA256 分别在 `results/input_receipt.json`、`results/preflight.json`、`results/selection_manifest.json`。本地参考核对不是对不可访问 Drive Stage3 CPU 重放的逐位复现，详见报告。

## 复现到新目录

在包含上述资产的新工作树仓库根目录执行。不要删除或覆盖本目录已有结果。`run.py` 会拒绝非空输出目录。

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/local_control_response_rollout_v1
REPLAY_RESULTS=$(mktemp -d /tmp/local-control-replay.XXXXXX)
"$PY" "$EXP/run.py" \
  --data-root /home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1 \
  --output "$REPLAY_RESULTS"
"$PY" "$EXP/analyze.py" --results "$REPLAY_RESULTS"
"$PY" "$EXP/verify.py" --results "$REPLAY_RESULTS"
"$PY" "$EXP/verify_curves.py" --results "$REPLAY_RESULTS"
"$PY" "$EXP/check_inputs.py" --results "$REPLAY_RESULTS" \
  --data-root /home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1
"$PY" "$EXP/figures.py" --results "$REPLAY_RESULTS" --figures "$REPLAY_RESULTS/figures"
"$PY" -m pytest -q "$EXP/test_metrics.py"
```

仅复核现有归档，无需 GPU 推理：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_rollout_v1/archive.py --verify
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_rollout_v1/verify.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_rollout_v1/verify_curves.py
```

CSV/gzip 压缩头和日志时间可能随重跑变化；数值复核比重新生成文件的字节相等更适合跨运行比较。现有归档字节则受 SHA256 保护。跨设备浮点预测可能有微小差异，参考核对容差固定为 2e-5；冗余事件分段一致性在同次运行中要求逐位相等。

## 文件与字段

- `config.json`：固定模型/指标/对齐/近零门限及展示规则；不重新定义第一步筛选。
- `run.py`：来源核验、固定成员打包、旧参考回归、实际未来控制自主推理。
- `analyze.py`：六分量响应、逐时间点指标、曲线指标、依赖性分层。
- `verify.py`：独立括号插值、直接按航次/连通分量求平均，与主实现分离。
- `verify_curves.py`：显式区间积分复核、非重叠与分区检查。
- `check_inputs.py`：改变未来状态/相位/标签，验证八个输入完全不变；记录角速度源时间偏移并复核权重哈希。
- `figures.py`：9 张 PNG、展示实例 ID 和精简表；展示实例不用于改变评价队列。
- `test_metrics.py`：插值不外推、近零、符号、抵消、权重、种子汇总与 A/B 对换等 8 项测试。
- `results/{windows,pairs,cohorts}.csv`：`window_id/flight_id/split`、`pair_uid=split:pair_id`、A/B rollout 下标及每个固定队列成员关系。用这三表追溯所有结果；不同 split 的数字 pair_id 不能单独当全局键。
- `results/native_rollouts.npz`：原八项模型输入，13 维原生状态真值/初值与各模型预测，微秒时间轴及 future_count；变长尾部填充不得作有效样本。
- `results/response_curves.csv.gz`：逐对、逐种子、六分量、20 ms 网格与共同端点的 real/pred/error；A/B 各自真实/预测增量、增量误差和绝对状态误差。`supported=false` 行有意保留缺失值，不能填补或外推。
- `results/point_responses.csv.gz`：100/200/300/400/500 ms 和 `supported_endpoint`。
- `results/summary.csv`：完整端点/时刻汇总；`seed=pooled` 平均误差而非集成预测；`weighting=pair/flight/component`。计数 n 包含种子重复，`pairs`/`supported_pairs` 才是对数。
- `results/curve_metrics.csv`、`curve_summary.csv`：完整支持时域 p 响应 RMSE、幅度、绝对峰值时间及曲线相关；峰值与相关中位数是 pair-seed 描述统计。
- `results/{per_group,leave_one_flight_out,direction_sensitivity,cohort_dependencies}.csv`：逐航次/日期/配置/依赖分量、移除单航次、近零门限及独立性限制。
- `results/{time_support,rate_clock_audit,packing_checks}.csv`：真实端点支持、源角速度时钟和起点差异。
- `results/{main_p_time_seed_table,endpoint_table,curve_table,pair_endpoint_mse}.csv`：报告可直接使用的精简表。
- `results/{preflight,input_causality_check,independent_verification,independent_curve_verification}.json`：可审计验证记录；相应 `*.log` 保存执行输出。
- `archive_manifest.json`：本实验全部受管产物的 SHA256，排除自身和缓存；`archive.py --verify` 只核对本归档，不打开任何数据集。

完整曲线包括原生记录在评价网格上的插值，不是未来真值输入。TARGET 起点和因果 initial 的差异可能放大响应增量误差；`causal_real`/`causal_origin_rmse` 仅作另一锚定的诊断，不替换主定义。Hold 的响应差恒为零，方向指标零表示弃判。图中的近零阴影是分析门限，不能解释为传感器噪声带。
