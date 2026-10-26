# Frozen local control swap sensitivity v1

基准 `exp/local-control-response-timing-slowfast-v1` / `9c9cb02098a29186922d9dcb061d3776f690605e`。新分支 `exp/local-control-response-swap-sensitivity-v1`。主报告：[report_zh.md](report_zh.md)。

研究对象、索引、模型全部冻结。仅 train/validation，test 封存。没有训练、模型／归一化修改、重新匹配或未来真值／相位输入。C/H 是内部预测差异，不能当作实测控制因果效应。

## 数据与代码

- 历史数据：`/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1`，仅登记的 samples、native_actuator_motors、native_actuator_servos × train/validation。本轮 samples 使用历史输入／时钟列白名单。
- 固定队列与旧数组：`../local_control_response_rollout_v1/results/{windows,pairs,cohorts}.csv`、`native_rollouts.npz`。
- 模型接口：直接导入旧 rollout `run.py` 的 checkpoint 恢复、原八张量输入白名单及 Stage3 `response.predict`；原 `splice_tape/event_pack` 与 `exact_zoh_merge_us_v1`。不另实现模型。
- 训练参考：`../local_control_response_scope_v1/results/train/{candidate_index,command_events}.csv.gz`，仅用于输入支持描述，不改变队列。
- 前轮时钟／事实诊断：`../local_control_response_timing_slowfast_v1/results/`，直接引用保存数据，不重新滤波。
- 六个权重路径／SHA256 见 `results/source_receipt.json`；归一化 SHA256 见 `inference_checks.json`。数据及额外来源见 `support_receipt.json`，环境和依赖源代码见 `execution.json`。

## 复现

无需安装依赖。使用现有 `flap-train-gpu`，已运行 Python 3.11 / Torch 2.7.0+cu128、RTX 4090。必须从本分支 worktree 根目录执行。先阅读适用 AGENTS.md。

为了保护已提交结果，`run.py` 拒绝写入非空输出目录。完整复现用新目录；所有脚本都遵循同一环境变量：

```bash
export LOCAL_SWAP_OUTPUT=/tmp/local-swap-repro/results
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_swap_sensitivity_v1/run.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_swap_sensitivity_v1/support.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_swap_sensitivity_v1/analyze.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_swap_sensitivity_v1/verify.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_swap_sensitivity_v1/plot.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_swap_sensitivity_v1/describe.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q experiments/local_control_response_swap_sensitivity_v1/test_contract.py
```

只检查提交产物不需要 GPU：在未设置 `LOCAL_SWAP_OUTPUT` 的新 shell 中运行 `verify.py` 和 pytest。若在同一 shell 中，先 `unset LOCAL_SWAP_OUTPUT`。分析脚本仅重算指定输出目录的派生产物，不修改任何上游文件。不同批量／GPU 浮点结果允许原先冻结的 2e-5 事实一致性容差；本机全批事实回放实际为 0。

## 文件导航

| 文件 | 内容与键 |
|---|---|
| `config.json` | 固定参数、300 ms 主时间、控制支持边界、方向／分母阈值、支持度容差 |
| `run.py` | 来源检查、六模型事实门禁、452 个四组合冻结推理、相同历史／隐藏态／相位检查 |
| `support.py` | 仅训练参考的粗联合覆盖；历史源时钟提取；输入范围，不根据模型结果筛选 |
| `analyze.py` | 原生时间插值、Y/CA/CB/HA/HB/C/H/F、全分组／种子／权重汇总 |
| `verify.py` | 独立括号插值、图遍历权重归约、原生通道 ZOH 并集检查 |
| `plot.py`, `describe.py` | 全队列图、预先指定病例、输入波形、可复算报告事实；仅引用旧慢变结果 |
| `results/cases.csv` | `case_id` 对应 NPZ 首维；pair_uid、cell、receiver/donor window/flight、split、平移、有效步数 |
| `results/{windows,pairs,cohorts}.csv` | 原冻结索引，不重选；cohort 为多重成员关系 |
| `results/inputs.npz` | 原八个模型张量 + valid_steps/receiver_index/native_time_us；只读来源历史，无未来真值 |
| `results/predictions.npz` | 六个 `GRU-Control_17` 等键，形状 `[452,K,13]`；状态为位置、NED v、四元数、FRD pqr |
| `results/window_points.csv.gz` | 四组合各自的六维运动增量；case_id、pair、cell、种子、时间、支持标记 |
| `results/decomposition_curves.csv.gz` | 全 20 ms 曲线与共同受支持端点，六个输出，所有分解项及原事实 real_F |
| `results/point_decomposition.csv.gz` | 100/200/300/400/500 ms 与端点；500 ms 明确保留 NaN/unsupported |
| `results/control_differences.csv` | 真正施加的四坐标差的精确时间均值／RMS／持续时间，gA/gB/g |
| `results/summary.csv`, `summary_300ms.json` | 每个 cohort × model × seed/pooled × output × time × weighting；方向、近零、规模、增益、抵消 |
| `results/direction_sensitivity.csv` | p 的 0.05/0.1/0.2 阈值、有效数量与弃判比例 |
| `results/per_group.csv` | 300 ms p：逐航次／日期／控制配置／依赖分量；各种子及 pooled |
| `results/leave_one_flight_out.csv` | 训练主队列逐航次移除的描述性敏感性，不是置信区间 |
| `results/seed_consistency.csv` | 各对、输出、时间、效应的三种子符号与幅值范围 |
| `results/all_case_diagnostics.csv` | 全部主／背景对 300 ms p 的近零、反号、历史主导、事实错误标记，未只保留成功案例 |
| `results/covariate_summary.csv` | 训练主对输入差幅值／持续时间／历史匹配质量的固定分箱，不对验证调参 |
| `results/swap_support.csv` | 每组合的训练范围、粗联合均值覆盖、未知／警告标记，不用于排除 |
| `results/training_reference.csv.gz` | 支持检查训练参考的固定 candidate/window/flight 与四坐标未来均值 |
| `results/state_phase_clock_audit.csv` | 测量、发布、源时钟、年龄及已知／未知标记，附旧起点 p 诊断 |
| `results/{time_contract,step_contract,time_support}.csv` | 各窗口时钟／原生步／相位及各对共同有效末端 |
| `results/prior_factual_context_300ms.csv` | 从前轮直接提取的 A/B/D 旁证，不是新滤波、无交换真值 |
| `results/{source_receipt,support_receipt,inference_checks,independent_verification,report_facts}.json` | 来源、实际检查和可复算报告事实 |
| `figures/*.png` | 10 张 PNG，四个指定病例每个另有完整四组合 p/q/r 图 |
| `*.log`, `execution.json`, `archive_manifest.json` | 实际命令／依赖／验证及全部产物 SHA256 |

## 解释约束

- `native_time_us` 相对各 receiver 位置测量 t0，不是相对决策；控制按决策对齐。保留 receiver 的 t0→decision 原控制前缀。
- 同 receiver 的 own/swap 使用相同原生 dt。末步超过 decision+500 ms 时裁掉该完整步，不能缩短步长或外推 donor 命令。NPZ 尾部 dt=0 的 padding 输出不是有效状态，只读取 `valid_steps` 以内；零时长事件也不得用来计算控制范围。
- `supported_endpoint` 是各对不同的共同端点。`window_points` 的单窗口支持和 `point_decomposition` 的四组合共同支持含义不同；端点比较请用后者。
- `seed=pooled` 合并种子结果，不先平均预测再算指标，不把三个种子当统计独立证据。
- 权重按有效对重新计算。双有效方向条件为 `abs(CA)>floor & abs(CB)>floor`；`both_active_n` 是对×种子记录数，`both_active_pairs` 是不同对数。NoControl 无双有效时方向率为 NaN。
- 增益分母是实际应用的有符号差动均值，abs<0.02 弃判；方向阈值不同于增益分母保护。四路波形均保留，增益没有独立物理控制意义。
- C/H 尺度分数不是方差解释率；F 与 real_F 的比较才是观察性事实评价，绝不把 real_F 当 CA/CB 的真实交换标签。
- 粗支持检查用完整 donor 500 ms 命令均值，不是完整时间波形或相位条件支持。边际范围、条件均值警告与因果支持是不同层次。

固定分箱定义位于 `analyze.py`：差动 RMS 0/0.1/0.2/0.3/∞，持续时间 0/100/200/300 ms，历史角速率 RMS 差 0/0.25/0.5/0.75/∞、姿态差 0/5/10/15/∞。这些仅为训练描述，不用于选择、调参或验证病例优化。
