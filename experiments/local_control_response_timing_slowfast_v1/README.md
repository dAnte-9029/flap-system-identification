# Timing and slow/fast diagnostics on frozen local responses

入口：[中文报告](report_zh.md)。基准 `c6580f7a9369db2de721cfd3e31b339c95d317f2`，分支 `exp/local-control-response-timing-slowfast-v1`。

本实验只后处理固定预测和获准 train/validation 的原生观测。无模型实例、GPU 推理、训练、换控或 test 读取。历史数据集为 `velocity_rate_multistep_v1/main_50hz`；不切换 canonical dataset。全部新增产物位于本目录，旧实验保持不变。

## 固定分析定义

- A：上一轮原位置时间网格与原瞬时起点；原结果先复现，再开始新分析。
- B：原生角速度按测量时间插值到模型名义物理时间，各曲线减去其物理 t0 起点，预测仍原封不动。
- C/D：A/B 的 270 ms 中心矩形平均，保留原各自瞬时锚点；积分整个窗口，不作边界填补或外推。
- `fast_A/fast_B`：原始减去慢分量，不保证正交能量分解，不等同于纯翼拍作用。
- `C_cycle/D_cycle`：从 C/D 减去首个完整窗中心 W/2 的值，比较后续周期均值变化；不是原 t0 响应。
- 270 ms = 训练主窗口历史频率中位数的倒数取到 10 ms；340 ms = 最小历史频率的倒数取到 10 ms，固定敏感性。只按历史频率定参数，不看预测成绩或验证结果。
- 主四口径比较为同一 300 ms、同一冻结队列；方向门限 0.1 rad/s，另保留 0.05/0.2。每种子单列，随后汇总，非集成预测。
- 位置时钟、角速度测量/发布、模型步起点相位与步终点输出区分。编码器/频率没有独立测量 timestamp_sample 时标记未知，用驱动源时间描述，不把 −1 转成虚假年龄。
- 真值插值是离线诊断。模型八项输入及预测未修改。中心平均有效时刻需要未来半窗，不是在线因果估计。

## 输入与环境

使用 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，版本见 `dependencies.json`，不安装依赖。

先核对上一轮 `experiments/local_control_response_rollout_v1/archive_manifest.json` 全部条目。主要复用 `native_rollouts.npz`、`response_curves.csv.gz`、固定 `windows/pairs/cohorts.csv`、时钟审计、指标与运行记录。上游选择仍是 `local_control_response_scope_v1`。

本地数据目录 `/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1`：只打开 `samples`、`native_vehicle_angular_velocity`、`native_encoder_count`、`native_flap_frequency` 的 train/validation Parquet，各自必须符合原 manifest SHA256。只提取固定 225 窗口所在航次及原有完整支持区间，不读取原始 ULog 或封存文件。物理插值不跨 20 ms 原生角速度缺口；模型/原目标网格不跨 40 ms。

注意：部分窗口原预测最后时刻在原生角速度右侧没有观测支持。保留缺失标志，四口径主要用全部支持的 300 ms；端点另保存纯时间戳决定的共同原生端点，最多缩短 4.990 ms，不能直接与原端点汇总混比。

## 在新输出目录复现

在这个分支的仓库根目录运行；变量只改变新结果输出路径，不修改本目录已有归档。基准门禁还会确认冻结 config 与训练历史频率规则一致。

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/local_control_response_timing_slowfast_v1
export LOCAL_DIAG_OUTPUT=$(mktemp -d /tmp/local-control-timing.XXXXXX)
export LOCAL_DIAG_FIGURES="$LOCAL_DIAG_OUTPUT/figures"
"$PY" "$EXP/baseline.py"
"$PY" "$EXP/prepare.py"
"$PY" "$EXP/analyze.py"
"$PY" "$EXP/diagnostics.py"
"$PY" "$EXP/fast_summary.py"
"$PY" "$EXP/verify.py"
"$PY" "$EXP/verify_extra.py"
"$PY" "$EXP/figures.py"
"$PY" -m pytest -q "$EXP/test_diagnostics.py"
```

不设置上述变量时，脚本使用本目录的 `results/figures`。不要在归档目录无意覆盖已有结果。为独立复现创建新输出目录即可，不必删除旧文件。数值复核是主要复现检查；压缩头时间、浮点归约顺序等可能使重跑字节哈希不同。当前归档自身的字节哈希由 `archive_manifest.json` 保护：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/local_control_response_timing_slowfast_v1/archive.py --verify
```

## 代码与结果

| 文件 | 内容 |
|---|---|
| `baseline.py` / `baseline_check.json` | 原归档字节核验；从保存数组复算旧逐点结果和全部 p 汇总；历史频率参数门禁 |
| `prepare.py` / `source_receipt.json` | 获准原生资料的只读提取与输入字段 SHA256 |
| `common.py` | 严格边界插值与分段线性曲线的完整窗积分 |
| `analyze.py` | 所有固定对、种子、p/q/r 的原始、慢变、快残差与 cycle 口径 |
| `diagnostics.py` | 匹配物理端点、方向分类迁移、时钟关联、逐航次移除与峰值汇总 |
| `fast_summary.py` | 同一 300 ms 的快残差完整种子/队列/权重汇总 |
| `verify.py` | 独立解析原函数积分、直接按航次/图分量求平均；不调用主指标函数 |
| `verify_extra.py` | 独立共同端点核对、相位公式、原文件与模型八项输入未变 |
| `test_diagnostics.py` | 10 项边界、信号、起点、分区、只读/无推理及未知时钟检查 |
| `figures.py` | 7 张 PNG；实例固定为 124/336/412、验证44及上一轮中位数105/183，未新挑选成功对 |

`results/` 中：

- `origin_clock_audit.csv`、`future_clock_audit.csv`、`pair_clock_audit.csv`：时间戳、源样本、各起点值与配对偏差；`alignment_decomposition.csv` 将参考变化拆为起点项和未来取样项。
- `native_diagnostic_inputs.npz`、`native_rate_extract.csv.gz`：本轮所用原生 p/q/r 及实际测量时间；每窗相对位置 t0，原始绝对时刻和发布时刻也保留。原模型预测仍引用上游包，不重复推理。
- `phase_forecast_audit.csv`、`frequency_audit.csv`：同航次相对编码器进度、原模型固定频率相位外推、源时间年龄及测量时刻是否已知；不用于移动预测曲线。
- `time_support.csv`：每窗起点/未来物理支持状态与过滤后的有效中心范围；不静默剔除。
- `responses.csv.gz`：逐对、时间、种子、变量及方法的真实/预测响应；`supported=false` 为缺失，不得补齐。`width_s=0` 为完整原始时域，0.27/0.34 为共同滤波时域。
- `points.csv.gz` / `summary.csv`：200/300 ms 和各时域末端。`domain=full` 与 `common_filter` 的 `endpoint` 不同，不得跨时域直接当同一时刻比较。
- `comparison_300ms.csv`：主窗口、同一 300 ms 的完整 p 汇总。
- `matched_endpoint_points.csv`、`matched_endpoint_summary.csv`：A/B 使用同一原生支持端点，同一固定对，独立于上面的旧端点统计。
- `peak_curve_metrics.csv.gz`、`peak_summary.csv`：逐对原始/慢变/快残差峰值符号、幅度、时间、边界标志及曲线 RMS。峰值汇总为 pair-seed 描述性统计；不据此拟合或修正时移。
- `fast_summary.csv`：p/q/r 快残差在 300 ms 的方向、尺度、增益等，含每个种子和三种权重。
- `direction_transitions.csv`：全分母分类计数。`seed=pooled` 的 `seed_rows` 是种子重复数，不能当独立对数；同时提供 `pairs`。`direction_transition_details.csv` 保留主对 ID。
- `clock_error_associations.csv`、`per_group.csv`、`leave_one_flight_out.csv`：描述性相关、航次/日期/配置/连通分量及单航次影响，均无显著性或置信区间。
- `one_period_repeatability.csv`：在完整约 1 s 历史—未来片段，以固定历史周期估计残差重复性；不是仅未来预测指标，也未搜索最佳滞后。
- `baseline_check.json`、`independent_verification.json`、`extra_verification.json`：复核结果。各脚本对应 `.log`，执行范围见 `execution_record.json`。

源配对身份映射继续使用上游 `results/{windows,pairs,cohorts}.csv`（哈希已登记），`pair_uid=split:pair_id`，不可只以数字 ID 合并分区。权重沿用逐对、航次等权和共享航次连通分量等权，子集不可相加当独立证据。
