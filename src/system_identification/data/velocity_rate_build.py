"""Builder, audit, provenance, and Chinese report for velocity_rate_multistep_v1."""
from __future__ import annotations

from collections import Counter
import json
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from system_identification.data.velocity_rate_dataset import (
    STATE, CONTROL, TARGET, PHASE, FIELDS, file_hash, write_json, flight_date,
    extract_flight, build_windows, interval_overlap, quantiles, VelocityRateDataset,
)


def prepare_split(config, repository, source):
    frames = {name: pd.read_csv(source / path) for name, path in config["source_files"].items()}
    raw = frames["flights"]
    for _, group in raw.groupby(["sha256", "flight_index"]):
        if group[["start_us", "end_us"]].drop_duplicates().shape[0] != 1:
            raise ValueError("duplicate flight identities have conflicting reviewed boundaries")
    flights = raw.drop_duplicates(["sha256", "flight_index"]).copy()
    if len(flights) != config["expected_flights"]:
        raise ValueError("reviewed flight count changed; configuration must be versioned")
    flights["flight_id"] = flights.sha256 + ":" + flights.flight_index.astype(str)
    flights["flight_date"] = flights.log_path.map(flight_date)
    owners = {}
    for part, dates in config["split_dates"].items():
        for date in dates:
            if date in owners:
                raise ValueError("date split overlap")
            owners[date] = part
    if not set(flights.flight_date) <= set(owners):
        raise ValueError("an approved flight date has no frozen split")
    flights["split"] = flights.flight_date.map(owners)
    if flights.groupby("sha256").split.nunique().max() != 1:
        raise ValueError("same raw log in multiple splits")
    inherited = {}
    for rel in config["inherit_split_files"]:
        path = repository / rel
        data = yaml.safe_load(path.read_text())
        inherited[rel] = {"sha256": file_hash(path), "content": data}
        if "partitions" in data:
            for old_split, paths in data["partitions"].items():
                for path in paths:
                    expected = "test" if old_split == "sealed_test" else old_split
                    if owners.get(flight_date(path)) != expected:
                        raise ValueError("old frozen split identity was reassigned")
        else:
            for path in data["sealed_test"] + data["reserved_day19"]:
                if owners.get(flight_date(path)) != "test":
                    raise ValueError("reserved log/date reassigned")
    for date in config["frozen_test_dates"]:
        if owners[date] != "test":
            raise ValueError("frozen test date assigned to development")
    for name in ("exclude", "review"):
        frame = frames[name]
        unknown = set(zip(frame.sha256, frame.flight_index)) - set(zip(flights.sha256, flights.flight_index))
        if unknown or (frame.end_us < frame.start_us).any():
            raise ValueError(f"invalid or unmatched {name} intervals")
    manifest = dict(dataset_id=config["dataset_id"], frozen_before_extraction=True,
                    policy="whole date, whole log, whole physical flight; recovered logs inherit date",
                    dates=config["split_dates"], inherited=inherited,
                    test_materialization="user-authorized schema/quality QA only",
                    test_prior_exposure="September reserved dates were previously inspected; not claimed never seen",
                    assignments=flights[["flight_id", "sha256", "flight_index", "log_path",
                                         "flight_date", "split"]].to_dict("records"))
    return flights.sort_values(["flight_date", "log_path", "flight_index"]), frames, manifest


def merge_parquets(paths, destination):
    """One row group per flight: filters in the loader do not load an entire split."""
    writer = None
    try:
        for path in paths:
            frame = pd.read_parquet(path)
            if not len(frame):
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                schema = table.schema
            writer.write_table(table.cast(schema))
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no rows for required artifact: {destination.name}")


def check_frames(samples, windows, record, config):
    """Vectorized exhaustive QA; no metrics or model selection."""
    t = samples.state_measurement_us.to_numpy()
    pub = samples.timestamp_us.to_numpy()
    if not ((np.diff(t) > 0).all() and (np.diff(pub) > 0).all()):
        raise AssertionError("state order")
    for prefix in ("attitude", "rate", "motor", "servo", "encoder", "frequency"):
        source = samples[f"{prefix}_publication_us"].to_numpy()
        if not (source[source >= 0] <= pub[source >= 0]).all():
            raise AssertionError(f"history publication leakage: {prefix}")
    if not len(windows):
        return
    a = windows.start_sample.to_numpy(dtype=int)
    o = windows.origin_sample.to_numpy(dtype=int)
    b = windows.end_sample.to_numpy(dtype=int)
    seg = samples.segment_id.to_numpy()
    assert (seg[a] >= 0).all() and (seg[a] == seg[b]).all()
    assert (windows.history_count.to_numpy() == o-a+1).all()
    assert (windows.future_count.to_numpy() == b-o).all()
    assert (t[o] - t[a] >= windows.history_us).all()
    assert (t[o+1] > pub[o]).all()
    assert np.allclose((t[b]-t[o])*1e-6, windows.actual_horizon_s, atol=1e-12)
    assert np.all(np.abs(t[b]-t[o]-windows.horizon_us)
                  <= config["groups"][record["sampling_group"]]["horizon_tolerance_us"])
    intervals = np.array(record["exclude_intervals_us"]).reshape(-1, 2)
    assert not interval_overlap(windows.support_start_us, windows.support_end_us, intervals).any()
    assert (windows.support_start_us >= record["start_us"]).all()
    assert (windows.support_end_us <= record["end_us"]).all()
    # Cumulative masks check every intermediate row, not only window endpoints.
    invalid = np.r_[0, np.cumsum(~samples.valid_core.to_numpy())]
    assert ((invalid[b+1]-invalid[a]) == 0).all()
    for name in ("target_attitude_measurement_us", "target_rate_measurement_us"):
        bad_edges = np.r_[0, np.diff(samples[name]) <= 0]
        count = np.r_[0, np.cumsum(bad_edges)]
        assert ((count[b+1]-count[o+1]) == 0).all()


def coverage(samples, windows):
    counts = []
    for (group, history, horizon), w in windows.groupby(["sampling_group", "history_us", "horizon_us"]):
        counts.append(dict(sampling_group=group, history_us=int(history), horizon_us=int(horizon),
                           windows=len(w), flights=int(w.flight_id.nunique()),
                           future_count_quantiles=quantiles(w.future_count),
                           actual_horizon_s_quantiles=quantiles(w.actual_horizon_s),
                           review_windows=int(w.review.sum())))
    valid = samples.valid_core.to_numpy()
    dt = samples.dt_measurement_s.to_numpy()
    same = (samples.flight_id == samples.flight_id.shift()) & (samples.segment_id == samples.segment_id.shift())
    edge = valid & samples.valid_core.shift(fill_value=False).to_numpy() & same.to_numpy()
    duration = float(np.nansum(dt[edge]))
    controls = {}
    for col in CONTROL:
        x = samples.loc[samples.valid_control, col].to_numpy()
        controls[col] = dict(quantiles=quantiles(x), saturation_fraction=float(np.mean(
            (x >= .99) | ((x <= .01) if col == CONTROL[0] else (x <= -.99)))) if len(x) else None)
    union = np.zeros(len(samples), dtype=bool)
    primary = windows.loc[windows.horizon_us.isin([100000, 200000])]
    offsets = {}
    for fid, s in samples.groupby("flight_id", sort=False):
        offsets[fid] = int(s.index[0])
    delta = np.zeros(len(samples)+1, dtype=np.int64)
    for fid, w in primary.groupby("flight_id", sort=False):
        offset = offsets[fid]
        np.add.at(delta, offset + w.start_sample.to_numpy(int), 1)
        np.add.at(delta, offset + w.end_sample.to_numpy(int)+1, -1)
    union = np.cumsum(delta[:-1]) > 0
    return dict(samples=len(samples), valid_samples=int(valid.sum()),
                valid_sample_fraction=float(valid.mean()), continuous_valid_duration_s=duration,
                primary_window_union_duration_s=float(np.nansum(dt[edge & union])),
                flights=int(samples.flight_id.nunique()), flights_with_windows=int(windows.flight_id.nunique()),
                windows=len(windows), windows_by_task=counts,
                invalid_counts_overlapping={k: int((~samples[k]).sum()) for k in
                                           ["valid_history_state", "valid_control", "valid_target_state",
                                            "valid_encoder", "valid_frequency"]},
                excluded_rows=int(samples.excluded.sum()), control_gap_rows=int(samples.control_gap.sum()),
                review_rows=int(samples.review.sum()), controls=controls,
                frequency_hz_quantiles=quantiles(samples.loc[samples.valid_frequency, "flap_frequency_hz"]),
                speed_m_s_quantiles=quantiles(np.linalg.norm(samples[STATE[3:6]], axis=1)),
                rate_rad_s_quantiles=quantiles(np.linalg.norm(samples[STATE[10:13]], axis=1)),
                nav_state_counts={str(k): int(v) for k, v in samples.nav_state.value_counts().items()},
                mode_change_rows=int(samples.mode_change.sum()), high_dynamics_rows=int(samples.high_dynamics.sum()))


def native_control_coverage(output, part):
    result = {}
    for topic, fields, names in (
        ("actuator_motors", ["control[0]"], ["motor"]),
        ("actuator_servos", ["control[0]", "control[1]", "control[2]"],
         ["left_elevon", "right_elevon", "rudder"]),
    ):
        data = pd.read_parquet(output / f"native_{topic}_{part}.parquet")
        data = data.loc[data.in_reviewed_flight].copy()
        same = (data.flight_id == data.flight_id.shift()).to_numpy()
        for field, name in zip(fields, names):
            value = data[field].to_numpy(dtype=float)
            finite = np.isfinite(value)
            difference = np.r_[np.nan, np.diff(value)]
            edges = same & finite & np.r_[False, finite[:-1]]
            edge_dt = np.r_[np.nan, np.diff(data.timestamp_us)]*1e-6
            saturation = (value >= .99) | ((value <= .01) if name == "motor" else (value <= -.99))
            result[name] = dict(
                native_rows=len(data), finite_rows=int(finite.sum()),
                finite_fraction=float(finite.mean()), value_quantiles=quantiles(value),
                saturation_fraction=float(saturation[finite].mean()),
                changed_edge_fraction=float((np.abs(difference[edges]) > 1e-5).mean()),
                absolute_change_quantiles=quantiles(np.abs(difference[edges])),
                recorded_dt_s_quantiles=quantiles(edge_dt[edges]),
                note="recorded allocated commands, not all unlogged actuator updates or measured angles")
    return result


def report_zh(output, quality, manifest):
    lines = [
        "# 速度与角速度多步预测数据集 v1",
        "",
        "独立原生时间数据集；无网络、无训练、无 MPC，无加速度/角加速度标签。",
        "监督参考是 PX4 EKF 输出，不是独立物理真值。没有修改 EKF 观测使其满足离散积分器。",
        "",
        f"经审阅清单去重后 **{quality['approved_flights']}** 个飞行片段，"
        f"总审阅时长 **{quality['reviewed_duration_s']/3600:.3f} 小时**。"
        f"50Hz / 10Hz 分组：{quality['sampling_groups']}。",
        "",
        "| 划分 | 飞行数 / 有窗飞行数 | 连续可用时长(s) | 原生样本 | 可用样本比例 | 窗口数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for part, q in quality["partitions"].items():
        lines.append(f"| {part} | {q['flights']} / {q['flights_with_windows']} | "
                     f"{q['continuous_valid_duration_s']:.2f} | {q['samples']} | "
                     f"{q['valid_sample_fraction']:.2%} | {q['windows']} |")
    lines += [
        "",
        "上表时长只累计同一连续有效段内相邻原生状态间隔，不累加重叠窗口。"
        "丢失比例必须区分样本无效、窗口边缘、EXCLUDE、复位和采样不完整；"
        "逐飞行与逐划分计数见 quality_summary.json，重叠无效原因不可相加。",
        "",
        "## 100～200ms 序列完整性",
        "",
        "| 划分 | 组 | 历史(ms) | 目标(ms) | 窗口 | 未来步数 min/median/max | 实际时域 min/median/max(ms) |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for part, q in quality["partitions"].items():
        for r in q["windows_by_task"]:
            if r["horizon_us"] not in [100000, 200000]:
                continue
            steps = r["future_count_quantiles"]
            horizon = r["actual_horizon_s_quantiles"]
            lines.append(f"| {part} | {r['sampling_group']} | {r['history_us']/1000:g} | "
                         f"{r['horizon_us']/1000:g} | {r['windows']} | "
                         f"{steps[0]:g}/{steps[2]:g}/{steps[4]:g} | "
                         f"{horizon[0]*1000:.2f}/{horizon[2]*1000:.2f}/{horizon[4]*1000:.2f} |")
    lines += [
        "",
        "每个保留窗口有完整历史、初始状态、每一步原生 dt、四路控制和完整 p/v/q/ω 目标；"
        "无跨日志、复位、状态缺口或 EXCLUDE。目标姿态/角速度由最近测量时刻原生观测配对，"
        "没有插值；重复使用同一姿态或角速度观测的状态行不能进入完整窗口。"
        "50Hz 配对容差 12ms、10Hz 容差 30ms，实际偏移保存在每行中。"
        "低频组不能称为 20ms 自主递推验证：100ms 通常一步，200ms 通常两步。",
        "",
        "## 时间、相位与控制契约",
        "",
        "- 递推 dt 来自 vehicle_local_position.timestamp_sample 的真实差值；"
        "决策时间是该消息 timestamp。历史 q/ω/动作按发布时间向后保持，保存年龄与测量时间。"
        "初始 q/ω 可能早于初始 p/v；这是观测异步性，不隐式插值或用目标反填初态。",
        "- 未来 p/v 原样保存；未来 q/ω 的离线配对只在 targets 命名空间。"
        "输入/目标测量时刻和发布时间均可追溯。四元数 Hamilton wxyz，FRD→NED，ω 为 FRD rad/s；"
        "v_B=R_NBᵀ v_N，转换时使用单位化旋转副本，原始四元数不改写。",
        "- 17 个已标识固件的相关消息定义、AS5600 驱动、WingPhase 源文件逐提交审计，内容一致。"
        "恢复日志 target_1723_reconstructed.ulg 没有 ver_sw，不借用其他日志固件身份；"
        "其参数、字段和时间质量可检查，但 firmware_verified=false，保留为 provenance REVIEW；"
        "依赖固件发布上界证明的历史编码器相位/频率掩码为 false，原始数据与未来目标仍保留。",
        "- wing_phase.timestamp 回填编码器时刻，并非模块发布时刻。历史输入禁止使用其值。"
        "encoder_count 与 flap_frequency 的驱动 timestamp 是采集后、发布前的时间；"
        "以同一串行驱动下一条已记录消息的时间作为前一条已发布的保守上界，"
        "只在上界不晚于当前决策时使用前一条值。没有可用上界的末条不可作历史输入。"
        "原生发布时间未直接记录的事实仍明确保留，绝不把上界称为实测发布时间。",
        "- 相对相位由整条日志第一个编码器计数为基准计算，保留计数、基准时间、未缠绕相位和 sin/cos。"
        "窗口不重新归零；原始 wing_phase 和 Hall 原生记录保留用于目标/审计。"
        "所有日志 FLAP_RATIO≈7.90909；本实验依日志参数，不套用四月 ratio8 修正，"
        "这也不等于独立测量验证了真实硬件比或跨日志绝对翼相位零点。",
        "- 控制顺序固定为 motor / left_elevon / right_elevon / rudder，来自 "
        "actuator_motors[0]、actuator_servos[0:3]；是分配器发布的归一化命令，非舵角实测。"
        "PWM 功能、反向与限幅参数逐日志保存。u[k] 只取发布时间≤测量 t[k] 的命令，"
        "区间内其余已记录命令作为明确提供的条件回放事件带，绝不取未来区间均值冒充 u[k]。",
        "- 记录受日志采样率约束；不能证明日志保留了每次硬件命令更新，"
        "也不声称改变控制后的反事实动作响应已验证。模式切换与 |ω|>2rad/s 高动态保留标记。",
        "",
        "## 划分与输入输出",
        "",
        "日期、日志、物理飞行整体划分；9月8日、9月19日保留测试身份，新增10月13日测试；"
        "9月7日/17日及新增9月26日/10月11日验证，其余明确列于配置。恢复日志按所属日期绑定，"
        "逐原生状态指纹和 boot 身份检查跨集合重复；只读已审阅清单指定日志。"
        "测试数据按本次授权做构建/结构 QA，不训练、不据此调整阈值。不能称历史测试从未被查看。",
        "",
        "```python",
        "from system_identification.data.velocity_rate_dataset import VelocityRateDataset",
        "ds = VelocityRateDataset('dataset/velocity_rate_multistep_v1',",
        "    split='train', sampling_group='main_50hz', history_us=500000, horizon_us=200000)",
        "batch = ds[0]",
        "x = batch['inputs']    # history [H,20], masks, ages, initial_state [13], future dt/control",
        "y = batch['targets']   # state [K,13], masks, next v_N/omega_B, phase/frequency targets",
        "```",
        "",
        "状态顺序：p_N(3)、v_N(3)、q_NB(wxyz,4)、ω_B(3)。history 另加四路控制、"
        "相对相位 sin/cos 和频率。联合或解耦模型均可切片这些通道。长度可变，"
        "下游自行按真实长度分桶或 padding+mask。只把 inputs 送入模型；未来真实状态、相位、频率"
        "只在 targets。未来 dt 是本记录的采样计划，不是模型状态特征。",
        "递推后续状态须由模型预测反馈；本数据 API 不提供 teacher-forcing 输入。"
        "多步损失可对 targets.state 全序列的 p/v/R/ω 计算，具体网络、积分器和权重尚未固定。",
        "归一化仅由 train 的有效原生样本计算并保存 normalization.json；"
        "没有利用验证或测试数据拟合。所有 Parquet 本地保留，源码、配置、摘要与哈希可提交。",
        "",
        "扩展 500ms/1s/2s 的分组窗口统计、控制范围/饱和、频率和飞行工况覆盖见 quality_summary.json。"
        "已保存完整原生控制事件，窗口索引不复制数百万段重叠序列；加载器按 flight_id 惰性读取。",
        "native 表额外保存至多两条前置、一条后置边界记录，仅供来源核对，"
        "in_reviewed_flight=false；所有训练窗口的完整信号支持区间仍须位于审阅飞行边界内。",
    ]
    lines += [
        "",
        "## 数据损失与控制覆盖",
        "",
        "| 划分 | 连续时长损失 | 有效控制样本 | 有效历史编码器相位 | 有效历史频率 | 扑翼频率 min/median/max(Hz) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for part, q in quality["partitions"].items():
        n = q["samples"]
        bad = q["invalid_counts_overlapping"]
        freq = q["frequency_hz_quantiles"]
        lines.append(f"| {part} | {q['unusable_continuous_duration_fraction']:.3%} | "
                     f"{1-bad['valid_control']/n:.3%} | {1-bad['valid_encoder']/n:.3%} | "
                     f"{1-bad['valid_frequency']/n:.3%} | {freq[0]:.3f}/{freq[2]:.3f}/{freq[4]:.3f} |")
    totals = Counter()
    for row in quality["window_rejections"]:
        totals.update({k: v for k, v in row.items() if k != "flight_id"})
    lines += [
        "",
        f"共评估 {totals['candidate']:,} 个候选窗口，保留 "
        f"{sum(q['windows'] for q in quality['partitions'].values()):,} 个。"
        f"按互斥的拒绝顺序：边缘/时域支持不足 {totals['duration_or_edge']:,}；"
        f"下一测量尚未晚于决策 {totals['next_target_not_future_at_decision']:,}；"
        f"EXCLUDE {totals['exclude']:,}；复位 {totals['reset']:,}；"
        f"其他无效/缺口/重复监督观测 {totals['invalid_or_gap_or_nonunique_target']:,}。",
        "",
        "| 划分 | 发布控制 | 原生记录数 | 有限值覆盖 | min / median / max | 饱和/端点比例 | 相邻记录变化比例 |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for part, q in quality["partitions"].items():
        for name, c in q["native_control_coverage"].items():
            v = c["value_quantiles"]
            lines.append(f"| {part} | {name} | {c['native_rows']} | {c['finite_fraction']:.2%} | "
                         f"{v[0]:.3f}/{v[2]:.3f}/{v[4]:.3f} | {c['saturation_fraction']:.2%} | "
                         f"{c['changed_edge_fraction']:.2%} |")
    lines += [
        "",
        "饱和/端点阈值：电机 ≤0.01 或 ≥0.99，舵面 |u|≥0.99；变化阈值 |Δu|>1e-5。"
        "这些是覆盖统计，不是模型激励充分性或可辨识性证明。原生控制记录间隔分位数、"
        "状态速度/角速度范围、导航模式计数与高动态标记见 JSON，保留极端值及质量标记。",
        "",
        "## 重建与测试",
        "",
        "环境固定为 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，依赖版本记录于 manifest。"
        "先运行 `scripts/audit_velocity_rate_firmware.py --inventory <flight_provenance.json> "
        "--output <源码快照目录> --fetch-missing`，也可直接复用交付的 firmware_sources。"
        "然后运行 `scripts/build_velocity_rate_dataset.py --firmware-sources "
        "<源码快照目录> --output <新的空输出路径>`。默认配置位于 "
        "`configs/data/velocity_rate_multistep_v1.yaml`。已有非空版本拒绝覆盖。",
        "验证命令：`python -m pytest -q tests/test_velocity_rate_dataset.py`。"
        "构建器对所有窗口执行结构/因果/EXCLUDE 校验，并对 60 个划分×采样组×历史×时域组合加载抽查。"
        "测试包含未来真实状态/相位/频率篡改不改变 inputs 的负向泄漏检查。",
    ]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n")


def build(repository, config_path, destination, firmware_sources):
    config = yaml.safe_load(config_path.read_text())
    config["split_dates"] = {p: [str(d) for d in days] for p, days in config["split_dates"].items()}
    config["frozen_test_dates"] = [str(d) for d in config["frozen_test_dates"]]
    output = (destination or repository / config["output_root"]).resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite existing version: {output}")
    staging = output.with_name("." + output.name + ".building")
    staging.mkdir(parents=True, exist_ok=True)
    source = Path(config["source_root"])
    source_hashes = {name: file_hash(source / rel) for name, rel in config["source_files"].items()}
    if source_hashes != config["source_sha256"]:
        raise ValueError("reviewed source CSV hashes changed; refuse silent dataset drift")
    code_paths = [Path(__file__), Path(__file__).with_name("velocity_rate_dataset.py"),
                  repository / "scripts/build_velocity_rate_dataset.py",
                  repository / "scripts/audit_velocity_rate_firmware.py",
                  repository / "scripts/verify_velocity_rate_dataset.py",
                  repository / "src/system_identification/data/trajectory_dataset.py"]
    build_id = {str(p.relative_to(repository)): file_hash(p) for p in code_paths}
    identity = dict(config_hash=file_hash(config_path), sources=source_hashes, code=build_id)
    marker = staging / "build_identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("staging build belongs to different inputs/code; use a new staging directory")
    write_json(marker, identity)
    flights, inputs, split = prepare_split(config, repository, source)
    write_json(staging / "split_manifest.json", split)  # frozen BEFORE state extraction
    shutil.copy2(config_path, staging / "build_config.yaml")
    (staging / "source_inputs").mkdir(exist_ok=True)
    for key, rel in config["source_files"].items():
        shutil.copy2(source / rel, staging / "source_inputs" / Path(rel).name)
    firmware_records = json.loads((firmware_sources / "source_manifest.json").read_text())
    for r in firmware_records:
        if r["status"] != "verified_exact_commit":
            raise ValueError(f"missing exact firmware source: {r}")
        if file_hash(firmware_sources / r["commit"] / r["source"]) != r["sha256"]:
            raise ValueError("firmware snapshot hash mismatch")
        if config["verified_firmware_source_sha256"].get(r["source"]) != r["sha256"]:
            raise ValueError("firmware timing or message contract changed; requires explicit audit")
    for commit in {r["commit"] for r in firmware_records}:
        sources = {r["source"] for r in firmware_records if r["commit"] == commit}
        if sources != set(config["verified_firmware_source_sha256"]):
            raise ValueError("incomplete firmware source audit")
    shutil.copytree(firmware_sources, staging / "firmware_sources", dirs_exist_ok=True)
    verified = {r["commit"] for r in firmware_records}
    records, rejected = [], []
    caches = staging / ".cache"
    caches.mkdir(exist_ok=True)
    for n, row in enumerate(flights.itertuples(), start=1):
        cache = caches / row.flight_id.replace(":", "_")
        done = cache / "record.json"
        if done.exists():
            record = json.loads(done.read_text())
            count = json.loads((cache / "rejections.json").read_text())
        else:
            cache.mkdir(exist_ok=True)
            intervals = {}
            for name in ("exclude", "review"):
                f = inputs[name]
                intervals[name] = f.loc[(f.sha256 == row.sha256) & (f.flight_index == row.flight_index),
                                        ["start_us", "end_us"]].to_numpy(np.int64)
            samples, native, record = extract_flight(row, config, intervals["exclude"], intervals["review"])
            record["firmware_verified"] = record["firmware"] in verified
            if record["firmware"] and not record["firmware_verified"]:
                raise ValueError(f"unverified identified firmware: {record['firmware']}")
            samples["firmware_verified"] = record["firmware_verified"]
            samples["provenance_review"] = not record["firmware_verified"]
            windows, count = build_windows(samples, record, config)
            if not len(windows):
                raise ValueError(f"flight has no supported windows: {row.log_path}")
            windows["firmware_verified"] = record["firmware_verified"]
            windows["provenance_review"] = not record["firmware_verified"]
            check_frames(samples, windows, record, config)
            samples.to_parquet(cache / "samples.parquet", index=False, compression="zstd")
            windows.to_parquet(cache / "windows.parquet", index=False, compression="zstd")
            for name, frame in native.items():
                frame.to_parquet(cache / f"native_{name}.parquet", index=False, compression="zstd")
            write_json(cache / "rejections.json", count)
            write_json(done, record)
        records.append(record)
        rejected.append(dict(flight_id=row.flight_id, **count))
        print(f"[{n}/{len(flights)}] {row.split} {record['sampling_group']} "
              f"{row.log_path}: {record['valid_samples']}/{record['samples']} valid", flush=True)
    # Physical boot identity and duplicate/recovery evidence never span splits.
    boots = {}
    for r in records:
        if r["boot_time_utc_us"] and r["sys_uuid"]:
            key = (r["sys_uuid"], r["boot_time_utc_us"])
            if key in boots and boots[key] != r["split"]:
                raise ValueError("physical boot identity leaks across partitions")
            boots[key] = r["split"]
    partitions, fingerprints = {}, {}
    for part in ("train", "validation", "test"):
        selected = [caches / r["flight_id"].replace(":", "_") for r in records if r["split"] == part]
        for name in ["samples", "windows", *("native_" + n for n in FIELDS)]:
            merge_parquets([c / f"{name}.parquet" for c in selected], staging / f"{name}_{part}.parquet")
        s = pd.read_parquet(staging / f"samples_{part}.parquet")
        w = pd.read_parquet(staging / f"windows_{part}.parquet")
        fingerprints[part] = np.unique(pd.util.hash_pandas_object(
            s[["state_measurement_us", *STATE[:6]]], index=False).to_numpy())
        partitions[part] = coverage(s, w)
        partitions[part]["native_control_coverage"] = native_control_coverage(staging, part)
        if part == "train":
            columns = STATE + CONTROL + PHASE
            normalization = {}
            for c in columns:
                ok = s.valid_core & np.isfinite(s[c])
                if c in PHASE[:2]:
                    ok &= s.valid_encoder
                if c == PHASE[2]:
                    ok &= s.valid_frequency
                values = s.loc[ok, c].to_numpy(dtype=float)
                if len(values) < 2:
                    raise ValueError(f"no train normalization support: {c}")
                normalization[c] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)),
                                        count=len(values), constant=bool(values.std() < 1e-12))
            write_json(staging / "normalization.json", dict(
                partition="train", fit_scope="unique valid native rows, not overlapping windows",
                columns=normalization, state_columns=STATE, control_columns=CONTROL, phase_columns=PHASE))
        print(f"assembled {part}: {len(s)} rows, {len(w)} windows", flush=True)
    duplicates = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        duplicates[f"{a}:{b}"] = len(np.intersect1d(fingerprints[a], fingerprints[b], assume_unique=True))
    if any(duplicates.values()):
        raise ValueError(f"duplicate native state fingerprints across partitions: {duplicates}")
    basic = [{k: r[k] for k in (
        "flight_id", "sha256", "log_path", "flight_index", "flight_date", "split", "sampling_group",
        "start_us", "end_us", "reviewed_duration_s", "native_rate_hz", "firmware", "firmware_verified",
        "reconstructed_source", "transmission_ratio", "samples", "valid_samples", "phase_origin_count",
        "phase_origin_timestamp_us", "absolute_phase_zero_verified")} for r in records]
    pd.DataFrame(basic).to_csv(staging / "flights.csv", index=False)
    write_json(staging / "flight_provenance.json", records)
    quality = dict(approved_flights=len(flights), reviewed_duration_s=float(flights.duration_s.sum()),
                   sampling_groups=dict(Counter(r["sampling_group"] for r in records)),
                   partitions=partitions, window_rejections=rejected,
                   firmware_identity_missing=[r["log_path"] for r in records if not r["firmware_verified"]],
                   cross_split_duplicate_state_fingerprints=duplicates,
                   qa="exhaustive timestamps/windows/exclusions; loader smoke; no training")
    for part, q in partitions.items():
        duration = sum(r["reviewed_duration_s"] for r in records if r["split"] == part)
        q["reviewed_duration_s"] = duration
        q["unusable_continuous_duration_fraction"] = 1 - q["continuous_valid_duration_s"]/duration
    write_json(staging / "quality_summary.json", quality)
    manifest = dict(
        dataset_id=config["dataset_id"], schema_version=1, config=config,
        source_files={k: {"path": str(source / config["source_files"][k]), "sha256": v}
                      for k, v in source_hashes.items()},
        builder_source_sha256=build_id,
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip(),
        python_executable=sys.executable, state_columns=STATE, control_columns=CONTROL,
        package_versions={p: importlib.metadata.version(p) for p in
                          ["numpy", "pandas", "pyarrow", "pyulog", "PyYAML"]},
        history_columns=STATE + CONTROL + PHASE, target_columns=TARGET,
        frames={"position_velocity": "NED m,m/s", "quaternion": "Hamilton wxyz body FRD to NED",
                "angular_velocity": "body FRD rad/s"},
        alignment={"history": "latest proven-available native record at local-position publication",
                   "target": "native local-position measurement; nearest native q/omega measurement, no interpolation",
                   "dt": "native local-position measurement differences",
                   "encoder_frequency_availability": "next serial-producer timestamp upper bound",
                   "wing_phase": "targets/native audit only; actual publication timestamp unavailable",
                   "test": "dataset construction QA only; preserved historical reservation identity"},
        model_contract={"prediction": "next velocity_ned and angular_velocity_body directly",
                        "state_loss_targets": "entire future p,v,q,omega sequence",
                        "autonomous_feedback": True, "future_truth_input_allowed": False,
                        "network_or_loss_weights_fixed": False, "acceleration_labels": False,
                        "future_control": "recorded command conditional replay; not counterfactual validation"},
        physical_truth=False, control_counterfactual_validated=False,
        artifact_sha256={},
    )
    report_zh(staging, quality, manifest)
    for name, old_hash in source_hashes.items():
        if file_hash(source / config["source_files"][name]) != old_hash:
            raise ValueError("reviewed input changed during build")
    for path in sorted(staging.iterdir()):
        if path.is_file() and path.name not in {"dataset_manifest.json", "build_identity.json"}:
            manifest["artifact_sha256"][path.name] = file_hash(path)
    for dirname in ("source_inputs", "firmware_sources"):
        for path in sorted((staging / dirname).rglob("*")):
            if path.is_file():
                manifest["artifact_sha256"][path.relative_to(staging).as_posix()] = file_hash(path)
    write_json(staging / "dataset_manifest.json", manifest)
    # Dataset-load checks cover every split/group/task, including the sparse cohort.
    checks = []
    for part in ("train", "validation", "test"):
        for group in config["groups"]:
            for h in config["history_us"]:
                for k in config["horizon_us"]:
                    ds = VelocityRateDataset(staging, part, group, h, k, allow_test=True, verify_hashes=False)
                    if not len(ds):
                        raise ValueError(f"missing requested cohort: {part}/{group}/{h}/{k}")
                    for index in sorted({0, len(ds)//2, len(ds)-1}):
                        item = ds[index]
                        x, y = item["inputs"], item["targets"]
                        assert y["state"].shape == (len(x["future_dt_s"]), 13)
                        assert y["state_mask"].all()
                        assert x["future_control"]["step_control_mask"].all()
                        assert x["future_control"]["event_mask"].all()
                        assert np.isfinite(x["initial_state"]).all()
                        assert np.isclose(sum(x["future_dt_s"]), item["metadata"]["actual_horizon_s"])
                    checks.append(dict(split=part, group=group, history_us=h, horizon_us=k, windows=len(ds)))
    write_json(staging / "validation.json", dict(status="passed", loader_cohorts=checks,
               all_windows_validated=True, source_inputs_unchanged=True, duplicate_state_fingerprints=duplicates))
    manifest["artifact_sha256"]["validation.json"] = file_hash(staging / "validation.json")
    write_json(staging / "dataset_manifest.json", manifest)
    shutil.rmtree(caches)
    staging.rename(output)
    print(f"COMPLETE {output}", flush=True)
