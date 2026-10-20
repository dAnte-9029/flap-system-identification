"""Append descriptive observation consistency diagnostics; never filter the data."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from system_identification.data.velocity_rate_dataset import file_hash, write_json
from system_identification.data.velocity_rate_kinematics import endpoint_residual


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    args = p.parse_args()
    root = args.dataset
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    rows = []
    for part in ("train", "validation", "test"):
        for name in ("samples", "windows"):
            filename = f"{name}_{part}.parquet"
            assert file_hash(root / filename) == manifest["artifact_sha256"][filename]
        samples = pd.read_parquet(root / f"samples_{part}.parquet")
        windows = pd.read_parquet(root / f"windows_{part}.parquet")
        windows = windows.loc[(windows.history_us == 500000) & windows.horizon_us.isin([100000, 200000])]
        by_flight = dict(tuple(samples.groupby("flight_id", sort=False)))
        for (fid, horizon), w in windows.groupby(["flight_id", "horizon_us"], sort=False):
            s = by_flight[fid].reset_index(drop=True)
            position, attitude = endpoint_residual(s, w.reset_index(drop=True))
            rows.append(dict(split=part, flight_id=fid, sampling_group=w.sampling_group.iloc[0],
                             horizon_us=int(horizon), windows=len(w),
                             position_endpoint_rmse_m=float(np.sqrt(np.mean(position**2))),
                             attitude_endpoint_rmse_deg=float(np.sqrt(np.mean(attitude**2)))))
    per_flight = pd.DataFrame(rows)
    per_flight.to_csv(root / "kinematic_consistency_per_flight.csv", index=False)
    summary = []
    for (part, group, horizon), frame in per_flight.groupby(["split", "sampling_group", "horizon_us"]):
        r = dict(split=part, sampling_group=group, horizon_us=int(horizon),
                 flights=len(frame), windows=int(frame.windows.sum()))
        for col in ("position_endpoint_rmse_m", "attitude_endpoint_rmse_deg"):
            r[col] = dict(mean=float(frame[col].mean()), sample_std=float(frame[col].std(ddof=1)))
        summary.append(r)
    write_json(root / "kinematic_consistency.json", dict(
        use="observation consistency diagnostic only, not physical trajectory error or a training label",
        integrator="trapezoidal v_N, right-multiplied body-rate midpoint exponential",
        sampling="actual local-position measurement dt, paired q/omega timing offsets retained",
        aggregation="per-flight window endpoint RMSE, across flights mean and sample standard deviation",
        history_us=500000, model_training=False, selection_or_filtering=False, summary=summary))
    report = root / "report_zh.md"
    marker = "\n## 观测内部运动学一致性诊断\n"
    text = report.read_text().split(marker)[0]
    lines = [marker, "",
             "以下只检查原样 EKF 状态序列：以目标初态出发，梯形积分目标 v_N、"
             "中点角速度指数映射积分姿态。使用实际状态测量 dt，姿态/角速度的原生配对时间偏移"
             "仍包含在残差中。这不是模型误差、真实轨迹误差或物理准确度验证；不删数据、不改标签，"
             "也不为后续网络固定积分器。历史固定500ms，统计100/200ms终点；窗口重叠，"
             "先按飞行计算终点RMSE，再报告各飞行的均值 ± 样本标准差。",
             "",
             "| 划分 | 组 | 时域(ms) | 飞行数 | 位置终点残差(m) | 姿态终点残差(deg) |",
             "|---|---|---:|---:|---|---|"]
    for r in summary:
        ep, er = r["position_endpoint_rmse_m"], r["attitude_endpoint_rmse_deg"]
        lines.append(f"| {r['split']} | {r['sampling_group']} | {r['horizon_us']/1000:g} | "
                     f"{r['flights']} | {ep['mean']:.5f} ± {ep['sample_std']:.5f} | "
                     f"{er['mean']:.5f} ± {er['sample_std']:.5f} |")
    lines += ["", "重建后执行 `python scripts/audit_velocity_rate_kinematics.py <数据集目录>` "
              "复现本附录；再执行 `python scripts/verify_velocity_rate_dataset.py <数据集目录>` "
              "进行独立逐行和全窗口核验。"]
    report.write_text(text + "\n".join(lines) + "\n")
    for name in ("kinematic_consistency_per_flight.csv", "kinematic_consistency.json", "report_zh.md"):
        manifest["artifact_sha256"][name] = file_hash(root / name)
    manifest["postbuild_audit_source_sha256"] = {
        str(path.relative_to(ROOT)): file_hash(path) for path in
        [Path(__file__), ROOT / "src/system_identification/data/velocity_rate_kinematics.py"]}
    write_json(root / "dataset_manifest.json", manifest)
    print(f"observation-only diagnostics: {len(per_flight)} flight/horizon records")


if __name__ == "__main__":
    main()
