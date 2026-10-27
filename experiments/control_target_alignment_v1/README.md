# 动力学训练目标与控制响应参考的一致性审计

入口：[中文报告](report_zh.md)、[时间合同](time_contract_zh.md)。本实验训练次数 **0**，无模型加载、GPU推理、Loss修改或test读取。

基准 `d8181c294bb4f5af628822ff21d6b6d1ed005c2b`，分支 `exp/control-target-alignment-v1`。全部新文件位于此目录，旧归档保持不变。

## 执行

在独立复现副本中执行以下命令；归档目录内的同名结果会被脚本覆盖，不要在需要保留原归档时直接重跑。可复制本实验目录到同一仓库 `experiments/` 下的新名称，脚本按自身路径解析输出。

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
E=experiments/control_target_alignment_v1
"$PY" "$E/prepare.py"
"$PY" "$E/oracles.py"
"$PY" "$E/full_audit.py"
"$PY" "$E/compare_models.py"
"$PY" "$E/describe.py"
"$PY" "$E/extra.py"
"$PY" "$E/verify.py"
"$PY" -m pytest -q "$E/test_audit.py"
"$PY" "$E/plot.py"
"$PY" "$E/report.py"
"$PY" "$E/archive.py"
```

环境使用现有 flap-train-gpu，不安装依赖。CPU按航次矢量化全量处理，float64评价，保留原float32训练缓存的精确表示。只读取明确的 train/validation 文件；不遍历或哈希封存数据。来源哈希在 `results/source_receipt.json`。

## 数据与结果索引

- `oracles.py`：复用冻结插值、270ms平均和原依赖权重；完整113对、225窗口、A/B/D、O0–O3；`oracle_curves/points.csv.gz`、`oracle_summary.csv`、`oracle_per_group.csv`。
- `window_decomposition.csv.gz` / `pair_decomposition.csv.gz`：起点、未来取样、连接边界与增量误差分解。`decomposition_summary.csv`记录交叉矩，未假设两项独立。
- `full_audit.py`：全部107,920 train、36,332 validation H1000窗口；完整缓存/源标签身份检查，逐窗口支持与时钟、p/q/r误差。输出 `full_window_audit.csv.gz`、`full_per_flight.csv`、`full_summary.csv`、`full_coverage.json`。
- `identity_clock_audit.csv` / `origin_support_availability.csv`：固定窗口精确关联、因果初值、标签起点及原生插值左右括号。221/225窗口可精确对应H1000；另4个保留原H500源样本身份，没有近邻补配。
- `compare_models.py`：只读归档6epoch M0W0/M1W0 seed17、24epoch M0/M1三种子；不混合训练预算。`model_error_decomposition.csv.gz`、`model_comparison_300.csv`、单窗口绝对A误差及逐航次关联。
- `case_oracles_300.csv`、`case_models_300.csv`、`figures/case_*.png`：八个预定病例，含唯一validation主病例；图中validation三个fold模型分别展示。
- `oracle_direction_counts.csv`：有效方向、近零真值、接近零预测及错误数量，避免把微弱同向当充分幅度复现。
- `leave_one_flight_out.csv`：46个航次逐一移除的描述性敏感性，不是重新选择主样本。
- `candidate_increment_conflict.csv` / `candidate_admission.json`：候选增量监督的离线目标冲突及准入建议；未实现生产Loss、未选择lambda或拟合尺度。
- `verify.py`：独立SciPy插值、解析分段积分、按航次直接平均和BFS分量求权；363个确定性窗口覆盖全121航次，复核全量处理结果。
- `test_audit.py`：边界、缺口、常量/周期信号、起点连接、身份/划分、源哈希、全量聚合、候选公式梯度及无训练检查。
- `archive_manifest.json`：本实验文件哈希；`results/execution_record.json`：环境、范围、测试和停止边界。

H1000缓存和原始授权Parquet是外部只读资产，路径在 `common.py` / source_receipt。复现需要这些既有资产及基准分支中的上游归档；缺失或哈希不符即失败，不重建数据、不换模型。

B不是无滤波延迟的绝对物理真值。Oracle使用真实未来，不是可部署预测器或性能上界。所有配对仍是闭环观察性差异，48对不是48次独立干预，三折开发数据已反复使用。
