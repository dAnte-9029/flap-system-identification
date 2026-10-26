# 正式三折跨航次留出比较

基准 `f1324dfaaa807297d83b1d246cf21e362d5140bf`，分支 `exp/structured-control-effectiveness-group-holdout-v1`。本目录独立保存正式训练及评价；模型定义直接导入上轮已冻结的 `../structured_control_effectiveness_v1/model.py`，不复制或修改架构。

完整结论见 `report_zh.md`。`results/training_completion.json` 是实际训练完成清单，不能将上游原封复制的 `holdout_protocol.json` 中历史 `executed:false` 字段当成本轮状态。

## 数据及合同

- 指定历史版本 `velocity_rate_multistep_v1/main_50hz`，V0 时间合同，train/validation 原划分。test 不读取、不散列、不解封。
- 原始缓存位置与 SHA256 见 `common.py`、`results/source_receipt.json`；保留原航次登记、窗口及原生 dt。机器需要原 `velocity-rate-dataset-v1` worktree 中已核验的 train/validation NPZ。没有自动重建数据、下载权重或安装依赖的回退。
- 三折索引、48 主对和11依赖分量均从试训练阶段逐字节复制。每折先切出 fit indices，再调用原 `fit_normalization`；全 train 缓存只作为只读容器。归一化函数接收到的张量不含留出窗口或 validation。
- 每折 M0/M1/M2 共享折内 input/output/loss scales；原全训练 normalization 从未用于构造正式模型。`normalization_receipt.json` 记录全部来源航次、索引哈希及独立 NumPy/SciPy 复核。
- `config.json` 保存启动时的配置字节不再改动。历史描述字段 `input_output_normalization`、`patience`、`evaluation_horizons_ms` 不被本训练器使用；实际分别为折内重算、无早停、训练1000ms／局部评价500ms，见 `results/resolved_contract.json`。`control_forecast` 的 GRU16 描述只适用于 M0；M1/M2 使用冻结的两个精确 ZOH 时间矩。
- 固定 seed 17/29/43、24 epoch、batch1024、AdamW(.001,.0001)、clip5、cosine T_max24。best 只按原27航次 validation 等航次损失、继承的最小改善1e-4规则保存；同时保留 last。不同折归一化 loss 不直接平均用于排名。

## 复现顺序

以下命令在本 worktree 根目录执行，使用现有环境，不安装依赖。已有结果不覆盖：完整重跑应从上述基准新建单独worktree，只复制本目录的Python脚本和config.json，创建空results目录；不要复制已有folds/jobs/results输出。`prepare.py` 遇到已有 normalization 目录会失败。

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/structured_control_effectiveness_group_holdout_v1
mkdir -p "$EXP/results"
$PY "$EXP/prepare.py" > "$EXP/prepare.log" 2>&1
$PY -m pytest -q "$EXP/test_contract.py" > "$EXP/tests_pretraining.log" 2>&1
```

只有上一条测试成功才建立门禁记录（本次记录含实际7项通过）：

```bash
$PY - <<'PY'
import json
from pathlib import Path
p=Path('experiments/structured_control_effectiveness_group_holdout_v1/results/pretraining_tests.json')
p.write_text(json.dumps({'passed':True,'tests':7,'log':'tests_pretraining.log'},indent=2)+'\n')
PY
```

先用 `nvidia-smi` 核查空闲显存及其他进程。两终端各运行一个 worker，自动领取不同 fold/seed；同一 fold/seed 的三个模型留在同一卡上。没有终止其他进程或因资源改变 batch size。

```bash
$PY "$EXP/train.py" --device cuda:0 > "$EXP/worker_gpu0.log" 2>&1
# 另一个终端：
$PY "$EXP/train.py" --device cuda:1 > "$EXP/worker_gpu1.log" 2>&1
```

`status.py` 查看完成数。中断时，先确认原 worker 已退出，再使用 `train.py --device cuda:N --resume-job J` 恢复那个 job。last 保存 optimizer/scheduler/RNG/完整轮记录，恢复至最后已完成轮；不重抽种子、不挑选重复训练。已完成模型按哈希检查后跳过。不要同时恢复同一 job。`jobs/jobJ/owner.json`、`done.json`、可能的 `failure.json` 保留运行证据。

全部27次完成后：

```bash
$PY "$EXP/support.py" > "$EXP/support.log" 2>&1
$PY "$EXP/evaluate.py" --device cuda:1 > "$EXP/evaluation.log" 2>&1
$PY "$EXP/analyze.py" > "$EXP/analysis.log" 2>&1
$PY "$EXP/verify.py" > "$EXP/verification.log" 2>&1
$PY "$EXP/plot.py" > "$EXP/plot.log" 2>&1
$PY "$EXP/describe.py" > "$EXP/description.log" 2>&1
$PY -m pytest -q "$EXP/test_contract.py" "$EXP/test_results.py" > "$EXP/tests_final.log" 2>&1
```

`support.py` 只使用 CPU，可在训练同时运行。本次训练启用原脚本的 TF32；评价关闭 TF32。运行环境、GPU占用和源码版本分别见环境记录、`results/gpu_before.json`、每次训练 `provenance.json`。

## 结果导航与统计定义

- `folds/foldN/normalization*.json`：折内尺度与独立核验；`training/M*/seed*/{best,last}.pt`：27组共54权重文件及逐轮 JSONL、配置、来源信息。best/last 中保存完整配置和 normalization。
- `results/predictions_foldN.npz`：原四组合输入 case_id 对应的13维原生预测；`window_points.csv.gz` 保留 receiver/donor/window/cell。
- `factual_{curves,points}.csv.gz`：A/B/D、六轴、折／种子／pair／时间及支持掩码。A 原始标签，B 原生角速度统一物理时间参考；D 对 B 和模型分别施加固定270ms中心平均，保留瞬时起点，仅作离线诊断。速度只保留 A。没有外推500ms或滤波边界。
- `internal_{curves,points}.csv.gz`：YAA/YAB/YBA/YBB、CA/CB/C/H及闭合量；没有真实 swapped 标签。四个输入只在事件上替换；历史、初值、dt、相位外推保持 receiver 合同。
- `factual_summary.csv`、`internal_summary.csv`：冻结主队列及所有质量／背景子集；300ms提供逐对、航次等权、分量等权。按 pair 所涉两端航次的度数倒数和加权；分量内每对权重为分量对数倒数。其他时刻保留航次等权。
- `paired_changes.csv.gz`、`paired_summary.csv`：M1−M0、M2−M0 严格同 pair/seed/fold 配对。`gained/lost_pair_seed_rows` 在 pooled 行是三种子的计分行数，不是独立对数。
- `per_group_300.csv`、`component_assignments.csv`：全部11分量、46航次及日期／配置描述；组内逐对等权。航次/日期/配置组可重叠，不能相加。`seed_consistency.csv` 保留内部贡献跨种子符号差异。
- 三种子各自统计再报告 pooled 描述；pooled 将每个种子指标的基础计分行等权纳入，不先平均轨迹、不选优种子。种子不是新增物理独立样本。
- `heldout_trajectory.csv.gz`：三折全部留出窗口的自主1s事实轨迹误差；先对每窗口全部有效原生步取RMS，再按航次平均。`trajectory_per_flight.csv` 与 `trajectory_summary.csv` 保留物理单位，避免跨折归一化尺度混淆。
- `fold_control_support.csv` 是折内训练边际范围；旧 `swap_support.csv` 是历史全训练条件支持记录，两者不得混称。边际范围内不证明 joint 支持或反事实可信。无样本因此被删除。
- `unevaluable_background_pairs.csv`：5个训练跨折背景对不重新配对；48主对各在指定折评价一次。其余2个验证背景对不属于训练留出协议；validation:44 每折模型仅单独展示，已用于 checkpoint 选择，不能作为独立测试排名。
- `illustration_selection.json` 明确固定病例及按全样本误差变化选出的最佳／最差插图规则，仅用于展示；全部失败样本仍在数据表中。
- `training_completion.json`、`independent_verification.json`、`inference_checks.json`、测试日志与归档 manifest 为验收入口。

源码依赖包括当前仓库原训练／模型／积分器、事件规范化及前序评价函数。无需训练历史 checkpoint，无需访问 test。试训练6轮 checkpoint 不参与本次正式推理。

报告完成并确认最终测试通过后，运行 `archive.py` 生成全部交付的SHA256及最终完成记录。初次流水线的验收计数断言失败保留在原日志中，已修正后的测试及 `results/completion_receipt.json` 是最终状态；复现时不会重造该历史失败。
