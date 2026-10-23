# Reproduce local control response scope v1

本轮只筛选和统计，不运行模型。中文结论见 [report_zh.md](report_zh.md)。所有新增内容局限本实验目录，基准为 `91f772448b78fd8690df83b9f98177d1b897218e`。

## Run in a fresh destination

使用现有 `flap-train-gpu` 环境，不安装依赖。数据根必须包含历史 registry 对应的原始共享清单及八个 train/validation Parquet；脚本在读取前核验登记哈希。原日志路径从获准 main_50hz 清单派生，仅读文件头参数和全文件字节哈希。缺少/损坏注册数据时失败；缺失原日志头会明确标为 unknown，不能声称同控制配置。

从 worktree 根执行，`OUT` 必须是新的空目录。训练命令返回成功后再执行验证；训练写入 frozen_selection，验证核对代码和配置指纹。

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/local_control_response_scope_v1
DATA=/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1
OUT=/tmp/local-control-response-scope-reproduction-v1
export MPLCONFIGDIR=/tmp/local-control-mpl
"$PY" "$EXP/analyze.py" --split train --data-root "$DATA" --output-root "$OUT"
"$PY" "$EXP/analyze.py" --split validation --data-root "$DATA" --output-root "$OUT"
"$PY" "$EXP/describe.py" --data-root "$DATA" --results "$OUT"
"$PY" "$EXP/verify.py" --data-root "$DATA" --results "$OUT"
"$PY" -m pytest -q "$EXP/test_scope.py" experiments/control_identifiability_audit_v1/test_audit.py -k 'not artifact_statistics_and_boundaries'
git diff --check
```

`analyze.py` 拒绝覆盖非空 split 目录；描述脚本只重建指定结果目录的描述表/图。复现 CSV 数值/身份应一致；gzip 包装时间、机器绝对路径、运行时间及冻结时间可能不同，不能以整个压缩包哈希相同替代数值验证。

## Index contract

- `candidate_id` 是 `candidate_index.csv.gz` 的本 split 行号；`window_id` 是原数据集稳定身份。主模式与其他全部原窗口字段保留。
- `pair_id` 是同 split `all_history_pairs.csv.gz` 的行号；`i/j` 指向 candidate_id，只有不同原日志才可能配对。不要把 train 与 validation 的整数 ID 混用。
- `selected_pairs.csv` 是各工况/方向各自去重的 cohort membership；不同 cohort 可重叠，不可求和。`flight_disjoint` 标记同一队列的贪心航次互斥选择。
- 使用主方案时取 frozen_selection 中的 AUTO_stable/differential，再连接主索引；备选 AUTO_stable/motor。不要从 validation 图表重选。
- `description/background_index.csv` 的各行定义见 report 的背景表；它们各自重新去重，不等于主队列中的子集。
- `command_events.csv.gz` 中原始通道和 `[motor,common,differential,rudder]` 坐标不要混淆。所有幅值均为归一化命令，无物理舵角或推力量纲。
- `all_eligible_metadata.csv.gz` 只是局部资格审查辅助文件，本地保留、可重建、不提交；固定工况全部候选索引、事件、匹配和选择表均提交。原始数据和模型不复制入实验目录。
- `sources/stage*_report.txt` 为 Drive 原报告本地快照，不重复发布全文；提交对应来源 JSON、哈希和本轮必要研究结论。重取快照需原 Drive 访问权限，不影响本轮数据统计复现。

## Validation and artifacts

`tests.log` 是最终 20 passed / 1 deselected；`tests_initial.log` 保留旧全产物测试在隔离工作区缺少路径的失败记录。`results/verification.json` 是独立复核数量与原生 ZOH 统计的结果。`results/description/asset_receipt.json` 记录仅做字节校验的原 checkpoint 和 Drive 报告快照。

`archive.py` 生成本实验代码、配置、来源与产物 SHA256 清单，并区分提交文件与本地可重建文件。发布 SHA 由 Git 提交和远端核对给出，避免在自身提交文件中循环嵌入 SHA。所有源数据、旧实验、AGENTS.md 均保持原状。
