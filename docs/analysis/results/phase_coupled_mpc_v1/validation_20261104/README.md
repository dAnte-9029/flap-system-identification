# PA-HBD 两种子验证与 1s GRU-Control 对照

本目录归档2026-11-04已执行的验证结果，没有启动新训练。

- [A/B 原固定验证集报告](report_zh.md)：39航次、156窗口，含两种角速度变化同向性定义。
- [A/B/C 共同范围报告](gru1s/report_zh.md)：27个main_50hz航次、108窗口，C为既有1s训练的GRU-Control；种子17、29。

本地原始分析目录：`/home/zn/pahbd-runs/ab-validation-20261104-0826`。
冻结正式运行目录：`/home/zn/pahbd-runs/ab-compiled-dual-20261103-1812`。
PA-HBD分析代码版本：`dba2b61`。checkpoint、选模步骤和数据身份见`summary.json`、`gru1s/preparation.json`及各模型`verification.json`。

归档包括报告、图、汇总/逐航次/逐种子CSV、输入成员索引、执行检查和实际使用的Python脚本。脚本是该次运行的原样快照，保留本机路径、CPU绑定及外部checkpoint/数据依赖，不是独立可移植的评估工具；原目录中的失败中间版本未归档。

未提交的本地材料：训练checkpoint、原始数据、输入/预测NPZ、逐窗口明细`direction_windows.csv`与`error_windows.csv`、运行日志及中间推理版本。上述文件均保留原位。仅检出本目录不能独立重跑GPU推理。原报告中的绝对本地路径也需要原主机数据。

复现须另建输出目录，保持本目录的父子结构，并提供上述外部依赖。A/B依次使用`analyze.py`、`direction_infer.py`、`direction_analyze.py`；共同范围使用`gru1s/prepare.py`、`gru1s/infer.py`的A/B模式，C最终结果使用`gru1s/infer_gru_archived_batch.py`，再运行共同范围的两个分析脚本。`infer.py`的C逐窗口结果不是最终报告所用版本。需要原PA-HBD预测NPZ才能执行共同范围的旧输出一致性检查。

PA-HBD编译推理使用已验证的`/home/zn/anaconda3/envs/pahbd-compile-v4/bin/python`；GRU及CPU分析使用`/home/zn/anaconda3/envs/flap-train-gpu/bin/python`。所有正式训练保持暂停，封存test未参与本次评估。
