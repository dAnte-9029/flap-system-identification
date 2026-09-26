# Git 交付范围

结论见 [report.md](report.md)，候选使用边界见 [model_card_zh.md](model_card_zh.md)。本轮研究交付完成，控制可用性尚未得到验证。

Git 保存本轮代码、测试、协议、紧凑结果表、审计记录、候选模型及支持参考。`delivery/control-dynamics-candidate-v1.tar.gz` 是自包含推理包；解压后可按包内说明运行 `run_example.py`，不需要原始飞行数据。它包含打包时的代码和报告快照，后续实验结论以本目录报告为准。

大型逐样本预测、闭环轨迹和已解压的重复包未纳入 Git；路径、大小和 SHA256 见 [git_artifact_scope.json](git_artifact_scope.json)。这些文件仍保留在原工作区。历史 completion/audit 文件记录的是原工作区完整产物的验证结果，不代表仅 clone 仓库便能通过全部产物核验。

完整实验复现还需要各 protocol 中固定哈希的原 train/validation 数据、准备缓存、源模型和源配置。数据及既有配置的其他本地改动不属于本次提交。实验入口拒绝覆盖已有结果；重跑应保留历史目录，并在独立工作副本中准备所需输入和新的输出位置。不得用其他版本数据或 sealed test 替代缺失输入。

本次提交前针对性测试：34 passed；`git diff --cached --check` 通过。
