# 扑翼周期预测研究归档

[实验报告](report_zh.md)给出三个种子的预测误差、同容量消融和留出整拍的分量检验。[读出模型](readout/model.py)和[保留的 seed 17 权重](readout/original_runs/incumbent17/corrected.pt)可直接配套加载。模型输入包括原有 180 维决策历史、14 维因果谐波摘要、起点控制、到各查询时刻的控制积分及两路原生时间。摘要的实现见[准备程序](../../../experiments/periodic_motion/prepare_native.py)。

原始强基线和本轮训练器保留在分支的文件树中。互斥的两种端到端模型保留各自原始 Git 提交，可用以下命令读取。

```bash
git show 3cb9e766707c6c89f2ece6e12c973b249db686cd:experiments/autopilot_flight_dynamics/model.py
git show 38aa995b23b75752ddbaeec2785209babc048ab5:experiments/autopilot_flight_dynamics/model.py
```

[训练记录](studies.json)列出三个 study。各 study 归档了模型权重、训练曲线、逐架次误差和源码提交；[六指标与步数](accuracy.json)可定位每次运行。历史运行记录中的绝对路径对应原研究机器。重做训练或逐采样点诊断需另行准备 `velocity_rate_multistep_v1` 的训练和开发数据、命令带及因果历史缓存，并按记录建立工作树。相位参考程序使用 `origin/research/slow-fast-separation-validation-ab-v1` 的提交 `9832d312de774c420c381754004bbd8fd12e68a6`。原始日志、大型数据缓存和可再生成的逐点数组留在研究机。已归档的各翼拍误差表和汇总足以复核报告的分量统计。

[论文复现记录](../../reproductions/rep-bc7ae8dfcdef4cb6/reproduction_report.md)附有本地核心实验、固定版本及许可。论文 PDF 和作者的完整仓库保留在原有论文目录。需要重新运行核心试验时，按复现记录取得作者固定版本，放在该复现的 `work/neural_prophet` 位置。
