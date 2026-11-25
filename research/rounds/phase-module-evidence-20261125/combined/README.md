# 组合模型推理

WingbeatPredictor 从本轮已保存的两个头载入 seed 17、29 或 43。继承的半秒决策历史为 26 个统一时刻、30 个字段，采样稀疏的速度由质量位屏蔽。原有输入中的 history194 由 6 个历史快照及 14 个一阶历史摘要组成。pool 和 actual_time 由原八个查询的原生测量时刻、各查询前已发布的控制命令构造。网络输出给定这些实录命令的速度和角速度增量，形状 [B, 8, 2, 3]，单位 m/s、rad/s，两路分别按各自的测量时间读取。

```python
from predictor import WingbeatPredictor

model = WingbeatPredictor.from_seed(17, device="cuda:0")
delta = model(history194, pooled_commands, initial_command, actual_time, harmonic_coefficients)
```

harmonic_coefficients 来源为 waveform/experiment.py 的 extract(history26, main_50hz_mask)。数组顺序为 [B, 流, 机体轴, 谐波次数, 正余弦]；谐波次数依次为 2、3。调用者将数组转成模型设备上的 float32 张量。零系数会令对应流的高阶增量严格为零。跨主干只需高阶头时，使用 waveform/coherence/inference.py 的 HarmonicTransport，它返回可直接相加的增量。

在项目训练环境运行本目录的 evaluate.py 可重新得到四项对照。121/39 架次的本地数据缓存及旧强主干来自本轮上游研究。三种子均已重算，单独启用两个头时的逐架次 MSE 与各自独立程序一致。另在 41 个相同开发窗口扰动后半段命令，三种子在前四个查询的实际输出改变量均为零。
