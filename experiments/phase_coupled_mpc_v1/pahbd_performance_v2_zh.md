# PA-HBD GPU 性能重构与实测

基线 `330257f223544366549f12212812af8eb6973bb7`；优化分支 `perf/pahbd-kernels-v2`，独立 worktree `pahbd-kernels-v2`。原训练工作区、队列、checkpoint 均未改动；没有正式训练、test 访问、A/B/C 性能排名或 push。

## 结论

默认 width=128、latent=12、float64、相同 seed=17、真实共享 Hall train 窗口下，新旧预测最大绝对差 `9.79e-13`，Loss 差 `1.05e-12`，参数梯度差 `1.51e-11`。数学合同保持一致，但不是逐位一致或完整长程训练参数一致的证明。

同一 GPU、同一性能核的端到端实测提速 **1.10–1.32 倍**。原六项固定预算重新估计为 **约6.27天**，相同测量条件下原实现为7.98天。此前约13天的估计受到资源共享、CPU 调度及小样本影响，不能将其与6.27天之比作为代码加速比。

## 根因与修改

- 每次传播重新扫描完整控制带，产生大量 `torch.equal` GPU 同步。现在 CPU 规范化一次，以 tensor identity/version、时间序列和 requires_grad 判定缓存有效性；精确 ZOH 区间表按原生事件/查询边界缓存，不合并不同可求导控制变量。
- 固定频率系数反复 `float(bf)`，固定 PSD 反复检查。现在固定系数只同步一次，加载/设备/精度变化触发缓存失效；每次 Belief 修正后同步调度频率，随后用固定相位方程的 float64 CPU 镜像维护子步调度，预测状态及其梯度仍在 GPU 上传播。
- 每子步两次4×4 Van Loan 矩阵指数。现在采用解析 `F` 与 `Qd=∫FQFᵀdt`；小参数使用稳定级数，精确浮点 dt 为缓存键，不量化步长。缓存有上限，保留协方差与 Q 的梯度；仅固定过程噪声走预验证路径。
- 原生观测逐条执行状态与翼拍残差。现在批量计算残差，保留每目标分量权重、source noise、时域权重，以及先来源族再求均值的归约顺序含义；跨翼拍姿态仍用群增量。
- 合并历史观测合法性传输及有限梯度检查，减少 Python 布尔同步。A/B 分支、观测时刻、相位保护、SO(3)、模型宽度和优化器未改变。

同样3个真实积分子步的有界 CUDA trace：kernel launch **1326→1002**，stream synchronize **96→12**，`_local_scalar_dense` **30→0**，矩阵指数及特征值检查各 **6→0**。这些是采样区间计数，不是整个窗口的总计。

缓存命中的100次相位协方差调用，profiler CPU total **88.785→6.379ms**，CUDA total **12.445→1.525ms**。2s窗口独立 Loss trace 的 CPU total：状态项 **211.7→1.04ms**，翼拍项 **241.1→1.27ms**；profiler 有记录开销，组件加速比不能直接替代端到端比值。

## 真实 GPU 对照与耗时

物理 GPU 1，`CUDA_VISIBLE_DEVICES=1`，同一 RTX 4090；环境 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`。机器为 i9-14900K，最终两版均固定 CPU 12；未固定核心的初步结果波动大，仅作诊断。没有改动其他任务的资源设置。

每种模式、每个时域使用相同真实窗口和参数初始化；1次完整 CUDA 前后向预热，2次同步计时。最终优化版每次重新加载窗口，并清空步长缓存，避免仅报告重复窗口热缓存收益。原版没有这些缓存。下表包含加载、历史重放、未来积分、完整 Loss、反向、裁剪及 Adam 更新的分段实测之和；阶段间同步有测量开销。预测/梯度对照时不推进参数，Adam 更新在该组合计时结束后独立测量。

|模式|时域|原版 秒/步|新版 秒/步|加速比|
|---|---:|---:|---:|---:|
|A|200ms|4.721|4.276|1.10×|
|A|500ms|6.889|5.700|1.21×|
|A|1s|10.064|8.216|1.22×|
|A|2s|17.448|13.174|1.32×|
|B|200ms|4.947|4.288|1.15×|
|B|500ms|6.862|5.611|1.22×|
|B|1s|10.174|7.928|1.28×|
|B|2s|17.239|13.366|1.29×|

详细分阶段 wall time、CPU time、CUDA event elapsed、显存峰值与窗口 ID 见 `pahbd_performance_v2_summary.json`。CUDA event elapsed 含 CPU 发射造成的 GPU 等待，不等同于 kernel 忙碌时间；kernel 时间另见 profiler。

四个真实窗口的子步数为443/617/920/1529，新旧每个 dt 完全一致。GPU A/B × 四时域的 x（p/v/q/ω）、f、φ、h、variance、联合相位协方差、状态/翼拍/方差 Loss、所有参数梯度，全部通过 `atol=rtol=1e-8` 对照；实际误差远小于阈值。A forcing/control 梯度非零，B forcing 保留、显式 control 梯度为零。

56项测试通过，包括解析协方差不同 dt/零及近零系数/相关 Q、前向与相关梯度、半群和半正定性，控制缓存失效、候选控制重复反向，以及原有 SO(3)、等价 ZOH 分段、Hall 因果性和 checkpoint 回归。CPU子步对照与GPU比较脚本均已保留。

## 编译与进一步并行化

对真实窗口的稳定导数核心测试 `torch.compile(fullgraph=True,dynamic=False)`。Dynamo 捕获1张图、0次图中断；Inductor/Triton 在首轮编译报 `constexpr has no attribute is_block`，未进入稳定运行，因而无法给出重编译次数或吞吐收益。没有改依赖、没有编造编译加速、没有把编译路径接入正式训练。诊断脚本保留编译探测选项以复现失败。

剩余主要开销为串行 Lie/Belief 状态依赖、反向传播和大量小 kernel。保持当前每窗口一次 SGD/Adam 更新时，直接并行计算下一窗口会使用尚未更新的参数，改变训练轨迹；不能简单改成 minibatch。可保持语义的下一步是 CPU 预取，以及保留 ragged 原生时刻的验证窗口并行，但数据加载占比较低，验证也仅占总预算一部分；本轮不声称未经测量的并行收益。没有启用 float32、放大步长或删减监督。

## ETA 与边界

相同固定验证成员中，选择四时域各一个窗口调用真实 `validation_selection_score`：原版16.393s，新版9.962s，score分别83.72831578251952与83.72831578251956。以四窗平均推算156窗完整验证，训练均衡平均为7.820s/步：

`6 × (10944 × 7.820 + 12 × 156 × 9.962/4) ≈ 6.27天`。

这是固定性能核、无 GPU 争抢的小样本外推，未运行六项完整训练；训练窗来自一个合格航次的四时域，验证也是一个航次的四个固定成员。全航次分布、长期资源竞争、checkpoint/I/O 开销均可能改变 ETA。不能保证任意未测窗口在浮点 ceil 边界的调度逐位一致，也不宣称长期参数或控制性能等价已获证明。

本地原始证据：`/home/zn/pahbd-runs/perf-v2-20261029/`。`baseline_fixed` 对应统一计时脚本的基线，`optimized_fresh` 为最终新窗口测量；其余目录保留 profiling、热缓存诊断、编译失败和数值证据。最初全量 profiler 因事件记录开销过大终止，后改为有界采样，不采用该未完成结果计算吞吐。

复现入口：`scripts/benchmark_pahbd_performance.py`（`--source` 指定待测worktree，`--output` 必须新目录，`--fresh-window --cpu 12` 测新窗口）、`scripts/compare_pahbd_performance.py`、`scripts/check_pahbd_schedule.py`、`scripts/benchmark_pahbd_validation.py`、`scripts/profile_pahbd_loss.py`。每次只运行一个 GPU 诊断进程。正式训练由用户另行决定。
