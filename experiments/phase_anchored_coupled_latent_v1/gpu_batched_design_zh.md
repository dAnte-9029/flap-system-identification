# PACLD GPU 批量训练架构与验证

基线：`research/phase-anchored-coupled-latent-v1`，`525ced43fa1c7daded1447c639aa24146243b460`。
交付分支：`research/pacld-gpu-batched-v1`。测试在该分支未提交工作区执行；结果中的运行时 Git revision 是基线，新增实现由本报告所在提交标识。

用户指定的 tmux CPU 训练（PID 1689196，`--device cpu --steps 10944`）已 SIGINT 停止并确认退出。原训练目录未覆盖。本次只有数值验证、性能短测及少量训练更新，没有启动长期训练。

## 执行架构

- `data/pacld_batch.py`：独立 `BatchedWindow`，保留初始状态、历史 measurement/availability/source、原生目标时间、metadata、控制与监督权重。CPU 先按原 availability 合同过滤，再按 measurement/availability/source 排序；逐窗口按控制切换及查询边界计算 `ceil(span/0.005)` 等长子步，随后沿执行序号 padding。没有插值原生目标或未来 Hall 注入。
- `models/pacld/batched_types.py`：独立 Tensor `BatchedState`。窗口时钟保存在编译计划中，GPU 不使用 Python 标量状态时间。
- `models/pacld/batch.py`：每个执行槽对整个 batch 同时计算共享 Encoder 与 Slow/Fast heads。传播使用各行自己的 dt/u，padding 用有效 mask 选择原状态，包含协方差，避免零 dt 的归一化舍入改变 padding 状态。首个可靠 Hall 锚定、圆周创新、测量时刻回放保持原公式。
- FULL/BASE 由同一历史后验展开成 `2B`，BASE 仅关闭显式相位载波。共用同一组 Slow、Fast 与物理解码参数；没有第二套物理模型。完整原 CPU 实现未改动。
- 解析频率与二阶相位协方差传播替代每子步 Van Loan 矩阵指数；与原 FP64 矩阵指数对照。原生 Lie midpoint、四路控制、FRD→NED、Hamilton wxyz 及最大 5 ms 分步保持。
- 原生目标与跨拍配对仅在 CPU 整理；GPU 通过 gather/index_add 归约。FULL 保持 EKF/gyro 两族等权，Macro 保持来源等权。先独立形成窗口 Loss，再对所有窗口平均；无配对窗口 Macro 贡献为零，Macro 指标只除以有效窗口数。
- `torch.compile(fullgraph=True)` 只覆盖一个批量积分子步与一个批量观测更新，不编译不规则计划遍历。编译块长度为一个子步/观测槽，复用跨窗口与时域的图，避免展开整个长时域。支持 `--checkpoint-steps`，重算积分激活；数值测试验证其梯度不变。

## 训练、DDP 与恢复

`run_pacld_gpu.py` 支持原生数据、Adam、全局梯度裁剪、显式总 optimizer steps、windows_seen、完整固定 validation、checkpoint/恢复、计算与准备时间、峰值显存和 GPU 利用率采样。

`BucketSampler` 以原 FlightBalancedSampler 为全局抽样流，按 horizon 排队，保留未消费队列；同一全局 batch 不重复窗口，重复抽中项延后保留。所有 rank 运行相同全局流，再取互不重叠的等长切片。它改变更新内窗口组合与顺序，不能声称与原单窗口 Adam 训练轨迹相同。

DDP 使用一 GPU 一进程、NCCL。各 rank 的窗口平均 Loss 由 DDP 平均梯度，等价于全局窗口平均；FULL 与 Macro 在同一图。裁剪在 backward/NCCL 完成后。验证仅 rank 0 执行完整固定队列，再广播选择状态。

新 schema `pacld_gpu_batched_v1` 保存模型、Adam、采样器及 pending queues、optimizer_steps/windows_seen、配置、数据身份、precision/world_size、各 rank RNG 与选择状态。拒绝旧 CPU optimizer checkpoint、全局 batch/precision/world size 或配置不匹配的恢复。保持固定 156 窗口 validation 与原 FULL 的 flight/horizon 层级选择公式；训练与验证角色不变，未打开 sealed test。

## 数值证据

详细机器可读结果见 `gpu_batched_results/numerics.json`、`restore.json`、`ddp_parity.json`。

真实窗口覆盖 0.2/0.5/1/2 秒及两种原生采样组；合成工程测试补充混合长度、不同控制边界、首次 Hall 锚定、未来 availability 哨兵、padding、无 Macro、激活重算与采样器恢复。合成窗口不作为真实飞行精度或物理真值证据。

| 项目 | FP64 GPU 相对原 CPU | mixed 相对 FP64/原 CPU | 全 FP32 相对 FP64/原 CPU |
|---|---:|---:|---:|
| 四时域物理轨迹最大绝对差 | 6.63e-13 | 5.51e-7 | 2.13e-5 |
| Hall 相位最大绝对差/rad | 1.36e-12 | 2.25e-10 | 3.54e-5 |
| 关键参数全集梯度相对 L2 差（对 GPU FP64） | 0 | 9.58e-7 | 2.67e-6 |
| 一次 Adam 后参数最大差（对 GPU FP64） | 0 | 3.30e-6 | 1.07e-5 |

GPU FP64 相对原 CPU 的梯度最大绝对差不超过 3.71e-11。混合精度四时域 SO(3) 姿态分量误差不超过 3.71e-9 rad，四元数范数误差约 2.22e-16。FULL/Macro 分项及每个时域的数值记录均在 JSON 中。

选择 **mixed**：MLP 与动力学网络上下文/载波计算使用 FP32；物理状态积分、频率/相位传播、观测残差、协方差及 Loss 保持 FP64。不启用 BF16/TF32。短测所有优化更新有限；这些结果支持本次短时域计算精度，不构成长期训练稳定性保证。

B>1 批量梯度与独立窗口梯度平均的 FP64 对照通过；B=1、B>1 的完整 Belief、FULL/BASE 物理轨迹通过聚焦测试。双卡同 Global Batch 的 Loss 差 3.92e-7、梯度最大差 1.24e-5、梯度相对 L2 差 6.95e-7。单卡与双卡恢复后下一批索引完全一致，下一次 Adam 参数更新最大差均为 0。

实际单卡入口训练 2 步/16 窗口，完整 validation FULL 分数 66.724165；从 checkpoint 恢复到第 3 步/24 窗口后分数 65.663367。此处仅验证入口、数值与保存/选择流程，不能据此宣称模型效果提升。

## 吞吐与端到端开销

环境：PyTorch 2.7.0+cu128、两张 RTX 4090（驱动报告每张 49,140 MiB）、每个训练进程一个 CPU Torch 线程。下表 GPU 使用 mixed 与 compile，所有组合各两次真实 forward/backward/Adam；取第二次为短程稳态估计，未把首次编译摊入稳态。数值只支持这组短测，未给统计置信区间。CPU 使用保留的原 FP64 单窗口实现及相同损失/裁剪/Adam，不开启额外梯度诊断。

CPU 在各时域的同一前 8 个窗口上逐窗口执行；GPU B=8 与该集合完全配对，B=16/32 为扩展集合，不能把不同集合的吞吐比解释为严格配对加速。算子/数值微基准统一用 lambda_macro=0.1；实际训练入口使用冻结 calibration 的 lambda_macro=1.0，未修改正式 Loss。CPU 计时逐窗口重新初始化模型（初始化不计入计算耗时），GPU 两步复用同模型；这是执行吞吐比较，参数/梯度等价由另外的固定参数测试验证。CPU 一次更新处理 1 窗口，GPU 一次更新处理 B 窗口；主要比较有效窗口吞吐，不把更新/秒混同为训练工作量。

| Global B | 时域/s | 单卡窗口/s | 双卡窗口/s | 单卡更新/s | 双卡更新/s | 单卡准备/s | 双卡 rank0 准备/s | 单卡峰值/MiB | 双卡 rank0 峰值/MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.2 | 5.52 | 7.14 | 0.691 | 0.892 | 0.842 | 0.539 | 151.6 | 90.1 |
| 8 | 0.5 | 4.78 | 5.26 | 0.597 | 0.657 | 0.955 | 0.654 | 210.8 | 118.1 |
| 8 | 1 | 3.60 | 4.13 | 0.450 | 0.517 | 1.294 | 0.744 | 305.3 | 159.7 |
| 8 | 2 | 2.40 | 2.82 | 0.300 | 0.353 | 1.980 | 1.058 | 506.1 | 249.2 |
| 16 | 0.2 | 12.27 | 11.62 | 0.767 | 0.726 | 1.702 | 0.979 | 362.4 | 152.1 |
| 16 | 0.5 | 9.37 | 9.38 | 0.586 | 0.586 | 1.938 | 1.091 | 419.5 | 211.2 |
| 16 | 1 | 6.34 | 7.37 | 0.396 | 0.461 | 2.582 | 1.416 | 614.2 | 305.6 |
| 16 | 2 | 4.34 | 4.88 | 0.271 | 0.305 | 3.975 | 2.169 | 1019.2 | 506.2 |
| 32 | 0.2 | 20.24 | 21.86 | 0.633 | 0.683 | 3.383 | 1.805 | 669.1 | 292.8 |
| 32 | 0.5 | 16.59 | 18.40 | 0.518 | 0.575 | 4.021 | 2.212 | 871.4 | 420.1 |
| 32 | 1 | 11.53 | 13.02 | 0.360 | 0.407 | 5.000 | 2.683 | 1329.7 | 614.3 |
| 32 | 2 | 8.05 | 8.51 | 0.252 | 0.266 | 7.584 | 4.192 | 2193.1 | 1019.4 |

原 CPU 配对基线与单卡 B=8：

| 时域/s | CPU 窗口/s | 单卡计算吞吐比 | 单卡准备+稳态更新耗时/s | 单卡含准备窗口/s |
|---:|---:|---:|---:|---:|
| 0.2 | 0.742 | 7.44× | 2.290 | 3.493 |
| 0.5 | 0.459 | 10.42× | 2.629 | 3.043 |
| 1 | 0.302 | 11.95× | 3.515 | 2.276 |
| 2 | 0.171 | 14.05× | 5.315 | 1.505 |

准备时间包含原生窗口读取、CPU 计划编译和传输前准备（单卡 benchmark 的传输在该计时区间外）；表中“含准备”是阶段耗时相加估计，不含进程启动及 dataset 索引初始化，也不是持续新窗口队列的独立端到端实测。正式入口的 `training.jsonl` 另记录实际每批读取、编译、H2D、更新的端到端耗时。双卡 benchmark 的准备时间为 rank0，不能冒充跨 rank 最大耗时。

单卡 B=8 首次冷编译探测总更新 37.434 s，第二次 1.626 s；后续进程利用编译磁盘缓存，第一步仍含图加载/专门化开销，不能把各文件的第一步全称为冷编译时间。所有组合的首次、稳态、前向、反向及 Adam 分解见对应 JSON。FULL/BASE 在同一批维度并行，不能再把两者单独计时相加；Profiler 单列共享历史与联合 FULL/BASE 区间。

双卡稳态计算吞吐相对单卡约为 0.95–1.29 倍，未达到线性扩展；某些组合甚至略慢。该小模型与细粒度原生事件主要受主机调度/小算子启动影响，增加第二张 GPU 不会消除时间递推依赖。CPU 数据准备仍占明显比例，双卡可分担准备但需要根据真实训练日志判断总收益。推荐优先单卡 B=32；双卡入口可用，不能承诺两倍加速。

## GPU Profiler

原版全原生 0.5 秒单窗口与 compiled B=8 的 CUDA trace 都包含前向和反向。详细算子表/JSON 位于 `gpu_batched_results/profile_original.*`、`profile_batch.*`。全原版 trace 的 CPU 事件汇总曾占约 22 GB 主存，最终成功产出；复现脚本默认将原版限制为一次观测更新和 10 个子步，`--full-reference` 才采集整窗。不得把 Profiler 本身耗时作为训练吞吐。

| 指标 | 原版单窗口整窗 | 批量 B=8 整批 |
|---|---:|---:|
| CUDA kernel launches，前向+反向 | 700,388 | 115,178 |
| `_local_scalar_dense` | 3,979 | 0 |
| `cudaStreamSynchronize` | 10,822 | 0 |
| `matrix_exp` | 476 | 0 |

原版的 Python 串行事件、MLP 小算子反复启动、相位系数标量取回及协方差 PSD 检查/矩阵指数，均有代码和 trace 支持。批量热路径去除了这些标量取回与矩阵指数；仍保留不可并行跨越的时间递推，以及每个批量子步的 compiled block 调度。共享历史只运行一次，未来 FULL/BASE 联合推进。

Profiler 中 B=8 的 host 包围区间约为：history 0.827 s、联合 FULL/BASE 0.347 s、backward 1.435 s；这些是带跟踪开销的区间，不是独立 CUDA kernel 时间。嵌套标注不能相加当作 GPU 忙碌时间。该结果支持将后续性能重点放在事件块调度/图开销与 CPU 编译传输，而非 DataLoader workers；本次已有实际单卡吞吐提升，不退回 CPU。

实际随机批次训练入口的第二步：单卡 B=8 端到端 3.042 s（2.629 窗口/s），双卡同全局批次 9.218 s（0.868 窗口/s）。双卡短程入口存在编译/调度开销，不能用重复同 batch 的稳态微基准掩盖它。两者第 2 步完整 validation 分数分别 66.724165038 与 66.724165059。**DDP 数学与恢复通过，但本次未确立实际变化窗口队列的端到端优势，暂不作为默认正式配置。**

## 预算与正式命令（仅提供，未执行）

建议单卡 B=32、mixed、compile。若先匹配旧 10,944 个窗口的数据曝光预算，应为 **342 optimizer updates / 10,944 windows**，不是 10,944 次批量更新。Adam 更新次数与窗口组合均变了，该预算不保证与旧单窗口训练相同的收敛效果。粗略耗时可按各时域准备+稳态时间平均乘 342 估计；不把这种预算估计当实际长期测量。

当前冻结配置的 validation_interval=1000 仍按 optimizer updates 解释，342 步会在末步执行完整固定 validation；没有暗改验证间隔。若研究要求与原 1000 窗口间隔相近的中间 checkpoint，需要显式修改运行配置中的间隔；本次未替用户改变这一选择。保持原 10,944 optimizer updates 则是 350,208 个窗口，不能称为同预算。

在新 worktree 根目录执行：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld_gpu.py \
  --device 0 --batch-size 32 --precision mixed --compile --steps 342 \
  --output runs/pacld_gpu_b32_seed17_10944windows
```

双卡功能入口（global B=32，每卡 16；当前不推荐以它代替单卡默认）：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/run_pacld_gpu.py \
  --batch-size 32 --precision mixed --compile --steps 342 \
  --output runs/pacld_ddp_b32_seed17_10944windows
```

恢复时保持 global batch、precision、world size 和配置，添加 `--resume <GPU checkpoint>`，使用新的 `--output`，`--steps` 是累计总更新数。需要降低激活显存时添加 `--checkpoint-steps`，本次最大 B=32 尚不需要默认启用。

## 验证边界

- VERIFIED：FP64 数学等价、B=1/B>1 梯度、padding/availability、解析协方差、单/双卡恢复、固定 validation 与短程运行。
- SUPPORTED：本组原生窗口及四时域下 mixed 精度适用、单卡实际批量吞吐提高。
- UNRESOLVED：长期优化收敛、长期数值稳定性及 DDP 对变化窗口队列的稳态端到端收益。无长期训练或物理真值/因果控制结论。

模型仍是学习的整机状态动力学，物理坐标为 p_N[m]、v_N[m/s]、q_NB[Hamilton wxyz，FRD→NED]、omega_B[rad/s]；未引入质量/惯量或独立翼载荷真值，也未改变 Hall=0 的既有工作假设。

最终聚焦测试：`PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_pacld_gpu_batch.py`，**4 passed**。新增入口全部通过 py_compile。原生 CUDA B=1 最大 Belief 差 9.94e-13、轨迹差 5.10e-14、梯度差 3.42e-12，见 `b1.json`。实际双卡入口从第 2 步 checkpoint 恢复到第 3 步/24 窗口，156 窗口完整 validation 得分 **65.663367077**，对应单卡恢复为 **65.663367092**。

实际入口逐步端到端记录保存为 `gpu_batched_results/train_*.jsonl`；配置及固定 validation 成员保存在 `training_configuration.json`。GPU 利用率/显存采样摘要见 `gpu_telemetry_summary.json`，原始逐秒采样留在本 worktree 的 `runs/pacld_gpu_batched/gpu_samples.csv`。采样区间包含编译、读取、Profiler、验证与空闲，不能把全区间均值叫作稳态训练利用率；NCCL 等待也可能表现为高 GPU utilization。所有短测及采样进程已结束。完整 checkpoint 留在 `runs/pacld_gpu_batched/train_{single,resume,ddp,ddp_resume}/`，未覆盖 CPU 正式训练目录，未将二进制 checkpoint 推送到 Git。
