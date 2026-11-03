# PA-HBD v4 编译执行实验

## 已保存的正式训练

正式双卡队列 `/home/zn/pahbd-runs/ab-dual-20261103-1453` 已安全暂停，两个进程退出码均为75，四项后续任务仍pending。

- A_seed17：3056步，`A_seed17/pause_00003056.pt`。
- B_seed17：1226步，`B_seed17/pause_00001226.pt`。

两个checkpoint已只读加载验证，Adam状态完整，采样器draws与保存步数一致。较早A17的1824步checkpoint保留。本轮不恢复正式训练，不更新best，不访问封存test。

## 隔离与环境修复

原训练分支/工作树`perf/pahbd-throughput-v3`及`flap-train-gpu`环境不改动。新工作树`pahbd-compile-v4`基于8fbf41c；独立conda环境`/home/zn/anaconda3/envs/pahbd-compile-v4`由原环境克隆，保持PyTorch 2.7.0+cu128、Triton 3.3.0。

确认错误来自与PyTorch issue164700对应的Inductor常量索引代码生成：带Identity的整数表达式没有识别为常量，产生非法`tl.broadcast_to`。在克隆环境回移官方[PR165066](https://github.com/pytorch/pytorch/pull/165066)的整数表达式判断及打印修复。未修改模型运算来绕过错误。`patch_pahbd_compile_environment.py`限制只在指定隔离环境执行，原环境检查不含此修复；实际补丁保存在实验目录environment.patch。

修复后原导数探针成功编译（1图、0图中断）。这不是通过更换torch版本得到的速度差异；另测克隆环境eager以核对环境影响。

## 执行内核

`flow.py`保持原样，作为参考。新增`tensor_flow.py`，只有显式调用`enable_tensor_engine()`才启用，生产训练入口没有接入。Tensor核心借用原模型同一组Parameter，不复制网络、不增加state_dict键；原State/ControlTape/Observation接口与Loss/Adam/Checkpoint保持不变。

- `tensor`：纯Tensor eager；保留已有固定项缓存。
- `net`：编译SlowBackbone/ControlMixer组成的完整导数区域。
- `step`：编译单个原Lie-midpoint步，包含其反向计算。
- `block4/block8`：按原dt顺序执行4/8个原子步，不放大dt；不足整块使用单步，不越过控制事件、观测修正或查询边界。

事件/子步调度沿用参考公式，包括CPU频率镜像逐子步递推的舍入顺序。dt、衰减与协方差系数作为Tensor输入，不用Python浮点值特化出每个dt的独立图。禁止切断可训练状态或控制输入的梯度；当前只支持既定冻结相位系数合同。

## 测量与验证

所有GPU基准使用物理GPU0、float64、默认网络、相同seed17、同一批真实Hall合格train窗口。稳定计时固定CPU12；热身允许8/10/12/14核加速编译，与稳定计时分开。加载、前向、反向、CPU时间、CUDA elapsed、显存、编译计数均记录。CUDA elapsed包含CPU提交间隙，不解释为纯kernel忙碌时间。

完整对照包括A/B×200ms/500ms/1s/2s。快照保留belief与全部未来状态、Loss分量、所有参数梯度和完整dt序列；数值容差atol=rtol=1e-8，积分dt序列要求逐项精确一致。每个模式执行8次连续原Adam更新，比较模型、Adam状态、抽样序列和Loss轨迹。验证不等于承诺全程10944步逐位一致。

编译热身和图缓存加载不计入稳定速度；不同方法共享磁盘编译cache，因此首次调用时间会受已有缓存影响，不宣称都是全空缓存冷启动。记录热身前后unique_graphs，以检查稳定阶段是否继续重编译。

## 多子步与CUDA Graphs限制

初筛的200ms/2s真实窗口，其单个事件区间的原积分步数为1～7。因此block8在这些输入上完全回退到单步，不能声称测试到了真实八步块的加速；block4可覆盖部分区间。这是该固定窗口组的观察，不是所有航次的上界。

`reduce-overhead`模式在200ms窗口首次热身超过180秒工程测试上限仍未完成，实际在203秒时中止，未得到稳态吞吐。中断堆栈位于CUDA Graph Trees录制输出、检查存活Tensor存储别名的路径；不能归因于Triton编译失败，也不能声称该模式在所有场景无效。本次保留默认编译模式作为候选。

全部原始记录：`/home/zn/pahbd-runs/compile-v4-20261103-1700`。

## 实测结果

下表为完整window_loss前向（含历史Belief、未来积分、全部Loss）+反向，预加载同一真实窗口、预热后3次均值。它不是仅网络前向耗时，也不含加载/Adam。

| 模式 | 时域 | v3参考(s) | 编译step(s) | 加速比 | 编译block4(s) | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.2s | 2.707 | 1.069 | 2.53× | 1.041 | 2.60× |
| A | 0.5s | 3.496 | 1.260 | 2.77× | 1.242 | 2.81× |
| A | 1s | 4.977 | 1.670 | 2.98× | 1.738 | 2.86× |
| A | 2s | 8.682 | 2.540 | 3.42× | 2.556 | 3.40× |
| B | 0.2s | 2.873 | 1.027 | 2.80× | 0.995 | 2.89× |
| B | 0.5s | 3.702 | 1.325 | 2.79× | 1.249 | 2.96× |
| B | 1s | 5.331 | 1.819 | 2.93× | 1.774 | 3.01× |
| B | 2s | 8.734 | 2.779 | 3.14× | 2.595 | 3.37× |

连续8次原Adam更新覆盖四个时域各两次，包含原窗口加载、原抽样器、Loss、反向与更新，以及很小的末尾诊断快照写出开销：

| 模式 | v3总秒 | step总秒/加速 | block4总秒/加速 |
|---|---:|---:|---:|
| A | 46.455 | 14.211 / 3.27× | 14.401 / 3.23× |
| B | 46.579 | 15.344 / 3.04× | 14.063 / 3.31× |

结论：step和block4都获得约3倍端到端提速，当前样本不足以证明block4稳定优于step，默认编译单步是更简单的候选。初筛A的200ms/2s：纯Tensor eager为2.652/8.272秒，仅编译导数为1.726/5.173秒；单纯去掉State构造收益小，完整积分步的编译和反向融合贡献更大。

A的2s反向从5.671秒降至1.273秒，前向从3.011秒降至1.267秒。参考/step的PyTorch峰值已分配显存为109.8/96.4MiB（不含CUDA context与保留缓存）。加载每窗口约0.06～0.17秒。CPU/CUDA elapsed分项、每次重复、编译热身与计数见外部timings.json；汇总对照已保存compile_v4/comparison_summary.json。

step热身产生3图，block4为4图；四时域稳定重复阶段没有新增图。图数大于1主要是初始/后续状态requires_grad不同；这不代表图中断。fullgraph模式均成功。训练与无梯度验证可能各需不同图，不能承诺整个数据集永不触发新图。克隆环境eager与原环境在200ms/2s的预测、Loss、梯度逐位相同；其速度差在约5%以内，无法解释3倍收益。

严格数值对照全部通过：每个窗口的积分dt序列精确一致；belief及全部未来p/v/q/ω、h/f/φ、执行器、方差、相位协方差保持一致，q的一致性包含姿态R。所有Loss分量、所有参数梯度满足atol=rtol=1e-8；完整四时域对照的最大状态差约9e-15，最大梯度差低于5e-13。A的forcing/control梯度均非零；B的forcing保留、显式control梯度为零。8次Adam后参数最大差低于2.2e-14，Adam状态差低于1.2e-15，抽样顺序与sampler状态完全一致。属于严格数值等价，不能承诺逐位相同或全程10944步后相同。

## 时间外推与剩余边界

已测step的端到端每更新约1.78～1.92秒。按固定10944步、六模型外推，纯训练预算约32～35 GPU小时；理想双卡各跑三项约16～18小时，另加验证、编译与checkpoint时间。这里只是单GPU四窗口短测外推，尚未实测双卡新内核并发，也未覆盖114航次的窗口成本分布，不能把它称为完整队列的保证ETA。已完成的正式步数继续保留。

剩余开销仍包括不规则事件间的Python调度、Belief更新及Loss区域、不能跨事件拼接的短积分片段。未改动DataLoader/预取，因为当前主收益来自反向编译；未改batch、精度、步长、Loss、Adam或网络。尚未将实验内核接入正式调度器，也未恢复训练。

## 暂停恢复验证

直接只读加载正式A3056/B1226 checkpoint，在独立诊断目录分别比较：原实现连续5步；编译step更新3步→保存→重载→再更新2步→保存→重载。两者使用同一真实完整train采样器，随后用原固定validation成员中四时域各一个窗口检查验证入口，不做模型性能比较或best选择。

A到3061步：参数最大差1.11e-16、Adam状态5.88e-15、Loss轨迹1.42e-13、验证聚合值1.42e-14。B到1231步：参数最大差5.55e-17、Adam状态1.58e-15、Loss轨迹2.84e-14、验证聚合值差0。采样顺序、sampler、训练schedule、保存的Torch/所有CUDA RNG完全一致。

新增CPU单元测试6项通过，覆盖A/B完整合成窗口、原子步控制梯度、4/8步等价及batch输入。原训练的3056/1226保存点不因上述诊断步数而前移。

另对实际包含8个连续原始子步的合成状态运行GPU编译块，状态、参数梯度及候选控制梯度均通过1e-8容差；首次调用含编译31.36秒。这只证明8步块数值可用，不是其真实窗口吞吐证据。

复现完整窗口实验（输出目录必须不存在）：

```bash
cd /home/zn/flap-system-identification/.worktrees/pahbd-compile-v4
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=4 \
TORCHINDUCTOR_CACHE_DIR=/home/zn/pahbd-runs/compile-v4-20261103-1700/inductor-cache \
/home/zn/anaconda3/envs/pahbd-compile-v4/bin/python scripts/benchmark_pahbd_compile.py \
  --engine step --model-mode A --repeats 3 --updates 8 \
  --output /home/zn/pahbd-runs/compile-v4-reproduction-A
```

使用的是已克隆且回移修复的独立Python环境；原`flap-train-gpu`没有应用编译器补丁。主数据与Hall机械假设保持原合同。

## Profiler证据

同一真实200ms窗口（含历史重放，共443个原子步），独立于吞吐计时执行torch.profiler：

| 指标 | v3参考 | 编译step |
|---|---:|---:|
| CPU profiler事件 | 2,570,171 | 605,015 |
| CUDA设备事件（包含拷贝等，不能全称为kernel） | 531,638 | 189,618 |
| CUDA设备事件累计时长 | 787.75ms | 442.51ms |
| cudaLaunchKernel调用 | 468,622 | 98,698 |
| cuLaunchKernel调用（编译路径） | 未列入参考前20项 | 76,209 |

编译路径观察到443次CompiledFunction及443次CompiledFunctionBackward，证明反向也被捕获。参考侧最大的CPU自身耗时项是cudaLaunchKernel（812ms），随后是mul/div/copy；编译后CPU事件约减少76%、CUDA设备事件约减少64%。这些计数支持小算子及调度开销是主因。Profiler本身记录/解析数百万事件开销很大，以上profile耗时不能替代前述无profiler稳定计时，累计设备事件时间也不是端到端时间。两项profile均正常完成。
