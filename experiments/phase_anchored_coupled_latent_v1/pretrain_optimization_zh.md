# PACLD v1 正式训练前性能与稳定性修复

基线：`371765df7b94d89864d415fd19b6c31f68518705`；分支：`research/phase-anchored-coupled-latent-v1`。

**结论：具备按本轮配置启动seed17正式训练的工程准入条件，推荐CPU单线程；本轮没有启动长期训练。** 模型、Loss、固定lambda=1、Adam lr=3e-4、float64、最大5 ms积分、Hall准入、四时域、原生目标与FULL选择规则均保持不变。范数10裁剪是明确新增的优化稳定性设置，不与未裁剪训练声称等价。短运行不证明泛化或长期数值稳定。

## 1. 修改范围与原因

| 文件 | 改动 |
|---|---|
| `configs/pacld/default.json` | 独立梯度诊断间隔100；`max_grad_norm=10.0`；终端打印间隔100；LRU最多8航次、256 MiB |
| `data/pacld.py` | PACLD专用有界LRU缓存原生表、samples及encoder；统计命中、缺失、驱逐、Parquet读取、占用 |
| `data/pahbd.py` | 仅提取`_encoder_counts(fid)`读取hook；默认仍每窗读取并按timestamp排序，原PA-HBD行为不变 |
| `training/pacld.py` | 诊断调度、状态/梯度有限检查、全局L2裁剪、分段耗时、失败记录、checkpoint v2、低频终端日志 |
| `training/pacld_compatibility.py`、`configs/pacld/source_compatibility.json` | 分离模型/Loss/数据/Sampler/优化/评价版本与实际源码指纹、Git身份；未知源码拒绝恢复 |
| `scripts/run_pacld.py` | 保持CLI；smoke与正式入口均使用配置的诊断及裁剪设置；加载时核对Sampler索引 |
| `scripts/verify_pacld_pretrain.py` | 从指定Git基线提取原读取器和Loss，进行原生缓存核对和独立短程CPU/CUDA前后比较 |
| `tests/test_pacld_pretrain.py` | 诊断观察性、裁剪、恢复、源码兼容、LRU、异常停止定向测试 |

没有修改`models/pacld/`或PA-HBD模型/训练/评价公式。共享数据hook只是原读语句的提取，已与基线读取器逐字段比较。

## 2. 诊断降频与等价性

默认完整诊断执行在全局更新**1、101、201……**（已完成step对100取模为0），恢复不会重新从“首步”计数。其他更新仅执行总Loss backward，日志明确`gradient_norms=null`。每步FULL/Macro/total、有效配对数、窗口身份、裁剪与耗时仍写入`training.jsonl`；终端只打印首步、每100步及最后一步简要信息。

在相同模型、窗口、非空Adam状态和seed下，诊断开/关的总Loss、正常backward梯度、更新后模型参数和Adam状态均**CPU逐位一致**。另对Git基线与新实现关闭裁剪后的更新进行了逐位对照，也一致。独立`autograd.grad(..., retain_graph=True)`仅观察，不写入正常参数`.grad`，不会改变优化。

## 3. 梯度裁剪与异常停止

顺序为：检查总Loss及自主状态有限性/四元数单位范数 → 可选独立诊断 → 总Loss backward → 检查全部梯度有限 → `torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0, norm_type=2, error_if_nonfinite=True)` → Adam → 参数有限性检查。

所有具有梯度的可训练参数参与同一个全局范数，不分Slow/Fast单独裁剪。记录`pre_clip_grad_norm`、`max_grad_norm`、`clip_coefficient`和`gradient_clipped`。实际是否缩放按PyTorch的`min(1, max_norm/(norm+1e-6))`判定。配置可显式设JSON `null`关闭裁剪，不能用未声明阈值替换默认值。

定向验证：低于阈值时开/关裁剪的参数与Adam状态逐位一致；超阈值时梯度有限且范数不超过阈值；同配置保存/恢复后下一步更新及优化器状态逐位一致。真实优化组16/16步均触发范数10裁剪，说明此设置在初始化阶段有实质影响，不能称其为等价性能优化或已经调优的最佳参数。

非有限Loss/状态/梯度在optimizer前停止；参数更新后若非有限也立即停止，不保存坏checkpoint。正式循环写`failure.json`包含已完成step、尝试窗口和错误。已保存checkpoint保留；尚未checkpoint的更新在进程意外中断后不承诺保留。

## 4. 有界缓存与原生窗口一致性

LRU键为`(split, flight_id)`，每个读取器实例独立，默认同时受8航次和256 MiB约束。缓存仅存原生DataFrame、samples和encoder，不存训练窗口或隐状态。表只读复用；命中不再读Parquet，驱逐后重新读取。超过单项容量的航次只作为当前活动航次使用，不纳入保留缓存。

容量占用是`DataFrame.memory_usage(deep=True)`的表内存估计，**不是进程总RSS上限**。已有NativeWindows的split级窗口索引/Hall表初始化保持不变，不计入新增LRU容量；当前活动对象与其他训练张量由RSS实测覆盖。没有预物化343,108个训练窗口，没有改变Encoder可见信息。

固定真实main/sparse窗口，与Git基线原实现递归比较全部Window字段：物理bootstrap、观测值、时间、availability、mask、quality、variance、控制序列、Hall、cycle_pairs及全部metadata均相同。还比较了同一缓存航次的不同1秒窗口；跨split键、容量和超大单项行为另有定向测试。

| 场景 | 原实现读取次数 | 新读取次数 | 原耗时ms | 新耗时ms |
|---|---:|---:|---:|---:|
| cold_main | 8 | 8 | 126.31 | 112.65 |
| hit_main | 1 | 0 | 117.23 | 72.06 |
| cold_sparse | 8 | 8 | 71.24 | 71.66 |
| switch_hit_main | 8 | 0 | 100.86 | 67.45 |
| evict_with_third | 8 | 8 | 110.99 | 120.75 |
| reload_sparse | 8 | 8 | 69.06 | 75.53 |
| reload_main | 8 | 8 | 109.00 | 106.30 |

上述为最多2航次的定向LRU测试，包含冷读、命中、跨航次返回、驱逐和重载。命中后仍需构造原生观测/控制/配对对象，所以耗时不会降为零。OS页缓存未清空，这些不是磁盘冷缓存基准。

正式Sampler的前16次抽样来自16个不同航次，新缓存实际为**0命中、16缺失、8驱逐、128次Parquet读取**，最多保留8航次；保留表估计峰值150.11 MiB、结束141.02 MiB。不能把warm-hit的读取收益套用到这个跨航次序列。缓存在回访未驱逐航次时有效，但当前主要加速不是I/O。[全部核对结果](pretrain_optimization/cache_checks.json)。

## 5. 真正的连续短程吞吐

正式`FlightBalancedSampler`、seed17、同一前16窗、同一初始模型，四组独立临时运行：原版CPU、优化CPU、原版cuda:0、优化cuda:0；**每组16次真实更新，共64次对照更新**，不是一条64步正式训练。优化组在第8步保存并恢复，再完成后8步；中点检查下一Sampler抽样完全一致。

序列覆盖：main_50hz的0.2/0.5/1/2秒分别4/5/1/2窗；sparse_10hz的0.5/2秒分别3/1窗。每组16航次、包含15次航次切换。未人为补选窗口来制造缓存命中；也不能把这16窗视为所有航次/时域×采样组的精确分布代表。

原版使用基线原始Loss与每步诊断、不裁剪；优化版诊断只在首步执行、裁剪10。原有无Macro配对窗口不进行无用BASE传播的规则两组完全相同，**没有减少目标、改变步长或额外删除BASE传播**。由于裁剪不同，后续参数/Loss允许不同；其计算量合同相同。

硬件：Intel i9-14900K，CPU单线程；NVIDIA RTX4090 cuda:0。宿主检查显示两卡利用率0%、无compute进程后使用cuda:0。各组串行运行，无相互争抢。float64、width128、5 ms不变。

“更新s”包含实际Sampler抽样、Parquet读取/窗口构造、搬运、前向、反向、诊断、梯度检查/裁剪及Adam；启动、日志和checkpoint恢复另有整个16步墙钟。CPU RSS为5 ms采样峰值，CUDA为allocator峰值。

| 设备/版本 | 平均更新s | 窗口/s | 16步墙钟s（含日志/恢复） | CPU RSS MiB | CUDA峰值MiB |
|---|---:|---:|---:|---:|---:|
| cpu_baseline | 4.5189 | 0.2213 | 72.839 | 2577.68 | — |
| cpu_optimized | 2.6701 | 0.3745 | 44.070 | 2770.15 | — |
| cuda_baseline | 8.8047 | 0.1136 | 141.212 | 3543.14 | 158.16 |
| cuda_optimized | 5.1848 | 0.1929 | 83.979 | 3713.60 | 156.34 |

CPU更新吞吐约提升1.69倍，CUDA约1.70倍；计入日志及优化组中点恢复的整段墙钟，CPU约提升1.65倍、CUDA约1.68倍。优化CUDA仍约慢于优化CPU1.94倍，建议正式seed17使用CPU。

平均分段时间（秒/更新；优化组均值包含一次完整诊断）：

| 组 | 读取/构造 | 搬运 | Encoder | FULL传播 | BASE传播 | 反向 | 独立诊断 | 梯度检查/裁剪 | Adam |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu_baseline | 0.1195 | 0.0009 | 0.3365 | 0.2709 | 0.2576 | 1.3989 | 1.9852 | 0.0069 | 0.0011 |
| cpu_optimized | 0.1231 | 0.0009 | 0.3390 | 0.2811 | 0.2349 | 1.4011 | 0.1185 | 0.0044 | 0.0036 |
| cuda_baseline | 0.1053 | 0.0078 | 0.7005 | 0.5042 | 0.4713 | 2.9610 | 3.8588 | 0.0331 | 0.0043 |
| cuda_optimized | 0.1107 | 0.0078 | 0.6837 | 0.5137 | 0.4919 | 2.9460 | 0.2386 | 0.0288 | 0.0067 |

其余时间包括配对验证/Loss计算、状态检查和Python对象管理。读取/构造约0.11–0.12秒，CPU搬运约0.001秒，CUDA搬运约0.008秒；当前不是主瓶颈。串行细积分与反向图仍主导，GPU小kernel/主机调用与同步开销使其较慢。没有在本轮做批量化重构。

[CPU原版](pretrain_optimization/cpu_baseline.json)、[CPU优化](pretrain_optimization/cpu_optimized.json)、[CUDA原版](pretrain_optimization/cuda_baseline.json)、[CUDA优化](pretrain_optimization/cuda_optimized.json)保留每窗身份、完整阶段耗时、FULL/Macro/total、梯度和内存记录。基线JSON明确记录有效设置为诊断1/无裁剪/原读取器，避免与附带的新配置模板混淆。

## 6. 有限性、Loss与裁剪记录

四组均完成16步，Loss/梯度检查通过；优化组额外逐步检查自主状态、四元数与更新后参数，未触发异常。优化CPU总Loss范围约22.66–235.22，裁剪前全局梯度范数约12.40–248.74；这些来自不同航次/时域，不是收敛曲线。16步均裁剪，裁剪前范数仍完整记录。诊断执行步列表为`[1]`，其余`gradient_norms=null`，没有沿用旧诊断值。

CPU与CUDA优化组从相同状态出发、相同clip设置，其逐步Loss只存在浮点量级差异；没有将短Loss下降或裁剪行为当作泛化证据。本轮只读真实train用于新测量；不新增validation选择、不访问sealed test。

## 7. Checkpoint兼容与恢复

新schema为`pacld_training_v2`，保留模型、Adam、seed/RNG、Sampler状态、step、配置、数据合同、固定lambda与Git SHA，并增加：

- 模型语义`pacld_model_v1`、Loss语义`pacld_full_macro_v1`、数据/Sampler/评价语义版本，以及独立优化版本`pacld_adam_clip_v1`。
- 实际运行源码AST指纹与已审核兼容指纹表；忽略注释、docstring、行号，涵盖模型/相位/几何/控制合同、Loss、数据、Sampler、优化入口与评价。
- 明确的全局L2裁剪规则、阈值、Adam配置与先后次序，保存Sampler身份。

恢复核对保存与当前源码都属于同版本下的已审核指纹，核对模型/dtype/Adam嵌入配置、裁剪规则、数据身份和Sampler身份，再恢复RNG/采样器。仅`data_cache`、诊断间隔、终端打印间隔可作为运行配置变化兼容；Git SHA本身变化不构成拒绝理由，变化会记录到`resume_compatibility`和`run_contract.json`。

未知源码直接拒绝；已证明等价的日志/缓存/性能源码变动需在`source_compatibility.json`中留下审核记录并添加同版本指纹。模型/Loss数学变化必须新语义版本，不能借添加“兼容”指纹绕过。

本轮有一个实际兼容案例：性能测量后仅增加嵌入checkpoint模型/dtype/Adam一致性拒绝检查，未改update/Loss/Sampler；旧、新两个优化源码指纹被明确记录为同版本。两份真实CPU/CUDA第8步checkpoint均以最终代码重新加载成功，各恢复43组Adam状态。[恢复证据](pretrain_optimization/source_compatibility_restore.json)。

定向测试还确认相同设置恢复后的下一步参数与Adam状态逐位一致，Sampler下一抽样一致且不重新抽取首个已完成窗口。裁剪阈值改变、模型/Loss版本或未知指纹、数据/Sampler身份变化均拒绝。缺少源码与裁剪合同的旧v1 checkpoint不自动迁移；已有历史smoke文件未改写。正式checkpoint仍每1000步和最终步保存，validation成员/选择规则不变。

## 8. 定向验证结果

**23 passed in 37.68 s**：本轮新增检查与既有14项训练机制回归。包括诊断纯观察性、裁剪上下阈值、保存/加载/下一更新、真实Sampler顺序、兼容/不兼容源码与Git变化、配置矛盾拒绝、LRU命中/驱逐/跨split/容量、非有限状态先于optimizer停止。没有重跑第一阶段全套模型结构审计。

真实原生窗口完整一致性见第4节，独立四组连续短更新见第5节。模型数学、FULL/Macro定义、已冻结lambda校准文件及validation选择代码均未改变。使用指定`/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，未调整依赖。

## 9. 10,944步成本估计（非实际长训）

无额外独立诊断的15步，实测均值 **2.5724秒/步**，但样本时域比例不是长期均衡比例。按0.2/0.5/1/2秒分组的无诊断实测均值为：

| 时域s | 更新s | 实测窗数（不含首步诊断） |
|---:|---:|---:|
| 0.2 | 1.3307 | 4 |
| 0.5 | 1.9963 | 7 |
| 1 | 3.1823 | 1 |
| 2 | 5.3689 | 3 |

正式Sampler在长预算内按时域均衡，因此粗估取四时域均值 **2.9695秒/步**：

`10,944 × 2.9695 / 3600 ≈ 9.03小时`（训练更新主体）。

1、101……10901步共110次诊断，使用原版按时域均衡的诊断实测约2.267秒/次估计，额外约0.07小时；合计约 **9.1小时训练更新**。再加11次固定validation（每次156窗）、checkpoint、启动和系统波动，工程排期建议预留 **10–11小时**，这部分余量不是实测总工时。

以上为粗估，不是用0.2秒外推2秒。1秒仅有1个实测航次，16窗没有覆盖全部时域×采样组组合，冷/热页缓存、其他任务和后续数值路径均可能影响实际耗时；没有为本估计运行10,944步，也没有重跑完整156窗validation来伪装成完整成本测量。

## 10. 正式准入与命令（仅提供，不执行）

可以启动seed17的新正式实验：CPU、单线程、float64、5 ms、lr=3e-4、lambda=1、全局范数10、诊断100、固定FULL验证选择。这里的“可以”是工程链路和短程有限性准入，不承诺长期稳定、收敛、快慢职责可辨识或控制泛化；范数10仍是工程先验。

从本worktree运行，输出目录必须不存在；累计目标为10,944步。读取缓存和诊断低频化没有改变冻结数据、状态标签或输入因果边界。

```bash
# 正式启动：本轮未执行
PYTHONPATH=src MPLCONFIGDIR=/tmp/pacld-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld.py --mode train --source native --device cpu --steps 10944 --output runs/pacld_v1_seed17_10944

# 另一终端监控每步Loss、裁剪、耗时与缓存统计
 tail -F runs/pacld_v1_seed17_10944/training.jsonl

# 从实际保存的v2 checkpoint恢复；累计目标仍为10944，新输出目录
PYTHONPATH=src MPLCONFIGDIR=/tmp/pacld-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld.py --mode train --source native --device cpu --steps 10944 --checkpoint runs/pacld_v1_seed17_10944/step_00001000.pt --output runs/pacld_v1_seed17_10944_resumed
```

若只读诊断间隔/打印/缓存配置改变，恢复日志会记录差异；模型、Loss、clip、学习率、数据和验证选择不可静默更换。临时benchmark checkpoint只保留在`/tmp/pacld_pretrain_*_optimized_v1/midpoint.pt`，带专用benchmark身份，不是正式seed17起点。

复现测试：`PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_pacld_pretrain.py tests/test_pacld_training.py`。性能脚本以`--mode benchmark --variant baseline|optimized --device cpu|cuda:0 --steps 16 --output <新目录>`运行；CUDA须先确认设备空闲。所有原始数据、旧结果和工作区无关文件保持原样。本任务到此结束，不进入正式长期训练。
