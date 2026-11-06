# PACLD GPU 优化与终端监控

分支 `research/pacld-gpu-batched-v1`；基线 `1757211daadd9b2d93df6369e5bb99da7da4dd22`。测试在本提交的工作区执行，运行配置中 Git revision 因尚未提交而记录基线。未启动342步正式训练。原始数据、Hall合同、CPU配置及历史结果未修改，未访问sealed test。

## 四项修改

1. `predict_batch()`根据有效Macro配对索引是否为空选择传播：有配对返回2B行（FULL+BASE），无配对返回B行FULL。共享Encoder、Slow/Fast和统一Decoder不变，无新增模型。无Macro时Total等于FULL，Macro指标为N/A；梯度、有限性/姿态诊断和DDP归约兼容。两个旧数值诊断脚本也适配了B/2B输出。
2. 仅rank 0在初始化、每20次更新、验证完成后请求GPU遥测。后台单线程执行带2秒超时的`nvidia-smi`，主循环不等待；读取失败返回N/A。每步JSONL保留完整Loss、梯度、窗口索引、时间、显存及GPU采样字段，采样值在下次完成读取前复用，不冒充逐步实测。
3. 新增独立`configs/pacld/gpu.json`，GPU入口默认读取它，validation_interval=32；CPU default.json仍为1000。固定validation仍为156窗口、39航次、0.2/0.5/1/2秒，FULL层级分数选best；末步无论是否整除32都验证及保存。
4. 新增`progress.log`，与stdout同步输出普通文本并flush。第1步、每10步、恢复后首步、验证开始/逐批/结束、checkpoint成功保存、训练结束均有输出；仅rank 0写公共文件。训练JSONL不被文本日志替代。

状态单位保持p_N[m]、v_N[m/s]、Hamilton wxyz四元数（FRD→NED）、omega_B[rad/s]。模型仍是学习的整机状态动力学；没有增加质量、几何或独立翼载荷假设。

## ETA规则与恢复

保存最近40次有效更新的端到端耗时（含读窗、CPU计划准备、传输、forward/backward/Adam），每个新进程首步的编译/图加载耗时不进入滚动窗口，初始化也不进入更新均值。至少20个有效样本且变异系数小于0.75才显示更新ETA；这是显示稳定性规则，不影响训练。初始化/首次编译阶段显示WARMUP / ETA CALIBRATING。

ETA = 剩余更新数 × 滚动平均更新秒数 + 待完成验证次数 × 最近最多10次完整验证平均秒数。待完成次数包含最后不足32步的验证，以及当前更新已完成但尚未完成的验证。首次验证前明确标记pending calibration，不输出精确总ETA；首次完整验证后使用实测耗时。首次验证可能含eval图加载，初期估计可能偏保守。完成时刻由服务器本地`datetime.now()`计算，ETA不是完成时间保证。

checkpoint保存累计optimizer_steps/windows_seen、更新及验证计时样本、累计活动耗时、上次分数和best_step。恢复读取实际累计窗口数，不从0开始；不计停机等待时间。旧GPU checkpoint缺少监控计时可重新校准；配置、精度、全局batch、world size仍须匹配。

## 定向验证

执行：

```bash
PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q \
  tests/test_pacld_gpu_batch.py tests/test_pacld_monitor.py
```

共9项定向测试通过（GPU批量5项、监控4项，分别执行确认）。原4项批量数学测试通过；新增无Macro FP64轨迹/Loss/梯度/裁剪后Adam参数和一、二阶矩对照，ETA/恢复/末步验证计数、遥测超时N/A和CPU/GPU配置隔离测试通过。诊断脚本通过py_compile。未重跑全部精度方案或大规模Profiler。

## 32步真实短测

单卡GPU 0、RTX 4090、B=32、mixed、compile、seed=17；使用指定`flap-train-gpu`环境。32次更新处理1024窗口，其中7个batch无Macro。第32步完整156窗口验证FULL=54.307326，耗时39.36秒，保存`runs/pacld_gpu_monitor_32_seed17/step_00000032.pt`。末步训练FULL=69.544995、Macro=8.392251、Total=77.937246。

排除首步，31次更新平均端到端7.211秒/步，4.438窗口/秒；训练加验证的监控累计活动时间约4分32秒。不同预测时域耗时不同，此短测吞吐不能保证正式训练速度或收敛。首次恢复至33步/1056窗口完成156窗口验证，FULL=53.969854，较32步下降0.337472；仅为流程短测，不据此宣称模型改善。

检查恢复日志时修正了“末步更新已完成、末步验证尚未完成”的ETA计数，并补测试及重跑恢复检查；原短测输出保留。最终恢复目录为`runs/pacld_gpu_monitor_resume33_final_seed17`：从32步/1024窗口开始，完成33步/1056窗口；验证FULL仍为53.969854，耗时27.80秒，保存step_00000033.pt。更新完成但验证尚未开始时显示约39秒ETA，验证完成后归零；初始化则明确WARMUP。

## 无Macro收益与数值结果

`verify_pacld_monitor.py`选取真实train前32个0.2秒窗口，断言其自然有效Macro配对为0，未删除监督。旧版2B传播与新版本均使用同一初始参数、同一batch、mixed+compile；交替执行各5次，首轮图加载不计稳态，各4次取均值。每次从相同模型状态和新Adam开始，包含forward、backward、裁剪和Adam，不包含数据准备。没有并发GPU训练干扰此测量。

| 项目 | 实测 |
|---|---:|
| 原2B更新 | 1.40149 s |
| FULL-only更新 | 1.29763 s |
| 加速 / 耗时下降 | 1.0800× / 7.41% |
| FULL轨迹最大绝对差 | 7.11e-9 |
| Total Loss绝对差 | 2.01e-9 |
| 梯度最大绝对差 / 相对L2差 | 2.38e-7 / 6.44e-8 |
| Adam后参数最大绝对差 | 8.20e-8 |
| Adam一、二阶矩最大绝对差 | 2.98e-8 |

mixed的网络部分使用FP32，B/2B矩阵形状变化会改变舍入，不宣称逐位相等。FP64聚焦测试与上述mixed实测共同支持数值一致性。共享历史编码仍需运行，小模型也受调度开销影响，不能把删除BASE解读为整步2倍加速，或把这个单batch计算收益当整个训练收益。

文本面板实际格式化、文件写入/flush及stdout重定向文件写入的100次均值为0.0266 ms/完整面板；每10步一次，摊销约0.00266 ms/步。测量是本机页缓存文件写入，非fsync磁盘持久化或慢远程终端延迟；不包含异步遥测子进程消耗。日志代码不访问随机数或修改模型/optimizer。

双卡NCCL工程检查使用原脚本、global B=8、mixed、0.2秒无Macro真实batch：相对单卡Loss差0，梯度最大差6.20e-6、相对L2差5.30e-7；恢复后的下一批索引相同、下一次Adam参数最大差0。该检查不作为DDP吞吐结论。

原始数值记录留在`runs/pacld_gpu_monitor_checks/no_macro.json`和`ddp_no_macro.json`，checkpoint、逐步日志和微基准输出留在runs，不发布生成的大文件。可复现命令：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/verify_pacld_monitor.py \
  --output runs/pacld_gpu_monitor_checks_new/no_macro.json
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/verify_pacld_ddp.py \
  --batch-size 8 --horizon 200000 --parity \
  --output runs/pacld_gpu_monitor_checks_new/ddp_no_macro.json
```

## 实际可读日志摘录

以下来自第30步，不是示例假数据；首次完整验证前总ETA保持待校准：

```text
==============================================================
 PACLD GPU TRAINING | seed=17 | GPU 0 | Batch=32
==============================================================
 Status        : TRAINING | WARMUP / ETA CALIBRATING
 Progress      :   30 / 32 steps [ 93.8%]
 Windows       : 960 / 1024
 Time elapsed  : 00:03:35
 ETA remaining : N/A (dynamic estimate)
   Updates     : 00:00:14
   Validation  : N/A (pending calibration)
 Finish around : N/A (server local time)

 Performance
   Batch time  :     7.091 s/step
   Throughput  :     4.513 windows/s
   GPU usage   :        51%
   GPU memory  :     2.968 GiB

 Training Loss
   FULL        :    45.952
   Macro       : N/A
   Total       :    45.952
   Grad norm   :     8.338
   Clipped     : No

 Validation
   Last step   : N/A
   FULL score  : N/A
   Best score  : N/A (step N/A)
   Next val    : 32
 Checkpoint    : N/A
==============================================================
```

## 命令

以下命令在工作树根目录执行：

```bash
cd /home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1
```

实际32步短测监控（已完成，`tail -F`仍可查看）：

```bash
tail -n 45 -F runs/pacld_gpu_monitor_32_seed17/progress.log
```

正式单卡342步 / 10,944窗口命令，**仅提供，未执行**；输出目录须不存在：

```bash
nohup /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld_gpu.py \
  --config configs/pacld/gpu.json --device 0 --batch-size 32 \
  --precision mixed --compile --steps 342 \
  --output runs/pacld_gpu_b32_seed17_10944windows \
  > runs/pacld_gpu_b32_seed17_10944windows.console.log 2>&1 &
echo $! > runs/pacld_gpu_b32_seed17_10944windows.pid
```

监控及停止：

```bash
tail -n 45 -F runs/pacld_gpu_b32_seed17_10944windows/progress.log
kill -INT "$(cat runs/pacld_gpu_b32_seed17_10944windows.pid)"
```

停止会中断当前工作；恢复使用最后成功保存的checkpoint，未保存的更新不会保留。前台运行也可Ctrl-C。不要把`best.json`中的best checkpoint自动视为最近checkpoint；连续恢复应选最新成功保存的step文件。

恢复示例（从本次32步checkpoint继续到正式累计342步；**未执行**）：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld_gpu.py \
  --config configs/pacld/gpu.json --device 0 --batch-size 32 \
  --precision mixed --compile --steps 342 \
  --resume runs/pacld_gpu_monitor_32_seed17/step_00000032.pt \
  --output runs/pacld_gpu_b32_seed17_resume342
```

该命令仅再处理310步/9920窗口，累计仍为342步/10944窗口。DDP仍可用torchrun启动，全局batch须能整除world size；公共日志由rank 0输出。
