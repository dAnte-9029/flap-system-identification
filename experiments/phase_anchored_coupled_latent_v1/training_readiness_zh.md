# PACLD v1 训练体系与功能分工验证准入

基线：`31e5342a060bd628bde609532142cf4c3fea688f`；分支：`research/phase-anchored-coupled-latent-v1`。本报告、配置、源码和机器结果共同提交，实际源码提交由 `git log -1 -- src/system_identification/training/pacld.py` 定位。JSON中的Git身份记录的是本轮实现期间的基线，不伪称为已提交的新源码SHA。

**结论：训练、恢复、诊断和短程CPU/CUDA计算链路可运行；未完成正式训练，不能据此证明快慢功能分工、泛化优势或控制效能。** 只进行了3次真实train optimizer更新。0.2秒Macro监督极稀疏；0.5/1/2秒覆盖足以实施指定辅助Loss。GPU逐窗口执行比CPU慢，当前瓶颈与后续优化边界见性能部分。

## 1. 新增实现与第一阶段核查

- `data/pacld.py`：独立轻量只读视图，复用`NativeWindows`，只收录Hall合格train/validation；保留window/flight/segment、原生时间、控制、mask、质量、cycle_pairs。没有调用访问`config.latent`的PA-HBD包装器。`initial_state()`仍负责12/6隐状态零初始化。
- `training/pacld.py`：纯状态FULL Loss、来源均衡Macro Loss、train-only权重校准、独立梯度诊断、航次/时域均衡采样、固定验证选择、独立checkpoint和恢复。
- `evaluation/pacld.py`：原生多步指标、同一配对BASE/FULL跨拍误差、q/vD拍内误差与相位关联、隐状态/载波轨迹及冻结响应评价适配。
- `scripts/run_pacld.py`：smoke/train/validate/calibrate，CPU或显式CUDA设备、显式步数、独立输出、checkpoint恢复。`audit_pacld_pairs.py`完成覆盖普查；`verify_pacld_training.py`完成有界准入运行。
- `tests/test_pacld_training.py`及本目录机器结果；扩展`configs/pacld/default.json`，保持第一阶段模型配置数值不变。

已核查：一次`encode()`获得同一Belief对象，再分别传播FULL和BASE；解码器、隐状态演化和Lie-group积分完全共享。**没有修改`models/pacld/`、PA-HBD源码或数学模型。** 仅训练中无有效Macro配对时省略无用BASE传播；FULL照常训练。没有新增正则项、Slow/Fast标签或独立辅助预测Head。

## 2. Loss定义、单位与梯度

对原生时刻的观测e，采用残差`r=[Δp_N, Δv_N, Log(R_pred^-1 R_obs), Δomega_B]`，单位依次为m、m/s、rad、rad/s。令`w=mask*quality`，`sigma²`为观测固定噪声方差：

`ell_e = sum(w * r² / max(sigma², 1e-4)) / max(sum(w), 1)`。

**L_full**：先分别在EKF族（位置/速度和姿态）与gyro族内对事件平均，再两族等权，沿用PA-HBD已验证的纯状态定义。完全不调用其预测方差辅助项或默认cycle项。无有效mask的事件跳过；可学习预测方差不参与主监督分母。历史滤波中已有的不确定性机制保持原样。

**L_macro**：只验证并使用原`cycle_pairs`。位置/速度/角速度比较`(pred_j-pred_i)-(obs_j-obs_i)`；姿态比较`Log((R_obs_i^-1 R_obs_j)^-1 (R_pred_i^-1 R_pred_j))`。成对权重为两端mask乘积乘最小quality，固定噪声为两端方差之和（同样下限1e-4）。先在同一来源内对有效配对平均，再三个已有来源等权，避免高频gyro配对数淹没其他来源。保留实际时间差，不除以Δt改成变化率。

拒绝不同来源、非正/倒序时间、无效mask、错误availability、重复配对及不满足原相位匹配条件的配对。没有有效配对时Macro为`None`，不计入Macro汇总分母；总Loss只等于FULL，不伪造零误差。

`L_total = L_full + lambda_macro * L_macro`。两个Loss均通过真实自主传播作用于共享Encoder/Head、Slow、包络和唯一Decoder，无detach。BASE关闭显式载波，因此不要求其对关闭路径有梯度。每步记录两类独立梯度范数和模块范数，固定lambda不由梯度比或validation自动调整。

BASE保留包络、历史相位推断和物理状态反馈；Macro是辅助跨拍预测约束，**不是Slow真值监督**。未来记录控制仅用于事实条件预测，未来Hall、扑频或物理观测不会注入传播。

## 3. cycle_pairs全覆盖普查

普查对象：343,108个train窗口/114航次，112,933个validation窗口/39航次；0.5秒历史，四个指定时域，main_50hz与sparse_10hz，均`shared_periodic_eligible=True`。所有重叠窗口只作为覆盖计数，不当作独立统计样本。

[完整48行分组CSV](training_readiness/pair_coverage.csv)给出每个split×时域×采样组×来源的合格/有效航次数、含配对窗口数、配对数及比例。`valid_per_candidate`的分母是有离线相位且还有后续同来源目标的候选起点数；`window_ratio`是含有效配对窗口/合格窗口。

普查采用与原读取器相同的最近一周期后继、半最近原生相位间距容差和30 ms编码器括号规则，仅对**离线编码器相位**使用原有插值，不插值物理状态。全量按窗口扫描，逐航次检查相位单调性；16个时域/组/split代表窗口逐一与未修改的`NativeWindows`配对数一致，另有数值原算法对照测试。[读取器核对](training_readiness/pair_reader_checks.json)。没有放宽Hall质量阈值或改变冻结边界。

下表每个单元为“有效航次数 / 含配对窗口数 / 有效配对数”；LP同时覆盖位置与速度。

| split | 时域s | 采样组 | 合格窗口 | LP | 姿态 | 角速度 |
|---|---:|---|---:|---|---|---|
| train | 0.2 | main_50hz | 78,554 | 3 / 6 / 6 | 9 / 24 / 24 | 8 / 48 / 51 |
| train | 0.2 | sparse_10hz | 14,128 | 0 / 0 / 0 | 2 / 7 / 7 | 0 / 0 / 0 |
| train | 0.5 | main_50hz | 75,648 | 90 / 75,275 / 731,854 | 90 / 75,302 / 751,717 | 90 / 75,350 / 1,524,422 |
| train | 0.5 | sparse_10hz | 13,600 | 24 / 13,008 / 19,985 | 24 / 13,430 / 50,508 | 24 / 13,514 / 134,190 |
| train | 1 | main_50hz | 70,836 | 90 / 70,836 / 2,293,696 | 90 / 70,836 / 2,321,377 | 90 / 70,836 / 4,624,549 |
| train | 1 | sparse_10hz | 12,818 | 24 / 12,818 / 79,575 | 24 / 12,818 / 168,821 | 24 / 12,818 / 422,814 |
| train | 2 | main_50hz | 65,914 | 90 / 65,914 / 5,116,898 | 90 / 65,914 / 5,163,209 | 90 / 65,914 / 10,260,978 |
| train | 2 | sparse_10hz | 11,610 | 24 / 11,610 / 183,056 | 24 / 11,610 / 372,705 | 24 / 11,610 / 920,193 |
| validation | 0.2 | main_50hz | 27,775 | 0 / 0 / 0 | 1 / 2 / 2 | 1 / 1 / 1 |
| validation | 0.2 | sparse_10hz | 2,970 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| validation | 0.5 | main_50hz | 26,964 | 27 / 26,887 / 261,091 | 27 / 26,891 / 264,763 | 27 / 26,894 / 546,558 |
| validation | 0.5 | sparse_10hz | 2,429 | 12 / 2,219 / 3,060 | 12 / 2,391 / 7,880 | 12 / 2,416 / 21,403 |
| validation | 1 | main_50hz | 25,638 | 27 / 25,638 / 825,941 | 27 / 25,638 / 830,618 | 27 / 25,638 / 1,679,984 |
| validation | 1 | sparse_10hz | 1,751 | 12 / 1,751 / 10,742 | 12 / 1,751 / 22,237 | 12 / 1,751 / 55,839 |
| validation | 2 | main_50hz | 24,397 | 27 / 24,397 / 1,881,203 | 27 / 24,397 / 1,887,008 | 27 / 24,397 / 3,810,009 |
| validation | 2 | sparse_10hz | 1,009 | 12 / 1,009 / 15,984 | 12 / 1,009 / 31,918 | 12 / 1,009 / 78,580 |

0.2秒train仅66个窗口具有任一有效配对（main 59、sparse 7），validation仅3个main窗口，sparse为0；无法支持可靠的0.2秒Macro功能结论。0.5秒各来源含配对比例为91.35%–99.74%；1秒/2秒各来源均100%。因此Macro主要由0.5秒及更长窗口提供，0.2秒FULL训练保持正常。配对为近似相同机械相位，不能称为精确Hall零点截面。

全量逐窗口计数保留在本机`/tmp/pacld_pairs_task2/windows_{train,validation}.parquet`，可用覆盖脚本重新生成；交付仓库保存必要汇总和读取器检查，不添加新的冻结数据版本。

## 4. train-only初始lambda校准

seed17，未训练width128/float64模型。对有配对的train窗口按“时域×采样组轮转；组内航次固定随机排列；航次内窗口无放回”选64个窗口，覆盖46个航次及全部四个时域。0.2秒main选9个/9航次，sparse选其全部7个/2航次；其余六个时域/采样组各8个/8航次。此规则刻意保留稀少的0.2秒配对，但不把这些重复窗口视为独立证据。

[校准身份、每窗Loss和分布](training_readiness/calibration.json)：

| Loss | min | Q25 | median | Q75 | max |
|---|---:|---:|---:|---:|---:|
| full | 17.755898 | 41.240487 | 68.937257 | 89.789666 | 209.486101 |
| macro | 0.168078 | 3.463619 | 6.248055 | 10.121426 | 234.705090 |

按指定公式，未裁剪比值为1.103339，最终固定 **lambda_macro=1.0**。64窗校准耗时57.15秒（CPU，包含读取与前向，无optimizer更新）。没有用validation调整权重。该比例只做初始数值尺度校准；Loss尾部较宽，不能保证Macro在训练全过程都不主导，需继续查看已记录的梯度范数。

## 5. 定向测试与恢复

最终定向测试 **14 passed in 19.16 s**。覆盖纯状态Loss与原实现一致、两Loss分别有限/梯度可达、无配对FULL更新、非法配对拒绝、SO(3)相对旋转、单次encode与相同Belief、未来目标值不改变传播、ZOH等价分段、只读包装及sealed拒绝、checkpoint模型/Adam/RNG/采样器恢复后下一步数值一致，以及完整训练循环的保存/恢复。没有重复第一阶段全部结构审计。

checkpoint schema为`pacld_training_v1`，保存模型、Adam、配置、seed、Python/NumPy/Torch/CUDA RNG、采样器、当前step/最佳FULL指标、train/validation合同和固定窗口身份、lambda与Git身份。配置、数据身份或权重不兼容拒绝恢复；PA-HBD checkpoint拒绝加载。正式训练输出不覆盖既有目录/检查点。普通`train --steps N`中的N是含已完成步骤的**累计目标**。

实际3步readiness checkpoint在`/tmp/pacld_training_readiness_task2/readiness.pt`，已CPU重新加载：step=3、43组Adam参数状态、所有模型参数有限、lambda=1。此检查点保留readiness专用数据/验证身份，不冒充正式训练恢复点；正式恢复命令使用正式入口产生的检查点。

## 6. 实际短更新与验证

合成窗口完成FULL/Macro forward/backward；真实main_50hz和sparse_10hz均覆盖0.2秒无配对和0.5秒有配对情况。所有Loss和梯度有限。只进行了下列**3次真实优化更新**，第三次继承前两次的模型和Adam状态迁移到CUDA；合成单元测试中的optimizer步骤仅属工程测试。

| 步 | 设备/窗口 | FULL | Macro | FULL梯度L2 | Macro梯度L2 | 加权梯度比 | 实测更新s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | cpu / main_50hz 0.5s | 79.226701 | 22.598548 | 47.0008 | 14.5826 | 0.3103 | 3.931 |
| 2 | cpu / sparse_10hz 0.5s | 49.243684 | 4.233750 | 33.4579 | 2.6719 | 0.0799 | 3.245 |
| 3 | cuda:0 / sparse_10hz 0.5s | 48.691365 | 4.191791 | 32.8106 | 2.5940 | 0.0791 | 6.577 |

[更新记录及模块梯度](training_readiness/updates.json)。上述不同窗口Loss不应连接成收敛曲线。记录确认Macro梯度到达FULL继续使用的Decoder、Slow、包络及两个Head，而不是独立辅助Head。

短validation由默认seed1701固定成员中每个时域×采样组取第一个窗口，共8窗/2航次；预先确定，未按误差选取。前向评价耗时7.013秒，6窗有配对、647对有效、0对被拒绝。主FULL航次等权Loss=75.479632；BASE Macro=12.111175；FULL同配对误差=12.106323。0.2秒Macro缺失，不记零。

正式选择仅使用FULL主Loss：窗口→同航次同预测时域均值→航次内时域等权→航次等权。Macro独立报告，不参与最优checkpoint排序。正式验证固定每个航次/时域/采样组1窗，即39航次×4时域=156窗；本轮未将8窗短测试冒充完整156窗评价。

FULL原生时刻误差沿用PA-HBD：最近真实测量时刻±30 ms，p/v/R/omega分别为向量误差范数RMSE，航次等权；不插值。8窗短测试：

| 时域s | 位置m | 速度m/s | 姿态rad | 角速度rad/s |
|---:|---:|---:|---:|---:|
| 0.2 | 0.292879 | 1.220929 | 0.050450 | 0.685285 |
| 0.5 | 0.784599 | 2.129145 | 0.145618 | 0.716844 |
| 1 | 2.446715 | 4.082787 | 0.300378 | 0.759096 |
| 2 | 10.005048 | 10.604901 | 0.660420 | 1.025606 |

除2秒姿态只有1航次存在容差内观测外，其余单元均2航次；没有为稀疏组造观测。2秒误差很大，符合仅3次更新尚未训练好的事实，不能据此作模型性能排名。[完整验证汇总与逐窗结果](training_readiness/validation_summary.json)。

## 7. 功能分工接口与观测

同配对BASE/FULL物理增量RMSE按0.5/1/2秒分别输出，0.2秒缺失。例如2秒位置增量误差BASE=2.013971 m、FULL=2.013489 m；这是短测试数值，不是FULL优越性证据。

q指FRD机体系俯仰角速度，vD指NED下向速度。只用实际原生目标，分别报告稳定/控制变化样本的误差、FULL−BASE预测差异和误差对离线相位1/2/3阶sin/cos的描述性回归R²。至少8个样本、覆盖一周期且设计矩阵满秩才给R²，否则为缺失；稀疏采样的高阶结果仍可能混叠。离线相位只用于诊断，不进入模型。

控制变化标签仅为诊断：任一归一化命令相邻变化>0.02，随后0.06秒内为control_change；其余为stable。两个常数固定在配置中，不作Loss加权或数据准入标准，不能把stable解释为无任何控制变化。

| 采样组 | 变量 | 工况 | FULL RMSE | BASE RMSE |
|---|---|---|---:|---:|
| main_50hz | q | control_change | 0.490916 | 0.490941 |
| main_50hz | q | stable | 0.498403 | 0.498481 |
| main_50hz | vD | control_change | 0.811524 | 0.811175 |
| main_50hz | vD | stable | 0.881879 | 0.881410 |
| sparse_10hz | q | control_change | 0.556679 | 0.556735 |
| sparse_10hz | q | stable | 0.495417 | 0.495467 |
| sparse_10hz | vD | stable | 0.815688 | 0.816482 |

各采样组这里只含1个航次；sparse的vD没有control_change有效目标，明确缺失。不能据这些微小差异证明载波已学得周期价值。逐窗相位R²与有效样本数保存在验证JSON，未跨航次混合相位零位作相关性结论。

[完整隐状态原生查询轨迹](training_readiness/latent_traces.json.gz)记录z_s、c、Q(phi)c、f/phi、Slow/包络变化率、同一FULL状态下开/关载波的decoder输出差，以及两种实际轨迹上的输出差。反向包络振荡探针用包络二维角对phi的斜率，与理想抵消载波的−1/−2/−3比较。该8窗各模态斜率约−0.088到0.232；显式载波加速度各轴RMS约0.0023–0.0100 m/s²，当前分支作用较小。不能据初始化附近短窗宣布Fast失效或已实现物理解耦，也未设置额外collapse惩罚。

`evaluation.pacld.frozen_response(model, windows, source, split)`直接调用原`FrozenRollResponse`和`factual_predictor`，保留匹配对/窗口身份/指标；缺少冻结窗口时拒绝重选配对。**本轮没有运行冻结Control/NoControl响应评价，不输出控制响应结论。**

## 8. CPU/CUDA吞吐、内存与瓶颈

Intel i9-14900K、单CPU线程、float64、width128、最大5 ms积分步长。宿主检查时两张NVIDIA RTX4090利用率0%、无compute进程，选择cuda:0；受限环境中不可见GPU，实际CUDA测试经宿主执行完成，不把可用显存当作空闲证明。

下表为真实main_50hz窗口，数据已物化，含共享历史推断、必要的FULL/BASE传播、独立梯度范数诊断及总Loss backward；不含数据加载和optimizer.step。0.2秒该窗口没有Macro配对，省略BASE；其他时域完整双模式。RSS为采样到的整个进程峰值，包含数据与缓存，不能与CUDA allocator峰值等同。

| 设备 | 时域s | 前向s | 反向s | 独立梯度诊断s | 窗口/s | RSS峰值MiB | CUDA分配峰值MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 0.2 | 0.430 | 0.795 | 0.894 | 0.4719 | 1786.4 | — |
| cpu | 0.5 | 0.837 | 1.274 | 2.054 | 0.2401 | 1917.6 | — |
| cpu | 1 | 1.174 | 1.992 | 2.803 | 0.1675 | 2109.7 | — |
| cpu | 2 | 2.075 | 3.622 | 4.621 | 0.0969 | 2515.5 | — |
| cuda:0 | 0.2 | 0.871 | 1.596 | 1.605 | 0.2456 | 2965.0 | 50.3 |
| cuda:0 | 0.5 | 1.528 | 2.626 | 3.952 | 0.1234 | 3095.5 | 73.0 |
| cuda:0 | 1 | 2.298 | 4.278 | 5.314 | 0.0841 | 3292.0 | 100.7 |
| cuda:0 | 2 | 3.949 | 6.746 | 8.085 | 0.0532 | 3693.0 | 156.4 |

sparse_10hz的0.2/0.5秒CPU吞吐分别0.584/0.282窗口每秒；实际3步optimizer耗时见上表。[全部性能记录](training_readiness/performance.json)另保留CUDA reserved峰值和RSS增量。没有从0.2秒外推2秒成本，四个时域均实际运行。相同初始模型/窗口的CPU与CUDA FULL Loss最大绝对差5.68e-14，Macro最大差3.56e-15；这只是本组数值一致性检查。

**性能限制**：当前小型MLP在大量串行Lie子步/观测事件/目标时刻中被调用，两种轨迹分别传播；相位协方差传播和CPU/GPU同步检查也增加开销。每步独立FULL/Macro梯度范数还需两次autograd遍历，2秒CPU为4.62秒、CUDA为8.08秒。CUDA未获得批并行收益，反而约慢1.8–2倍；低显存占用不等于高吞吐。以上瓶颈解释由代码路径和实测支持，未执行kernel级归因分析。

本轮没有修改积分精度或为速度重写数学定义。后续可评估同时间网格批处理、缓存固定协方差转移算子、减少主机同步、诊断降频；任何实现需比较前向、Loss、梯度及optimizer更新等价性后再采用。本轮到准入实现为止。

## 9. 正式训练推荐与命令（未执行）

保留已验证配置：width128，z_s12/c6，float64，5 ms，Adam lr=3e-4，seed17，固定lambda=1，四时域航次均衡采样，0.5秒历史，两种原生采样组，Hall合格窗口；validation seed1701每航次/时域固定1窗，只以FULL选择checkpoint，每1000步验证保存。当前逐窗口实现建议先使用CPU单线程；CUDA入口可用但实测没有提速。示例累计10000步是可显式修改的预算，不是已证明足够收敛的步数。

从当前worktree运行，输出目录必须不存在：

```bash
# 正式训练示例：本轮没有执行
PYTHONPATH=src MPLCONFIGDIR=/tmp/pacld-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld.py --mode train --source native --device cpu --steps 10000 --output runs/pacld_v1_seed17

# 固定完整validation及功能诊断；使用上一命令实际产生的checkpoint
PYTHONPATH=src MPLCONFIGDIR=/tmp/pacld-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld.py --mode validate --device cpu --checkpoint runs/pacld_v1_seed17/step_00001000.pt --output runs/pacld_v1_seed17_validation_1000

# 恢复到累计10000步，另开输出目录，不覆盖既有结果
PYTHONPATH=src MPLCONFIGDIR=/tmp/pacld-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pacld.py --mode train --source native --device cpu --steps 10000 --checkpoint runs/pacld_v1_seed17/step_00001000.pt --output runs/pacld_v1_seed17_resumed
```

训练状态/步级Loss/梯度写入`training.jsonl`，固定validation与checkpoint分步保存，`best.json`只追踪FULL主指标。默认使用本次交付的校准JSON，不随validation调权。`--mode calibrate --coverage <train普查parquet>`可复现校准，但必须从seed17初始模型、独立新输出目录运行。

## 10. 证据边界与剩余风险

- **VERIFIED**：实际配对覆盖、损失与共享梯度路径、短CPU/CUDA更新、checkpoint恢复、四时域耗时与原生时间评价接口。没有访问sealed test，也没有拟合Slow/Fast标签。
- **SUPPORTED**：0.5秒以上可提供广泛跨拍辅助监督；当前逐窗口GPU吞吐不足以带来加速。覆盖窗口重叠很强，统计独立性仍以航次为基础。
- **UNRESOLVED**：Slow/Fast功能可辨识性、BASE辅助监督是否改善FULL泛化、包络反向振荡或Fast失效是否在正式训练后出现、控制可辨识性、长期稳定性和MPC收益。本轮诊断不把隐状态解释为唯一真实物理快慢态。
- 固定Hall零点仍是工作约定，20 ms发布保护仍非实测上界；稀疏组拍内相位诊断可能混叠，0.2秒Macro不能可靠比较；loss中位数校准不保证梯度始终平衡。
- 未训练的2秒预测仍有明显误差；8窗/2航次短validation不代表完整validation结论。未运行Control/NoControl冻结评价、正式长期训练或第三阶段。

执行记录：新增机制定向测试14项通过（19.16秒）；测试命令为`PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_pacld_training.py`。真实准入命令为`PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/verify_pacld_training.py --cuda-device cuda:0 --output /tmp/pacld_training_readiness_task2`（再次运行须换新输出目录）。
