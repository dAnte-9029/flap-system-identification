# PA-HBD Hall 机械相位合同修正与跨航次周期训练准入

日期：2026-10-29。实现基线：`14d4719`；本报告、配置、验证结果与修改后的代码一起本地提交。没有正式训练、GPU 优化器步骤、超参数搜索、性能排名或 test 数据访问。

## 结论

**共享周期 A/B 入口已通过准入检查，覆盖 9–10 月。** 每个时域 train 均有 114 个独立航次、17 个日期，validation 均有 39 个航次、4 个日期。准入以 Hall 数据质量和用户接受的机械工作约定为依据，不再要求空的人工证书。该结论有条件地采用 20 ms 发布延迟保护假设，不等于已取得真实在线发布时钟证明、独立翼角标定、周期控制增益有效性或 MPC 可用性证据。

## 相位定义、源码与原生数据事实

- **已核实**：冻结 `build_config.yaml` 中 WingPhase 源码 SHA256 为 `0a0ca2ee9da44daa21409dee24aa2901a8f04f7250649b7c7675ebe4ed67b945`。本次读取冻结 `firmware_sources/c9af571575af33d5403fbb479256848797c801e2/src/modules/wing_phase/WingPhase.cpp`，核对该身份；沿用已完成的 17 版本一致性证据，不重复全版本审计。实际仓库为 https://github.com/jiashunjiang2022/PX4-Autopilot 。当前 FUSION 的源码不同，未用它替代日志对应的冻结源码。
- 冻结源码由 Hall 时间在编码器相邻样本间插值得到 `_zero_count`，启动后第一次 Hall 锁定之前相位无效；存在 π/4 的 Hall 再触发保护，该保护不是上下扑切换标签。`phase_rad` 是最近 Hall 参考下的 [0,2π) 圆周相位；`phase_unwrapped_rad` 会随 Hall 重定相归零，不是全航次连续展开。
- 原生表直接保留 `phase_rad`、`phase_unwrapped_rad`、`phase_valid`、`encoder_total_count`；`timestamp_sample_us=-1`，没有 `availability_us`。数据生成器对 encoder_count/flap_frequency 使用下一次采样时间作为各自发布保护，对 wing_phase 没有建立该字段。原生 Hall 表直接来自日志 topic，没有由相位反推伪造 Hall 事件。
- **用户确认的工作假设**：Hall=0；AS5600 正向计数；0 后开始上扑。它由安装信息、固件和日志支持，但不等于独立翼角测量得到的绝对标定。不假设 π 是上下扑精确切换位置。
- 本合同显式覆盖冻结构建合同中旧的 `wing_phase_history_allowed=false` 和逐日志原点输入约定；冻结表、原始 ULog、split 和 REVIEW/EXCLUDE 规则均未改写。

## 因果合同与模型接通

WingPhase 以 5 ms 周期在工作队列运行，消息时间戳回填编码器采样时刻，之后才 publish。AS5600 的调度周期为 10 ms。这里采用 **availability = encoder timestamp + 20 ms**，即按两个名义编码器周期作延迟保护；这依赖“实际发布延迟不超过 20 ms”的研究假设。它不是实测上界，也不能从 5 ms 调度周期推出严格最坏延迟。异常长调度延迟仍是未消除的不确定性；该合同适用于有记录假设的离线研究，不能直接作为已证明的在线时序安全保证。

只有 guarded availability≤决策时刻的观测进入 Belief。为不越过冻结支持边界，历史起点不再强行借用更早的相位或控制：起点周期门关闭，在历史回放中第一条可靠 Hall 观测的测量时刻建立圆周相位与观测方差。随后在 SO(3) 历史回放中进行圆周残差更新，再由同一个 Phase Propagator 沿历史 ZOH 控制传播到决策时刻。初始化读数不重复用于收缩同一相位方差；频率/相位联合协方差继续按现有 SDE 传播。方差仍为未标定的建模假设。

模型内部 phi 是连续积分状态，可超过 2π；后续 Hall 观测只通过 atan2(sin Δ,cos Δ) 残差校正，不能拼接日志 unwrapped 字段。未来 rollout 的接口只有 Belief、查询时刻和候选控制，不注入未来相位、编码器或 IMU。未来状态与编码器只保留为独立损失目标/离线周期配对，不作为流模型输入。周期配对只用编码器相位差，原点不影响配对；不宣称精确翼角 Poincaré 截面。

平均动力学、周期 forcing、显式周期控制三个分支边界保持不变。A/B 的定义差异仍只有显式周期控制交互项，使用同一数值积分和数据合同。`gauge_offset` 仍是数学坐标变换，加载器不按航次设定或优化它。

## 数据质量准入与覆盖

质量判据固定在配置中：圆周相位增量减去编码器增量对应角度的残差绝对值≤0.15 rad；间隔≤50 ms；编码器不反向且计数率≤20 Hz 的宽松异常上限；相位有效且数值有限。异常或长缺测后须有 500 ms 连续恢复。0.15 rad、20 Hz 和 500 ms 是显式研究 QC 选择，不是硬件精度标定，也未根据模型预测误差调优。短缺测由自主传播跨越；正常 2π 跨越和小 Hall 重定相不视为故障。

共享窗口须在历史内有可用 Hall 初始化，历史至目标支持末尾满足连续性，并继承完整频率、冻结 reset、状态与 REVIEW 排除条件。没有独立 hall_event 的窗口不因此被排除。未来区间的传感器 QC 只用于离线训练窗口筛选，不作为部署时可预知的质量标志。

相位不合格而状态合格的窗口保留为 average_only；状态/固件/REVIEW 不满足原冻结准入的窗口不进入本研究加载器。正式 shared_periodic_AB 仅使用共享合格窗口，平均模型可通过显式 average_dynamics 配置使用保留数据，避免平均窗口淹没周期训练。

10 月数据属于冻结 `sparse_10hz`。加载器现同时读取 main_50hz 和 sparse_10hz，并采用各自既有的最大状态间隔（40/160 ms）与姿态年龄（50/100 ms），角速度仍遵守原 50 ms 限制。没有提高原有数据质量容忍度；仅移除了 main_50hz 的硬编码单组选取。稀疏组最长实际历史约 0.609 s，因此回放缓冲配置为 0.65 s。

| split | 时域 | 合格航次 | 合格窗口 | 平均专用窗口 | 初始化不可用/恢复中 | 区间异常/恢复中 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.2 s | 114 | 92,682 | 38,364 | 28,009 | 10,355 |
| train | 0.5 s | 114 | 89,248 | 40,987 | 27,890 | 13,097 |
| train | 1 s | 114 | 83,654 | 44,130 | 27,497 | 16,633 |
| train | 2 s | 114 | 77,524 | 48,713 | 27,246 | 21,467 |
| validation | 0.2 s | 39 | 30,745 | 14,770 | 9,771 | 4,999 |
| validation | 0.5 s | 39 | 29,393 | 15,951 | 9,742 | 6,209 |
| validation | 1 s | 39 | 27,389 | 17,375 | 9,665 | 7,710 |
| validation | 2 s | 39 | 25,406 | 19,118 | 9,636 | 9,482 |

这些窗口高度重叠，不能视为独立实验。以下按日期列出 200 ms 覆盖；全部四个时域及精确固件 SHA 的逐航次记录见附件 CSV。train 合格数据包含 15 个固件版本，validation 包含 5 个版本，二者使用同一 Hall 约定。

| split | 日期 | 状态合格航次 | 共享周期航次 | 共享窗口（200 ms） |
|---|---|---:|---:|---:|
| train | 2026-09-06 | 7 | 6 | 5,777 |
| train | 2026-09-11 | 3 | 3 | 2,229 |
| train | 2026-09-12 | 8 | 8 | 9,914 |
| train | 2026-09-13 | 6 | 6 | 6,537 |
| train | 2026-09-14 | 4 | 4 | 5,091 |
| train | 2026-09-15 | 6 | 4 | 2,742 |
| train | 2026-09-16 | 5 | 5 | 3,088 |
| train | 2026-09-18 | 3 | 3 | 1,586 |
| train | 2026-09-22 | 6 | 6 | 2,923 |
| train | 2026-09-23 | 9 | 9 | 4,108 |
| train | 2026-09-24 | 12 | 12 | 7,397 |
| train | 2026-09-25 | 4 | 4 | 2,719 |
| train | 2026-09-27 | 3 | 3 | 3,791 |
| train | 2026-09-28 | 8 | 8 | 9,813 |
| train | 2026-09-29 | 9 | 9 | 10,839 |
| train | 2026-10-10 | 6 | 6 | 3,969 |
| train | 2026-10-12 | 22 | 18 | 10,159 |
| validation | 2026-09-07 | 10 | 10 | 12,821 |
| validation | 2026-09-17 | 8 | 8 | 4,609 |
| validation | 2026-09-26 | 9 | 9 | 10,345 |
| validation | 2026-10-11 | 12 | 12 | 2,970 |

9 月 15 日六航次中，`60cc2db1…:1`、`22a70d54…:1` 分别有 1,178、1,084 个超过 0.15 rad 的参考创新，幅度达到约 ±π；恢复时间要求使这两航次的共享周期覆盖为零。其余四航次仍有 75–135 个超阈值事件，只保留稳定区间，200 ms 共 2,742 个窗口。具体时间戳在 `anomaly_locations.csv.gz` 中，可按 flight_id 定位。没有用 IMU、姿态相关性或预测 RMSE 调相。

跨日期源关系核对覆盖全部允许的 train/validation 原生相位表：有独立 Hall 记录时，从 Hall 两侧编码器插值得到事件计数，再用后继 wing_phase 推回事件相位；各航次合格配对的 95% 绝对圆周误差量级约 10^-8 rad（见 sensor_alignment.csv）。这是同一固件插值计算的可重现性，不是机械传感器精度或独立翼位真值。无独立 Hall 的日期用相位增量与编码器关系定位稳定区间；结论仍依赖已确认安装约定。10/10、10/11、10/12 均有 Hall 配对证据，不再因日期或缺人工证书被整体阻断。

## 已执行验证与代码变化

- `data/pahbd_phase.py`：以原生 Hall 质量与冻结窗口支持区间直接准入，删除空 certified_intervals 门槛及按日志计数锚证书机制。
- `data/pahbd.py`：Hall 圆周观测、20 ms 因果保护、分组原生状态合同、历史内初始化；保留原生目标、分段、时间顺序和 ZOH。
- `models/pahbd/belief.py`：首次可信 Hall 建立相位与方差，之后保留已有圆周残差和 SO(3) 更新；没有重设计 Backbone。
- `training/pahbd.py`、`run_pahbd.py`：按实际覆盖与共同 reference_id 准入，每一时域 train/validation 必须非空；正式入口和 formal_fit 使用 Hall-only 索引；相同 seed/index 的航次均衡、多时域采样不变。配置及 checkpoint 元数据包含机械假设、发布时间假设和未绝对标定限制；新 checkpoint schema 为 pahbd_v3_hall，避免旧相位语义的 checkpoint 被静默继续训练。
- CPU float64，seed=17，默认 width=128/latent=12：实际加载每个日期×时域一窗口，train 68 组、validation 16 组均符合运行时合同；全候选窗口的频率 bootstrap 检查没有缺失/过期窗口。
- 真实 train 三个代表窗口完成完整历史 Belief + 200 ms 原生目标自主预测 + backward，无 optimizer.step。首次 Hall 初始化与日志圆周相位完全一致；后续可信更新的连续 phi 单调推进；协方差正定；未来未可用观测污染后 Belief 完全不变；候选电机改变未来相位；A/B 使用相同控制时未来相位一致。

| 日期 | forcing 权重梯度 L1 | 显式 control 权重梯度 L1 | 历史 Hall 更新 |
|---|---:|---:|---:|
| 2026-09-06 | 47.655614 | 3.857274 | 48 |
| 2026-09-15 | 40.055514 | 1.713582 | 48 |
| 2026-10-12 | 34.444737 | 4.044952 | 58 |

这些梯度只证明计算图与周期分支已接通，不证明学得的动力学或控制效能准确。针对性回归涵盖 SO(3)、ZOH 等价性、解析相位积分、周期分支隔离、延迟观测、协方差、采样恢复、缺相位禁止准入及 Hall wrap/异常恢复。最终定向回归为 **21 passed, 4 deselected（16.41 s）**；此前含原生未来污染检查的一轮为 **22 passed, 3 deselected（85.03 s）**。未运行封存 test 或无关冻结控制实验测试。

复现（从本 worktree 运行，输出目录必须不存在）：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/verify_pahbd_hall.py --output /tmp/pahbd_hall_recheck
PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_pahbd.py tests/test_pahbd_readiness.py -k "not native and not frozen and not formal_entry and not checkpoint_sampler_optimizer"
```

正式入口已具备准入条件，但本轮没有执行 train 模式。后续是否存在跨航次可泛化的周期控制效能、是否优于 B、是否有 MPC/闭环价值，仍须正式训练及对应控制验证回答；本报告不作性能排名。

附件：[机器结果](hall_contract_v2/results.json)、[train 逐航次](hall_contract_v2/train_flights.csv)、[validation 逐航次](hall_contract_v2/validation_flights.csv)、[train 日期/固件](hall_contract_v2/train_dates_firmware.csv)、[validation 日期/固件](hall_contract_v2/validation_dates_firmware.csv)、[跨航次源关系](hall_contract_v2/sensor_alignment.csv)、[异常定位](hall_contract_v2/anomaly_locations.csv.gz)。
