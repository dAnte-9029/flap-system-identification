# 时间合同与消融准入（查看新预测前冻结）

证据入口：`results/firmware_sources.csv`、`source_evidence.csv`、`sources/` 中的源文件快照。覆盖冻结窗口的 11 个固件标识，16 个关键文件 × 11，共 176 个精确提交来源。各路径在这些提交中的内容哈希相同。来源优先本地 PX4 Git object、已登记的历史数据集源码缓存，其余从原 PX4 fork 的精确 commit URL 读取；没有修改 PX4 工作树。Stage4 报告从既有 scope worktree 缓存恢复并保存哈希，未重新读取原始 ULog。

## 源码证实的调用路径

1. `AS5600::RunImpl → read_raw_angle → I2C transfer(0x0C,2 bytes)`：读取 RAW_ANGLE 后才调用 `hrt_absolute_time()` 得到 `now`。累计展开 motor count；`dtheta / dt` 得到转速，alpha=0.2 的一阶低通形成 rpm_estimate，再按 `abs(rpm)/(60*FLAP_RATIO)` 得到扑翼频率。`flap_frequency.timestamp` 和 `encoder_count.timestamp` 都赋值为这个读数返回后的 now，然后依次 publish。驱动约 10 ms 调度；无独立 timestamp_sample。源时间戳有明确软件含义，但不是硬件采样中断时间。
2. `RPMCapture::gpio_interrupt_callback` 用 `hrt_absolute_time()` 记录 GPIO 上升沿，经过最小脉冲间隔判断；后续 Run 发布 `hall_event.timestamp=_hrt_timestamp`。它接近中断响应时间，不是日志写入时间，也不证明没有 GPIO/中断延迟。
3. `WingPhase::Run` 读取 encoder、frequency、Hall，使用前后编码器计数插值到 Hall 时间；Hall 重置 phase 零点，按 `4096*ratio` 算机械周期坐标。消息 timestamp 回填最近 encoder 的 timestamp，不是该模块真正 publish 的时间。异步到达与 Hall 插值意味着不能仅凭这个字段将 logged wing_phase 作为已知历史。
4. 当前**历史数据集**因此不使用 logged wing_phase 作为输入：`extract_flight` 从全日志首条 encoder count 定义相对原点，再用 `(count−origin)*2π/(4096*FLAP_RATIO)` 生成相位。它是 log-local 相位，没有跨日志共同的机械零点。本轮不改 ratio、不引入 Hall 对齐或 canonical 数据集的新相位。
5. `extract_flight` 将 encoder/frequency 的 `availability_us` 设为**下一条源 timestamp**，最后一条设为无穷大。理由是单串行 Run：下一次采集时间是上次 publish 完成的保守上界。`hold` 按 availability≤每行 state_publication 选最近过去值；`encoder_publication_us`/`frequency_publication_us` 保存的是这个构造上界，**不是实测 publication 时间**。上轮“发布年龄”必须按此含义理解，源年龄 14.73 ms 也不能等同物理测量延迟。
6. `VehicleAngularVelocity::Run` 从选定 gyro FIFO 或单样本读取，经过 notch/低通、旋转／校准与偏置处理，再 `CalibrateAndPublish`；timestamp_sample 传递输入样本时间，timestamp 在 publish 前用 hrt_absolute_time。滤波后值仍保留传感器采样戳，滤波的频率相关相位延迟没有变成另一个 timestamp。不能把它当成没有滤波延迟的瞬时机体真值。
7. `EKF2::Run` 令 `now=imu_sample_new.time_us`，EKF 更新后 `PublishLocalPosition(now)` 将 lpos.timestamp_sample 设为 now，`EstimatorInterface::getVelocity` 转到 output_predictor，`getPosition` 使用相应预测输出与坐标投影；位置／速度来自 EKF 输出预测器，而不是直接读取延迟融合状态；lpos.timestamp 为 publish 前系统时间（非 replay 分支）。因此这里的“位置测量时间”更准确地说是 EKF 输出使用的 IMU 时钟标签，不是 GPS/位置传感器的独立采样时刻。
8. 数据 `hold` 以每行位置 publication 作可用性判断，按 source/measurement 时间保存年龄；rate 是已发布的最近值，目标 TARGET 则按测量时间最近邻，仅供离线标签。`pack_windows` 把全历史 state/control/phase 打包，以历史末端 state 建立 initial，历史 dt 与未来 dt 均来自位置 sample 时间差。
9. GRU 对整段历史生成隐藏态，取最后 valid 行的 phase/frequency；未来在每一步**起点**使用 `phase0+2π*f0*elapsed`，状态输出在步终点。未来频率保持，不读取真实未来相位或状态。Control/NoControl 都使用相位；NoControl 同时去掉历史和未来控制。

## 证据边界

| 层次 | 可支持的陈述 |
|---|---|
| 源码及缓存日志确认 | 软件读数返回后的 encoder 源时间；频率低通与 ratio 定义；构造的 availability 上界；每行可用性；模型步起点相位；原生状态时钟；11 个版本相关源文件相同 |
| 合理但仍是模型假设 | 将 RAW_ANGLE 的有效物理时刻近似为读数返回后的软件源时间；用已经可用的低通频率作短期匀速相位传播 |
| 无法确认 | AS5600 内部采样/数字滤波延迟、I²C 事务内对应的精确物理时刻、传感器到机械翼拍的绝对相位；各日志实际 gyro 硬件/滤波群延迟的完整标定；实飞二进制及所有编译选项认证 |

Stage4 已提醒：日志 ver_sw 对应的源码和字段一致不等于实飞 binary、编译选项或全部 dirty 修改被认证。本轮也保留该限制。

## 冻结版本

- **V0 Original**：直接复用已保存 inputs/predictions 并做六个模型的同接口回归。控制、归一化、状态、相位、时间全不变。
- **V1_source_assumption**：唯一实际执行的改变版，明确是条件性软件源时间诊断，**不是已验证物理对齐**。对于历史每一行，只用该行决策时已可用的频率和 encoder：`δφ=2π*f_available*(state_sample−encoder_source)`；将保存的 sin/cos 转为 atan2 后加 δφ 再转回 sin/cos。每行都修正，历史末端和未来起点因此一致；频率值、mask、其余状态、initial、dt、控制事件全部原样。没有固定 offset 搜索、按误差选择参数或真实未来相位校正。
- **V1 物理对齐**：不准入；无法把内部传感器延迟证明为 0 或已知值。
- **V2 state-and-phase**：不准入。只检查每行决策前已发布的 native rate 是否能在目标位置时钟两侧提供括号。现有记录仅 1,899/6,011 历史行、65/225 起点满足；其余需外推。V0 本来已是过去值保持，重新命名 ZOH 不是新重建；引入角加速度外推则需额外高频动态／滤波假设，缺乏当前证据，不为完整队列强行实施。

相位传播参数只来自训练侧的历史时间与频率。验证仅应用此固定公式。输入变化和训练分布偏离另表保存，不作为删样依据。新表示并非模型训练时的合同，结果变化不能包装成权重性能提升或已校正物理控制增益。
