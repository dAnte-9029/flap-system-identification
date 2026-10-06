# 下一阶段：控制器组件、时钟误差与采集准备

本轮完成了本地控制器组件和采集预检工具，并用原三份开发日志验证。**尚未完成新的实机辨识，也未进入新动力学模型训练或控制器优化。** 目前最明确的技术结论：实飞角速度环约 400 Hz，日志约 100/50 Hz；简单把控制器调用频率提高到 400 Hz 并保持旧测量，不能稳定改善回放。

## 1. 实际控制周期已获得数值证据

目标固件的 Roll 原始积分更新为：

`delta_I_raw = max(0, 1 - (rate_error / radians(400))²) * FW_RR_I * rate_error * dt`

它在 anti-windup 和积分限幅之前记录。因此在 update_enabled 且分母绝对值>0.005 的样本中，可以反推该次实际 dt。本次仅将其用于时钟诊断，没有用 Roll 输出拟合模型或调控制参数。

|9.18 日志|有效诊断点|dt P1 / 中位数 / P99|约对应频率|
|---|---:|---|---:|
|log_0|7167|2.493 / 2.495 / 2.497 ms|400.8 Hz|
|log_1|6810|2.462 / 2.464 / 2.466 ms|405.8 Hz|
|log_2|7429|2.462 / 2.464 / 2.466 ms|405.8 Hz|

这比“从 topic 记录间隔猜控制频率”更直接，证据在 `controller_clock_evidence.csv`。仍不等于恢复了全部高频输入序列。

## 2. 已实现可复用的俯仰角速度控制组件

源码：`src/system_identification/control/px4_pitch.py`。

- 独立保存积分、空速一阶滤波、增益压缩的高通/能量滤波、上一输入及缓存增益；支持完整 snapshot/restore。
- 参数必须由实际飞行参数显式构造，没有静默补默认飞行参数。
- 修正上一阶段简化实现的 reset 语义：reset_integral 只清积分；landed/disabled 对内部 compression gain 的重置不等于清掉全部滤波状态或立即改写缓存输出增益。
- 输出使用本周期更新前的 I，之后执行带 anti-windup 的积分更新，符合目标源码顺序。
- `step` 必须给出实际控制更新 dt，超出源码允许的 2–40 ms 范围会报错；时钟调度/子步由调用方负责。
- 关闭 rates 时返回 torque=None，避免把“未实现的手动控制分支”伪装成零转矩控制。

该组件只实现 pitch rate arithmetic。没有宣称复现完整 PX4，没有实现 B2b Roll 状态机、导航/TECS、驱动器或动力学模型。B 组日志这里只用于俯仰组件跨飞行诊断。

验证包括 12 个针对性测试：积分更新顺序、anti-windup、reset 语义、完整状态恢复、时钟拒绝、草案事件分组和评估数据隔离。另从冻结 e624 源码提取原始 `GainCompression::update` 方法，使用最小 filter/math 适配层编译 C++，对照 Python 的 1200 次变步长更新。误差小于 3e-6；实际误差见 `gain_reference_check.json`。这仅验证该算术组件，不是整机固件编译验证。

## 3. 400 Hz 保持输入不等于补回缺失测量

以下三个版本使用同样的评分点：原 stage2、修正 reset 后仍在 gyro 记录网格更新、按反推周期执行并因果保持最新记录输入。都只在首点用记录 I 初始化；没有周期性拿真实 I 修正控制器。

|日志|stage2 全段 RMSE|reset 修正、记录网格|约 400 Hz 保持输入|
|---|---:|---:|---:|
|log_0|0.013408|0.013411|0.013876|
|log_1|0.014180|0.014141|0.016685|
|log_2|0.011512|0.011564|0.016167|

单位是归一化 pitch 转矩命令；前 5 s 按原规则不评分。只看 AUTO 段，400 Hz 版本相对原版分别为 0.011055 对 0.012748、0.012675 对 0.012561、0.013964 对 0.009098，收益并不一致。没有挑选 log_0 的改善作为整体成功。

因此保留这项负结果：**不要用高频零阶保持的回放结果宣称恢复了真实飞控响应。** 真实高频角速度、设定值、分配器反馈和完整初始滤波状态仍缺失。native 版本还使用从日志测得的周期及独立固定时钟相位，非精确飞控调度重放。

`replay_comparison.csv` 保存全部指标，`*_comparison.csv` 保存同点逐样本结果，`*_state_snapshots.json` 保存组件状态。

## 4. 已落地的辨识准备

- `configs/control/pitch_identification_v1.yaml`：离线试验草案。固定为 A 组控制器、pitch attitude reference 激励，0.5 s 前史、1 s 评价、1.2 s 保持、3 s 恢复；8 个计划飞行按 4 train / 2 validation / 2 reserved_evaluation 分组，每飞行 6 次正负平衡事件。这是可审阅草案，不是统计功效保证或可上传航线。
- 激励 amplitude_deg 仍为空，physical_surface_mapping_verified=false；其数值不能从模型或旧事件阈值反推。物理映射、已允许任务幅度和来源未确认前，没有生成具体飞行幅值。
- `configs/control/identification_log_admission.yaml` 仅准入本次已审计的三份 9.18 开发日志及 SHA256。
- `scripts/check_pitch_identification_logs.py` 实现哈希准入、固件/参数核对、各 topic 记录频率和最大间隔预检，输出逐条失败原因和不可执行的相对时间草案。评估角色不会被该工具读取。

已对三份旧日志执行 dry run，均未通过新的采集预检；B 组还因 FLAP_SLOW_EN 不符合 A 组要求被明确标记。不是旧日志突然“不可用”，而是它们不能满足此次更高要求的精确控制重放/外生辨识证据。

**该工具目前是采集预检，不是完整的辨识验收器。** 周期序号、完整状态和外生激励事件证据的固件消息尚未实现，因此 identification_ready 与 exact_controller_replay_ready 始终为 false；不会仅凭高记录频率自动放行。

gain_compression 在当前固件主动约 100 ms 发布一次；logger 提高记录频率不会让它变成 400 Hz。下一步固件记录方案见 `telemetry_contract.md`。

## 5. 本地复核方法与下一步条件

测试：

```bash
MPLCONFIGDIR=/tmp/flap-stage2-matplotlib /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_px4_pitch_controller.py tests/test_pitch_identification_precheck.py
```

采集预检输出目录必须不存在，例如：

```bash
MPLCONFIGDIR=/tmp/flap-stage2-matplotlib /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/check_pitch_identification_logs.py --output /tmp/pitch-acquisition-review
```

下一步真正依赖的实物信息是：物理舵面对应/反向及可用范围、允许的俯仰参考机动幅度、实际补充采集。若要精确周期回放，还需实现并验证下述同步日志契约。现阶段可以使用本轮组件继续搭接口和做软件验证，不能据此冻结“控制可用”的动力学性能或开展新控制器实飞比较。

本轮未读取封存测试或 9.19 候选，未修改外部 PX4/原始日志/旧模型，未安装依赖。
