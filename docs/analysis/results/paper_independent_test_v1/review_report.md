# Paper Step7：冻结方案与留出日期评价

## 1. 冻结时间与数据访问

主性能结论在两个留出日期得到支持：500 ms三项动力学指标均有3/3配对seed支持GRU；按每flight先平均三seed误差后，Sep8的6/6和Sep19的22/22架flight均支持GRU。这不是每个窗口都改善，也不是基于独立窗口假设的显著性结论。

History结论需要收窄到日期、指标与时域：500 ms下H1→H26在Sep8速度/姿态/角速度改善6.97%/0.66%/2.90%，Sep19改善13.05%/11.03%/8.06%。Sep8姿态仅2/3seed、4/6flight同方向，不能称稳定全面改善。H13→H26在Sep8速度和角速度分别为−1.10%和−0.57%，即H13略好；Sep19三项边际改善为2.91%/0.07%/0.92%，姿态flight方向11胜11负。约240 ms上下文已获取大部分500 ms收益的叙述仍可保留，但更长历史持续稳定更好的说法不受支持。到1 s，Sep8 H1→H26速度和角速度分别−0.94%和−3.05%，对应结论在该条件下未得到支持；Sep19的速度/姿态均值收益也缩小至不足1%。不切换H26，不修改旧Mixed分类，不把历史跨度解释为物理时间常数。

500 ms未来命令信息价值在两日期ALL和high组均得到支持；三项动力学均为3/3seed且每flight三seed均值方向全部正向。ALL速度/姿态/角速度改善为Sep8的19.36%/16.34%/9.15%及Sep19的18.24%/15.43%/17.30%；high组分别15.34%/19.90%/13.42%及22.03%/18.88%/18.44%。不能扩大为所有时域、组别和flight皆改善：Sep8 100 ms ALL角速度由Hold略好0.28%，high组也有0.57%的反向均值；500 ms low组两日期各有姿态flight反例，Sep8 middle速度也有局部反例。Hold无对应反事实实测轨迹，结论仍是增量预测信息。

相对优势与绝对日期变化必须分开：H26在Sep19的500 ms速度、姿态误差较validation ALL分别高29.02%和21.65%，但角速度误差反而低11.40%。因此不能声称所有状态都同等跨日期稳定或全部退化。独立测试主表可作为论文性能叙述的主要依据，适用范围限定于原质量规则准入的两日期日志。

方案于2026-09-22T06:09:10.808392+00:00冻结；本轮Sep8首次读取2026-09-22T06:09:24.239195+00:00，Sep19首次读取2026-09-22T06:09:32.901130+00:00。先通过validation/合成接口检查再开封。

Sep8是主独立测试，Sep19是补充留出日期，均按预定方案执行且分开报告。旧manifest明确Sep8曾接受描述性质量审计；Sep19原inventory已有文件身份记录。因此不称为原始日志首次被任何人读取。没有在已知冻结实验manifest/对应输出目录中发现这15模型此前的测试性能评价记录，但不对未记录的外部访问作绝对保证。两日期本轮均已使用，不能再称尚未打开。

## 2. 数据质量与实际覆盖

Sep8: 6/6 flights admitted; 871 fixed origins.

Sep19: 22/23 flights admitted; 3202 fixed origins.

| date | session | flight_id | status | final_origins | reason |
| --- | --- | --- | --- | --- | --- |
| Sep8 | 2026.9.8 | 2026.9.8/log_0_2026-9-8-05-42-48.ulg | admitted | 111 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_1_2026-9-8-05-49-28.ulg | admitted | 165 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_2_2026-9-8-05-56-46.ulg | admitted | 156 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_4_2026-9-8-06-38-56.ulg | admitted | 149 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_6_2026-9-8-06-46-18.ulg | admitted | 142 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_7_2026-9-8-06-55-26.ulg | admitted | 148 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_0_2026-9-19-14-31-44.ulg | admitted | 160 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_1_2026-9-19-14-40-12.ulg | admitted | 126 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_2_2026-9-19-14-45-56.ulg | excluded_by_frozen_gate | 0 | insufficient contiguous airborne data: duration<30 or stride10 windows<10 |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_3_2026-9-19-14-56-32.ulg | admitted | 141 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_4_2026-9-19-15-15-06.ulg | admitted | 140 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_5_2026-9-19-15-23-48.ulg | admitted | 133 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_6_2026-9-19-15-28-00.ulg | admitted | 119 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_7_2026-9-19-15-39-08.ulg | admitted | 150 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_8_2026-9-19-15-47-10.ulg | admitted | 164 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_0_2026-9-19-17-14-26.ulg | admitted | 141 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_1_2026-9-19-17-21-44.ulg | admitted | 142 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_2_2026-9-19-17-26-42.ulg | admitted | 141 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_3_2026-9-19-17-38-46.ulg | admitted | 126 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_4_2026-9-19-17-44-06.ulg | admitted | 148 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_5_2026-9-19-17-49-50.ulg | admitted | 155 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_9_2026-9-19-17-05-08.ulg | admitted | 154 | — |
| Sep19 | 9.19数据 | 9.19数据/log_0_2026-9-19-05-53-40.ulg | admitted | 153 | — |
| Sep19 | 9.19数据 | 9.19数据/log_1_2026-9-19-06-01-30.ulg | admitted | 144 | — |
| Sep19 | 9.19数据 | 9.19数据/log_2_2026-9-19-06-07-52.ulg | admitted | 145 | — |
| Sep19 | 9.19数据 | 9.19数据/log_3_2026-9-19-06-14-30.ulg | admitted | 156 | — |
| Sep19 | 9.19数据 | 9.19数据/log_4_2026-9-19-06-19-40.ulg | admitted | 153 | — |
| Sep19 | 9.19数据 | 9.19数据/log_5_2026-9-19-06-26-44.ulg | admitted | 150 | — |
| Sep19 | 9.19数据 | 9.19数据/log_6_2026-9-19-06-31-46.ulg | admitted | 161 | — |

Sep8沿用v2、Sep19沿用v3既有准入条件，具体差异在注册方案中预先写定，不是看测试误差后放宽或收紧。Sep19各文件夹是同一日期的session来源，不是多个独立日期。所有模型共享H26合法origin及50步future，短history仅取精确后缀；control、labels、dt一致。未因MLP无需历史或较短horizon增加窗口。

实际历史跨度：

| date | session | history_steps | n_origins | mean | std | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | ALL | 1 | 871 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sep8 | ALL | 5 | 871 | 0.080031 | 0.0033829 | 0.068519 | 0.078513 | 0.078885 | 0.07981 | 0.079864 | 0.088746 | 0.089936 |
| Sep8 | ALL | 13 | 871 | 0.24001 | 0.0043806 | 0.22932 | 0.23614 | 0.23697 | 0.23948 | 0.23957 | 0.24945 | 0.24959 |
| Sep8 | ALL | 26 | 871 | 0.49998 | 0.0039926 | 0.48875 | 0.49276 | 0.49886 | 0.49896 | 0.50262 | 0.5089 | 0.50908 |
| Sep19 | ALL | 1 | 3202 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sep19 | ALL | 5 | 3202 | 0.080017 | 0.0032918 | 0.068948 | 0.078779 | 0.079509 | 0.07979 | 0.07986 | 0.088814 | 0.090124 |
| Sep19 | ALL | 13 | 3202 | 0.24 | 0.0043393 | 0.22923 | 0.23216 | 0.23932 | 0.23949 | 0.23957 | 0.24947 | 0.24982 |
| Sep19 | ALL | 26 | 3202 | 0.50001 | 0.0039629 | 0.48879 | 0.49281 | 0.49892 | 0.49899 | 0.50239 | 0.50895 | 0.50931 |

## 3. 主模型独立测试

两日期500 ms的三项动力学指标均保留GRU相对MLP的均值优势。

| date | model | condition | group | horizon_s | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | B0 kinematic | Actual | ALL | 0.5 | 0.3628 | 1.1462 | 23.9791 | 0.9779 |
| Sep8 | MLP | Actual | ALL | 0.5 | 0.1636 ± 0.0037 | 0.4921 ± 0.0159 | 4.5510 ± 0.0677 | 0.6576 ± 0.0060 |
| Sep8 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.1205 ± 0.0036 | 0.3533 ± 0.0132 | 3.6977 ± 0.1867 | 0.5298 ± 0.0082 |
| Sep19 | B0 kinematic | Actual | ALL | 0.5 | 0.4769 | 1.4227 | 22.8819 | 1.1018 |
| Sep19 | MLP | Actual | ALL | 0.5 | 0.2847 ± 0.0013 | 0.7036 ± 0.0034 | 6.9392 ± 0.0204 | 0.7459 ± 0.0020 |
| Sep19 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.2097 ± 0.0016 | 0.4610 ± 0.0112 | 4.9615 ± 0.0364 | 0.5286 ± 0.0043 |

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | attitude_error_deg | 6.9392 | 4.9615 | 28.5 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.74595 | 0.52859 | 29.138 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | position_rmse_m | 0.28466 | 0.20969 | 26.335 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.70362 | 0.46098 | 34.485 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | attitude_error_deg | 4.551 | 3.6977 | 18.75 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.65764 | 0.5298 | 19.44 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | position_rmse_m | 0.16357 | 0.12052 | 26.315 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.49209 | 0.35328 | 28.208 | 3 | 0 | 6 | 0 | 6 | 871 |

## 4. History结论是否保留

| date | model | condition | group | horizon_s | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | GRU/H1 | Actual | ALL | 0.5 | 0.1380 ± 0.0022 | 0.3797 ± 0.0010 | 3.7222 ± 0.0665 | 0.5456 ± 0.0024 |
| Sep8 | GRU/H5 | Actual | ALL | 0.5 | 0.1272 ± 0.0042 | 0.3651 ± 0.0089 | 3.9062 ± 0.0747 | 0.5340 ± 0.0023 |
| Sep8 | GRU/H13 | Actual | ALL | 0.5 | 0.1227 ± 0.0009 | 0.3494 ± 0.0051 | 3.7016 ± 0.1078 | 0.5268 ± 0.0074 |
| Sep8 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.1205 ± 0.0036 | 0.3533 ± 0.0132 | 3.6977 ± 0.1867 | 0.5298 ± 0.0082 |
| Sep19 | GRU/H1 | Actual | ALL | 0.5 | 0.2495 ± 0.0010 | 0.5302 ± 0.0015 | 5.5766 ± 0.0457 | 0.5749 ± 0.0058 |
| Sep19 | GRU/H5 | Actual | ALL | 0.5 | 0.2260 ± 0.0012 | 0.5056 ± 0.0022 | 5.1653 ± 0.0313 | 0.5561 ± 0.0051 |
| Sep19 | GRU/H13 | Actual | ALL | 0.5 | 0.2142 ± 0.0006 | 0.4748 ± 0.0056 | 4.9652 ± 0.0475 | 0.5335 ± 0.0051 |
| Sep19 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.2097 ± 0.0016 | 0.4610 ± 0.0112 | 4.9615 ± 0.0364 | 0.5286 ± 0.0043 |

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H13_to_H26 | Sep19 | ALL | 0.5 | attitude_error_deg | 4.9652 | 4.9615 | 0.073583 | 2 | 1 | 11 | 11 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.53349 | 0.52859 | 0.9177 | 2 | 1 | 16 | 6 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | position_rmse_m | 0.21418 | 0.20969 | 2.0944 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.47478 | 0.46098 | 2.9063 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H13_to_H26 | Sep8 | ALL | 0.5 | attitude_error_deg | 3.7016 | 3.6977 | 0.10508 | 2 | 1 | 4 | 2 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.52681 | 0.5298 | -0.56667 | 1 | 2 | 1 | 5 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | position_rmse_m | 0.12268 | 0.12052 | 1.7582 | 2 | 1 | 5 | 1 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.34942 | 0.35328 | -1.104 | 1 | 2 | 4 | 2 | 6 | 871 |
| H1_to_H26 | Sep19 | ALL | 0.5 | attitude_error_deg | 5.5766 | 4.9615 | 11.03 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.57493 | 0.52859 | 8.0597 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | position_rmse_m | 0.24951 | 0.20969 | 15.958 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.53016 | 0.46098 | 13.049 | 3 | 0 | 21 | 1 | 22 | 3202 |
| H1_to_H26 | Sep8 | ALL | 0.5 | attitude_error_deg | 3.7222 | 3.6977 | 0.65858 | 2 | 1 | 4 | 2 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.54564 | 0.5298 | 2.9036 | 3 | 0 | 5 | 1 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | position_rmse_m | 0.13795 | 0.12052 | 12.635 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.37975 | 0.35328 | 6.9688 | 3 | 0 | 5 | 1 | 6 | 871 |

Sep8 H1_to_H26：velocity_rmse_m_s: 6.97%，seed 3/3、flight 5/6支持比较条件；attitude_error_deg: 0.66%，seed 2/3、flight 4/6支持比较条件；body_rate_rmse_rad_s: 2.90%，seed 3/3、flight 5/6支持比较条件。

Sep8 H13_to_H26：velocity_rmse_m_s: -1.10%，seed 1/3、flight 4/6支持比较条件；attitude_error_deg: 0.11%，seed 2/3、flight 4/6支持比较条件；body_rate_rmse_rad_s: -0.57%，seed 1/3、flight 1/6支持比较条件。

Sep19 H1_to_H26：velocity_rmse_m_s: 13.05%，seed 3/3、flight 21/22支持比较条件；attitude_error_deg: 11.03%，seed 3/3、flight 22/22支持比较条件；body_rate_rmse_rad_s: 8.06%，seed 3/3、flight 22/22支持比较条件。

Sep19 H13_to_H26：velocity_rmse_m_s: 2.91%，seed 3/3、flight 19/22支持比较条件；attitude_error_deg: 0.07%，seed 2/3、flight 11/22支持比较条件；body_rate_rmse_rad_s: 0.92%，seed 2/3、flight 16/22支持比较条件。

H1→H26的正值支持当前数据上历史上下文的预测价值，负值则在该条件下未得到支持。H13→H26须逐日期和指标判断，不能预设长history一定更好。H26不切换；旧validation Mixed分类不回写；有效上下文跨度不等于物理记忆常数。

## 5. Future-control结论是否保留

| date | model | condition | group | horizon_s | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.1205 ± 0.0036 | 0.3533 ± 0.0132 | 3.6977 ± 0.1867 | 0.5298 ± 0.0082 |
| Sep8 | Standard GRU/H26 | Actual | low | 0.5 | 0.1023 ± 0.0024 | 0.2712 ± 0.0140 | 3.3242 ± 0.2472 | 0.5214 ± 0.0077 |
| Sep8 | Standard GRU/H26 | Actual | middle | 0.5 | 0.1138 ± 0.0021 | 0.3547 ± 0.0065 | 3.5281 ± 0.1961 | 0.5137 ± 0.0110 |
| Sep8 | Standard GRU/H26 | Actual | high | 0.5 | 0.1649 ± 0.0115 | 0.4651 ± 0.0383 | 4.8100 ± 0.0709 | 0.5449 ± 0.0093 |
| Sep8 | Standard GRU/H26 | Hold | ALL | 0.5 | 0.1368 ± 0.0039 | 0.4381 ± 0.0163 | 4.4201 ± 0.2186 | 0.5831 ± 0.0118 |
| Sep8 | Standard GRU/H26 | Hold | low | 0.5 | 0.1204 ± 0.0037 | 0.3574 ± 0.0214 | 3.7431 ± 0.2891 | 0.5591 ± 0.0109 |
| Sep8 | Standard GRU/H26 | Hold | middle | 0.5 | 0.1317 ± 0.0032 | 0.4491 ± 0.0099 | 4.2230 ± 0.2315 | 0.5684 ± 0.0114 |
| Sep8 | Standard GRU/H26 | Hold | high | 0.5 | 0.1775 ± 0.0092 | 0.5494 ± 0.0339 | 6.0050 ± 0.1015 | 0.6293 ± 0.0143 |
| Sep19 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.2097 ± 0.0016 | 0.4610 ± 0.0112 | 4.9615 ± 0.0364 | 0.5286 ± 0.0043 |
| Sep19 | Standard GRU/H26 | Actual | low | 0.5 | 0.1974 ± 0.0058 | 0.3949 ± 0.0300 | 3.9179 ± 0.0543 | 0.4665 ± 0.0082 |
| Sep19 | Standard GRU/H26 | Actual | middle | 0.5 | 0.1956 ± 0.0018 | 0.4118 ± 0.0099 | 4.5606 ± 0.0438 | 0.5093 ± 0.0050 |
| Sep19 | Standard GRU/H26 | Actual | high | 0.5 | 0.2238 ± 0.0038 | 0.5655 ± 0.0174 | 6.2085 ± 0.0349 | 0.6063 ± 0.0044 |
| Sep19 | Standard GRU/H26 | Hold | ALL | 0.5 | 0.2254 ± 0.0010 | 0.5638 ± 0.0077 | 5.8669 ± 0.0459 | 0.6392 ± 0.0025 |
| Sep19 | Standard GRU/H26 | Hold | low | 0.5 | 0.2054 ± 0.0060 | 0.4267 ± 0.0321 | 4.2742 ± 0.0949 | 0.5219 ± 0.0029 |
| Sep19 | Standard GRU/H26 | Hold | middle | 0.5 | 0.2086 ± 0.0013 | 0.4982 ± 0.0089 | 5.2524 ± 0.0624 | 0.6177 ± 0.0040 |
| Sep19 | Standard GRU/H26 | Hold | high | 0.5 | 0.2468 ± 0.0044 | 0.7252 ± 0.0228 | 7.6533 ± 0.0327 | 0.7434 ± 0.0028 |

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | attitude_error_deg | 5.8669 | 4.9615 | 15.432 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.63916 | 0.52859 | 17.299 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | position_rmse_m | 0.22545 | 0.20969 | 6.9879 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.56382 | 0.46098 | 18.241 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | high | 0.5 | attitude_error_deg | 7.6533 | 6.2085 | 18.878 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | body_rate_rmse_rad_s | 0.74339 | 0.60632 | 18.438 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | position_rmse_m | 0.2468 | 0.22384 | 9.3034 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | velocity_rmse_m_s | 0.72519 | 0.56546 | 22.026 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | low | 0.5 | attitude_error_deg | 4.2742 | 3.9179 | 8.3359 | 3 | 0 | 21 | 1 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | body_rate_rmse_rad_s | 0.52194 | 0.46654 | 10.613 | 3 | 0 | 22 | 0 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | position_rmse_m | 0.20537 | 0.19735 | 3.9019 | 3 | 0 | 21 | 1 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | velocity_rmse_m_s | 0.42668 | 0.39494 | 7.4393 | 3 | 0 | 22 | 0 | 22 | 603 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | attitude_error_deg | 5.2524 | 4.5606 | 13.171 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | body_rate_rmse_rad_s | 0.61773 | 0.50931 | 17.551 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | position_rmse_m | 0.20864 | 0.19559 | 6.2527 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | velocity_rmse_m_s | 0.49818 | 0.41184 | 17.331 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | attitude_error_deg | 4.4201 | 3.6977 | 16.343 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.58312 | 0.5298 | 9.1455 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | position_rmse_m | 0.13683 | 0.12052 | 11.92 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.43807 | 0.35328 | 19.355 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | high | 0.5 | attitude_error_deg | 6.005 | 4.81 | 19.9 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | body_rate_rmse_rad_s | 0.62932 | 0.54489 | 13.417 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | position_rmse_m | 0.17747 | 0.16491 | 7.0761 | 3 | 0 | 5 | 1 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | velocity_rmse_m_s | 0.54943 | 0.46513 | 15.343 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | low | 0.5 | attitude_error_deg | 3.7431 | 3.3242 | 11.191 | 3 | 0 | 5 | 1 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | body_rate_rmse_rad_s | 0.55911 | 0.52141 | 6.7434 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | position_rmse_m | 0.12038 | 0.10232 | 15.003 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | velocity_rmse_m_s | 0.35738 | 0.27124 | 24.104 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | attitude_error_deg | 4.223 | 3.5281 | 16.455 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | body_rate_rmse_rad_s | 0.56835 | 0.51375 | 9.6077 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | position_rmse_m | 0.13172 | 0.11379 | 13.617 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | velocity_rmse_m_s | 0.44908 | 0.35471 | 21.013 | 3 | 0 | 5 | 1 | 6 | 379 |

正值支持实际未来命令的增量预测信息；负值需要明确限制。ALL不等于全部flight/window改善；high组也不自动代表收益最大。low/短horizon反例完整保留。控制组使用冻结训练阈值，仅是离线分组，不是在线可提前知道的工况分类器。

## 6. 与validation的相同点和差异

| test_date | validation_cohort | model | horizon_s | metric | test_mean | test_seed_sd | validation_mean | validation_seed_sd | test_over_validation_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | ALL | B0 | 0.5 | position_rmse_m | 0.3628 | — | 0.41818 | — | 0.86757 |
| Sep8 | ALL | B0 | 0.5 | velocity_rmse_m_s | 1.1462 | — | 1.1707 | — | 0.97905 |
| Sep8 | ALL | B0 | 0.5 | attitude_error_deg | 23.979 | — | 23.245 | — | 1.0316 |
| Sep8 | ALL | B0 | 0.5 | body_rate_rmse_rad_s | 0.97789 | — | 1.066 | — | 0.91736 |
| Sep8 | ALL | MLP | 0.5 | position_rmse_m | 0.16357 | 0.003654 | 0.24067 | 0.0018522 | 0.67964 |
| Sep8 | ALL | MLP | 0.5 | velocity_rmse_m_s | 0.49209 | 0.015907 | 0.54218 | 0.009692 | 0.9076 |
| Sep8 | ALL | MLP | 0.5 | attitude_error_deg | 4.551 | 0.067684 | 5.4859 | 0.045136 | 0.82959 |
| Sep8 | ALL | MLP | 0.5 | body_rate_rmse_rad_s | 0.65764 | 0.0060157 | 0.7526 | 0.0014554 | 0.87382 |
| Sep8 | ALL | H26 | 0.5 | position_rmse_m | 0.12052 | 0.0036177 | 0.1924 | 0.0018568 | 0.62643 |
| Sep8 | ALL | H26 | 0.5 | velocity_rmse_m_s | 0.35328 | 0.013161 | 0.35729 | 0.0071649 | 0.98879 |
| Sep8 | ALL | H26 | 0.5 | attitude_error_deg | 3.6977 | 0.18666 | 4.0784 | 0.046072 | 0.90666 |
| Sep8 | ALL | H26 | 0.5 | body_rate_rmse_rad_s | 0.5298 | 0.0082155 | 0.59658 | 0.0028141 | 0.88806 |
| Sep19 | ALL | B0 | 0.5 | position_rmse_m | 0.47688 | — | 0.41818 | — | 1.1404 |
| Sep19 | ALL | B0 | 0.5 | velocity_rmse_m_s | 1.4227 | — | 1.1707 | — | 1.2153 |
| Sep19 | ALL | B0 | 0.5 | attitude_error_deg | 22.882 | — | 23.245 | — | 0.98436 |
| Sep19 | ALL | B0 | 0.5 | body_rate_rmse_rad_s | 1.1018 | — | 1.066 | — | 1.0336 |
| Sep19 | ALL | MLP | 0.5 | position_rmse_m | 0.28466 | 0.0012817 | 0.24067 | 0.0018522 | 1.1828 |
| Sep19 | ALL | MLP | 0.5 | velocity_rmse_m_s | 0.70362 | 0.0033878 | 0.54218 | 0.009692 | 1.2978 |
| Sep19 | ALL | MLP | 0.5 | attitude_error_deg | 6.9392 | 0.020443 | 5.4859 | 0.045136 | 1.2649 |
| Sep19 | ALL | MLP | 0.5 | body_rate_rmse_rad_s | 0.74595 | 0.0019967 | 0.7526 | 0.0014554 | 0.99116 |
| Sep19 | ALL | H26 | 0.5 | position_rmse_m | 0.20969 | 0.0016474 | 0.1924 | 0.0018568 | 1.0899 |
| Sep19 | ALL | H26 | 0.5 | velocity_rmse_m_s | 0.46098 | 0.011184 | 0.35729 | 0.0071649 | 1.2902 |
| Sep19 | ALL | H26 | 0.5 | attitude_error_deg | 4.9615 | 0.036435 | 4.0784 | 0.046072 | 1.2165 |
| Sep19 | ALL | H26 | 0.5 | body_rate_rmse_rad_s | 0.52859 | 0.0042755 | 0.59658 | 0.0028141 | 0.88604 |

比值>1表示测试绝对误差高于开发validation均值；相对优于MLP与绝对误差跨日期升高可以同时成立，不能互相替代。小seed SD不能证明跨日期泛化稳定。

## 7. 不利结果、失败与边界

模型/日期/条件工程失败数：0。完整失败记录见artifact状态；不丢弃失败origins后报告成功子集。

主要动力学的非正向聚合比较如下，均保留；负值表示对应验证结论在该测试条件下未得到支持：

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Actual_vs_Hold | Sep8 | ALL | 0.1 | body_rate_rmse_rad_s | 0.54588 | 0.5474 | -0.27813 | 1 | 2 | 4 | 2 | 6 | 871 |
| Actual_vs_Hold | Sep8 | high | 0.1 | body_rate_rmse_rad_s | 0.59533 | 0.59873 | -0.57079 | 1 | 2 | 2 | 4 | 6 | 228 |
| Actual_vs_Hold | Sep8 | middle | 0.1 | body_rate_rmse_rad_s | 0.51616 | 0.51685 | -0.13291 | 2 | 1 | 3 | 3 | 6 | 416 |
| H13_to_H26 | Sep8 | ALL | 0.2 | body_rate_rmse_rad_s | 0.53351 | 0.5337 | -0.035689 | 2 | 1 | 3 | 3 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.52681 | 0.5298 | -0.56667 | 1 | 2 | 1 | 5 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.34942 | 0.35328 | -1.104 | 1 | 2 | 4 | 2 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.60405 | 0.60658 | -0.41821 | 2 | 1 | 2 | 4 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.58864 | 0.60658 | -3.0472 | 1 | 2 | 1 | 5 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | velocity_rmse_m_s | 0.59678 | 0.60238 | -0.93839 | 2 | 1 | 5 | 1 | 6 | 871 |

全部flight和seed反例分别保存在paired_flight_exceptions.csv、paired_seed_exceptions.csv。数据准入被排除的flight不属于已评价样本，不能称全部原始航次都得到验证。

## 8. 可用于论文Results的表述

On Sep8, the signed mean reductions in velocity, attitude and body-rate error of Standard GRU/H26 relative to the MLP were 28.21%, 18.75% and 19.44%, respectively, at the nominal500ms horizon (positive denotes lower GRU error). On Sep19, the signed mean reductions in velocity, attitude and body-rate error of Standard GRU/H26 relative to the MLP were 34.49%, 28.50% and 29.14%, respectively, at the nominal500ms horizon (positive denotes lower GRU error). Errors were computed within each flight, averaged equally across flights within each date, and summarized across three seeds. These are descriptive held-out-date results; no window-independence significance test was used.

## 9. Discussion限制

The test dates were selected and the model checkpoints, preprocessing gates and analysis rules were frozen before this evaluation. Sep8 had prior descriptive quality-audit exposure and Sep19 had prior file-inventory exposure; we do not claim first-ever access to the raw logs. The inherited v2 and v3 flight-admission gates differ, and results apply to eligible windows of these dates, not all recordings or the flight envelope. The MLP/GRU comparison changes capacity and architecture; controlled history comparisons supply separate evidence. Both Actual and Hold forecasts use the same logged-trajectory truth, without matched counterfactual Hold flights. Neither relative accuracy nor small seed SD establishes arbitrary-action causal fidelity, generalization to arbitrary winds or airframes, long-horizon stability, or closed-loop benefits. Sep8 and Sep19 are now used test data for this model version and cannot subsequently be treated as unopened independent tests after development on their results.

## 10. 是否具备预测实验整理写作条件

已形成按日期的主性能、history、Actual/Hold主表及全时域补充结果，可整理预测实验章节；科学结论应逐日期、指标和工况限定，独立测试表优先，validation表保留为开发证据。若存在工程失败则对应单元保留缺项，不伪装完整成功。三seed SD仅描述初始化变化。

## 11. 下一步问题，但不执行

进入闭环仿真或实机验证前，需要独立验证输入响应的方向、时延与幅值，明确状态估计/坐标/控制分配和时间接口，考虑执行器约束及闭环误差累积，再定义有界时域和安全退出的控制验证方案。本轮没有开展MPC/RL、闭环控制、实机激励或任何训练。之后若利用这两日期改模型，必须如实记为已用于开发的数据，新的独立结论需要其他未用于开发的证据。无自动commit/push；建议提交：feat: add frozen held-out flight evaluation。
