# Step 4 — Dynamics / Observability / Phase Audit

**唯一第一优先级：P2 改 derivative/state-transition target。第二优先级：P1 修 phase representation，但必须保留/解决 per-log unknown zero，不能直接把日志phase当统一机械phase。**

理由：2-step increment监督的小型受控对照在3个seed的validation上改善同一increment指标；直接加入经train offset调整的phase在未知validation offset下反而退化。强phase同步证明信号可重复，不等于已证明phase替换是主要解法。没有训练Main V3、没有修改Main V2架构/权重、没有打开sealed test或启动RL。

## 输入与证据范围

首先完整读取Step 3报告及loss审计。精确源码路径见 `docs/audits/2026-09-15_main_v2_phase_observability_audit.md`。所有Step 3文件hash保留并复核；本轮结果单独保存。

**口径纠正：3.30 rad/s²是recursive全时域预测std，不是teacher-state一步std。** 本轮每个合法时点用真实当前状态和26点合法过去reset，只step一次；train 43,057点、validation 40,085点，要求两侧至少25步且不跨segment。validation逐点预测std约12.08、raw target约30.20 rad/s²，仍偏弱，但不能以3.30描述一步预测。

Linear acceleration指NED速度差分，不是IMU specific force；angular acceleration为body-FRD角速度差分。统计结论是对已记录/估计的状态，不把phase-locked estimator vibration自动升级为全部真实刚体运动。

实际duration范围（每个segment原生时间；过滤诊断才使用20ms网格）：
| split | steps | min_s | max_s | mean_s |
| --- | --- | --- | --- | --- |
| train | 1 | 0.0099 | 0.0300 | 0.0200 |
| train | 2 | 0.0296 | 0.0499 | 0.0400 |
| train | 5 | 0.0887 | 0.1098 | 0.1000 |
| train | 10 | 0.1896 | 0.2096 | 0.2000 |
| train | 25 | 0.4890 | 0.5091 | 0.5000 |
| validation | 1 | 0.0099 | 0.0299 | 0.0200 |
| validation | 2 | 0.0296 | 0.0499 | 0.0400 |
| validation | 5 | 0.0887 | 0.1098 | 0.1000 |
| validation | 10 | 0.1896 | 0.2096 | 0.2000 |
| validation | 25 | 0.4890 | 0.5090 | 0.5000 |

history5/13/26分别覆盖4/12/25个实际转移，约0.08/0.24/0.50秒；不是5/13/26乘20ms。完整dt、increment时长在actual_durations.csv。

## Q1：多少phase-synchronous component？

按每个log独立3阶phase harmonic regression，前半时间拟合、后半时间检验，中间purge。下表为原生采样及真实训练窗口上的日志等权R²。cycle-domain24bins/完整周期分析另列，不能将重采样平滑后的R²冒充raw R²。

| split | representation | signal | heldout_r2 |
| --- | --- | --- | --- |
| train | t0_relative | ax_n | -0.0080 |
| train | t0_relative | ay_n | -0.0048 |
| train | t0_relative | az_n | -0.0025 |
| train | t0_relative | p | -0.0014 |
| train | t0_relative | p_dot | -0.0011 |
| train | t0_relative | q | -0.0039 |
| train | t0_relative | q_dot | -0.0012 |
| train | t0_relative | r | -0.0102 |
| train | t0_relative | r_dot | -0.0009 |
| train | within_log_native | ax_n | -0.0134 |
| train | within_log_native | ay_n | -0.0303 |
| train | within_log_native | az_n | 0.9047 |
| train | within_log_native | p | 0.3923 |
| train | within_log_native | p_dot | 0.1575 |
| train | within_log_native | q | 0.4512 |
| train | within_log_native | q_dot | 0.5747 |
| train | within_log_native | r | 0.0251 |
| train | within_log_native | r_dot | 0.0898 |
| validation | t0_relative | ax_n | -0.0035 |
| validation | t0_relative | ay_n | -0.0055 |
| validation | t0_relative | az_n | 0.0008 |
| validation | t0_relative | p | -0.0002 |
| validation | t0_relative | p_dot | 0.0002 |
| validation | t0_relative | q | -0.0048 |
| validation | t0_relative | q_dot | 0.0011 |
| validation | t0_relative | r | -0.0391 |
| validation | t0_relative | r_dot | -0.0003 |
| validation | within_log_native | ax_n | -0.1058 |
| validation | within_log_native | ay_n | -0.0289 |
| validation | within_log_native | az_n | 0.9296 |
| validation | within_log_native | p | 0.4385 |
| validation | within_log_native | p_dot | 0.2514 |
| validation | within_log_native | q | 0.5666 |
| validation | within_log_native | q_dot | 0.6379 |
| validation | within_log_native | r | 0.0358 |
| validation | within_log_native | r_dot | 0.2398 |

三维导数的逐点phase-only heldout R²与Main V2自身phase component：
| split | signal | component_heldout_r2 | component_truth_std | component_pred_std |
| --- | --- | --- | --- | --- |
| train | angular | 0.2933 | 27.8927 | 11.5939 |
| train | linear | 0.7468 | 10.6345 | 1.8248 |
| validation | angular | 0.4038 | 22.1277 | 8.4955 |
| validation | linear | 0.7980 | 10.3219 | 1.9562 |

validation：p/q/r分别约0.439/0.567/0.036，ax/ay/az约−0.106/−0.029/0.930，p_dot/q_dot/r_dot约0.251/0.638/0.240（各log R²等权，不是合并方差百分比）。负值表示该固定phase模板没有跨时间预测能力。线性导数整体phase-only R²约0.798、角导数约0.404；不同采样权重下数字不能混用。

1×/2×/3×阶数完整结果在harmonic_regression.csv；q_dot在cycle域的validation heldout R²从1阶约0.096升到2阶0.625、3阶0.703，证明不能仅保留fundamental。每log均值/std及十周期block bootstrap CI在phase_curves.csv，图为phase_domain_train/validation.png。CI只描述日志内block重复性，不覆盖跨flight未知offset或非平稳环境。

## Q2：t0-relative是否破坏physical phase consistency？ YES

encoder total_count → 4096×FLAP_RATIO换算 → log-local stored phase（原点为首个有限对齐encoder count）→ 每个history减t0 phase → rollout保持该anchor → sin(phi−anchor)/cos(phi−anchor)。stored phase不是per-segment归零，但network phase确实per-window归零。

同一物理phase在不同anchor窗口中不保证同sin/cos；所有t0 sin/cos都为(0,1)，可以对应完全不同wing位置。对跨log还额外存在unknown mechanical zero。Q1表中t0-relative R²接近0，相比日志内phase显著丢失直接同步位置；但历史状态仍能间接承载phase，因此不能据此称全部观测信息被删除。

只在train日志早半周期估计常数offset，再在晚半周期检查跨log导数曲线一致性：
| log_id | phase_offset_rad | heldout_cross_log_dispersion_before | heldout_cross_log_dispersion_after |
| --- | --- | --- | --- |
| 2026.8.10-8.20/log_14_2026-8-19-06-27-52.ulg | 0.0000 | 184.8786 | 117.9188 |
| 2026.8.10-8.20/log_15_2026-8-19-06-52-42.ulg | 2.8798 | 184.8786 | 117.9188 |
| 2026.8.10-8.20/log_18_2026-8-19-07-03-30.ulg | 2.0944 | 184.8786 | 117.9188 |
| 2026.8.10-8.20/log_1_2026-8-19-17-20-36.ulg | 3.9270 | 184.8786 | 117.9188 |
| 2026.8.10-8.20/log_5_2026-8-19-18-31-42.ulg | 1.5708 | 184.8786 | 117.9188 |
| 2026.8.10-8.20/log_6_2026-8-19-18-40-42.ulg | 0.7854 | 184.8786 | 117.9188 |

离散15°offset使晚半周期跨log平均方差从184.88降至117.92，约36.2%。仍有明显剩余差异；这个训练内改善不是validation推广。validation没有拟合任何部署offset；对其单独phase回归只用于Q1描述性重复性分析。

## Q3：raw变化多少是物理、多少是噪声？

不能严格分成两个百分比。至少有大量可跨时间预测的phase-synchronous记录分量，尤其q_dot二/三阶谐波；大于10Hz不能全当noise。约40%的三维角导数heldout预测解释度是“实测可重复性”的证据，不是独立传感器确认的物理variance下界；剩余约60%混合非平稳、higher harmonics、未建模状态和测量/差分噪声。

raw与各离线target的总体std、teacher-state误差：
| split | target | signal | truth_std | pred_std | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| train | D0_raw | angular | 39.4842 | 15.4608 | 33.8042 | 0.2670 |
| train | D1_lp4 | angular | 8.5103 | 15.4608 | 11.6061 | -0.8599 |
| train | D1_lp6 | angular | 16.1095 | 15.4608 | 11.5314 | 0.4876 |
| train | D1_lp8 | angular | 19.3628 | 15.4608 | 13.3927 | 0.5216 |
| train | D1_lp12 | angular | 25.4868 | 15.4608 | 19.2060 | 0.4321 |
| train | increment2 | angular | 29.4226 | 15.4608 | 22.3066 | 0.4252 |
| train | increment5 | angular | 15.0438 | 15.4608 | 12.9197 | 0.2624 |
| train | increment10 | angular | 6.4804 | 15.4608 | 15.2733 | -4.5547 |
| validation | D0_raw | angular | 30.2043 | 12.0752 | 25.8950 | 0.2650 |
| validation | D1_lp4 | angular | 6.7920 | 12.0752 | 9.7130 | -1.0451 |
| validation | D1_lp6 | angular | 11.2459 | 12.0752 | 9.5271 | 0.2823 |
| validation | D1_lp8 | angular | 15.3492 | 12.0752 | 11.5371 | 0.4350 |
| validation | D1_lp12 | angular | 20.4614 | 12.0752 | 15.9940 | 0.3890 |
| validation | increment2 | angular | 22.7273 | 12.0752 | 17.5700 | 0.4024 |
| validation | increment5 | angular | 11.3811 | 12.0752 | 11.1616 | 0.0382 |
| validation | increment10 | angular | 5.5791 | 12.0752 | 11.9959 | -3.6232 |

D1为4阶Butterworth zero-phase，uniform20ms内部网格后回原生timestamp再差分，cutoff4/6/8/12Hz；边界剔除25步。**仅用于离线target diagnosis，不能作为simulator inference输入。** D2_center2/5/10用于居中导数积分分析；increment2/5/10为向前真实状态差/真实duration，两者的时间定位不同，不混称。

## Q4：优先哪种supervision target？

**优先短时间2-step state increment（约0.04秒），保留pointwise state约束，先做与Main V2架构固定的target实验。** 不是要求NN std达到raw30，不建议把6Hz低通导数直接设为新的唯一真值。

积分一致性结果如下，velocity结果也保存在CSV：
| split | target | steps | rmse | r2 |
| --- | --- | --- | --- | --- |
| train | D0_raw | 2 | 0.0000 | 1.0000 |
| train | D0_raw | 5 | 0.0000 | 1.0000 |
| train | D0_raw | 10 | 0.0000 | 1.0000 |
| train | D0_raw | 25 | 0.0000 | 1.0000 |
| train | D1_lp12 | 2 | 0.4966 | 0.7828 |
| train | D1_lp12 | 5 | 0.4505 | 0.8857 |
| train | D1_lp12 | 10 | 0.4477 | 0.8531 |
| train | D1_lp12 | 25 | 0.4288 | 0.8736 |
| train | D1_lp4 | 2 | 0.9905 | 0.2040 |
| train | D1_lp4 | 5 | 0.9938 | 0.5067 |
| train | D1_lp4 | 10 | 0.8980 | 0.4844 |
| train | D1_lp4 | 25 | 0.8854 | 0.5141 |
| train | D1_lp6 | 2 | 0.8759 | 0.3681 |
| train | D1_lp6 | 5 | 0.6687 | 0.7615 |
| train | D1_lp6 | 10 | 0.7968 | 0.5878 |
| train | D1_lp6 | 25 | 0.7233 | 0.6613 |
| train | D1_lp8 | 2 | 0.7404 | 0.5408 |
| train | D1_lp8 | 5 | 0.5484 | 0.8346 |
| train | D1_lp8 | 10 | 0.6326 | 0.7225 |
| train | D1_lp8 | 25 | 0.5994 | 0.7599 |
| train | D2_center10 | 2 | 1.0939 | 0.0336 |
| train | D2_center10 | 5 | 1.2184 | 0.2626 |
| train | D2_center10 | 10 | 1.0621 | 0.2950 |
| train | D2_center10 | 25 | 1.0266 | 0.3534 |
| train | D2_center2 | 2 | 0.5875 | 0.7125 |
| train | D2_center2 | 5 | 0.5653 | 0.8326 |
| train | D2_center2 | 10 | 0.5876 | 0.7688 |
| train | D2_center2 | 25 | 0.6278 | 0.7404 |
| train | D2_center5 | 2 | 0.9114 | 0.3065 |
| train | D2_center5 | 5 | 0.7421 | 0.7075 |
| train | D2_center5 | 10 | 0.7985 | 0.5709 |
| train | D2_center5 | 25 | 0.7620 | 0.6205 |
| validation | D0_raw | 2 | 0.0000 | 1.0000 |
| validation | D0_raw | 5 | 0.0000 | 1.0000 |
| validation | D0_raw | 10 | 0.0000 | 1.0000 |
| validation | D0_raw | 25 | 0.0000 | 1.0000 |
| validation | D1_lp12 | 2 | 0.3387 | 0.8636 |
| validation | D1_lp12 | 5 | 0.3319 | 0.9114 |
| validation | D1_lp12 | 10 | 0.3296 | 0.9074 |
| validation | D1_lp12 | 25 | 0.2992 | 0.8943 |
| validation | D1_lp4 | 2 | 0.8292 | 0.1811 |
| validation | D1_lp4 | 5 | 0.8097 | 0.4770 |
| validation | D1_lp4 | 10 | 0.8526 | 0.3976 |
| validation | D1_lp4 | 25 | 0.6711 | 0.4945 |
| validation | D1_lp6 | 2 | 0.7312 | 0.3661 |
| validation | D1_lp6 | 5 | 0.5961 | 0.7144 |
| validation | D1_lp6 | 10 | 0.7464 | 0.5323 |
| validation | D1_lp6 | 25 | 0.5664 | 0.6329 |
| validation | D1_lp8 | 2 | 0.5436 | 0.6494 |
| validation | D1_lp8 | 5 | 0.4256 | 0.8534 |
| validation | D1_lp8 | 10 | 0.5139 | 0.7701 |
| validation | D1_lp8 | 25 | 0.4201 | 0.7907 |
| validation | D2_center10 | 2 | 0.9546 | -0.0863 |
| validation | D2_center10 | 5 | 1.0119 | 0.1852 |
| validation | D2_center10 | 10 | 1.0329 | 0.1237 |
| validation | D2_center10 | 25 | 0.8052 | 0.2812 |
| validation | D2_center2 | 2 | 0.4529 | 0.7563 |
| validation | D2_center2 | 5 | 0.4510 | 0.8374 |
| validation | D2_center2 | 10 | 0.4906 | 0.7993 |
| validation | D2_center2 | 25 | 0.4415 | 0.7771 |
| validation | D2_center5 | 2 | 0.6921 | 0.4317 |
| validation | D2_center5 | 5 | 0.5802 | 0.7300 |
| validation | D2_center5 | 10 | 0.6721 | 0.6174 |
| validation | D2_center5 | 25 | 0.5436 | 0.6593 |

D0积分精确等于原状态增量是望远镜求和的代数恒等式，不证明无噪声。D1_lp6在validation约0.04/0.2/0.5秒角增量R²仅0.366/0.532/0.633，lp12约0.864/0.907/0.894；强低通确实删掉了可积累的state变化。2/5/10step平均若被当瞬时导数再积分也有平滑/时移误差。2step监督的优势由下面同输入、同架构、同预算模型对照支持；未证明可直接改善5秒simulator。

## Q5：当前input/history是否近似可辨识？ PARTIAL

固定train reference stride5、每log200个等间隔query、k5/20；同log排除±2秒，跨log单独检验。全部标准化只用train；neighbor_ids.csv保存每个query与邻居身份。

| representation | target | conditional_variance_ratio | knn_r2 | distance_median |
| --- | --- | --- | --- | --- |
| actuator | D0_raw | 0.6129 | 0.1116 | 0.5833 |
| actuator | D1_lp6 | 0.5555 | 0.0565 | 0.5833 |
| actuator | increment2 | 0.5179 | 0.2971 | 0.5833 |
| gru26 | D0_raw | 0.5317 | 0.2185 | 0.5943 |
| gru26 | D1_lp6 | 0.3094 | 0.5400 | 0.5943 |
| gru26 | increment2 | 0.4520 | 0.3927 | 0.5943 |
| gru26_airdata | D0_raw | 0.5220 | 0.2278 | 0.6061 |
| gru26_airdata | D1_lp6 | 0.2990 | 0.5335 | 0.6061 |
| gru26_airdata | increment2 | 0.4457 | 0.3997 | 0.6061 |
| gru26_phase | D0_raw | 0.5367 | 0.2262 | 0.6161 |
| gru26_phase | D1_lp6 | 0.3067 | 0.5480 | 0.6161 |
| gru26_phase | increment2 | 0.4547 | 0.3924 | 0.6161 |
| history13 | D0_raw | 0.5745 | 0.2033 | 0.6398 |
| history13 | D1_lp6 | 0.4107 | 0.4304 | 0.6398 |
| history13 | increment2 | 0.5188 | 0.3375 | 0.6398 |
| history26 | D0_raw | 0.5829 | 0.1725 | 0.6553 |
| history26 | D1_lp6 | 0.4345 | 0.3913 | 0.6553 |
| history26 | increment2 | 0.5368 | 0.2906 | 0.6553 |
| history5 | D0_raw | 0.6248 | 0.2173 | 0.6285 |
| history5 | D1_lp6 | 0.4214 | 0.4673 | 0.6285 |
| history5 | increment2 | 0.5356 | 0.3637 | 0.6285 |
| instant | D0_raw | 0.6043 | 0.1267 | 0.5256 |
| instant | D1_lp6 | 0.5355 | 0.0773 | 0.5256 |
| instant | increment2 | 0.5039 | 0.3235 | 0.5256 |

GRU26相对instant改善filtered/increment的邻域预测：validation raw R²约0.127→0.218，lp6约0.077→0.540，increment2约0.324→0.393。13点通常优于26点flattened history，不能据此说“history无用”，也不能说history越长越能解决missing state。不同维数邻域距离有集中效应，未做相同流形维数证明。

严格“几乎相同”输入证据不足：validation k20、RMS标准化距离≤0.25时所有表示均无query；≤0.5时GRU仅30/1000个。不能从剩余conditional variance直接证明非Markov或必须latent。same_log/other_log结果有差异但邻域密度不同；date与split完全混杂，不能识别wind/date因果。

## Q6：最可能缺什么？按证据排序

1. **可靠phase reference/一致phase表示**：同步结构强且确有t0重锚、跨log零点不确定。它是有直接证据的表示缺口，但受控probe尚未证明简单替换能泛化。
2. **airspeed/wind**：日志实际存在Pitot速度与estimated wind，不是完全缺传感器；额外context使raw角conditional ratio仅约0.532→0.522、kNN R²0.218→0.228，证据弱，不能宣布它主导误差。
3. **actual servo state**：没有feedback，只有命令和输出命令；高tail误差关联见actuator_associations.csv，不能区分lag/load/saturation/dead-zone，更不能据此改time constants。
4. **motor/load**：pack current/voltage、raw RPM存在，但不能当真实motor torque/wing deformation。额外信息的邻域指标改善很小。
5. **wing/aero memory及其他latent**：本轮没有直接传感器或可识别的因果证据，不以剩余方差当作证明。

可用telemetry的原生publication rate和全日志最大gap（不能把全日志gap都当飞行核心掉线；core freshness另列）：
| topic | rate_min | rate_max | max_recorded_gap_s |
| --- | --- | --- | --- |
| actuator_outputs | 9.8992 | 10.0203 | 0.4267 |
| airspeed | 16.6984 | 16.9067 | 0.4242 |
| airspeed_validated | 10.0000 | 10.0000 | 0.5000 |
| battery_status | 10.0000 | 10.0000 | 0.4308 |
| rpm | 100.0000 | 100.0000 | 0.4200 |
| vehicle_acceleration | 0.9995 | 1.0005 | 1.0057 |
| vehicle_angular_velocity | 100.1904 | 101.4405 | 0.4293 |
| wind | 20.0381 | 20.2864 | 100.7092 |

airspeed_quality_input的source=1且valid=1，validated多为sensor1；validation最后一log有约7.6% source=-1。字段存在/valid只表示发布契约，不证明校准精度；wind是估计量，GPS与local velocity来自导航链，均非独立空气速度真值。wind最长发布gap可约100秒，全日志range需结合core freshness。

servo_feedback、ESC report/status在11日志均不存在。battery current为pack ADC量；日志间current均值有差异，validation一log约7.61A，其他约2.5–3.5A，不能假定同频率等于同负载或把电流差直接归于负载。

新增load channels的validation角动态kNN：
| representation | conditional_variance_ratio | knn_rmse |
| --- | --- | --- |
| plus_all_load_channels | 0.5317 | 27.0569 |
| plus_pack_voltage_current | 0.5368 | 27.2235 |
| plus_raw_rpm | 0.5270 | 27.1686 |

## Q7：command sensitivity可信？ PARTIAL，尚无真实Jacobian验证

固定600 train和500 validation operating points（每log100等间隔），±0.001 normalized command中心差分。sym同时改变左右，diff左右反向；没有用结果挑点。完整输出和motor/tail/speed/body-rate/frequency分桶见command_sensitivity.csv、sensitivity_bins.csv。

| channel | step | output | median | p1 | p99 | max_abs | near_zero_fraction | positive_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| motor | 0 | frequency_dot | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| motor | 1 | frequency_dot | 3.3089 | 1.6716 | 4.8817 | 4.9450 | 0.0000 | 1.0000 |
| motor | 5 | frequency_dot | 9.6750 | 6.4266 | 10.7306 | 10.8184 | 0.0000 | 1.0000 |
| rudder | 0 | r_dot | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| rudder | 1 | r_dot | 0.5500 | -0.0899 | 1.0137 | 1.4172 | 0.0000 | 0.9820 |
| rudder | 5 | r_dot | 0.7010 | -0.7786 | 1.4310 | 1.9062 | 0.0000 | 0.9040 |
| tail_diff | 0 | p_dot | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| tail_diff | 1 | p_dot | -1.1904 | -3.0611 | 1.0686 | 3.4847 | 0.0000 | 0.1040 |
| tail_diff | 5 | p_dot | -1.8603 | -3.9527 | 1.1693 | 5.2948 | 0.0000 | 0.0920 |
| tail_sym | 0 | q_dot | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| tail_sym | 1 | q_dot | 1.9698 | 0.0720 | 4.0876 | 4.6020 | 0.0000 | 0.9920 |
| tail_sym | 5 | q_dot | 2.2454 | -2.3004 | 4.1301 | 4.5178 | 0.0000 | 0.9240 |

step0导数对command严格零，原因是proxy在step末更新。step1 motor→frequency-dot为正且非零；此时motor→rigid-body仍零（结构只直接作用frequency），后续只通过状态间接影响。sym→q-dot、rudder→r-dot大多同号但少数变号，diff→p-dot有明显符号分布。没有经过机械舵面符号/真实响应实验标定，不能把正/负直接判wrong sign。零cross-axis响应多由mask强制，不是从数据证明该通道真实不存在；这些限制不支持宣称控制因果已学会。

## 小型受控对照（先审计后训练）

B0 current26-point input→raw；B1同输入→2step increment/duration；B2更换history phase为log-phase加train-only offset→raw；B3 phase+increment。相同420→64→64→6 MLP、30epoch、batch512、AdamW lr0.001、seed415/416/417、同训练rows/原始target scales；没有预训练GRU权重作为probe输入。第一70%每train log训练、末30%purge2s作训练内诊断，正式train/validation flight分配没有变化。validation offset未知设0，绝不调其标签offset。

| scope | model | evaluation_target | signal | rmse | r2 | pred_std | truth_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_temporal_holdout | B0 | D0_raw | angular | 22.8799 | 0.5477 | 27.9076 | 34.7974 |
| train_temporal_holdout | B0 | D0_raw | linear | 3.9188 | 0.8421 | 9.3366 | 9.9061 |
| train_temporal_holdout | B0 | increment2 | angular | 17.7078 | 0.5140 | 27.9076 | 26.2311 |
| train_temporal_holdout | B0 | increment2 | linear | 3.9611 | 0.8245 | 9.3366 | 9.5103 |
| train_temporal_holdout | B1 | D0_raw | angular | 26.8502 | 0.3889 | 23.3760 | 34.7974 |
| train_temporal_holdout | B1 | D0_raw | linear | 4.9821 | 0.7459 | 9.2362 | 9.9061 |
| train_temporal_holdout | B1 | increment2 | angular | 13.7531 | 0.6998 | 23.3760 | 26.2311 |
| train_temporal_holdout | B1 | increment2 | linear | 3.7092 | 0.8455 | 9.2362 | 9.5103 |
| train_temporal_holdout | B2 | D0_raw | angular | 22.8119 | 0.5538 | 28.8747 | 34.7974 |
| train_temporal_holdout | B2 | D0_raw | linear | 4.0211 | 0.8321 | 9.5836 | 9.9061 |
| train_temporal_holdout | B2 | increment2 | angular | 18.5574 | 0.4688 | 28.8747 | 26.2311 |
| train_temporal_holdout | B2 | increment2 | linear | 4.0320 | 0.8182 | 9.5836 | 9.5103 |
| train_temporal_holdout | B3 | D0_raw | angular | 27.0129 | 0.3814 | 23.6878 | 34.7974 |
| train_temporal_holdout | B3 | D0_raw | linear | 4.8815 | 0.7536 | 9.3136 | 9.9061 |
| train_temporal_holdout | B3 | increment2 | angular | 14.0767 | 0.6869 | 23.6878 | 26.2311 |
| train_temporal_holdout | B3 | increment2 | linear | 3.6108 | 0.8517 | 9.3136 | 9.5103 |
| validation_unknown_offset | B0 | D0_raw | angular | 23.1293 | 0.4108 | 22.9547 | 30.1897 |
| validation_unknown_offset | B0 | D0_raw | linear | 4.3952 | 0.8349 | 10.2786 | 10.8448 |
| validation_unknown_offset | B0 | increment2 | angular | 18.1497 | 0.3607 | 22.9547 | 22.7205 |
| validation_unknown_offset | B0 | increment2 | linear | 4.9327 | 0.7749 | 10.2786 | 10.4023 |
| validation_unknown_offset | B1 | D0_raw | angular | 24.1772 | 0.3567 | 19.3573 | 30.1897 |
| validation_unknown_offset | B1 | D0_raw | linear | 4.7746 | 0.8053 | 9.8441 | 10.8448 |
| validation_unknown_offset | B1 | increment2 | angular | 14.4931 | 0.5901 | 19.3573 | 22.7205 |
| validation_unknown_offset | B1 | increment2 | linear | 3.9039 | 0.8583 | 9.8441 | 10.4023 |
| validation_unknown_offset | B2 | D0_raw | angular | 25.5798 | 0.2802 | 22.0750 | 30.1897 |
| validation_unknown_offset | B2 | D0_raw | linear | 5.4502 | 0.7468 | 9.1832 | 10.8448 |
| validation_unknown_offset | B2 | increment2 | angular | 19.8550 | 0.2336 | 22.0750 | 22.7205 |
| validation_unknown_offset | B2 | increment2 | linear | 5.5907 | 0.7096 | 9.1832 | 10.4023 |
| validation_unknown_offset | B3 | D0_raw | angular | 24.8295 | 0.3226 | 18.3197 | 30.1897 |
| validation_unknown_offset | B3 | D0_raw | linear | 5.9100 | 0.7005 | 8.7060 | 10.8448 |
| validation_unknown_offset | B3 | increment2 | angular | 15.6315 | 0.5246 | 18.3197 | 22.7205 |
| validation_unknown_offset | B3 | increment2 | linear | 5.0270 | 0.7659 | 8.7060 | 10.4023 |

同target、配对seed百分比变化（负数改善；seed范围不是跨flight置信区间）：
| scope | candidate | reference | target | signal | mean_change_pct | seed_min | seed_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_temporal_holdout | B1 | B0 | increment2 | linear | -6.3543 | -7.6352 | -5.1684 |
| train_temporal_holdout | B1 | B0 | increment2 | angular | -22.3304 | -23.1834 | -21.7401 |
| train_temporal_holdout | B2 | B0 | D0_raw | linear | 2.6114 | 2.2135 | 3.2256 |
| train_temporal_holdout | B2 | B0 | D0_raw | angular | -0.2990 | -1.7466 | 0.9539 |
| train_temporal_holdout | B3 | B1 | increment2 | linear | -2.6633 | -6.2092 | -0.2288 |
| train_temporal_holdout | B3 | B1 | increment2 | angular | 2.3516 | 1.2672 | 3.0449 |
| validation_unknown_offset | B1 | B0 | increment2 | linear | -20.8339 | -23.3640 | -19.1660 |
| validation_unknown_offset | B1 | B0 | increment2 | angular | -20.1693 | -22.9198 | -18.4544 |
| validation_unknown_offset | B2 | B0 | D0_raw | linear | 24.0216 | 19.4308 | 27.4754 |
| validation_unknown_offset | B2 | B0 | D0_raw | angular | 10.6040 | 9.2147 | 12.4370 |
| validation_unknown_offset | B3 | B1 | increment2 | linear | 28.7544 | 25.7021 | 32.0801 |
| validation_unknown_offset | B3 | B1 | increment2 | angular | 8.0222 | 3.7457 | 14.7065 |

B0本身已能在validation预测线性raw动态R²约0.835、角动态约0.411，明显强于冻结Main V2一步的约0.079/0.265。这说明合法history中有更多可用信息，不能直接判state整体不可观测或数据全噪声；但probe同时改变训练任务/映射结构，不能单独归因GRU容量。B1对共同2step target优于B0；B2对未知zero的validation退化，无法支持当前直接phase替换。所有probe只做局部回归，没有5秒simulator评估，不作候选晋级。

## Q8：长期漂移最有证据的前三个来源

1. **当前训练得到的局部transition没有利用可预测动态变化**：teacher-state已衰减，phase均值幅值偏弱；小模型在同合法history上能预测更多变化。原因可能包括监督对象与recurrent representation，不能仅凭此判容量不足。
2. **phase位置表示不一致及跨log reference不确定**：有直接源码与phase-domain证据；但history可补部分信息，故没有证明其为唯一或首要可修复因果。
3. **raw native差分含大量快速变化且监督时间尺度不匹配**：滤波/积分与increment probe显示target选择重要；噪声与高阶可重复分量不能严格分离。

**下一步只选P2为第一优先级**：短时state-increment物理监督的固定Main V2实验，继续保留raw/filtered/phase同期诊断；P1为第二优先，必须先解决可部署的reference契约。P3/P4/P5没有足够提升证据，P6可为跨log机械相位和独立舵面/气流验证服务，但当前不优先替换网络或启动RL。

## Readiness / 限制

Simulator structural readiness: **100%**

Dynamics fidelity readiness: **60%**

RL-ready: **NO**

沿用Step 3十项证据清单，不因诊断或局部probe提高分数。本轮没有证明长期fidelity改善，没有确认独立物理真值，也没有完成可靠command因果标定。

## 一条完整复现命令

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_step4_all.py \
  --output /tmp/main-v2-step4/results --artifacts /tmp/main-v2-step4/probes --device cuda:1
```

使用空目录，先phase/derivative/telemetry/teacher/NN/sensitivity审计，再补充诊断、12个小模型、报告和回归。所有过滤参数、source hashes、IDs、seed与step数均保存。测试和git diff检查结果见verification.json。