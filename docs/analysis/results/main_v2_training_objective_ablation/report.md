# Main V2 training-objective ablation — Step 3


保持架构、模拟器、actuator time constants、归一化与数据划分不变；全部候选重新训练。sealed test未打开。

## 当前loss与预诊断
详见 `docs/audits/2026-09-15_main_v2_loss_audit.md`。原训练为50个实际dt转移，约1秒，非固定50×0.02积分。

实际训练时长分布（秒）：
| steps | count | min | max | p1 | p99 | mean | median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 4214.0000 | 0.9880 | 1.0081 | 0.9881 | 1.0081 | 1.0000 | 0.9980 |
| 100 | 4214.0000 | 1.9911 | 2.0110 | 1.9911 | 2.0063 | 2.0001 | 1.9963 |

冻结train的归一化状态loss量级（选择prefix权重之前）：
| steps | position | velocity | attitude | body_rate | phase | frequency |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.0033 | 0.0796 | 0.0209 | 0.1767 | 0.0023 | 0.0004 |
| 25 | 0.0153 | 0.0873 | 0.0566 | 0.1898 | 0.0054 | 0.0004 |
| 50 | 0.0778 | 0.1376 | 0.1128 | 0.2000 | 0.0153 | 0.0004 |
| 100 | 0.7203 | 0.3543 | 0.2460 | 0.2135 | 0.0451 | 0.0004 |

A0相对于旧checkpoint最大参数差：0.0；因此本次重训成功复现旧权重，并非复制旧checkpoint。

局部一步预测已经低估角加速度标准差（验证三轴约40%/40%/32%），递推进一步衰减；下一步omega本身较接近测量部分来自输入的真实omega，不能据此称一步导数准确。GRU norm从0.2秒约2.21降到5秒约1.78，但其temporal std约1.1并未消失，不能简单归因为hidden collapse。

原生角加速度差分训练谱中10Hz以上功率约48%；使用两步对齐delta-omega辅助，避免直接回归所有高频差分。不是振幅奖励，也不是证明高频均为噪声。

## 实验与训练预算
| experiment | steps | base_epochs | actuator_epochs | wall_time_s | base_windows_per_s | actuator_windows_per_s | peak_gpu_bytes | base_final_loss | actuator_final_loss | checkpoint_sha256 | base_optimizer_steps | actuator_optimizer_steps | legacy_checkpoint_max_parameter_difference | device | concurrent_jobs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_baseline_retrain | 50 | 40 | 25 | 1013.7547 | 306.8688 | 227.5276 | 144401920 | 0.6101 | 0.5671 | cfbff6f384fffe7f04a2735861bc729f7c276022ff09271827d3170c960904ff | 680 | 425 | 0.0000 | cuda:1 | 4 |
| A1_longer_rollout | 100 | 40 | 25 | 2003.1074 | 148.1217 | 121.9622 | 200081920 | 1.8312 | 1.6499 | c48af41028b039b93c36721446c6921e8e49479318c22272e238c7ae5daf37a1 | 680 | 425 | nan | cuda:0 | 4 |
| A2_multi_horizon | 100 | 40 | 25 | 2177.4742 | 135.6765 | 112.8113 | 200081920 | 0.4392 | 0.4117 | bb61a0a361986f21f9d352c411dae61a791bff95ca9d971c7d883d013b73e497 | 680 | 425 | nan | cuda:1 | 4 |
| A3_dynamic_delta | 100 | 40 | 25 | 2039.9810 | 150.4454 | 114.7641 | 200081920 | 0.6317 | 0.5972 | 2d143ac04c6e0bab894e2f795e13172a2652ef822eef5781395ec6ed43ccd4be | 680 | 425 | nan | cuda:0 | 4 |

四组均4,214 train窗口、40+25 epoch、seed17/29、batch256、最终epoch checkpoint。两阶段分别训练backbone、冻结该backbone后训练actuator residual。GPU并发壁钟/吞吐受共享负载影响，不是独占硬件速度对比。串行pilot在完成checkpoint前停止并保留记录，不计入正式矩阵。

## Q1：一步还是递推？
两者都有。一步角加速度幅值/偏差已有问题；free-run后状态动态进一步变弱。不能从这种相关时序证明loss是唯一原因。

## Q2：长horizon是否改善？
A1对A0的逐flight置信区间与完整变化见下表。

| candidate | reference | horizon_s | metric | change_pct | ci95_lower | ci95_upper | improved_flights | n_flights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1_longer_rollout | A0_baseline_retrain | 2.0000 | velocity_m_s | 7.9889 | 5.6261 | 10.8154 | 0 | 5 |
| A1_longer_rollout | A0_baseline_retrain | 2.0000 | attitude_deg | 5.4134 | 2.5513 | 8.2341 | 0 | 5 |
| A1_longer_rollout | A0_baseline_retrain | 3.0000 | velocity_m_s | 8.0948 | 3.7333 | 12.9017 | 0 | 5 |
| A1_longer_rollout | A0_baseline_retrain | 3.0000 | attitude_deg | 7.3619 | 3.4588 | 11.3660 | 0 | 5 |
| A1_longer_rollout | A0_baseline_retrain | 5.0000 | velocity_m_s | 8.1620 | 4.7348 | 11.3179 | 0 | 5 |
| A1_longer_rollout | A0_baseline_retrain | 5.0000 | attitude_deg | 6.4924 | 3.4271 | 9.5622 | 0 | 5 |

## Q3：multi-horizon是否优于单一horizon？
A2对A1（同100步）的直接对照：

| candidate | reference | horizon_s | metric | change_pct | ci95_lower | ci95_upper | improved_flights | n_flights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2_multi_horizon | A1_longer_rollout | 0.2000 | velocity_m_s | -40.1519 | -41.3978 | -38.8537 | 5 | 5 |
| A2_multi_horizon | A1_longer_rollout | 0.2000 | attitude_deg | -15.2793 | -18.6941 | -11.7014 | 5 | 5 |
| A2_multi_horizon | A1_longer_rollout | 0.5000 | velocity_m_s | -5.8438 | -7.7151 | -3.6891 | 5 | 5 |
| A2_multi_horizon | A1_longer_rollout | 0.5000 | attitude_deg | -2.5958 | -4.3474 | -0.8504 | 4 | 5 |
| A2_multi_horizon | A1_longer_rollout | 2.0000 | velocity_m_s | -3.8719 | -5.4798 | -2.1214 | 5 | 5 |
| A2_multi_horizon | A1_longer_rollout | 2.0000 | attitude_deg | -0.1969 | -3.5858 | 3.3758 | 2 | 5 |
| A2_multi_horizon | A1_longer_rollout | 3.0000 | velocity_m_s | -4.2837 | -7.5910 | -0.6664 | 4 | 5 |
| A2_multi_horizon | A1_longer_rollout | 3.0000 | attitude_deg | -2.7639 | -6.9477 | 1.8621 | 3 | 5 |
| A2_multi_horizon | A1_longer_rollout | 5.0000 | velocity_m_s | -6.6263 | -10.2898 | -2.5212 | 5 | 5 |
| A2_multi_horizon | A1_longer_rollout | 5.0000 | attitude_deg | -5.8306 | -9.8849 | -1.2534 | 4 | 5 |

## Q4：dynamic loss是否恢复动态？
A3相对于A2只增加对齐两步delta-omega，lambda由train量级固定。不能把比值略增等同于恢复真实动态：

| experiment | horizon_s | signal | median | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| A2_multi_horizon | 0.2000 | omega | 0.7157 | 0.4872 | 0.9585 |
| A2_multi_horizon | 0.5000 | omega | 0.6097 | 0.4559 | 0.7989 |
| A2_multi_horizon | 1.0000 | omega | 0.5040 | 0.3556 | 0.6825 |
| A2_multi_horizon | 2.0000 | omega | 0.4110 | 0.2826 | 0.5875 |
| A2_multi_horizon | 3.0000 | omega | 0.3702 | 0.2591 | 0.5512 |
| A2_multi_horizon | 5.0000 | omega | 0.3283 | 0.2331 | 0.4938 |
| A2_multi_horizon | 5.0000 | omega_last1s | 0.1681 | 0.1005 | 0.3910 |
| A3_dynamic_delta | 0.2000 | omega | 0.7293 | 0.5022 | 0.9741 |
| A3_dynamic_delta | 0.5000 | omega | 0.6155 | 0.4514 | 0.8065 |
| A3_dynamic_delta | 1.0000 | omega | 0.4905 | 0.3480 | 0.6637 |
| A3_dynamic_delta | 2.0000 | omega | 0.3814 | 0.2698 | 0.5387 |
| A3_dynamic_delta | 3.0000 | omega | 0.3375 | 0.2421 | 0.4761 |
| A3_dynamic_delta | 5.0000 | omega | 0.2910 | 0.2102 | 0.4004 |
| A3_dynamic_delta | 5.0000 | omega_last1s | 0.1070 | 0.0707 | 0.1684 |

噪声检查包括angular acceleration RMSE、PSD偏差和train-PSD的95%功率频率以上能量，详见各模型summary/derivatives/spectra。

| experiment | angular_accel_rmse | high_frequency_energy_ratio | spectral_relative_l1 |
| --- | --- | --- | --- |
| A0_baseline_retrain | 30.0012 | 0.0000 | 0.9670 |
| A1_longer_rollout | 30.0420 | 0.0000 | 0.9835 |
| A2_multi_horizon | 30.0433 | 0.0002 | 0.8934 |
| A3_dynamic_delta | 29.9201 | 0.0001 | 0.9371 |

## Q5：改善来源
所有模型积分代码完全相同，所以没有“更准确的积分器”这一实现改变。候选间angular acceleration误差/幅值、hidden统计、相位同步和旋转误差分解见对应CSV；只是伴随变化，不能未经干预把某项判为因果来源。

相位误差由频率积分误差加实测phase/frequency不一致项精确重建：冻结模型5秒两项RMS约1.90 rad和0.13 rad，重建误差约3e-6 rad。初始frequency输入真值，initial error为0；微小持续frequency bias依然能积累相位漂移。固定phase anchor与真实dt均保持，未发现积分bug。相关性按每个horizon分别计算，见phase_correlations.csv，不混合时间来制造相关。

## Q6：最佳实验与A0百分比变化
最佳只按预先声明的2/3/5秒velocity/attitude日志等权RMSE几何比值排序，不等于通过success gate。选择：**A3_dynamic_delta**。负数是改善。

| experiment | horizon_s | position_m_change_pct | velocity_m_s_change_pct | attitude_deg_change_pct | body_rate_rad_s_change_pct | frequency_hz_change_pct | phase_rad_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3_dynamic_delta | 0.2000 | -29.2004 | -25.1938 | 1.4081 | -5.6514 | -4.7536 | -4.8670 |
| A3_dynamic_delta | 0.5000 | -10.4984 | 1.9755 | 2.5451 | -6.4028 | 0.0692 | -0.2485 |
| A3_dynamic_delta | 1.0000 | 0.4671 | 2.0321 | 4.6130 | -3.2888 | 1.0382 | 2.3036 |
| A3_dynamic_delta | 2.0000 | 1.5573 | 2.4593 | 5.0482 | 0.8092 | 1.8108 | 3.5221 |
| A3_dynamic_delta | 3.0000 | 1.6481 | 2.5091 | 4.2682 | 1.0655 | 1.9829 | 5.0691 |
| A3_dynamic_delta | 5.0000 | 0.8136 | 0.2114 | -0.1819 | 0.3782 | 1.9564 | 2.1422 |

最佳完整日志等权RMSE：

| horizon_s | position_m_equal_log_rmse | velocity_m_s_equal_log_rmse | attitude_deg_equal_log_rmse | body_rate_rad_s_equal_log_rmse | frequency_hz_equal_log_rmse | phase_rad_equal_log_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 0.0619 | 0.3769 | 3.2728 | 0.6674 | 0.1833 | 0.2151 |
| 0.5000 | 0.1824 | 0.7048 | 6.5906 | 0.6947 | 0.1871 | 0.3357 |
| 1.0000 | 0.5734 | 1.1794 | 10.9990 | 0.7248 | 0.1990 | 0.6035 |
| 2.0000 | 1.9660 | 2.1471 | 18.9236 | 0.7695 | 0.2061 | 0.9465 |
| 3.0000 | 4.3396 | 3.3305 | 28.3170 | 0.7824 | 0.2080 | 1.2409 |
| 5.0000 | 11.9033 | 5.0622 | 45.4490 | 0.7909 | 0.2056 | 1.6608 |

## Q7：最佳body-rate variation
| experiment | horizon_s | signal | median | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| A3_dynamic_delta | 0.2000 | omega | 0.7293 | 0.5022 | 0.9741 |
| A3_dynamic_delta | 0.5000 | omega | 0.6155 | 0.4514 | 0.8065 |
| A3_dynamic_delta | 1.0000 | omega | 0.4905 | 0.3480 | 0.6637 |
| A3_dynamic_delta | 2.0000 | omega | 0.3814 | 0.2698 | 0.5387 |
| A3_dynamic_delta | 3.0000 | omega | 0.3375 | 0.2421 | 0.4761 |
| A3_dynamic_delta | 5.0000 | omega | 0.2910 | 0.2102 | 0.4004 |
| A3_dynamic_delta | 5.0000 | omega_last1s | 0.1070 | 0.0707 | 0.1684 |

5秒prefix median=0.290975；末1秒median=0.106986。必须结合PSD与导数误差，不能以振幅增加本身论成功。

## 稳定性
| experiment | numeric_failures | support_failures | clipping_failures |
| --- | --- | --- | --- |
| A0_baseline_retrain | 0 | 255 | 0 |
| A1_longer_rollout | 0 | 438 | 0 |
| A2_multi_horizon | 0 | 199 | 0 |
| A3_dynamic_delta | 0 | 231 | 0 |

Support是训练min/max支持域筛查，不是适航硬包线；保留有限失败轨迹计算误差。

## Q8：下一阶段
C. objective 基本无效，说明主要瓶颈不是训练 horizon/loss

该判断仅覆盖本轮1/2秒、固定四组目标与单seed政策，不能证明所有可能的loss无效。3秒会改变原共同训练起点，本轮未加入；5秒训练是否需要仍未证明，不预设curriculum必然有效。

统计限制：仅5条flight，cluster bootstrap为3125种重采样；没有独立多seed复验，也没有对模型选择/多重比较做校正。即便CI排除0，也只作本矩阵的条件性证据，不宣称跨seed/新策略显著。短期任一指标退化>10%应明确警告，见下表：

| experiment | horizon_s | position_m_change_pct | velocity_m_s_change_pct | attitude_deg_change_pct | body_rate_rad_s_change_pct | frequency_hz_change_pct | phase_rad_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3_dynamic_delta | 0.2000 | -29.2004 | -25.1938 | 1.4081 | -5.6514 | -4.7536 | -4.8670 |
| A3_dynamic_delta | 0.5000 | -10.4984 | 1.9755 | 2.5451 | -6.4028 | 0.0692 | -0.2485 |

## Readiness
Simulator structural readiness: 100%

Dynamics fidelity readiness: 60%

RL-ready: NO

沿用Step 2十项证据清单，不是成功概率；未证明项不加分。

## 复现
```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_step3.py \
  --results /tmp/main-v2-step3/results --artifacts /tmp/main-v2-step3/artifacts
```
训练前自动诊断，四个GPU作业隔离运行，完成后统一benchmark和报告。目录必须为空。

代表轨迹按各模型5秒姿态误差best/median/worst固定选择；不同模型可能不同起点，仅用于描述。统计对照始终使用相同722起点。所有图见本目录。

## 结果解释与失效边界

**延长训练时域：** 长期六项中0/6项改善，百分比变化范围5.41%至8.16%；CI排除零且改善的有0/6项。

**Multi-horizon：** 长期六项中6/6项改善，百分比变化范围-6.63%至-0.20%；CI排除零且改善的有4/6项。 这只是相对于2秒均匀监督；相对于原A0：长期六项中0/6项改善，百分比变化范围0.28%至5.21%；CI排除零且改善的有0/6项。

**Dynamic loss：** 长期六项中6/6项改善，百分比变化范围-1.30%至-0.12%；CI排除零且改善的有2/6项。

A2_multi_horizon：5秒prefix=0.3283，末1秒=0.1681。

A3_dynamic_delta：5秒prefix=0.2910，末1秒=0.1070。

不能把局部幅值改善描述为完全无效，但本矩阵没有候选同时改善长期velocity/attitude并恢复动态；Q8的C针对本轮主要长期目标，不证明loss不可能是瓶颈。A3与A2的对照也不支持新增delta项恢复了长期角运动。

短期所有实验的>10%退化告警（正数为退化）：

| experiment | horizon_s | metric | degradation_pct |
| --- | --- | --- | --- |
| A1_longer_rollout | 0.2000 | position_m | 13.3294 |
| A1_longer_rollout | 0.2000 | velocity_m_s | 16.9471 |
| A1_longer_rollout | 0.2000 | attitude_deg | 19.9438 |
| A1_longer_rollout | 0.5000 | position_m | 11.4949 |
| A1_longer_rollout | 0.5000 | phase_rad | 11.4143 |

角加速度全时域向量RMSE（rad/s²）及标准差；该原生差分含噪声，不能把接近零预测的低MSE当作已解决动态：
| experiment | one_step_rmse | full_rmse | pred_std | truth_std |
| --- | --- | --- | --- | --- |
| A0_baseline_retrain | 27.1507 | 30.0012 | 3.2963 | 30.0934 |
| A1_longer_rollout | 28.1795 | 30.0420 | 1.4037 | 30.0934 |
| A2_multi_horizon | 27.1961 | 30.0433 | 5.9509 | 30.0934 |
| A3_dynamic_delta | 27.1911 | 29.9201 | 4.7731 | 30.0934 |

姿态漂移链的离线代数分解：
| experiment | horizon_s | rotation_component_rms | body_derivative_component_rms | vector_sum_rms |
| --- | --- | --- | --- | --- |
| A0_baseline_retrain | 0.2000 | 0.0887 | 10.0783 | 10.0796 |
| A0_baseline_retrain | 1.0000 | 0.2337 | 10.5251 | 10.5262 |
| A0_baseline_retrain | 5.0000 | 0.6073 | 10.6816 | 10.6850 |
| A3_dynamic_delta | 0.2000 | 0.1664 | 7.2685 | 7.2739 |
| A3_dynamic_delta | 1.0000 | 0.3750 | 8.9582 | 8.9721 |
| A3_dynamic_delta | 5.0000 | 0.6661 | 10.4230 | 10.4372 |

这里使用(Rpred−Rtruth)×a_pred作为旋转项；分解依赖参考加速度的选择，两项相关，不是因果贡献百分比。当前旋转项远小于body-derivative项，不能宣称“姿态误差主导速度误差”的完整链条已经被证明。一步导数偏弱、递推角动态衰减、姿态和位置误差增长有共现证据。

5秒相位相关性（固定horizon；相关不是因果）：
| experiment | horizon_s | target | pearson_abs_phase | frequency_integral_error_rms | measured_phase_frequency_residual_rms | integration_reconstruction_max_error |
| --- | --- | --- | --- | --- | --- | --- |
| A0_baseline_retrain | 5.0000 | body_rate_error | 0.1376 | 1.8959 | 0.1315 | 0.0000 |
| A0_baseline_retrain | 5.0000 | angular_acceleration_rmse | 0.2775 | 1.8959 | 0.1315 | 0.0000 |
| A0_baseline_retrain | 5.0000 | linear_acceleration_rmse | 0.2064 | 1.8959 | 0.1315 | 0.0000 |
| A1_longer_rollout | 5.0000 | body_rate_error | 0.1586 | 1.7704 | 0.1315 | 0.0000 |
| A1_longer_rollout | 5.0000 | angular_acceleration_rmse | 0.2731 | 1.7704 | 0.1315 | 0.0000 |
| A1_longer_rollout | 5.0000 | linear_acceleration_rmse | 0.2165 | 1.7704 | 0.1315 | 0.0000 |
| A2_multi_horizon | 5.0000 | body_rate_error | 0.1654 | 1.8490 | 0.1315 | 0.0000 |
| A2_multi_horizon | 5.0000 | angular_acceleration_rmse | 0.3031 | 1.8490 | 0.1315 | 0.0000 |
| A2_multi_horizon | 5.0000 | linear_acceleration_rmse | 0.1951 | 1.8490 | 0.1315 | 0.0000 |
| A3_dynamic_delta | 5.0000 | body_rate_error | 0.1519 | 2.0158 | 0.1315 | 0.0000 |
| A3_dynamic_delta | 5.0000 | angular_acceleration_rmse | 0.3234 | 2.0158 | 0.1315 | 0.0000 |
| A3_dynamic_delta | 5.0000 | linear_acceleration_rmse | 0.2790 | 2.0158 | 0.1315 | 0.0000 |

最佳候选的5秒连续变量分桶（bin 0/1/2为train分位递增，非动作标签）：
| variable | bin | n_rollouts | velocity_m_s_equal_log_rmse | attitude_deg_equal_log_rmse |
| --- | --- | --- | --- | --- |
| vertical_speed | 0 | 234 | 5.2519 | 47.7611 |
| vertical_speed | 1 | 265 | 5.1804 | 46.5881 |
| vertical_speed | 2 | 223 | 4.6204 | 40.1958 |
| abs_body_yaw_rate | 0 | 283 | 4.9752 | 44.3547 |
| abs_body_yaw_rate | 1 | 239 | 5.0799 | 45.9466 |
| abs_body_yaw_rate | 2 | 200 | 5.1664 | 46.2601 |
| abs_roll_deg | 0 | 280 | 4.5970 | 40.8523 |
| abs_roll_deg | 1 | 267 | 5.0381 | 45.1608 |
| abs_roll_deg | 2 | 175 | 5.7947 | 52.8134 |
| pitch_deg | 0 | 117 | 5.0251 | 44.1734 |
| pitch_deg | 1 | 298 | 4.7032 | 41.4539 |
| pitch_deg | 2 | 307 | 5.4585 | 49.9667 |
| motor_command | 0 | 368 | 4.5739 | 40.2858 |
| motor_command | 1 | 268 | 4.9877 | 44.9508 |
| motor_command | 2 | 86 | 6.7027 | 61.1318 |
| tail_command_magnitude | 0 | 188 | 4.3581 | 38.3982 |
| tail_command_magnitude | 1 | 314 | 4.6366 | 40.1957 |
| tail_command_magnitude | 2 | 220 | 6.2359 | 57.6612 |

分桶边界沿用Step 2 summary.json中的bin_edges_train_tertiles。各桶flight构成可能不同，只作条件误差描述，不据此推断控制因果。完整0.5/1/2/5秒各模型数据见regime_summary.csv。

逐10步非重叠块的角加速度、omega variation以及hidden norm/variation见attenuation_blocks.csv与attenuation_and_hidden.png；prefix统计见各模型dynamics_statistics.csv，避免用累积std掩盖末段衰减。