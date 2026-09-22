# Paper Step6：冻结MLP三seed比较

## 1. 论文问题与核心结论

补齐三seed后，500 ms三项主要动力学指标的均值仍支持Standard GRU/H26优于当前MLP。这是冻结模型家族与matched-budget协议下的验证集比较；不预设所有seed、flight和时域都胜出。

## 2. 实验设置、复用与覆盖

只新增MLP seed23与42，各自随机初始化；seed17和H26三个seed直接复用。B0是单个确定性参照，没有复制seed或人为标准差。41架训练flight/28,293 origins；17架验证flight/2,582 origins；Sep7为9架/1,481，Sep17为8架/1,101。所有模型origins、标签、Actual控制和native dt一致。MLP为5,703参数、GRU为21,383参数。40+25 epochs、两阶段重建AdamW、7215次更新、最后epoch65，不按验证误差选模。归一化沿用原B1六个冻结buffer，未重新fit。内部硬编码seed17的旧factory未修改，新入口保留原类及构造顺序，明确设置23/35、42/54。

## 3. 500 ms主结果

| Cohort | Horizon [s] | Model | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.5 | B0 Kinematic | 0.4182 | 1.1707 | 23.2455 | 1.0660 |
| ALL | 0.5 | B1 MLP | 0.2407 ± 0.0019 | 0.5422 ± 0.0097 | 5.4859 ± 0.0451 | 0.7526 ± 0.0015 |
| ALL | 0.5 | B2 Standard GRU / H26 | 0.1924 ± 0.0019 | 0.3573 ± 0.0072 | 4.0784 ± 0.0461 | 0.5966 ± 0.0028 |

GRU相对MLP的velocity、attitude、body-rate均值改善分别为34.10%、25.66%、20.73%。计算先flight内RMSE，再flight等权，再3seed均值及sample SD(ddof=1)；百分比由聚合误差计算，未平均窗口百分比。

## 4. seed与flight方向

Velocity RMSE [m/s]：500 ms ALL有3/3个seed、17/17架flight支持GRU，0架支持MLP、0架持平。

Sep7改善32.89%，Sep17改善35.31%；正值为GRU更低，负值为MLP更低。

Attitude geodesic RMS [deg]：500 ms ALL有3/3个seed、17/17架flight支持GRU，0架支持MLP、0架持平。

Sep7改善19.83%，Sep17改善30.21%；正值为GRU更低，负值为MLP更低。

Body-rate RMSE [rad/s]：500 ms ALL有3/3个seed、17/17架flight支持GRU，0架支持MLP、0架持平。

Sep7改善16.83%，Sep17改善25.15%；正值为GRU更低，负值为MLP更低。

| cohort | horizon_s | metric | relative_gain_pct | seeds_gru_better | seeds_mlp_better | flights_gru_better | flights_mlp_better | n_flights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.5 | attitude_error_deg | 25.656 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | body_rate_rmse_rad_s | 20.732 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | position_rmse_m | 20.057 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | velocity_rmse_m_s | 34.102 | 3 | 0 | 17 | 0 | 17 |
| Sep17 | 0.5 | attitude_error_deg | 30.207 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | body_rate_rmse_rad_s | 25.151 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | position_rmse_m | 32.262 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | velocity_rmse_m_s | 35.31 | 3 | 0 | 8 | 0 | 8 |
| Sep7 | 0.5 | attitude_error_deg | 19.825 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | body_rate_rmse_rad_s | 16.834 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | position_rmse_m | 12.791 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | velocity_rmse_m_s | 32.888 | 3 | 0 | 9 | 0 | 9 |

flight方向先对每架flight的三个seed误差取均值，再计算MLP−GRU。正值支持GRU。匹配seed编号只表示配对训练重复，不表示不同架构有相同初始权重或随机轨迹。

500 ms反向或持平flight：

无。

## 5. Sep7/Sep17及100 ms至1 s

Velocity RMSE [m/s]相对改善从100 ms的72.70%变化至1 s的24.84%；四个时域并非单调增加，不将其概括为时域越长优势必然越大。

Attitude geodesic RMS [deg]相对改善从100 ms的30.90%变化至1 s的21.57%；四个时域并非单调增加，不将其概括为时域越长优势必然越大。

Body-rate RMSE [rad/s]相对改善从100 ms的24.09%变化至1 s的19.21%；四个时域并非单调增加，不将其概括为时域越长优势必然越大。

| cohort | horizon_s | metric | relative_gain_pct | seeds_gru_better | seeds_mlp_better | flights_gru_better | flights_mlp_better | n_flights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.1 | attitude_error_deg | 30.895 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.1 | body_rate_rmse_rad_s | 24.092 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.1 | position_rmse_m | 28.502 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.1 | velocity_rmse_m_s | 72.697 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | attitude_error_deg | 30.719 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | body_rate_rmse_rad_s | 25.022 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | position_rmse_m | 28.335 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | velocity_rmse_m_s | 54.43 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | attitude_error_deg | 25.656 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | body_rate_rmse_rad_s | 20.732 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | position_rmse_m | 20.057 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | velocity_rmse_m_s | 34.102 | 3 | 0 | 17 | 0 | 17 |
| ALL | 1 | attitude_error_deg | 21.568 | 3 | 0 | 17 | 0 | 17 |
| ALL | 1 | body_rate_rmse_rad_s | 19.21 | 3 | 0 | 17 | 0 | 17 |
| ALL | 1 | position_rmse_m | 16.473 | 3 | 0 | 16 | 1 | 17 |
| ALL | 1 | velocity_rmse_m_s | 24.839 | 3 | 0 | 17 | 0 | 17 |
| Sep17 | 0.1 | attitude_error_deg | 34.535 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.1 | body_rate_rmse_rad_s | 27.925 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.1 | position_rmse_m | 48.957 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.1 | velocity_rmse_m_s | 71.98 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | attitude_error_deg | 35.232 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | body_rate_rmse_rad_s | 29.898 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | position_rmse_m | 46.011 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | velocity_rmse_m_s | 50.744 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | attitude_error_deg | 30.207 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | body_rate_rmse_rad_s | 25.151 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | position_rmse_m | 32.262 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | velocity_rmse_m_s | 35.31 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | attitude_error_deg | 24.859 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | body_rate_rmse_rad_s | 20.983 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | position_rmse_m | 25.928 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | velocity_rmse_m_s | 27.986 | 3 | 0 | 8 | 0 | 8 |
| Sep7 | 0.1 | attitude_error_deg | 26.971 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.1 | body_rate_rmse_rad_s | 20.564 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.1 | position_rmse_m | 17.474 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.1 | velocity_rmse_m_s | 73.372 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | attitude_error_deg | 25.323 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | body_rate_rmse_rad_s | 20.569 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | position_rmse_m | 18.462 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | velocity_rmse_m_s | 57.989 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | attitude_error_deg | 19.825 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | body_rate_rmse_rad_s | 16.834 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | position_rmse_m | 12.791 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | velocity_rmse_m_s | 32.888 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 1 | attitude_error_deg | 17.458 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 1 | body_rate_rmse_rad_s | 17.584 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 1 | position_rmse_m | 9.8432 | 3 | 0 | 8 | 1 | 9 |
| Sep7 | 1 | velocity_rmse_m_s | 21.406 | 3 | 0 | 9 | 0 | 9 |

完整绝对误差与跨seed离散程度见下表；不能将总体改善替代每个cohort、时域和flight的具体方向。

| Cohort | Horizon [s] | Model | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.1 | B0 Kinematic | 0.0587 | 0.8196 | 4.1554 | 1.2121 |
| ALL | 0.1 | B1 MLP | 0.0494 ± 1.36e-05 | 0.5308 ± 0.0013 | 2.0366 ± 0.0095 | 0.7184 ± 0.0031 |
| ALL | 0.1 | B2 Standard GRU / H26 | 0.0353 ± 0.0002 | 0.1449 ± 0.0046 | 1.4074 ± 0.0206 | 0.5453 ± 0.0026 |
| ALL | 0.2 | B0 Kinematic | 0.1378 | 0.7480 | 9.3133 | 1.2005 |
| ALL | 0.2 | B1 MLP | 0.0982 ± 0.0002 | 0.4289 ± 0.0037 | 3.1704 ± 0.0144 | 0.7553 ± 0.0066 |
| ALL | 0.2 | B2 Standard GRU / H26 | 0.0704 ± 0.0005 | 0.1954 ± 0.0038 | 2.1965 ± 0.0385 | 0.5663 ± 0.0028 |
| ALL | 0.5 | B0 Kinematic | 0.4182 | 1.1707 | 23.2455 | 1.0660 |
| ALL | 0.5 | B1 MLP | 0.2407 ± 0.0019 | 0.5422 ± 0.0097 | 5.4859 ± 0.0451 | 0.7526 ± 0.0015 |
| ALL | 0.5 | B2 Standard GRU / H26 | 0.1924 ± 0.0019 | 0.3573 ± 0.0072 | 4.0784 ± 0.0461 | 0.5966 ± 0.0028 |
| ALL | 1 | B0 Kinematic | 1.2249 | 2.0733 | 46.7779 | 1.1262 |
| ALL | 1 | B1 MLP | 0.5675 ± 0.0071 | 0.8618 ± 0.0143 | 8.5328 ± 0.1523 | 0.7687 ± 0.0048 |
| ALL | 1 | B2 Standard GRU / H26 | 0.4740 ± 0.0063 | 0.6477 ± 0.0070 | 6.6925 ± 0.1321 | 0.6211 ± 0.0108 |
| Sep17 | 0.1 | B0 Kinematic | 0.0473 | 0.8067 | 4.0547 | 1.2504 |
| Sep17 | 0.1 | B1 MLP | 0.0368 ± 0.0002 | 0.5470 ± 0.0018 | 2.2452 ± 0.0272 | 0.7317 ± 0.0070 |
| Sep17 | 0.1 | B2 Standard GRU / H26 | 0.0188 ± 0.0003 | 0.1533 ± 0.0048 | 1.4698 ± 0.0125 | 0.5274 ± 0.0027 |
| Sep17 | 0.2 | B0 Kinematic | 0.1187 | 0.7733 | 9.3406 | 1.1662 |
| Sep17 | 0.2 | B1 MLP | 0.0748 ± 0.0002 | 0.4477 ± 0.0059 | 3.6687 ± 0.0360 | 0.7661 ± 0.0060 |
| Sep17 | 0.2 | B2 Standard GRU / H26 | 0.0404 ± 0.0008 | 0.2205 ± 0.0009 | 2.3762 ± 0.0567 | 0.5370 ± 0.0007 |
| Sep17 | 0.5 | B0 Kinematic | 0.3985 | 1.2780 | 23.4183 | 1.0921 |
| Sep17 | 0.5 | B1 MLP | 0.1909 ± 0.0020 | 0.5775 ± 0.0067 | 6.5473 ± 0.0512 | 0.7494 ± 0.0025 |
| Sep17 | 0.5 | B2 Standard GRU / H26 | 0.1293 ± 0.0040 | 0.3736 ± 0.0143 | 4.5695 ± 0.1054 | 0.5609 ± 0.0046 |
| Sep17 | 1 | B0 Kinematic | 1.2449 | 2.2176 | 47.3359 | 1.1321 |
| Sep17 | 1 | B1 MLP | 0.4970 ± 0.0070 | 0.9556 ± 0.0181 | 10.0683 ± 0.1660 | 0.7813 ± 0.0056 |
| Sep17 | 1 | B2 Standard GRU / H26 | 0.3682 ± 0.0141 | 0.6882 ± 0.0119 | 7.5654 ± 0.3508 | 0.6174 ± 0.0161 |
| Sep7 | 0.1 | B0 Kinematic | 0.0689 | 0.8311 | 4.2450 | 1.1780 |
| Sep7 | 0.1 | B1 MLP | 0.0607 ± 0.0001 | 0.5164 ± 0.0016 | 1.8512 ± 0.0063 | 0.7066 ± 0.0016 |
| Sep7 | 0.1 | B2 Standard GRU / H26 | 0.0501 ± 0.0001 | 0.1375 ± 0.0045 | 1.3519 ± 0.0288 | 0.5613 ± 0.0029 |
| Sep7 | 0.2 | B0 Kinematic | 0.1547 | 0.7254 | 9.2891 | 1.2309 |
| Sep7 | 0.2 | B1 MLP | 0.1190 ± 0.0002 | 0.4121 ± 0.0019 | 2.7275 ± 0.0145 | 0.7457 ± 0.0072 |
| Sep7 | 0.2 | B2 Standard GRU / H26 | 0.0970 ± 0.0003 | 0.1731 ± 0.0063 | 2.0368 ± 0.0230 | 0.5923 ± 0.0050 |
| Sep7 | 0.5 | B0 Kinematic | 0.4357 | 1.0754 | 23.0919 | 1.0428 |
| Sep7 | 0.5 | B1 MLP | 0.2849 ± 0.0025 | 0.5108 ± 0.0132 | 4.5424 ± 0.0500 | 0.7554 ± 0.0024 |
| Sep7 | 0.5 | B2 Standard GRU / H26 | 0.2485 ± 0.0008 | 0.3428 ± 0.0010 | 3.6419 ± 0.0567 | 0.6283 ± 0.0026 |
| Sep7 | 1 | B0 Kinematic | 1.2072 | 1.9450 | 46.2818 | 1.1209 |
| Sep7 | 1 | B1 MLP | 0.6301 ± 0.0097 | 0.7785 ± 0.0113 | 7.1679 ± 0.1413 | 0.7576 ± 0.0042 |
| Sep7 | 1 | B2 Standard GRU / H26 | 0.5681 ± 0.0035 | 0.6118 ± 0.0035 | 5.9166 ± 0.0650 | 0.6244 ± 0.0073 |

全部时域的flight例外如下；未删除。配对seed例外数量为0。

| flight_id | cohort | horizon_s | metric | absolute_gain | relative_gain_pct |
| --- | --- | --- | --- | --- | --- |
| 2026.9.7/log_30_2026-9-7-06-58-22.ulg | Sep7 | 1 | position_rmse_m | -0.00055103 | -0.039619 |

MLP速度端点误差从100 ms到200 ms下降，随后升高，不能把其误差描述为随时域单调增长。初始状态、prefix与native时间对齐检查通过；本轮不将这种形状归因为已被识别的物理机制。

## 6. 可采用的Results表述

Under the frozen matched-budget protocol, the Standard GRU/H26 and memoryless MLP were evaluated using three training seeds (17, 23, 42) on the same 2,582 validation origins from 17 flights. At the nominal 500 ms horizon, the GRU reduced the mean velocity, attitude, and body-rate errors by 34.10%, 25.66%, and 20.73%, respectively, relative to the MLP. Errors were first computed within each flight, averaged equally across flights, and then summarized across seeds. Reported dispersions are sample standard deviations across three seeds. These are descriptive validation results for the frozen model families.

## 7. Discussion限制

The MLP and GRU differ in architecture and parameter count (5,703 versus 21,383); this comparison therefore cannot isolate history as the cause of all performance differences. The separate H1–H26 experiment provides the controlled history-length evidence. Matching seed identifiers pairs training repetitions, not initial weights or identical random trajectories. Three seeds offer limited coverage of optimization variability, and overlapping windows are not independent trials. All models replay logged future commands; these validation errors do not establish causal responses to arbitrary actions, independent test generalization, long-horizon simulation validity, or closed-loop control performance. No sealed or reserved test was accessed.

容量和架构同时不同，因此不能用本实验单独证明“历史导致全部改善”，也不能推广为所有GRU普遍优于所有MLP。历史的单因素证据由既有H1→H26同结构消融承担，正式Mixed分类不变。这里不做window独立性显著性检验，不把3seed SD称为置信区间、模型不确定性或独立测试证据。

## 8. 训练与工程核验

| model | seed | status | final_epoch | final_train_loss | training_time_s | checkpoint_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| B1_MLP | 17 | reused | 65 | 0.72186 | 1396.7 | 2e82f6b6a5380883bfc78eaa419c4f1a258a9fd06cba31017f4151610a94bef7 |
| B1_MLP | 23 | complete | 65 | 0.71851 | 1348.8 | c9fca6b6435d8046ed64e7f9154314f7b9e8917dbc2c02720c5f63646787ae57 |
| B1_MLP | 42 | complete | 65 | 0.72172 | 1345 | 0483807dacbae37b849f4e8a14445522d050c2349e0956fb8c9334fe4d396dae |
| B2_StandardGRU | 17 | reused | 65 | 0.34286 | 1816.4 | f9895418b7dd2a2757dde8d740109ab42dae258c593fd5902f0076a3faeb0e26 |
| B2_StandardGRU | 23 | reused | 65 | 0.34216 | 5353.4 | 1cfbb865db55e3fc7c248eff4de95892fa9ee9155ee8e81c0318a9021b7d8b03 |
| B2_StandardGRU | 42 | reused | 65 | 0.34177 | 5334.5 | 98fd6266bdc8926055c7f8b0dc2f37a4333fa92f55f336fb4cba4d3480324f86 |

仅有两次新增正式训练，无失败seed静默重跑。旧checkpoint、结果与源文件hash复核；有限性、future-label poisoning、prefix、四元数、步数和聚合测试见tests.json/sanity_checks.json。旧seed17没有历史final validation loss记录，保留空值；新seed最终validation loss仅供报告。

## 9. 论文表图

主表：paper_table_500ms.csv；补表：paper_table_all_horizons.csv、paired_seed_differences.csv、paired_flight_differences.csv。主图建议velocity/attitude/body-rate随horizon曲线，以及500 ms全部flight配对差值图。position同样保留，不筛选“好看”指标或seed。本轮已补齐B0/MLP/GRU的Overall Prediction Performance验证集主表，独立test尚未执行。

## 10. 下一步建议与停止边界

可进入最终评价方案冻结与独立测试准备：明确主表、统计单位、checkpoint及报告范围，再决定开启独立测试。这里只提出建议，不执行。H26主模型不变、history Mixed不变、future-control结论不变；sealed Sep8/reserved Sep19的数据、预测、结果均未打开。没有启动其他模型、消融、控制或RL实验，没有自动commit/push。
