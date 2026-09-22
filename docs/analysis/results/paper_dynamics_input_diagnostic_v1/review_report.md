# Paper Step5：未来控制输入预测价值与误差演化

## 1. 本实验回答的论文问题

实际未来命令在总体及高控制变化组均降低三项主要动力学预测误差，支持其具有增量预测信息。

问题严格限定为：同一历史、起点和冻结动力学模型下，知道后续实际命令，是否比假设命令维持当前值更有助于预测原始真实飞行？这是已知输入回放的信息价值诊断，不是训练优化或干预辨识。

## 2. 实验设置与覆盖情况

500 ms low/middle/high分别888/1,240/454个origins，全部覆盖17flights。High组包含Sep7的185个、Sep17的269个origins，每flight为8–58个。覆盖广，但某些flight的组内样本较少，等flight加权不会消除其估计波动。

Standard GRU64 / H26，seed17/23/42，last epoch65，原归一化不变。41个train flights/28,293origins仅提供固定控制变化分位阈值；17个validation flights/2,582origins全部保留，Sep7为9flights、Sep17为8flights。Actual完整预测复用；Hold仅从第2条未来命令起改变4通道，history及其末尾u_t、初始状态、phase、dt不变。控制量为归一化分配命令，不是实际舵角或扑频。

| horizon_s | step | train_q25 | train_q75 | grouping_available | n_train_origins |
| --- | --- | --- | --- | --- | --- |
| 0.100000 | 5 | 0.069504 | 0.125937 | True | 28293 |
| 0.200000 | 10 | 0.110089 | 0.193606 | True | 28293 |
| 0.500000 | 25 | 0.173570 | 0.321677 | True | 28293 |
| 1.000000 | 50 | 0.254164 | 0.502741 | True | 28293 |

| cohort | group | n_origins | n_flights | E_median | E_p95 |
| --- | --- | --- | --- | --- | --- |
| ALL | ALL | 2582 | 17 | 0.211485 | 0.530446 |
| Sep7 | ALL | 1481 | 9 | 0.181135 | 0.488635 |
| Sep17 | ALL | 1101 | 8 | 0.241152 | 0.574079 |
| ALL | low | 888 | 17 | 0.130879 | 0.169141 |
| Sep7 | low | 687 | 9 | 0.126867 | 0.169107 |
| Sep17 | low | 201 | 8 | 0.142983 | 0.169706 |
| ALL | middle | 1240 | 17 | 0.230001 | 0.307120 |
| Sep7 | middle | 609 | 9 | 0.221687 | 0.305914 |
| Sep17 | middle | 631 | 8 | 0.235392 | 0.307588 |
| ALL | high | 454 | 17 | 0.420190 | 0.842672 |
| Sep7 | high | 185 | 9 | 0.427928 | 0.821190 |
| Sep17 | high | 269 | 8 | 0.414407 | 0.854780 |

所有误差均为每flight内RMSE→flight等权平均→三seed mean/sampleSD(ddof=1)。某组没有样本的flight不参与该组误差，零计数保存在control_group_per_flight.csv。±seedSD不是预测置信区间或模型不确定性。

## 3. 500 ms主结果

总体velocity/attitude/body-rate的改善分别17.43%/13.24%/7.34%；三个seed全部同方向，先三seed平均后的17个flight也全部同方向。Position为辅助指标，改善5.03%，其中16/17flights同方向。不能把这些方向计数当作基于独立window的显著性检验。

| condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- |
| Actual | 0.19240 ± 0.00186 | 0.35729 ± 0.00716 | 4.07840 ± 0.04607 | 0.59658 ± 0.00281 |
| Hold | 0.20259 ± 0.00228 | 0.43269 ± 0.00802 | 4.70068 ± 0.06310 | 0.64385 ± 0.00929 |

| cohort | group | metric | actual | hold | absolute_gain | relative_gain_pct | seeds_improved | flights_improved | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | attitude_error_deg | 4.078397 | 4.700676 | 0.622279 | 13.238070 | 3.000000 | 17.000000 | 17.000000 | 2582 |
| ALL | ALL | body_rate_rmse_rad_s | 0.596577 | 0.643854 | 0.047277 | 7.342886 | 3.000000 | 17.000000 | 17.000000 | 2582 |
| ALL | ALL | position_rmse_m | 0.192395 | 0.202593 | 0.010197 | 5.033428 | 3.000000 | 16.000000 | 17.000000 | 2582 |
| ALL | ALL | velocity_rmse_m_s | 0.357288 | 0.432695 | 0.075407 | 17.427311 | 3.000000 | 17.000000 | 17.000000 | 2582 |

gain=Hold−Actual；正值有利于Actual。相对增益按聚合误差计算，以Hold为分母；分母0时不可定义。每个seed及先三seed平均后的每flight配对差值保存在paired_differences.csv，没有对大量windows做独立t检验。

## 4. 控制变化分组结果

High组velocity/attitude/body-rate改善16.92%/17.58%/7.48%，seed方向均3/3；flight方向分别15/17、17/17、14/17。Low组也有12.21%/9.57%/3.86%的改善，所以价值不只存在于high组。不能声称控制变化越大、所有指标的百分比收益越大：velocity在middle为18.81%，略高于high；body rate也是middle高于high。High组velocity绝对收益0.1035 m/s，高于low的0.0407 m/s。

| group | condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- |
| high | Actual | 0.22688 ± 0.00220 | 0.50810 ± 0.01441 | 5.06294 ± 0.02050 | 0.66879 ± 0.00277 |
| high | Hold | 0.24185 ± 0.00245 | 0.61156 ± 0.00777 | 6.14294 ± 0.00360 | 0.72290 ± 0.00826 |
| low | Actual | 0.18298 ± 0.00320 | 0.29291 ± 0.01103 | 3.40530 ± 0.04863 | 0.55851 ± 0.00483 |
| low | Hold | 0.18998 ± 0.00422 | 0.33363 ± 0.01735 | 3.76577 ± 0.10836 | 0.58094 ± 0.01154 |
| middle | Actual | 0.18354 ± 0.00133 | 0.32167 ± 0.00384 | 3.97694 ± 0.04722 | 0.57555 ± 0.00416 |
| middle | Hold | 0.19231 ± 0.00185 | 0.39621 ± 0.00539 | 4.51025 ± 0.06788 | 0.62876 ± 0.00960 |

| cohort | group | metric | actual | hold | absolute_gain | relative_gain_pct | seeds_improved | flights_improved | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | high | attitude_error_deg | 5.062936 | 6.142940 | 1.080004 | 17.581230 | 3.000000 | 17.000000 | 17.000000 | 454 |
| Sep17 | high | attitude_error_deg | 5.771198 | 7.002653 | 1.231456 | 17.585558 | 3.000000 | 8.000000 | 8.000000 | 269 |
| Sep7 | high | attitude_error_deg | 4.433369 | 5.378751 | 0.945381 | 17.576222 | 3.000000 | 9.000000 | 9.000000 | 185 |
| ALL | high | body_rate_rmse_rad_s | 0.668793 | 0.722897 | 0.054104 | 7.484390 | 3.000000 | 14.000000 | 17.000000 | 454 |
| Sep17 | high | body_rate_rmse_rad_s | 0.626949 | 0.709076 | 0.082128 | 11.582343 | 3.000000 | 7.000000 | 8.000000 | 269 |
| Sep7 | high | body_rate_rmse_rad_s | 0.705987 | 0.735182 | 0.029195 | 3.971111 | 3.000000 | 7.000000 | 9.000000 | 185 |
| ALL | high | velocity_rmse_m_s | 0.508097 | 0.611564 | 0.103467 | 16.918454 | 3.000000 | 15.000000 | 17.000000 | 454 |
| Sep17 | high | velocity_rmse_m_s | 0.506143 | 0.667533 | 0.161390 | 24.177080 | 3.000000 | 8.000000 | 8.000000 | 269 |
| Sep7 | high | velocity_rmse_m_s | 0.509834 | 0.561814 | 0.051980 | 9.252222 | 3.000000 | 7.000000 | 9.000000 | 185 |

E_K是相对于起点命令的标准化变化RMS，不是total variation。各horizon分别用train的25%/75%分位数分组，组别随horizon可能变化。阈值、identity案例在Hold结果产生前冻结，未看误差调阈值。若阈值相等，则该horizon只保留ALL并记录分布。组别使用已实现的未来命令，不能称为提前可知的在线分类器。

## 5. 不同预测时域及完整误差过程

首步预测完全相同。100/200/500/1000 ms的velocity收益分别2.32%/6.19%/17.43%/23.59%；attitude为7.73%/8.26%/13.24%/17.41%；body rate为−0.17%/2.74%/7.34%/22.01%。100 ms body-rate略偏向Hold（Actual0.54533、Hold0.54439 rad/s），已完整保留；不是所有时间点都支持正结果。500 ms区间加权误差也支持Actual，三项改善为11.99%/10.95%/3.65%，但幅度不能与端点指标混用。

实际step25累计时间的中位数0.499094 s、范围0.488664–0.518876 s；step50中位数0.997967 s、范围0.987696–1.027915 s。未来命令的相对信息增益在较长预测时域更明显，与既有history消融“历史上下文相对收益在短时更强”是两个不同的诊断结论，不相互替代。

| step | condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- |
| 5 | Actual | 0.03534 ± 0.00016 | 0.14493 ± 0.00456 | 1.40739 ± 0.02065 | 0.54533 ± 0.00264 |
| 5 | Hold | 0.03538 ± 0.00016 | 0.14837 ± 0.00472 | 1.52534 ± 0.01911 | 0.54439 ± 0.00123 |
| 10 | Actual | 0.07038 ± 0.00050 | 0.19543 ± 0.00377 | 2.19649 ± 0.03851 | 0.56628 ± 0.00285 |
| 10 | Hold | 0.07096 ± 0.00053 | 0.20832 ± 0.00436 | 2.39426 ± 0.02490 | 0.58221 ± 0.00539 |
| 25 | Actual | 0.19240 ± 0.00186 | 0.35729 ± 0.00716 | 4.07840 ± 0.04607 | 0.59658 ± 0.00281 |
| 25 | Hold | 0.20259 ± 0.00228 | 0.43269 ± 0.00802 | 4.70068 ± 0.06310 | 0.64385 ± 0.00929 |
| 50 | Actual | 0.47401 ± 0.00635 | 0.64775 ± 0.00702 | 6.69249 ± 0.13211 | 0.62107 ± 0.01084 |
| 50 | Hold | 0.53543 ± 0.00758 | 0.84772 ± 0.00877 | 8.10351 ± 0.15571 | 0.79634 ± 0.00729 |

1–50步曲线保留原生step，横轴为参与origin累计dt的中位数；原始时间分布另存，既不重采样也不假装各origin共享同一timestamp。分组曲线始终使用固定500ms组别。首步使用相同命令，因此预测一致；后续差异才可能体现future tape的作用。误差端点不必随时间单调增加。

辅助区间轨迹误差公式：每origin的I²=Σ(dt×该步误差²)/Σdt，从第1步到K步，排除t0；flight内对origin的I²等权平均后开方，再flight等权、seed汇总。使用右端点矩形权重。该指标补充整段误差，不替代冻结的四个端点主指标。

| step | condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- |
| 5 | Actual | 0.02417 ± 0.00008 | 0.10725 ± 0.00310 | 1.02725 ± 0.01239 | 0.53943 ± 0.00158 |
| 5 | Hold | 0.02419 ± 0.00008 | 0.10903 ± 0.00321 | 1.08775 ± 0.01222 | 0.54024 ± 0.00148 |
| 10 | Actual | 0.04388 ± 0.00027 | 0.14943 ± 0.00373 | 1.52573 ± 0.02403 | 0.55165 ± 0.00148 |
| 10 | Hold | 0.04411 ± 0.00028 | 0.15538 ± 0.00403 | 1.65427 ± 0.01778 | 0.55640 ± 0.00150 |
| 25 | Actual | 0.11041 ± 0.00091 | 0.23937 ± 0.00434 | 2.72076 ± 0.02962 | 0.56654 ± 0.00066 |
| 25 | Hold | 0.11433 ± 0.00107 | 0.27197 ± 0.00449 | 3.05520 ± 0.03063 | 0.58797 ± 0.00427 |
| 50 | Actual | 0.25253 ± 0.00320 | 0.39847 ± 0.00630 | 4.32718 ± 0.04634 | 0.58477 ± 0.00297 |
| 50 | Hold | 0.27809 ± 0.00375 | 0.50018 ± 0.00652 | 5.12906 ± 0.08253 | 0.65897 ± 0.00628 |

已有history报告显示history相对收益集中于较短horizon；本轮仅引用该冻结结论，不重新运行其他H或修改Mixed分类。

## 6. 动态响应案例

案例结果刻意完整保留：原Step1固定案例在500 ms四项端点误差均由Hold更低；Sep7 high案例四项均Actual更低；Sep17 high案例velocity/attitude/position由Hold更低，body rate由Actual更低。这些图说明总体统计优势不保证单条响应轨迹更准确，不以个案替代17flight配对统计，也不删换不利案例。

采用Step1固定origin，以及Sep7/Sep17各自500ms high组按(log_id,segment_id,start_sample_in_segment)排序的中位origin；均预先固定seed17。缺失cohort案例时记录缺失，不用误差选替代样本。所有存在的案例同时展示Actual/Hold命令、真实运动及两种预测；Euler仅辅助可视化。

| case | window_id | seed | step | metric | actual | hold | absolute_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | position_rmse_m | 0.397782 | 0.391630 | -0.006152 |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | velocity_rmse_m_s | 0.250229 | 0.188781 | -0.061448 |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | attitude_error_deg | 2.649345 | 2.303632 | -0.345714 |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | body_rate_rmse_rad_s | 0.387868 | 0.377270 | -0.010597 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | position_rmse_m | 0.116554 | 0.161476 | 0.044922 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | velocity_rmse_m_s | 0.319508 | 0.548358 | 0.228849 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | attitude_error_deg | 4.927704 | 6.108864 | 1.181159 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | body_rate_rmse_rad_s | 1.386037 | 1.407350 | 0.021313 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | position_rmse_m | 0.084275 | 0.065705 | -0.018570 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | velocity_rmse_m_s | 0.462596 | 0.243905 | -0.218691 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | attitude_error_deg | 5.621190 | 5.295020 | -0.326170 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | body_rate_rmse_rad_s | 0.529446 | 0.590129 | 0.060683 |

## 7. 能够支持的论文结论

两个cohort均支持500 ms三项dynamics的正向总体增益。Sep7与Sep17的velocity增益分别10.79%和23.32%，attitude分别12.18%和14.16%，body rate分别3.39%和11.89%；输入信息价值具有session和state依赖，不应以ALL平均掩盖。

实际未来命令在总体及高控制变化组均降低三项主要动力学预测误差，支持其具有增量预测信息。

具体效应大小、seed方向和flight方向见主表及分组表；不设“必须改善X%”的科学通过阈值。工程核验通过与科学结果是否支持正向信息增益分别报告。

## 8. 不能支持的结论及限制

Actual和Hold都与同一条原始Actual飞行轨迹比较。Hold是信息受限的预测对照，没有对应真实Hold飞行反事实ground truth；因此不能由其误差断言Hold模拟轨迹本身物理上错误。日志处于闭环，控制命令可能携带状态反馈和策略相关信息。不能直接证明任意候选动作的因果响应、实机控制成功、闭环改善或长期RL仿真能力。有限flight/两session及3seeds也限制外推。

## 9. 对应论文章节与可直接采用的表述

Results: Contribution of Future Control Inputs — 使用paper_table_500ms.csv、paper_table_500ms_gains.csv及control_group_500ms图。Results: Prediction Horizon and Dynamic Response — 使用error_evolution_ALL、fixed_groups及全部三个case图。Discussion — 明确known-input replay与因果响应验证的区别。

> At the nominal 500 ms horizon, replaying the logged future commands reduced velocity RMSE from 0.4327 to 0.3573 m/s (17.43%), attitude geodesic RMS error from 4.7007° to 4.0784° (13.24%), and body-rate RMSE from 0.6439 to 0.5966 rad/s (7.34%) compared with holding the origin command. All three seeds favored actual-command replay on these aggregate metrics. After averaging seeds within each flight, the improvement direction was also consistent across all 17 validation flights. In the high-control-change subset (454 origins spanning all 17 flights), the corresponding reductions were 16.92%, 17.58% and 7.48%. These findings support incremental predictive information from future logged commands within the observed flight distribution; they do not imply improvement for every individual trajectory.

> Both conditions were evaluated against the same observed trajectory generated under the logged commands. Hold is therefore an information-limited forecast control, not a counterfactual flight with matched ground truth. Closed-loop feedback and correlated state estimates can contribute to the predictive association. These results do not establish causal accuracy for arbitrary control actions, closed-loop improvement, or suitability for long-horizon simulation/RL. Control groups use realized future commands offline; three-seed SD describes initialization sensitivity, not predictive uncertainty or a confidence interval.

## 10. 下一步建议，但不自动执行

若论文只主张当前日志分布中的已知输入预测价值，本轮证据可直接使用，无需为该限定结论立即新采数据。若要扩展到任意动作响应或控制用途，则需要专门的真实输入激励实验：在可安全执行的匹配工况下验证命令改变后的响应方向、时延和幅值，因为本轮Hold没有反事实实测。是否开展由用户决定，本轮未启动训练、输入激励采集或其他实验。

封存状态：sealed Sep8/reserved Sep19仍未打开。所有输出位于本实验新目录，旧分类与阈值保持原样。本轮不自动commit/push。建议commit：`feat: add frozen future-control information diagnostic`。
