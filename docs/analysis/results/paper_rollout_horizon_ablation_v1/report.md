# Training Rollout Horizon Ablation — 开发阶段验证结果

## 1. 研究定位与数据访问
本轮在已知旧独立测试结果后开展候选模型开发。prior_heldout_results_known=true；heldout_data_accessed_this_run=false。本轮仅显式加载原train/validation数据与K50 validation预测；未调用独立测试入口，也未读取Sep8/Sep19数据、预测或指标。不能称为新独立测试。H26/K50保持正式冻结主模型，旧history Mixed及future-control结论不修改。

## 2. 冻结比较
CausalHistoryTrajectoryModel / GRU64 / H26 / 21,383参数；同一41 flights、28,293 train origins及17 flights、2,582 validation origins（Sep7/Sep17）。使用原50步数据对象、历史与冻结normalization，仅新模型训练前向和监督取25步。K50三个seed只读复用；K25三个seed17/23/42均随机初始化，未从K50微调。

40 epochs AdamW lr3e-4 +25 epochs重建AdamW lr5e-4；batch256、weight decay1e-5、clip5，每epoch111更新，共7,215。base/continuation seed为s/s+12；last epoch65，无validation选模。相同optimizer预算不等于相同转移数、监督状态数、FLOPs或耗时，不据历史耗时计算严格加速率。

## 3. 损失范围及共同目标
完整公式见loss_definition.md。原位置/速度/姿态/角速度/相位/频率损失均取1..K平均，continuation额外0.2 frequency MSE同样取1..K；两步增量仍为t0到t2，原scales/weights不变。无teacher forcing、额外detach或第25步重置。各K原训练loss不可直接排名；common_validation_objectives.csv在相同validation上分别列L25与L50（含最终频率项），仅作辅助。

## 4. 500 ms ALL主结果
误差单位依次为m、m/s、deg、rad/s。先flight内vector RMSE/姿态geodesic RMS，再flight等权，再三seed mean±sample SD（ddof=1）。

| model | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- |
| K25 | 0.1978 ± 0.0018 | 0.3705 ± 0.0076 | 4.2845 ± 0.0372 | 0.5747 ± 0.0009 |
| K50 | 0.1924 ± 0.0019 | 0.3573 ± 0.0072 | 4.0784 ± 0.0461 | 0.5966 ± 0.0028 |

K25相对K50动力学改善率：attitude_error_deg: -5.05%; body_rate_rmse_rad_s: +3.67%; velocity_rmse_m_s: -3.70%。正值表示K25更好。

## 5. 1 s扩展结果
所有K25均连续自主递推50步、返回含起点51状态，未读取第25步真值。

| model | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- |
| K25 | 0.5062 ± 0.0101 | 0.7168 ± 0.0161 | 7.2818 ± 0.1768 | 0.6328 ± 0.0111 |
| K50 | 0.4740 ± 0.0063 | 0.6477 ± 0.0070 | 6.6925 ± 0.1321 | 0.6211 ± 0.0108 |

动力学改善率：attitude_error_deg: -8.81%; body_rate_rmse_rad_s: -1.88%; velocity_rmse_m_s: -10.67%。完整绝对差值与相对差值见relative_improvement.csv；负改善率为退化，不省略。

## 6. Seed、flight、cohort与短时域
下表为三个seed均值下的物理误差改善率，逐seed差值见paired_seed_differences.csv；flight方向先取三seed误差均值，不代表全部窗口改善。

| cohort | horizon_s | metric | K25 | K50 | absolute_gain | relative_gain_pct |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.1000 | attitude_error_deg | 1.3480 | 1.4074 | 0.0594 | 4.2177 |
| ALL | 0.1000 | body_rate_rmse_rad_s | 0.5190 | 0.5453 | 0.0263 | 4.8242 |
| ALL | 0.1000 | position_rmse_m | 0.0353 | 0.0353 | -0.0000 | -0.0216 |
| ALL | 0.1000 | velocity_rmse_m_s | 0.1350 | 0.1449 | 0.0099 | 6.8580 |
| ALL | 0.2000 | attitude_error_deg | 2.1557 | 2.1965 | 0.0408 | 1.8557 |
| ALL | 0.2000 | body_rate_rmse_rad_s | 0.5357 | 0.5663 | 0.0306 | 5.4008 |
| ALL | 0.2000 | position_rmse_m | 0.0709 | 0.0704 | -0.0005 | -0.6867 |
| ALL | 0.2000 | velocity_rmse_m_s | 0.1856 | 0.1954 | 0.0098 | 5.0325 |
| ALL | 0.5000 | attitude_error_deg | 4.2845 | 4.0784 | -0.2061 | -5.0527 |
| ALL | 0.5000 | body_rate_rmse_rad_s | 0.5747 | 0.5966 | 0.0219 | 3.6669 |
| ALL | 0.5000 | position_rmse_m | 0.1978 | 0.1924 | -0.0054 | -2.8144 |
| ALL | 0.5000 | velocity_rmse_m_s | 0.3705 | 0.3573 | -0.0132 | -3.6959 |
| ALL | 1.0000 | attitude_error_deg | 7.2818 | 6.6925 | -0.5893 | -8.8055 |
| ALL | 1.0000 | body_rate_rmse_rad_s | 0.6328 | 0.6211 | -0.0117 | -1.8840 |
| ALL | 1.0000 | position_rmse_m | 0.5062 | 0.4740 | -0.0322 | -6.7947 |
| ALL | 1.0000 | velocity_rmse_m_s | 0.7168 | 0.6477 | -0.0691 | -10.6657 |
| Sep17 | 0.1000 | attitude_error_deg | 1.4191 | 1.4698 | 0.0507 | 3.4504 |
| Sep17 | 0.1000 | body_rate_rmse_rad_s | 0.5044 | 0.5274 | 0.0230 | 4.3586 |
| Sep17 | 0.1000 | position_rmse_m | 0.0186 | 0.0188 | 0.0002 | 0.9286 |
| Sep17 | 0.1000 | velocity_rmse_m_s | 0.1400 | 0.1533 | 0.0133 | 8.6816 |
| Sep17 | 0.2000 | attitude_error_deg | 2.4079 | 2.3762 | -0.0317 | -1.3347 |
| Sep17 | 0.2000 | body_rate_rmse_rad_s | 0.5159 | 0.5370 | 0.0211 | 3.9362 |
| Sep17 | 0.2000 | position_rmse_m | 0.0406 | 0.0404 | -0.0002 | -0.4372 |
| Sep17 | 0.2000 | velocity_rmse_m_s | 0.2108 | 0.2205 | 0.0098 | 4.4262 |
| Sep17 | 0.5000 | attitude_error_deg | 4.8626 | 4.5695 | -0.2931 | -6.4142 |
| Sep17 | 0.5000 | body_rate_rmse_rad_s | 0.5505 | 0.5609 | 0.0104 | 1.8552 |
| Sep17 | 0.5000 | position_rmse_m | 0.1358 | 0.1293 | -0.0065 | -5.0208 |
| Sep17 | 0.5000 | velocity_rmse_m_s | 0.4009 | 0.3736 | -0.0273 | -7.2970 |
| Sep17 | 1.0000 | attitude_error_deg | 8.2881 | 7.5654 | -0.7227 | -9.5522 |
| Sep17 | 1.0000 | body_rate_rmse_rad_s | 0.6404 | 0.6174 | -0.0231 | -3.7338 |
| Sep17 | 1.0000 | position_rmse_m | 0.4127 | 0.3682 | -0.0445 | -12.0832 |
| Sep17 | 1.0000 | velocity_rmse_m_s | 0.7782 | 0.6882 | -0.0900 | -13.0779 |
| Sep7 | 0.1000 | attitude_error_deg | 1.2848 | 1.3519 | 0.0670 | 4.9593 |
| Sep7 | 0.1000 | body_rate_rmse_rad_s | 0.5320 | 0.5613 | 0.0293 | 5.2131 |
| Sep7 | 0.1000 | position_rmse_m | 0.0502 | 0.0501 | -0.0002 | -0.3384 |
| Sep7 | 0.1000 | velocity_rmse_m_s | 0.1306 | 0.1375 | 0.0069 | 5.0510 |
| Sep7 | 0.2000 | attitude_error_deg | 1.9316 | 2.0368 | 0.1052 | 5.1640 |
| Sep7 | 0.2000 | body_rate_rmse_rad_s | 0.5533 | 0.5923 | 0.0390 | 6.5813 |
| Sep7 | 0.2000 | position_rmse_m | 0.0978 | 0.0970 | -0.0008 | -0.7790 |
| Sep7 | 0.2000 | velocity_rmse_m_s | 0.1632 | 0.1731 | 0.0099 | 5.7189 |
| Sep7 | 0.5000 | attitude_error_deg | 3.7706 | 3.6419 | -0.1287 | -3.5342 |
| Sep7 | 0.5000 | body_rate_rmse_rad_s | 0.5962 | 0.6283 | 0.0321 | 5.1047 |
| Sep7 | 0.5000 | position_rmse_m | 0.2530 | 0.2485 | -0.0045 | -1.7941 |
| Sep7 | 0.5000 | velocity_rmse_m_s | 0.3435 | 0.3428 | -0.0007 | -0.2071 |
| Sep7 | 1.0000 | attitude_error_deg | 6.3873 | 5.9166 | -0.4708 | -7.9567 |
| Sep7 | 1.0000 | body_rate_rmse_rad_s | 0.6260 | 0.6244 | -0.0016 | -0.2582 |
| Sep7 | 1.0000 | position_rmse_m | 0.5894 | 0.5681 | -0.0213 | -3.7481 |
| Sep7 | 1.0000 | velocity_rmse_m_s | 0.6623 | 0.6118 | -0.0505 | -8.2539 |

| horizon_s | cohort | metric | seed_wins | seed_n | flight_wins | flight_n |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | ALL | position_rmse_m | 1 | 3 | 8 | 17 |
| 0.1000 | ALL | velocity_rmse_m_s | 3 | 3 | 16 | 17 |
| 0.1000 | ALL | attitude_error_deg | 3 | 3 | 16 | 17 |
| 0.1000 | ALL | body_rate_rmse_rad_s | 3 | 3 | 17 | 17 |
| 0.1000 | Sep7 | position_rmse_m | 0 | 3 | 2 | 9 |
| 0.1000 | Sep7 | velocity_rmse_m_s | 3 | 3 | 8 | 9 |
| 0.1000 | Sep7 | attitude_error_deg | 3 | 3 | 8 | 9 |
| 0.1000 | Sep7 | body_rate_rmse_rad_s | 3 | 3 | 9 | 9 |
| 0.1000 | Sep17 | position_rmse_m | 2 | 3 | 6 | 8 |
| 0.1000 | Sep17 | velocity_rmse_m_s | 3 | 3 | 8 | 8 |
| 0.1000 | Sep17 | attitude_error_deg | 3 | 3 | 8 | 8 |
| 0.1000 | Sep17 | body_rate_rmse_rad_s | 3 | 3 | 8 | 8 |
| 0.2000 | ALL | position_rmse_m | 1 | 3 | 6 | 17 |
| 0.2000 | ALL | velocity_rmse_m_s | 3 | 3 | 14 | 17 |
| 0.2000 | ALL | attitude_error_deg | 3 | 3 | 11 | 17 |
| 0.2000 | ALL | body_rate_rmse_rad_s | 3 | 3 | 16 | 17 |
| 0.2000 | Sep7 | position_rmse_m | 0 | 3 | 2 | 9 |
| 0.2000 | Sep7 | velocity_rmse_m_s | 3 | 3 | 7 | 9 |
| 0.2000 | Sep7 | attitude_error_deg | 3 | 3 | 8 | 9 |
| 0.2000 | Sep7 | body_rate_rmse_rad_s | 3 | 3 | 8 | 9 |
| 0.2000 | Sep17 | position_rmse_m | 1 | 3 | 4 | 8 |
| 0.2000 | Sep17 | velocity_rmse_m_s | 3 | 3 | 7 | 8 |
| 0.2000 | Sep17 | attitude_error_deg | 0 | 3 | 3 | 8 |
| 0.2000 | Sep17 | body_rate_rmse_rad_s | 3 | 3 | 8 | 8 |
| 0.5000 | ALL | position_rmse_m | 0 | 3 | 3 | 17 |
| 0.5000 | ALL | velocity_rmse_m_s | 0 | 3 | 3 | 17 |
| 0.5000 | ALL | attitude_error_deg | 0 | 3 | 0 | 17 |
| 0.5000 | ALL | body_rate_rmse_rad_s | 3 | 3 | 16 | 17 |
| 0.5000 | Sep7 | position_rmse_m | 0 | 3 | 3 | 9 |
| 0.5000 | Sep7 | velocity_rmse_m_s | 2 | 3 | 3 | 9 |
| 0.5000 | Sep7 | attitude_error_deg | 0 | 3 | 0 | 9 |
| 0.5000 | Sep7 | body_rate_rmse_rad_s | 3 | 3 | 9 | 9 |
| 0.5000 | Sep17 | position_rmse_m | 0 | 3 | 0 | 8 |
| 0.5000 | Sep17 | velocity_rmse_m_s | 0 | 3 | 0 | 8 |
| 0.5000 | Sep17 | attitude_error_deg | 0 | 3 | 0 | 8 |
| 0.5000 | Sep17 | body_rate_rmse_rad_s | 3 | 3 | 7 | 8 |
| 1.0000 | ALL | position_rmse_m | 0 | 3 | 2 | 17 |
| 1.0000 | ALL | velocity_rmse_m_s | 0 | 3 | 0 | 17 |
| 1.0000 | ALL | attitude_error_deg | 0 | 3 | 0 | 17 |
| 1.0000 | ALL | body_rate_rmse_rad_s | 0 | 3 | 3 | 17 |
| 1.0000 | Sep7 | position_rmse_m | 0 | 3 | 2 | 9 |
| 1.0000 | Sep7 | velocity_rmse_m_s | 0 | 3 | 0 | 9 |
| 1.0000 | Sep7 | attitude_error_deg | 0 | 3 | 0 | 9 |
| 1.0000 | Sep7 | body_rate_rmse_rad_s | 1 | 3 | 3 | 9 |
| 1.0000 | Sep17 | position_rmse_m | 0 | 3 | 0 | 8 |
| 1.0000 | Sep17 | velocity_rmse_m_s | 0 | 3 | 0 | 8 |
| 1.0000 | Sep17 | attitude_error_deg | 0 | 3 | 0 | 8 |
| 1.0000 | Sep17 | body_rate_rmse_rad_s | 0 | 3 | 0 | 8 |

不根据某个seed或cohort优势更换模型。不把三个seed SD视为置信区间，也不宣称统计显著；小差异需结合方向不一致和跨seed波动解释。

## 7. 可用于Results的表述
With architecture, history, normalization, training origins and optimizer-update budget fixed, we compared training unrolls of 25 and 50 native steps using three seeds. At the nominal 500 ms validation horizon, the relative error reductions of K25 were attitude_error_deg: -5.05%; body_rate_rmse_rad_s: +3.67%; velocity_rmse_m_s: -3.70%. At 1 s, the corresponding reductions were attitude_error_deg: -8.81%; body_rate_rmse_rad_s: -1.88%; velocity_rmse_m_s: -10.67%. Both candidates were evaluated using uninterrupted 50-step autonomous prediction. These results describe training-horizon sensitivity under the current development protocol rather than a universally optimal unroll length.

## 8. Discussion与限制
不同K的原始训练目标覆盖范围不同，最终训练loss不能作为同口径优劣证据。固定更新数不意味着固定计算量；历史K50训练环境与本轮耗时不能严格比较。任何500 ms收益均需同时衡量1 s代价、cohort和flight反例；三seed为描述性重复。此次实验发生在已知Sep8/Sep19结果之后，不能据validation优势宣称新独立测试验证、长时稳定、闭环控制或RL改进；后续确认候选需优先使用预先保留的新飞行数据。

## 9. 工程验证与下一步
测试记录见tests.json与sanity_checks.json，训练记录及checkpoint hashes见training_runs.csv，输入范围和注册见protocol.json。无自动重跑、删seed或筛窗口。建议先审阅本表中的时域折中；结构解耦实验如获后续授权，应另行冻结单独研究问题、预算和新数据确认边界，不把本轮validation结果用于声称结构改进已获独立验证。本轮不执行下一实验，不切换主模型。

## 10. 完成后结果解读（2026-09-26核验）
本轮不支持将K25作为500 ms综合性能改进：速度与姿态分别退化3.70%和5.05%，角速度改善3.67%。500 ms三个配对seed均支持上述各指标方向；按flight三seed均值，K25速度仅3/17更好、姿态0/17更好、角速度16/17更好。角速度唯一反例是Sep17 log_13_06-52-30，差值仅+0.000125 rad/s，应保留但不夸大。

Sep7的500 ms速度差异仅-0.21%改善率，且2/3 seed方向反而支持K25，不应声称该cohort存在稳定速度劣势；Sep17速度退化7.30%，三个seed及8/8 flights均支持K50。姿态在Sep7/Sep17分别退化3.53%/6.41%，角速度分别改善5.10%/1.86%。

100 ms三项动力学误差平均改善6.86%（速度）、4.22%（姿态）、4.82%（角速度）；200 ms对应5.03%、1.86%、5.40%。但200 ms姿态在Sep17退化1.33%，不能将ALL均值改善写成两个cohort全面改善。

1 s三项均退化：速度+0.06909 m/s（10.67%），姿态+0.58930 deg（8.81%），角速度+0.01170 rad/s（1.88%）。三个seed的ALL方向一致；flight均值下速度/姿态17/17支持K50，角速度14/17支持K50。Sep7的1 s角速度差异仅0.26%，不能夸大这一小差异。

因此，这不是“500 ms三项一起改善、1 s退化”，而是100–200 ms平均动力学收益及500 ms角速度局部收益，对应500 ms速度/姿态及1 s表现的代价。继续保留H26/K50更符合当前500 ms主目标，不建议仅凭本轮结果优先推进K25或切换主模型。

共同验证目标三seed均值：K25/K50的L25分别为0.227508/0.246912，L50为0.381655/0.379568。这说明更低的区间平均L25并不保证500 ms端点速度和姿态更低；损失混合不同状态、相位与频率，且其窗口等权口径与主指标flight等权不同，不能替代物理单位结果。

工程状态：三个K25各完成65 epochs/7,215 updates；独立复算1,632个per-flight端点指标单元通过，所有显式登记输入与输出hash匹配。复用未变化代码的22项通过测试。训练日志沿用原helper，逐epoch保存total、original_loss聚合项、两项增量、梯度范数及峰值显存；original_loss内部六项及额外频率MSE没有逐epoch单独列出，这是日志粒度限制，不能宣称保存了每个内部子项的独立曲线。未为补日志重训。

建议下一项结构解耦研究若获授权，继续以K50为冻结对照，单独预注册结构变量和预算，不将K25训练范围与结构改变同时混入比较；候选确认优先采用预先保留的新飞行数据。本轮不启动该实验。
