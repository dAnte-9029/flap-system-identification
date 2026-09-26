# 9.6–9.19数据充分性复核

## 结论：先利用已有数据，不宜笼统判为“不够”
**现有数据已支持当前日志分布下的短时预测模型及论文验证；但尚未充分验证持续、候选控制输入的真实响应。** 这是不同用途的证据要求，不是所有数据都缺激励。此前仅凭validation孤立片段较少就优先建议补飞，不够完整；训练集仍有值得提取的响应片段，应先利用。

本轮汇总完整归档与已准入窗口，不训练、不评价模型。新增读取已在Paper Step7打开的Sep8/Sep19缓存，只计算控制覆盖，不读取预测和性能指标，不改变任何划分。此前报告“当时未访问”保持原样，本轮明确记录缓存再访问事实。两日期不能再称未接触测试。

## 数据清单
归档92个ULog文件，SHA去重90份，9.12有两份重复文件。文件名刷新结果见inventory_refresh.json；哈希来自既有清单，本轮不重新解析90份原始日志。58架开发数据为41train/17validation；测试准入为Sep8六架、Sep19二十二架，共86架。另有Sep19一架被冻结质量门槛排除、三份旧排除或未准入日志。无9.9/9.10归档，不能把日期范围写成逐日连续采集。

| filename_date | role | current_status | unique_logs |
| --- | --- | --- | --- |
| 2026-09-06 | not_admitted | existing_exclusion_or_unassigned | 1 |
| 2026-09-06 | train | admitted | 6 |
| 2026-09-07 | not_admitted | existing_exclusion_or_unassigned | 1 |
| 2026-09-07 | validation | admitted | 9 |
| 2026-09-08 | Sep8 | admitted | 6 |
| 2026-09-08 | not_admitted | existing_exclusion_or_unassigned | 1 |
| 2026-09-11 | train | admitted | 3 |
| 2026-09-12 | train | admitted | 8 |
| 2026-09-13 | train | admitted | 6 |
| 2026-09-14 | train | admitted | 4 |
| 2026-09-15 | train | admitted | 6 |
| 2026-09-16 | train | admitted | 5 |
| 2026-09-17 | validation | admitted | 8 |
| 2026-09-18 | train | admitted | 3 |
| 2026-09-19 | Sep19 | admitted | 22 |
| 2026-09-19 | Sep19 | excluded_by_frozen_gate | 1 |

三份未准入为9.6 log_29（旧别名log_15，既有有效时长/比例不合格）、9.8 log_3（同类既有质量排除）、9.7 log_0（既有未准入/未分配，本轮没有原始内容充分性结论）。Sep19被排除原因保留在Step7 test_flight_coverage.csv。未为了增加样本修改质量规则或将测试并入训练。因此本轮是“90份清单对账＋86架准入数据的控制覆盖”，不是宣称90份原始观测链都已经逐字段审完。

训练28293 origins，validation2582，Sep8 871，Sep19 3202。训练与验证的有效时长此前manifest分别5901.60秒和2632.55秒，窗口高度相关，不能当数万独立试验。

## 固定控制变化规则与去重
沿用上轮独立坐标：drive=motor；common=(left+right)/2；differential=(left-right)/2；rudder=第四维。均为归一化软件命令，不是舵角。common/differential scale由原control_std组合，是权重约定，不是另拟合的物理标定。

500ms变化量使用相对起点的RMS/冻结scale。目标通道>train q75，其他三通道≤各自train q25，称近似孤立。再要求abs(mean departure)/RMS≥0.8，称主要向单一方向变化。未修改阈值，全部来源于既有train；heldout未重拟合。

各flight/segment内按时间排序，贪心保留完整50步区间不重叠的候选；数量是描述性去重，不等于独立随机试验。表中flights是筛选后原候选的flight数；去重不会清空有候选的flight。该规则只描述前500ms，**不保证整个1s恒定输入、更不证明稳态响应**。同一片段不等于外生阶跃，闭环反作用、风和未记录变量仍可能混入。

| partition | channel | n_nonoverlap | nonoverlap_positive | nonoverlap_negative | flights | flights_both_signs |
| --- | --- | --- | --- | --- | --- | --- |
| train | drive | 169 | 79 | 90 | 37 | 30 |
| train | common | 68 | 41 | 27 | 29 | 12 |
| train | differential | 59 | 25 | 34 | 25 | 8 |
| train | rudder | 38 | 18 | 20 | 24 | 3 |
| validation | drive | 22 | 10 | 12 | 13 | 5 |
| validation | common | 16 | 5 | 11 | 9 | 2 |
| validation | differential | 8 | 3 | 5 | 4 | 2 |
| validation | rudder | 2 | 2 | 0 | 2 | 0 |
| Sep8 | drive | 5 | 2 | 3 | 3 | 1 |
| Sep8 | common | 2 | 1 | 1 | 2 | 0 |
| Sep8 | differential | 3 | 0 | 3 | 2 | 0 |
| Sep8 | rudder | 3 | 3 | 0 | 2 | 0 |
| Sep19 | drive | 23 | 10 | 13 | 12 | 6 |
| Sep19 | common | 1 | 1 | 0 | 1 | 0 |
| Sep19 | differential | 5 | 1 | 4 | 5 | 0 |
| Sep19 | rudder | 4 | 1 | 3 | 4 | 0 |

## 证据如何解释
1. **短时真实日志预测：有充分开展建模的实用基础。** 已有相同flight划分、三seed、history和实际未来命令诊断，以及已完成的留出日期评价。这里复用其存在与协议事实，不重新读取测试性能来调模型。当前模型失败不能自动归咎数据总量不够。
2. **训练集仍有可用输入变化。** 尤其rudder有38个去重候选，18正/20负，24架flight；仅3架同时含正负候选，提示工况/flight与符号混杂。common68、differential59、drive169个。应先检查这些片段的命令、响应和观测有效性，而不是直接宣称必须补飞。
3. **独立方向验证覆盖薄弱。** validation的rudder仅2个且全正；Sep8三个且全正；Sep19四个、1正3负、无同flight双符号。common在Sep19只有1个正向候选。此结果没有设“至少N个才合格”的事后阈值，只明确哪些交叉比较无法进行。
4. **闭环数据并非天然不可辨识。** 既有conditional_excitation_v1对41train飞行的crossfit检查发现，state-only动作R² common≈0.385、diff≈0.426、rudder≈0.108；1s变化/动作std约0.568、0.851、0.482，不能说尾翼没有变化。加入动作历史后的高R²主要反映连续性，残差不等于外生激励。近似孤立筛选不是MIMO辨识的必要条件，未入选片段也不是废数据。
5. **观测链缺口尚未排除。** 前两轮对17validation+3Sep18的审计确认约100Hz角速度/50Hz状态、异步sample时刻与快速带宽敏感性；不能把该20架结论不加核验推广到全部90份。已有软件命令/PWM核对不等于实际舵角响应和物理符号全部确认。不能从动作幅值或频率覆盖直接推导执行器时间常数。

## 下一步顺序
先离线提取训练集已有的非重叠候选，特别是38个rudder、68个common，按固定identity顺序检查整段输入、初始工况、符号、响应、频率/相位与日志时刻，不以模型误差筛选。需要同时保留耦合输入和响应的自然变异；验证片段只做冻结规则的复核，测试片段不转入候选训练。若同工况双向复核、实际舵面链或持续区间仍缺证据，再补对应实验。

已确认的覆盖缺口足以**开始制定有条件的补充激励草案**，但不支持宣布整批数据不能建模、直接重采所有数据或立刻实飞。草案见excitation_plan_draft.md；它以现有候选复核为先决步骤，优先补rudder双向/持续时间与common跨flight验证，不改变模型或飞控。

## 检查与限制
原registry manifest和两个冻结cache身份核验；8个train/validation isolated计数与上轮精确一致；3项测试覆盖去重边界、输入不修改、统计恒等式；旧输入hash运行后不变。本轮只新增分析产物，没有模型训练/推理、划分变更、原始文件修改或commit/push。
