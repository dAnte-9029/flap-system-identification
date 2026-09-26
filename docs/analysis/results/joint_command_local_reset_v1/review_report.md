# 持续联合命令：连续递推与短时实测重初始化

本轮复用冻结三个seed的全部原始预测和既有local100ms预测，不训练、不新推理、不访问Sep8/Sep19。沿用Stabilized/Mission及上一轮持续同向命令规则，不按误差筛选：主组分别462origin/17flight和597origin/12flight；两模式全部窗口的结果也保留。

## 比较含义
continuous从原t0用真实历史/初始状态出发，之后自主递推。local100ms在各offset0/5/…/45用该时刻真实状态和真实H26历史重新编码，再自主预测5步。共同终点为5/10/…/50，同一实测标签、同一段实际四通道命令与native dt。

500ms终点的local大约从400ms重启；1s终点大约从900ms重启。它获得额外观测且预测年龄更短，**不是公平的主模型排行榜，也不是只替换一个状态量的因果实验**。重初始化同时更新观测状态、历史隐表示及相位/频率参考，不能将误差下降比例解释为“纯积分漂移占比”。每个5步预测内部无teacher forcing，多个local段不拼接成一条自主轨迹。

## 持续组主结果
物理单位分别m/s、deg、rad/s。先flight内RMSE，再flight等权，再seed均值；完整sample SD见summary.csv，正reduction_pct表示局部预测评分较低。

| mode | pattern | step | cohort | metric | continuous | local100ms | reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mission | sustained | 25 | ALL | attitude_deg | 3.9874 | 1.3532 | 66.0630 |
| Mission | sustained | 25 | ALL | body_rate_rad_s | 0.5509 | 0.5120 | 7.0680 |
| Mission | sustained | 25 | ALL | velocity_m_s | 0.3069 | 0.1250 | 59.2602 |
| Mission | sustained | 50 | ALL | attitude_deg | 6.7677 | 1.3276 | 80.3831 |
| Mission | sustained | 50 | ALL | body_rate_rad_s | 0.5887 | 0.5288 | 10.1696 |
| Mission | sustained | 50 | ALL | velocity_m_s | 0.5993 | 0.1207 | 79.8576 |
| Stabilized | sustained | 25 | ALL | attitude_deg | 4.6799 | 1.4292 | 69.4605 |
| Stabilized | sustained | 25 | ALL | body_rate_rad_s | 0.6114 | 0.5700 | 6.7606 |
| Stabilized | sustained | 25 | ALL | velocity_m_s | 0.4492 | 0.1611 | 64.1420 |
| Stabilized | sustained | 50 | ALL | attitude_deg | 7.4081 | 1.4923 | 79.8555 |
| Stabilized | sustained | 50 | ALL | body_rate_rad_s | 0.6242 | 0.5346 | 14.3568 |
| Stabilized | sustained | 50 | ALL | velocity_m_s | 0.7574 | 0.1412 | 81.3592 |

## p/q/r各轴
| mode | pattern | step | cohort | metric | continuous | local100ms | reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mission | sustained | 25 | ALL | p_rad_s | 0.4790 | 0.4512 | 5.8078 |
| Mission | sustained | 25 | ALL | q_rad_s | 0.2206 | 0.1984 | 10.0932 |
| Mission | sustained | 25 | ALL | r_rad_s | 0.1527 | 0.1312 | 14.0920 |
| Mission | sustained | 50 | ALL | p_rad_s | 0.4937 | 0.4706 | 4.6602 |
| Mission | sustained | 50 | ALL | q_rad_s | 0.2650 | 0.1934 | 26.9949 |
| Mission | sustained | 50 | ALL | r_rad_s | 0.1703 | 0.1383 | 18.8145 |
| Stabilized | sustained | 25 | ALL | p_rad_s | 0.5236 | 0.4999 | 4.5351 |
| Stabilized | sustained | 25 | ALL | q_rad_s | 0.2592 | 0.2305 | 11.0853 |
| Stabilized | sustained | 25 | ALL | r_rad_s | 0.1608 | 0.1343 | 16.5018 |
| Stabilized | sustained | 50 | ALL | p_rad_s | 0.5118 | 0.4650 | 9.1323 |
| Stabilized | sustained | 50 | ALL | q_rad_s | 0.2997 | 0.2204 | 26.4520 |
| Stabilized | sustained | 50 | ALL | r_rad_s | 0.1747 | 0.1303 | 25.4379 |

## seed与flight方向
flight方向先对每架flight的三seed误差平均，再比较，不能解释为所有window都改善。

| mode | pattern | step | cohort | metric | unit | n | local_better | local_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mission | sustained | 25 | ALL | attitude_deg | seed | 3 | 3 | 0 |
| Mission | sustained | 25 | ALL | body_rate_rad_s | seed | 3 | 3 | 0 |
| Mission | sustained | 25 | ALL | velocity_m_s | seed | 3 | 3 | 0 |
| Mission | sustained | 50 | ALL | attitude_deg | seed | 3 | 3 | 0 |
| Mission | sustained | 50 | ALL | body_rate_rad_s | seed | 3 | 3 | 0 |
| Mission | sustained | 50 | ALL | velocity_m_s | seed | 3 | 3 | 0 |
| Stabilized | sustained | 25 | ALL | attitude_deg | seed | 3 | 3 | 0 |
| Stabilized | sustained | 25 | ALL | body_rate_rad_s | seed | 3 | 3 | 0 |
| Stabilized | sustained | 25 | ALL | velocity_m_s | seed | 3 | 3 | 0 |
| Stabilized | sustained | 50 | ALL | attitude_deg | seed | 3 | 3 | 0 |
| Stabilized | sustained | 50 | ALL | body_rate_rad_s | seed | 3 | 3 | 0 |
| Stabilized | sustained | 50 | ALL | velocity_m_s | seed | 3 | 3 | 0 |
| Mission | sustained | 25 | ALL | attitude_deg | flight_three_seed_mean | 12 | 12 | 0 |
| Mission | sustained | 25 | ALL | body_rate_rad_s | flight_three_seed_mean | 12 | 11 | 1 |
| Mission | sustained | 25 | ALL | velocity_m_s | flight_three_seed_mean | 12 | 12 | 0 |
| Mission | sustained | 50 | ALL | attitude_deg | flight_three_seed_mean | 12 | 12 | 0 |
| Mission | sustained | 50 | ALL | body_rate_rad_s | flight_three_seed_mean | 12 | 11 | 1 |
| Mission | sustained | 50 | ALL | velocity_m_s | flight_three_seed_mean | 12 | 12 | 0 |
| Stabilized | sustained | 25 | ALL | attitude_deg | flight_three_seed_mean | 17 | 17 | 0 |
| Stabilized | sustained | 25 | ALL | body_rate_rad_s | flight_three_seed_mean | 17 | 15 | 2 |
| Stabilized | sustained | 25 | ALL | velocity_m_s | flight_three_seed_mean | 17 | 17 | 0 |
| Stabilized | sustained | 50 | ALL | attitude_deg | flight_three_seed_mean | 17 | 17 | 0 |
| Stabilized | sustained | 50 | ALL | body_rate_rad_s | flight_three_seed_mean | 17 | 17 | 0 |
| Stabilized | sustained | 50 | ALL | velocity_m_s | flight_three_seed_mean | 17 | 17 | 0 |

## 实际发现与下一步
两模式持续组在500ms同终点，速度误差下降59.26%（Mission）/64.14%（Stabilized），姿态下降66.06%/69.46%，body-rate仅下降7.07%/6.76%。三seed均同方向。速度和姿态在12/12 Mission及17/17 Stabilized flights（三seed平均）均下降；body-rate分别11/12与15/17，保留不改善的flight。

到1s终点，速度和姿态下降约80%，角速度下降10.17%/14.36%。p仍是主要局部残差：500ms同终点的local p RMSE为0.4512/0.4999rad/s，较连续仅低5.81%/4.54%；1s也仍有0.4706/0.4650rad/s。因此当前证据不支持“只修长时积分漂移就能解决全部操纵响应”。

下一步优先定位既有100ms预测的角速度误差，尤其p的平均偏差与波形/幅度误差，并与此前测量带宽敏感性结果一起解释；不直接调rudder、actuator tau或控制器。速度/姿态较大的恢复与近期观测/较短递推有益一致，但本轮尚未分离哪个内部状态造成累积偏差。继续沿用联合命令及两种模式，不回到孤立rudder筛选。

## 解释限制
速度/姿态若随着重初始化显著恢复，说明引入近期真实观测和缩短预测长度有价值，与自由递推累积偏差相容；不能由此确定是积分器、GRU隐藏状态或控制映射哪一个根因。角速度若局部预测仍有较大残差，则仅靠周期重置并未解决其快速/局部波形问题，也不能自动归结为传感器噪声。

局部误差是100ms预测条件下的误差，不是不可消除噪声下限。全部数据是已开放validation，任何新设计都属于开发，不能称独立确认。没有修改当前H26/K50主模型，不开展闭环试验。

## 核验
三个测试覆盖终点索引、配对RMSE、不配对数组拒绝；逐offset核验全部local父origin/标签/controls/dt精确一致，offset0与原连续预测前5步数值一致，所有预测有限，输入hash前后不变。图横轴为原生step，实际endpoint和local年龄中位数保存在per_flight.csv，不假设严格50Hz。
