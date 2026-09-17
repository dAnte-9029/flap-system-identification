# Step 3 训练前审计与预注册

审计对象：Main V2 原始 1482b87 + Step 2 stateful adapter，旧权重不覆盖。
指定报告及summary/CSV已读取，train/validation样本和旧模型/模拟器源码SHA256与Step 2 manifest逐项一致。

## 原 objective

`training/trajectory_main_v1.py::trajectory_rollout_loss`，对 k=1..K 全部预测步等权平均，排除t0：

- position：mean ||p_hat-p||²，单位 m²；隐含尺度1 m。
- velocity：mean ||(v_hat-v)/2||²，NED，尺度2 m/s。
- attitude：mean 4(1-(q_hat·q)²)/0.35²；先分别归一化四元数、符号不敏感；小角近似是姿态角/0.35 rad 的平方。
- body rates：mean ||(omega_hat-omega)/2||²；尺度2 rad/s，三轴权重均1。
- phase：0.1 mean [2-2cos(phi_hat-phi)]，circular chord loss。
- frequency：0.1 mean [(f_hat-f)/3]²。

Main V2 residual阶段额外：0.20 mean (f_hat-f)²（**不除3**），因此总frequency系数为0.20+0.1/9；再加1e-3 mean(control_residual²) +1e-2 sum(tail_gate)。AdamW weight_decay=1e-5为优化器项，不是上述显式loss。

没有单独加权的one-step objective，没有显式 linear/angular acceleration supervision，也没有std、PSD或delta loss。one-step误差仅作为前缀内第一个状态误差出现。导数的训练集均值/标准差用于网络输出尺度，不等于导数监督。

默认实现K=50个原生转移；nominal 50Hz所以约1秒，不是每步强制0.02。模型积分逐步读取实际dt；loss按步平均而非dt时间加权。准确时长分布在protocol.json。既有选择是固定runner默认值和历史对照预算，并无证据表明1秒是最优物理时域。

## 训练与梯度

第一阶段：history-only 64维GRU+导数head，从随机初始化训练40 epoch，AdamW lr3e-4，seed17。
第二阶段：复制并冻结该backbone，训练drive/tail/gates 25 epoch，lr5e-4，seed29；drive tau0.10秒、tail tau0.04秒、history filter dt0.02秒均固定。两阶段batch256、gradient norm clip5。原4,214个训练窗口、训练统计和shuffle政策固定；不改train/validation日期。

历史只到t0，随后完整free-running；没有teacher forcing。所有预测积分、GRU转移、四元数与相位/frequency更新均在计算图中，不在rollout步间detach。第二阶段backbone参数不求梯度，但网络对预测输入的导数仍允许残差梯度跨步传播。频率/导数clamp的饱和区域会切断对应梯度；这不意味着冻结模型基准中已触发clamp（Step 2为零）。

每个ablation都重训第一阶段，再冻结其自己的backbone做第二阶段；不解冻原Main V2来改变训练政策，不用旧checkpoint代替A0。

## 训练前证据与选择

只用train确定辅助目标和权重。训练实测omega谱峰约4.297 Hz，但其95%功率范围上界约13.12 Hz；原生差分dot-omega的10Hz以上功率约47.9%，不能凭此保证高频全部物理可信。FFT前仅在诊断内部按每条独立连续片段重采样20ms，实际训练/模拟器仍使用原生dt，不跨段。

选择对齐两步增量 `omega[t+2]-omega[t]`，避免除以抖动dt放大高频；保留方向、时刻与pointwise状态loss，因此不奖励随机抖动或不匹配的振幅。两步增量仍不是“已证明无噪声”的标签，不做这种声明。没有增加频谱loss或修改五个状态权重。

A2/A3的0.2/0.5/1/2秒前缀权重取冻结模型train上相应total loss倒数后归一化：0.400268/0.319550/0.208413/0.071769，使初始各prefix贡献相当。每个prefix仍对内部所有步骤平均，故会重复加权早期状态，不是只监督四个终点。A3 delta权重0.581923，按冻结train预测使新增项与A2的body-rate项贡献相当。全部在正式训练前固定。

## 矩阵和判定

A0=原50步；A1=100步其他权重不变；A2=100步multi-prefix；A3=A2+两步delta-omega。全部40+25 epoch，固定最后epoch，不选最佳epoch；归一化与旧checkpoint逐位相同。只改objective/horizon。原4,214窗口只有100步未来；本轮不为了3秒组改变共同训练起点，3秒训练尚无证据，不预设需要5秒。

每组采用冻结722起点warm26、全部六个horizon、同commands/dt、相同support/clamp/numeric gates及regime bins。主要对照是重新训练A0，旧模型仅是reproduction参照。完整记录显存、时间、吞吐与step数。

判定：2/3/5秒velocity和attitude均下降；以五条flight为cluster的paired bootstrap报告95%CI，禁止把重叠窗口当独立样本。短期>10%退化明确警告。检查5秒prefix和末1秒variation是否更接近1，同时delta/angular-acceleration RMSE、实测训练谱上尾之外的新增能量不得恶化以掩盖噪声；frequency退化另列。单seed结果不宣称跨seed显著，选择偏差和5个cluster的统计限制必须保留。最终三选一结论仅在实验结果齐全后给出。

GPU：沙箱内查询失败；解除限制后确认cuda:1可用，正式训练只用GPU。未安装依赖或修改其他GPU进程。
