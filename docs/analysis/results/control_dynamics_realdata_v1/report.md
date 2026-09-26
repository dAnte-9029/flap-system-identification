# 真实数据控制动力学探索：阶段报告

状态：进行中，已保存通过短时预测门槛的冻结集成候选，尚未通过控制可用门槛。分支 `control-dynamics-realdata-v1`，起点 `e64a1ba`。现有论文基线及分区不变。本轮只读取原train/validation，不打开sealed test。历史独立测试评估的存在已知，本轮不声称完全盲测。

## 已完成：基线重新执行

StandardGRU64/H26/K50的17/23/42三个seed均在指定环境CPU重新推理，覆盖原17个validation flights、2,582 origins。当前GPU驱动不可用。checkpoint、manifest、分区文件、缓存和实现源文件哈希均核验。与冻结预测比较，最大位置差3.052e-5 m，其余状态差异在预先规定的rtol=1e-5、atol=1e-4内。未来标签污染不改变输出；50步预测与25步前缀逐值相等。后续探针还验证10步之后的命令变化不改变前10步输出。

下表先flight内vector RMS，再flight等权，再三个seed平均。它们是预测误差，不是干预舵效误差。

| 时域 | 速度 m/s | 姿态 deg | 角速度 rad/s |
| --- | ---: | ---: | ---: |
| 100 ms | 0.1449 | 1.4074 | 0.5453 |
| 200 ms | 0.1954 | 2.1965 | 0.5663 |
| 500 ms | 0.3573 | 4.0784 | 0.5966 |
| 1 s | 0.6477 | 6.6925 | 0.6211 |

完整seed、日期和flight结果见 `baseline/`。重现命令：

```bash
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/reproduce_control_dynamics_baseline.py --device cpu
```

入口拒绝覆盖已有结果；复现时在独立checkout保留输入产物、使用空结果目录。输入哈希固定在protocol。

## 假设与最小对照：轨迹误差相近不保证局部控制响应一致

每flight按既有identity顺序等距取最多16个起点，共272个，不按误差选择。将未来命令固定为t0命令，对drive/common/differential/rudder分别施加正负0.1倍训练坐标标准差，保持初始状态、H26历史和其他控制完全相同。测量速度与角速度的中心差分导数。三个seed使用相同探针，未拟合任何新参数。

227个起点通过初始状态特征及正负命令的train q01..q99边界筛查，覆盖17 flights；所有起点结果均保留。这个边界只检验各维边际覆盖，不保证联合分布、保持命令轨迹或后续状态在训练支持域内。因此不能把结果写成真实可用范围，也不能把相同符号理解为正确物理符号。

| 通道→轴 | 100 ms三seed方向一致率 | 200 ms | 500 ms | 1 s |
| --- | ---: | ---: | ---: | ---: |
| common→q | 100.0% | 78.6% | 84.2% | 78.0% |
| differential→p | 71.9% | 72.8% | 79.4% | 71.5% |
| rudder→r | 83.3% | 56.8% | 95.5% | 98.1% |
| drive→vz（导航系） | 86.9% | 89.3% | 92.5% | 91.0% |

比例先flight内计算再flight等权。所有六轴、每origin、每seed的导数及绝对增益/seed标准差均在 `response_probe/`；近零导数以1e-4报告，表中对应主轴没有该近零情况，但该数值筛查不是物理显著性门槛。幅值单位为(m/s或rad/s)/归一化命令。drive→vz不是气动力轴或推力标定。

保留该负面诊断：只依赖单seed预测导数设计控制有明显风险。不能从此确认哪个seed正确，不能据此强制所有模型导数同号，也不能估计真实执行器时延。三seed共用架构、训练数据和目标函数，一致也无法排除共同偏差。

```bash
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/probe_control_dynamics_response.py
```

第一次预检因对frozen dataclass属性使用增强赋值而退出，未生成预测。修复为数组切片原位修改后完整运行；失败protocol和堆栈保留在 `response_probe_failed_preflight/`，不作为有效实验。

## 第一轮提出的下一实验

下一步优先检验局部俯仰响应：以train flight分组交叉验证比较状态/历史回归和增加过去及当前命令的回归，检查条件激励、正负覆盖、跨flight增益与延迟稳定性；不得以未来实测状态为输入。先冻结选择规则再执行。100ms common→q在模型间一致，仅使其成为较合理的探索起点，不构成已经可控制的证据。

该局部对照及随后的候选对照现已完成，结果如下。现有pitch闭环脚本加载旧MainV2，不能直接作为当前StandardGRU验证。

## 第二轮：局部命令信息对照

`local_identification/protocol.json`在拟合前冻结。原41个训练flight按日志完整分成五折，所有归一化只用各fold拟合日志。固定岭惩罚0.01，flight等权，无超参数搜索。输入为当前及0/1/2/5/10/25步状态特征、严格过去1/2/5/10/25步命令；对照增加预测区间内已知命令的分箱均值。标签为未来速度和角速度相对起点的改变量，未输入未来实测状态。20/100/200/500ms仅为名义时域，实际dt分布见elapsed_time.csv。

| 俯仰q目标 | train OOF历史模型 | train OOF增加命令 | OOF改善flight | validation历史模型 | validation增加命令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100ms | 0.36847 | 0.36048 | 40/41 | 0.35261 | 0.35106 |
| 200ms | 0.41617 | 0.41042 | 33/41 | 0.40546 | 0.40434 |
| 500ms | 0.50717 | 0.48980 | 40/41 | 0.49206 | 0.48070 |

单位rad/s。预先要求100ms q的OOF平均误差改善且至少60%训练flight改善，结果通过：2.17%、40/41；validation改善0.44%、11/17。保留“命令提供额外预测信息”假设。条件命令预测残差及正负比例完整保留，但残差混合预测器误差和未观测反馈状态，不是有效因果工具变量。

**否定将该线性模型作为更优动力学候选。** 它在100ms validation q上误差0.35106，当前GRU三seed均值0.22417，明显更差。更关键的是，其100ms持续common命令导数在五折为+0.659至+0.889 rad/s/command，而GRU探针三个seed的均值为约−2.21/−1.73/−2.14。两者来自不同函数族和估计口径，没有真实反事实轨迹作裁决；不能选择一个符号当物理约束，也不能把各命令分箱系数解释成真实执行器时延。

这条负面结果改变后续方向：不靠放大舵效或强制导数符号改模型，先保留模型分歧并检查它对闭环的影响。

## 第三轮：冻结等权动力学集成候选

`ensemble_candidate/protocol.json`在预测前冻结。三个已有GRU在同一物理状态上计算机体系加速度、角加速度及频率导数，等权平均后沿原积分器前进一步；每个成员独立更新自己的隐藏状态。没有新增训练、调tau、修改物理参数、选择best seed或重拟合归一化。成员导数标准差作为诊断输出，尚未校准，不能当置信区间。该模型与直接平均最终轨迹不同，保留单一姿态与速度的积分一致性。

与三seed各自指标的均值比较：

| 时域 | 速度RMSE候选/基线 | 姿态RMSE候选/基线 | 角速度RMSE候选/基线 |
| --- | ---: | ---: | ---: |
| 100ms | 0.13850 / 0.14493 | 1.37376 / 1.40739 | 0.53939 / 0.54533 |
| 200ms | 0.18813 / 0.19543 | 2.13743 / 2.19649 | 0.56037 / 0.56628 |
| 500ms | 0.34588 / 0.35729 | 3.96519 / 4.07840 | 0.59031 / 0.59658 |
| 1s | 0.62275 / 0.64775 | 6.48852 / 6.69249 | 0.61002 / 0.62107 |

单位依次m/s、deg、rad/s。100/200ms的ALL/high_change × ALL/Sep7/Sep17 × 四指标共48项均改善，最小改善0.14%。100ms速度/姿态/角速度分别改善4.44%/2.39%/1.09%，通过“无超过1%退化，至少一个主指标改善1%”的预定预测门槛。500ms/1s只作扩展报告，没有用于选择。

暂时保留为**最佳短时预测候选**，控制晋升仍未通过。三个成员共用架构和数据，集成无法消除共同响应偏差；更小RMSE不解决局部回归与GRU的符号冲突，不支持实机、MPC或RL部署声明。

候选权重：`ensemble_candidate/model.pt`，包含三个成员完整权重和归一化buffer、成员种子、hidden_size与格式版本；其SHA256及所有结果哈希见completion.json。运行实现：`src/system_identification/models/control_ensemble.py`。单成员与原GRU逐值一致；重复成员一致性、未来命令前缀、显式状态续跑及非法dt拒绝通过测试；集成另通过未来标签污染及25/50步前缀检查。

```bash
OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/identify_control_dynamics_local.py
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_control_dynamics_ensemble.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_control_dynamics_local.py tests/test_control_derivative_ensemble.py
```

这两个新实验及原实验均拒绝覆盖已有输出；需要完整复现时使用保留输入产物的独立checkout和空对应输出目录。局部模型的信息边界/flight等权测试4项，集成模型测试4项均通过。

## 第四轮：固定PX4俯仰链路的受限闭环与跨模型重放

`px4_guarded_closed_loop/`保存事前protocol、控制器参数来源、51个固定起点、所有轨迹、命令tape及终止记录。使用已有e624俯仰速率控制组件与FlightAllocator，参数来自9月18日train日志的已审计快照；核验快照哈希，未调整增益、效能矩阵或物理参数。这不是完整PX4固件/外环/传感器仿真。模型20ms更新一次，控制组件每2.5ms更新一次，期间保持角速度；角加速度用过去差分。控制器I=0、空速滤波状态=trim，未重建完整飞行控制器历史。未来空速不可用时沿用PX4规定的trim fallback。

每flight三个按identity等距取出的验证起点，共51个；q目标为初始实测q加−0.2/0/+0.2 rad/s，其他命令保持。三个单seed与集成分别进行原命令保持/反馈，再将各模型的反馈命令原样重放到其他三个模型。没有对闭环结果优化控制器或选择seed。

### 支持范围与限制

训练数据建立18维联合邻域：机体系速度、角速度、机体系重力、扑频、四个命令坐标、过去五步命令变化。各维须位于train q01..q99内，标准化最近邻距离还须≤3.98577（训练五折整flight留出最近邻距离q99）。这是经验覆盖筛查，不是因果可辨识保证，也不验证未观测空速、舵角或隐藏状态。

核心边际范围如下；完整18维、均值/尺度和引用样本见feature_ranges.csv与support.npz，**不能仅按此表独立判断准入**。

| 量 | 下界 | 上界 |
| --- | ---: | ---: |
| 机体系vx / vy / vz (m/s) | 3.633 / −3.337 / 0.091 | 10.382 / 2.764 / 3.509 |
| p / q / r (rad/s) | −1.288 / −1.135 / −0.683 | 1.603 / 1.148 / 0.774 |
| 扑频 (Hz) | 2.311 | 4.726 |
| drive | 0.374 | 0.868 |
| common | −0.197 | 0.326 |
| differential | −0.704 | 0.586 |
| rudder | −0.289 | 0.237 |

common相对当前命令额外限制±0.01145385（0.1倍训练起点common标准差），保留起点differential/rudder/motor并检查原始舵面命令范围。反馈抗积分饱和使用上一次实际投影后分配残差。提出的状态—动作离域时不执行；下一状态离域或导数/扑频clipping时终止，不把剩余冻结状态算成成功。

### 结果：没有控制晋升证据

51个起点中23个通过初始联合支持检查，22个完成100/200/500ms，21个完成1s（集成模型，三个目标偏置均相同覆盖）。最初拒绝的28个案例、退出的案例全部保留。集成运行期间93.4%–97.1%的活动时刻触及额外命令限制，因此这里检验的是带严格约束的组件实验，不能评价完整PX4真实性能。

下表是完成配对的案例上先flight内平均再flight等权的q跟踪RMSE，单位rad/s。完成配对仅22/51（16 flights），不应掩盖准入拒绝。

| 目标偏置 | 100ms保持命令 | 100ms反馈 | 200ms保持命令 | 200ms反馈 |
| --- | ---: | ---: | ---: | ---: |
| −0.2 | 0.6690 | 0.6837 | 0.7061 | 0.7191 |
| 0 | 0.6042 | 0.6143 | 0.6570 | 0.6674 |
| +0.2 | 0.5775 | 0.5871 | 0.6455 | 0.6554 |

集成模型内反馈均值比保持命令略差；100ms逐起点改善计数分别2/22、5/22、8/22。不能为了让闭环通过而调增益、翻转分配符号或扩大动作范围。初始状态本身可能带快速运动、目标保持并非真实后续任务，故这些分数不等于实飞跟踪误差。

跨模型命令重放同时要求源反馈/源保持/目的重放/目的保持四条轨迹完成。100/200/500/1000ms可比较配对数分别792/792/792/756（总请求每个时域1836，重复起点和有序模型配对，不独立）。其中“在源模型有益、在目的模型有害”分别7/6/44/81项。该计数使用效果差值符号，包含微小差异，原始幅度保存在transfer_per_origin.csv，不能把计数当显著性或实机危害率。它表明部分收益依赖模型，长时域尤其脆弱。

`px4_guarded_summary/`分别保存所有请求的生存率、完成配对、共同有效时长结果；未将不同生存时间的RMSE直接混为总体成功指标。

## 第五轮：可信度审计与动作优化的共同偏差

`credibility/`在原2582个验证起点上，用固定支持筛查检查候选预测路径及已知命令，准入不读取未来真实状态/误差。100/200/500/1000ms完整预测前缀通过数为2007/1891/1678/1387，均覆盖17 flights。与闭环51个固定起点覆盖不同，不能混用分母。

在通过预测前缀支持检查的子集，100/200ms q RMSE为0.1897/0.2055 rad/s，但三seed平均预测标准差只有0.0424/0.0441 rad/s。实测误差落在±2倍seed标准差内的flight等权比例仅32.8%/32.7%。未经筛查的全验证集也仅33.3%/34.5%。**因此不能把集成分歧当作可靠置信区间；同架构、同数据的共同偏差明显。** 未拟合置信倍率或根据validation误差放宽支持门槛。

`planning_stress/`进一步做事前冻结的一次性受限命令搜索：保持common原值或±训练确定的epsilon，100/200ms，以terminal q目标平方误差选动作；对照seed17、集成、三seed最坏误差策略。只使用真实初始状态/历史与候选命令，没有未来状态输入；不是已经部署的滚动MPC。

集成策略在100ms自己的模型中，对三个目标偏置的22个可比较起点分别改善22/21/22个、恶化0个。但同一动作在train-only局部线性模型中分别恶化20/17/16个；200ms分别恶化21/17/16个。这个线性模型此前已被否定为更优预测模型，它不是因果真值或独立实飞测试。这里的价值是揭示：只在当前模型或同族成员上优化，可以获得看似一致的收益，却依赖未经数据证实的响应方向。最坏seed优化也不能消除共同偏差。

因此暂不批准基于该候选的控制器设计结论、实际MPC动作执行或RL单模型训练推广。100/200ms只是当前继续研究的预测时域；0.5/1s保持压力测试用途。未辨识物理执行器时延、持续舵效及其状态依赖，没有新增实飞。

## 复现与候选加载接口

候选加载与联合支持推理：`src/system_identification/integration/control_dynamics_candidate.py`。`load_candidate(path, expected_sha256)`先验证权重哈希，再严格恢复三个冻结成员；`reset`只接受H26历史及当前物理状态，`step(state, command, dt)`只接受显式当前记忆、命令和dt，返回下一状态及成员导数分歧，不读取数据集。`JointSupport`加载已拟合的support.npz数组，不在线拟合。

```bash
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_control_dynamics_closed_loop.py
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/report_control_dynamics_closed_loop.py
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_control_dynamics_credibility.py
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/stress_control_dynamics_planning.py
```

同样拒绝覆盖历史结果。候选加载哈希/权重、联合距离与边际筛查差别、phase anchor不变性及PX4相关21项测试通过。闭环与审计产物哈希在各completion.json记录。

## 第六轮：历史预测残差信息与经验误差范围

`past_residual_information/`在原train/validation全部起点上，从t−5步的真实历史及已发生命令做五步预测，其终点严格等于当前t0；只把当前已观测到的预测误差用作特征。各窗口的native dt单独记录。173个train起点、52个validation起点在旧预测的开始处不足26步历史，沿用标准因果mask左填充；不跨segment，不删除这些起点。

比较零修正、0..1正增益的持续残差修正、允许负增益的诊断。系数按训练flight等权拟合并做五折训练日志交叉验证；只有修正系数交叉拟合，基础GRU权重曾见过全部train日志，因此没有将它描述为完整模型的独立OOF验证。修正仅用于端点信息检验，没有冒充新的积分一致动力学模型。

预先要求100ms q的训练OOF误差至少改善5%，且至少60%训练flight改善。实际改善0.49%（32/41 flights），未通过；全train拟合后，validation q由0.218773变成0.219233 rad/s，略差。200ms的q正增益被截为0，没有改善。100ms p/r拟合符号为负，亦不支持统一的持续扰动解释。**否定本轮持续残差修正，不加入候选，不根据validation反向挑选修正系数。** 这也不证明残差一定是测量噪声，未观测扑翼运动或更复杂状态仍可能参与。

另外，以每个train flight的绝对预测误差95分位数取跨flight最大值，固定经验误差范围。q在100/200ms的阈值为0.527285/0.532996 rad/s，原validation的flight等权覆盖率98.55%/98.57%，最差flight为97.33%/96.73%。范围来自基础模型的训练残差，存在训练拟合偏差；这里只报告实测validation覆盖，没有分布无关或反事实保证。全六轴、四时域及逐flight覆盖均保存。

`planning_error_margin/`用固定阈值复核上一轮动作搜索。若两次预测各有±B误差，则预测目标误差改善要超过2B，才可能仅依据这个误差范围区分动作；反事实误差是否满足B本身仍未验证。396个可评估的模型内规划比较中，**0个超过该经验余量**。此前几乎总能找到的自身模型收益，远小于观测误差尺度；不能用seed一致或自身优化成功替代这个缺口。

`control_error_envelope.py`提供只读阈值查询和余量检查，对未评估时域拒绝外推；即使余量为正也不会自动标记控制有效。7项新增测试验证历史窗口边界、flight等权、正负残差区别和双预测误差余量，均通过。最初一次测试收集遇到脚本与测试同名冲突，重命名测试后解决，不影响实验计算。

```bash
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/test_control_dynamics_residual_information.py
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_control_planning_error_margin.py
```

## 当前尚缺交付与下一步

已完成基线、两类最小改进对照、闭环组件实验、跨模型重放、规划误差利用压力测试及支持/分歧审计；**仍未获得控制可用模型**。目前最佳候选是误差略小、可报告分歧和覆盖范围的冻结集成，原论文基线未替换。

历史残差修正和误差范围现已检验，控制门槛依然否定。剩余物理证据缺口必须保留：持续舵效和时延没有被现有闭环数据可靠辨识，全部反事实命令只有模型输出，实际动作效果缺少独立验证。继续降低同一验证集分数或调整控制器不能补上这些证据。

## 自包含候选交付与当前审计

- [候选说明](model_card_zh.md)，[独立推理包](delivery/control-dynamics-candidate-v1.tar.gz)，[当前证据核验记录](delivery_audit.json)。包约3.6MiB，SHA256为`94264491a356b95f34f4a64d00a32ef7e78a007f73b3d226efe15baa8c88996e`。
- 包含三个冻结权重、完整推理源代码快照、支持参考、经验误差范围、中文报告与一个真实当前/历史输入示例；示例未来命令为人工保持当前命令，不包含未来真值，期望输出是候选模型预测。
- 在`/tmp`中独立进程验证，并另外实际解压tar.gz后再次验证；六类预测与打包前逐值一致、未访问数据集。候选包可以直接加载，但这只证明可复现运行，不能证明控制有效。
- 当前32项针对性测试通过；10个实验及候选交付的输入/输出哈希核验通过；显式缓存只有原train/validation，41/17 flights互不重叠。sealed test未打开。
- 完整实验重跑仍需要原仓库中已登记、已校验的train/validation及既有checkpoint/cache；独立包提供推理重现，不假装包含全部训练数据。各入口拒绝覆盖历史实验。包内报告是打包时快照，后续工作以仓库当前报告为准。

```bash
# 完整实验产物的只读核验
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/verify_control_dynamics_delivery.py
# 解压包后，在任意工作目录执行（路径替换为实际解压位置）
/home/zn/anaconda3/envs/flap-train-gpu/bin/python /tmp/control-dynamics-candidate-v1/run_example.py --bundle /tmp/control-dynamics-candidate-v1
```

交付产物可用，控制可用性目标仍未成立。后续目标审计必须区分“研究代码与候选已交付”和“某个非平凡动作范围内的控制效果已有可信证据”，不能因打包和测试通过而自动将后者标记完成。

## 第七轮：配对响应对照，避免把保守余量当成不可能性证明

此前2B余量是不依赖误差相关性的保守条件，未通过不证明真实动作一定无效，也不证明所有已有数据都不可辨识。因此新增不同flight的观测配对，直接检查预测误差之差。匹配只用当前/过去状态、过去命令、当前模式及预测区间内的已知命令；不按响应结果选样本。

`matched_contrasts_scored/`保存完整结果，`matched_contrasts/`保留原先冻结的选择及原脚本。每个log/segment的历史25步+预测区间先去重叠，再按状态距离选不重复配对；要求不同flight、当前nav_state相同、非目标通道命令tape相近、目标通道剂量差超过train固定门槛。预注册三档60维标准化RMS距离0.25/0.5/1.0，每点只检查最近64个候选；这是确定性有限匹配算法，未声称穷尽所有可能的配对。

- 距离0.25时，train/validation均未找到合格配对。
- 距离0.5时，common在100/200/500ms的validation配对仅6/9/1个，未通过预定描述性数量门槛；不能根据少数符号一致案例宣称俯仰舵效已确定。
- 同一中等距离下，只有100ms rudder与200ms differential达到事前规定的train/validation数量及flight覆盖门槛。这是样本量门槛，不是控制晋升。
- 距离1.0时12个通道×时域组合都达到数量门槛，但允许的状态失配更大。部分预测的记录响应差异超过配对误差范围，说明不能由上一轮2B失败直接推断“没有可用响应信息”。这些差异仍混合状态差、惯性、隐藏扰动与共同输入，不是同一状态下已观测到的反事实舵效。

首次预检遇到无效数据行重复(-1)索引，显式筛选valid_core后修复；失败记录保留。首次评分存在重复解压NPZ的性能问题，经主动终止后保留原配对，修复为一次读取预测数组，仅重启评分，不重新选样本。原/现脚本、选择哈希与恢复原因均记录在两个protocol中。匹配函数的非重复配对、flight/mode边界、其他通道/时间条件及无效行处理测试5项通过。

## 第八轮：显式周期残差仍不足以解决俯仰问题

`phase_residual_screen/`对每个当前起点的H26角速度拟合截距、趋势及1阶/2阶相对扑翼相位谐波，再用冻结模型预测相位向前延伸。当前增量严格为0，未来实测相位/扑频不进入特征。原相位物理零位未确认，因此只作当前相对相位下的周期信息检验。

单个p/q/r修正增益由train的100/200ms目标共同拟合，五折整训练flight交叉验证；基础权重仍曾见全部train。选择规则先于validation评分冻结：q平均误差至少改善5%，至少60%训练flight改善，p/r不能出现超过1%退化。1阶/2阶在训练OOF的q改善仅0.16%/0.52%，均失败，预先选择baseline。

完整validation诊断仍保留：2阶q在100/200ms为0.21723/0.21319 rad/s，基线0.21877/0.21569；r也有一定改善，但没有以这些结果反向改变主门槛。该实验尚不是积分一致的新模型，已否定为本轮俯仰修复，不加入交付权重。3项测试验证零起点、已知周期信号、未来相位尾部不影响前缀及输入不变性。

## 尚可检验的路径：已有参考指令与反馈相关性

闭环数据中的控制命令可能与扰动相关。文献提出利用参考信号或控制器信息构造工具变量，但依赖相应系统/噪声和相关性条件，不能把方法名称当作因果验证。[Wang等，2023，原文III节](https://arxiv.org/html/2309.05916v1)。该论文研究LTI框架；本项目的非线性、混合模式PX4和驾驶员反馈不自动满足其条件。

作为可行性前置审计，`reference_availability/`仅打开原41个train日志，逐个核验登记raw哈希，不读取validation raw或sealed test。41个日志均记录manual_control_setpoint、input_rc、vehicle_attitude_setpoint和vehicle_rates_setpoint。跨日志记录间隔中位数分别约0.20042、0.50031、0.01996、0.01996s；这反映日志采样，不能当生产者实际更新率。RC_MAP_PITCH均为2。FLAP_SLOW_EN在5/14个日志为0/1，另外22个没有该字段，不能擅自补默认值。

这提供了一条尚未验证的方向：检查原Stabilized片段中当前/过去参考指令的来源、时间有效性和条件输入相关性，再决定能否进行参考辅助的局部辨识。不能将vehicle_rates_setpoint自动当外生信号，它可能由状态反馈计算；也不能把驾驶员指令默认与未来扰动独立。未来记录的参考信号若用于离线辨识统计，必须与模型部署输入严格分开，不能输入预测器。

```bash
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_control_matched_contrasts.py
# 本次保留原选择、只恢复评分所用命令
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_control_matched_contrasts.py --score-frozen
MPLCONFIGDIR=/tmp/flap-paper-mpl OPENBLAS_NUM_THREADS=2 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/screen_control_phase_residual.py
MPLCONFIGDIR=/tmp/flap-paper-mpl /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_control_reference_availability.py
```

这轮没有改变候选权重、支持门槛或控制器；原独立包仍是同一冻结版本。参考信号适用性随后按下述实验检验，不能把负面结果直接宣称为所有方法的不可能性。

## 第九轮：参考语义、局部工具变量与固定验证

`reference_semantics/`核验41个train原始日志，在当前Stabilized模式、当前manual有效且三个参考时刻满足新鲜度条件下保留6,477个起点。历史manual值未逐个要求valid，不能将筛选解释为全部历史参考均确认有效。对齐要求消息发布与sample时间均不晚于查询时刻，不使用未来参考。归档e624源码中的manual姿态映射与新鲜记录一致，但未证明所有日志固件相同。角速度参考明确由姿态反馈产生，不作为外生工具。

加入当前/过去姿态参考后，整flight交叉验证的100/200ms common命令预测误差下降5.43%/4.04%；这证明额外预测信息，不证明与未来扰动独立。驾驶员可能根据飞机运动修正参考，排除限制仍未成立。

`reference_iv_diagnostic/`用相同train子集作嵌套整flight交叉拟合：状态/过去命令及其他通道已知未来均值为干扰项，manual或attitude的当前/过去100/200ms值为候选工具，固定ridge=0.01，不约束增益符号。外层留出flight完全不进入内层拟合。推理只需要当前/过去状态、命令历史和候选未来命令，不需要未来参考或状态。

| 方法 | 时域 | 五折增益范围 | 相对无common输入的OOF误差变化 | 预定训练筛选 |
|---|---|---|---|---|
| manual参考 | 100ms | −4.901至−3.121 | 改善1.73%，39/41 flights改善 | 通过必要筛选 |
| attitude参考 | 100ms | −1.668至−0.046 | 改善0.67% | 未通过，弱于直接回归 |
| manual参考 | 200ms | −0.641至+2.165 | 退化0.13% | 未通过，符号不稳定 |
| attitude参考 | 200ms | +1.621至+3.892 | 改善0.47% | 未通过 |

系数单位为(rad/s)/归一化common命令；这是条件统计系数，不是确认的物理舵效。manual100ms投影命令RMS约0.0023–0.0025，远小于条件输入残差RMS约0.016–0.018；未给出弱工具稳健置信区间，不宣称显著因果效果。

只将训练筛选通过的manual100ms固定后，在`reference_iv_validation/`执行一次原validation全量预测对照。拟合和归一化仍只用6,477个train起点，validation没有读取raw参考日志；因此这里验证整个原validation的预测适用性，不能冒充与train参考新鲜度筛选完全相同的人群。

| 固定方法 | validation逐flight平均q100ms RMSE，rad/s |
|---|---:|
| 无common输入 | 0.375877 |
| 直接回归 | 0.375523 |
| manual参考辅助 | 0.376813 |
| 已交付冻结动力学集成 | **0.218773** |

验证预测门槛失败，不替换候选、不接入控制链路。该结果否定本轮固定方法作为更好预测器的晋升，不能证明参考辅助方法普遍无效，也不能从误差差异反推真实舵效符号。新增8项测试覆盖参考两时钟可用性、重复时间戳、四元数映射、已知内生合成例、flight等权、无工具信号拒绝和预测前缀边界，均通过。

```bash
MPLCONFIGDIR=/tmp/flap-paper-mpl OPENBLAS_NUM_THREADS=2 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_control_reference_semantics.py
MPLCONFIGDIR=/tmp/flap-paper-mpl OPENBLAS_NUM_THREADS=2 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/diagnose_control_reference_iv.py
MPLCONFIGDIR=/tmp/flap-paper-mpl OPENBLAS_NUM_THREADS=2 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_control_reference_iv.py
```

## 本轮研究交付结论

本轮已完成基线复现、可检验标准、多轮最小对照、受限PX4闭环与模型误差利用审计，并交付最佳预测候选及独立推理包。后续配对、周期残差和参考辅助实验未提供足以晋升的改进，全部保留协议与负面结果。当前结论仍为**控制可用性未成立**，不将研究交付完成等同于达成控制晋升。

候选适用于原数据支持范围内的100/200ms观测轨迹预测研究和控制方案离线排查；支持域检查、误差范围和拒绝条件见模型卡。500ms/1s仅作压力测试，不承诺反事实控制、实飞或RL训练有效性。剩余必要证据是可独立核验的动作响应方向/幅值/时延，以及在对应状态和动作范围内不依赖共同模型偏差的闭环收益；本轮已有数据分析没有建立这些证据。未来研究可以继续探索，但不放宽门槛、不借sealed test或控制器调参制造成功。
