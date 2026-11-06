# PACLD v1：结构、接口与定向验证

实现基线：`research/phase-coupled-mpc-v1`，`330257f223544366549f12212812af8eb6973bb7`。
独立分支：`research/phase-anchored-coupled-latent-v1`。
本目录报告与源码一起提交；实际交付提交可用 `git log -1 -- src/system_identification/models/pacld` 查得。

## 结构与单位

源码位于 `src/system_identification/models/pacld/`，配置为 `configs/pacld/default.json`。没有改动 PA-HBD 的源码、模型行为或训练入口。

| 模块 | 输入 | 输出与默认值 |
|---|---|---|
| State | 因果物理 bootstrap、历史观测 | x13：NED 位置 m、速度 m/s、Hamilton wxyz FRD→NED 四元数、FRD 角速度 rad/s；z_s12、c6 无量纲；f Hz、连续 phi rad、归一化尾舵 a3；方差、联合 f/phi 协方差、相位质量、当前/最近观测时刻 |
| 共享 Encoder | 14维切空间残差、mask、quality、log variance、availability−measurement、事件 dt、z_s、c、chi(x)、a、电机、f | 两层 width=128 SiLU 特征；独立 SlowHead12、FastHead6；另有物理校正及更新门。没有 GRU 或按窗口优化的自由参数 |
| 工况 chi(x) | x | v_B/(10 m/s)、omega_B/(3 rad/s)、机体系重力单位方向，共9维；不读取绝对 NED 位置 |
| Slow MLP | z_s、c、chi(x)、a、电机、f/(5 Hz) | dz_s/dt = 10 tanh(raw)，s^-1；不添加 Slow 导数最小化 loss |
| Fast 包络 MLP | z_s、chi(x)、a、电机、f/(5 Hz) | c_eq=2 tanh(raw)，lambda=0.5+19.5 sigmoid(raw) s^-1，K=tanh(raw) 6×3；dc/dt=lambda(c_eq−c)+K da/dt |
| 执行器 | 四路 ZOH 控制中的尾舵3路 | da/dt=(u_tail−a)/0.06 s；子步/半步用解析一阶解，无命令阶跃差分 |
| 相位载波 | 连续 phi、c | Q(phi)c，三个二维块分别旋转 phi、2phi、3phi；没有可学习零位 |
| 唯一共享 Decoder | chi(x)、z_s、c、z_f、a、电机、f/(5 Hz) | 非重力 FRD 加速度3维 m/s²、FRD 角加速度3维 rad/s²；非线性耦合，不是固定傅里叶加法 |
| Lie-midpoint Flow | State、ControlTape、原生查询时间、mode | 最大5 ms可配步长；NED重力+[R_NB accel_B]，右乘SO(3)指数映射并归一化；两种模式采用同一积分定义 |

网络动力学 MLP 为两层 width=128 SiLU 后线性输出。Encoder 的两个隐状态更新门均来自同一共享特征；Slow/Fast 目标分别以2倍 tanh初始化/校正。这只是历史推断的工程范围，**不保证传播中 z_s、c 始终在 ±2 内**。解码器垂向输出 bias 以 −9.80665 m/s² 初始化，仅作数值起点。范围、执行器时间常数、12/6隐维和噪声均为工程先验，不是测得的物理参数。

本模型输出整机有效加速度，不输出独立机翼载荷或力矩；无须虚构质量/惯量、气动力矩参考点或 DeLaurier 先验。x 沿用原有 native 状态参考，亦不新增质心杠杆臂修正。

## 因果推断与双模式

`initial_state(window, config)` 只移植已有因果物理 bootstrap，将两组隐状态置零，不读取目标值，也不读取 PA-HBD 的 h。历史回放沿用 measurement/availability 分离规则：先按 availability≤decision 过滤，再按 measurement 排序；在测量时刻用 SO(3)切空间残差修正，之后重放至决策时刻。0.65 s 缓冲之外的历史不接受。缺失信息通过 mask/quality 门控；零质量 Hall 不建立参考。

首个可靠 Hall 设置相位参考和方差，后续以 atan2(sin Δ,cos Δ)更新。继承 `configs/pahbd/phase_reference.json` 的 Hall=0工作约定和20 ms发布保护假设，不学习额外零位，不拼接日志 unwrapped 字段。无可靠参考时 FULL 的显式载波由因果相位质量门关闭；可靠原生窗口的质量为1，此时严格采用 Q(phi)c。

历史统一以 FULL 传播并推断一个 posterior。`predict_pair` 从这个完全相同的 posterior 出发：BASE 仅把 decoder 的显式 z_f 换成0；FULL 保留 Q(phi)c。双方共用全部参数、Slow/Fast方程、控制合同及积分方法。不同物理轨迹随后可反馈出不同隐状态轨迹。BASE 仍含历史相位推断信息与包络/物理反馈，**不是严格的物理平均模型，也不是完全无相位模型**。

包络和 Slow 动力学均不读取独立相位 sin/cos；物理状态本身仍可能含周期信息，不能据此宣称严格快慢解耦。电机直接进入包络、Slow和解码器，同时进入复用的 PhasePropagator。未来接口只接收 posterior、控制和查询时刻，不接受观测更新。

保留 PA-HBD 的未标定物理方差代理及频率/相位联合协方差传播/Joseph更新。PACLD当前使用确定性相位载波，未将协方差折算成非线性 decoder 的期望，也没有隐状态协方差或已标定概率覆盖结论。

## 后续调用接口

```python
import json
from system_identification.models.pacld import PACLD, Config
from system_identification.data.pahbd import NativeWindows

cfg = json.load(open('configs/pacld/default.json'))
model = PACLD(Config(**cfg['model'])).double()
# NativeWindows延续冻结数据身份、分段、采样组和Hall合同；只选择获授权的split。
data = NativeWindows(**cfg['data'], split='train')
w = data[0]
belief = model.encode(w)
full = model.flow.rollout(belief, times, candidate_controls, mode='FULL')
base = model.flow.rollout(belief, times, candidate_controls, mode='BASE')
belief, pair = model.predict_pair(w)  # 默认只读取targets的原生查询时间，不读取目标值
```

`candidate_controls` 采用现有 `ControlTape(times, values)`，顺序为电机/左水平尾/右水平尾/垂尾，决策相对时间单位为秒。时间必须有限且有序，状态时间须匹配传播起点，禁止反向传播；控制为右连续 ZOH。MPC候选控制须由调用者在决策时提供；训练事实回放的未来日志控制只是条件输入，不是已知未来观测。

查询沿同一条轨迹顺序推进，不对每个目标从头积分，不制造同步观测；sparse_10hz 保持原生目标时间。低层 State/Flow支持相同时间网格的批维；原生窗口仍是异步ragged接口，本轮未实现跨窗口批调度。模型参数可用标准 PyTorch state_dict保存/恢复；未新增正式训练调度、模型选择或checkpoint训练恢复协议。

## 已执行验证

CPU环境：`/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，PyTorch 2.7.0+cu128，但设备明确为CPU；float64、单线程、seed=17。PACLD smoke 无 optimizer.step。

最终 PACLD 定向测试 **10 passed in 5.63 s**。另运行 PACLD + PA-HBD 合成回归（`-k "not native and not frozen"`），**22 passed、3 deselected，15.44 s**；该筛选排除了两项名称含native的PACLD测试，已在最终10项完整定向测试中覆盖。PA-HBD既有回归包含合成数据的单步优化/恢复检查，不是PACLD正式训练。

定向测试覆盖：共享特征/双Head梯度；三模态旋转与跨2π连续；共享decoder/零载波等价；FULL相位响应与BASE显式相位隔离；全部控制通道及电机直接路径梯度；未来发布观测/目标污染；Hall首次锚定、圆周创新、零质量门控、延迟回放；四元数反号等价、单位范数、原生时间、ZOH端点；反向时间/非有限时刻拒绝；向量场公式；完整传播控制梯度有限差分；state_dict恢复。

独立合成 DOP853 对照仅替换已知测试向量场，检验积分器：最大步长10/5/2.5 ms下，13维状态向量误差范数分别为 `5.736295e-5 / 1.476632e-5 / 3.692537e-6`，体现二阶收敛。该混合单位数值仅是数值回归指标，不是飞行动力学精度。

默认 width=128 的完整共享历史 + BASE/FULL原生预测 + 联合loss反向结果见 [cpu_smoke.json](cpu_smoke.json)：

| 窗口 | 历史事件 | 未来原生查询 | 前向 s | 反向 s |
|---|---:|---:|---:|---:|
| 合成 | 44 | 34 | 0.316 | 0.990 |
| train main_50hz | 200 | 32 | 0.514 | 0.761 |
| train sparse_10hz | 121 | 15 | 0.401 | 0.638 |

真实窗口分别约0.503/0.599 s历史、0.206/0.209 s实际目标终点；所有状态/梯度有限、四元数单位范数，两个窗口的Hall质量均为1。两个Head及Slow、包络平衡值、变化率、瞬态耦合、decoder均有非零梯度。数据加载约0.680 s另计；这些单次CPU耗时不是性能基准。

使用现有NativeWindows的冻结manifest身份校验；窗口、航次、segment、decision、manifest身份记录在JSON。仅实体加载train数据，没有validation选择或sealed test访问。loss只用于计算图smoke，不是BASE/FULL预测质量比较，不据随机响应符号作物理判断。

复现（输出文件必须不存在）：

```bash
PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q -s tests/test_pacld.py
PYTHONPATH=src /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/verify_pacld.py --real-train --output /tmp/pacld_smoke_new.json
```

## 证据边界与下一阶段问题

**VERIFIED**：本轮结构、因果接口、梯度和短窗数值合同；实际测试计数见交付结果。**UNRESOLVED**：隐状态可辨识性、真正快慢功能分工、闭环数据下的控制可辨识性、多步泛化、长时间稳定性、FULL相对BASE的训练后收益及MPC闭环价值。不能把非零梯度、初始loss或平滑轨迹视为上述结论。

Hall零点尚非独立翼角标定，20 ms发布延迟保护不是实测上界。5 ms默认步长不保证任意高频/长期预测精度。c_eq有界并不限制强控制瞬态下c的所有数值；Slow速率有界也不保证无限时域状态有界。这些应在下一阶段明确训练协议和功能诊断后检验。本任务到实现与smoke为止，不自动开始训练。
