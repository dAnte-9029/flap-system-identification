# 三路线共用合同 v1
基线 9c15fffd24bfcacc7c57cd980cc149098cb83f82。各路线独立 worktree 从此基线建立；本目录为主 Agent 共用合同，其他 Agent 只读。Python /home/zn/anaconda3/envs/flap-train-gpu/bin/python。

仓库：/home/zn/flap-system-identification (origin dAnte-9029/flap-system-identification)，/home/zn/QgcLogs (origin dAnte-9029/QgcLogs)，/home/zn/PX4-Autopilot (实际 origin jiashunjiang2022/PX4-Autopilot)。后两者严格只读，禁止 checkout/fetch 写入，按日志 SHA 用 git show 读取源码。不能将当前 HEAD 视为实飞固件；缺失对象可读数据集 firmware_sources 精确快照，快照不含的源码应明确缺失或获取 exact SHA 的公开源码到自己 worktree。

数据族 velocity_rate_multistep_v1，注册表 configs/data/velocity_rate_dataset_registry.yaml。共享数据根 /home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1。仅 train，禁止 validation/test 数据读取与哈希。允许读取既有 split/provenance 元数据并立即过滤 train。遵守 build_config.yaml、dataset_manifest.json 和 native masks，保留 reviewed flight、gap/segment、availability、sample time；质量规则不得自行放宽。

主共同队列：CRM/A2 已有 11221 窗口、114 flights、17 dates；features.pkl 位于 /home/zn/flap-system-identification/.worktrees/pacld-control-identifiability-v1/experiments/phase_anchored_coupled_latent_v1/control_identifiability/，context.npy 同目录。A2 identity/features/pacld_train 位于 .worktrees/pacld-control-effectiveness-v2/experiments/phase_anchored_coupled_latent_v1/control_effectiveness_a2_v2。CRM windows.csv 位于 .worktrees/pacld-control-residual-moment-v1/experiments/phase_anchored_coupled_latent_v1/control_residual_moment_v1。必须按 flight,t 键连接，禁止默认行序。B 可使用这些 train flights 内的原生序列重建控制器，报告额外筛选和样本量。A/C 同样可访问上述已确认 train 原始日志。firmware_train.json 是主 Agent 逐日志核对结果；train_flights.json 是对应 train provenance，不可扩展其他 split。

控制 a=[motor,left,right,rudder]，left=actuator_servos.control[0]，right=control[1]，rudder=control[2]；d=(right-left)/2，s=(left+right)/2。均为 PX4 normalized command，不能称舵面实际角度或力矩。原控制 bounds [-1,1]，差动亦 [-1,1]，不能再任意缩放。CRM U[:,5] 为 [t,t+0.1s] ZOH 平均 d 减 t 前 baseline d；它是记录到的未来闭环控制描述量，非 t 已知动作。p=vehicle_angular_velocity.xyz[0]，FRD rad/s。目标 Δp=p(native sample nearest t+0.2s)-p(available at t)，精确时刻 t+200000+offset[0] us；不能强行插值到 0.2s 或混淆 timestamp 与 timestamp_sample。模式变化及已有 quality masks 沿用 A2；特征来自 t 前可用测量。

三个估计对象明确分开：O 条件闭环观察性响应；R 真正外部参考变化引起的整个闭环总响应；P 固定控制干预下的 plant 响应（必须声明控制波形、其他通道、初态和 horizon）。O/R/P 增益不得直接比较。第一阶段强不证明 IV exclusion；控制器重建好不证明 P 可辨识；观测残差分位数不证明噪声硬界。不得借助模拟 plant、NN 训练、checkpoint 修改或正式 PACLD 修改。跨日期用已有 train dates，窗口不是独立实验。

各 Agent 保存脚本、机器可读汇总与中文报告在自己 worktree；只提交自己新增脚本/聚合结果/报告到本地，逐窗口数据忽略，不推送。结论用 VERIFIED/SUPPORTED/HYPOTHESIS/UNRESOLVED，允许 NO-GO；缺失信息须直接报告。最终复现命令明确绝对 Python 和 cwd。数学不可辨识性证据可优于强造有限区间。

主 Agent 已执行核验：114 日志的 ver_sw 与 provenance 全部一致，15 个 SHA 全部存在本地 PX4 git 对象。features/context 身份哈希与既有 A2 合同一致，三个产物 flight,t 键集合完全相同；Y、U[:,5] 与 CRM逐键一致。实际目标 horizon 范围 [0.189061,0.209147] s；此差异必须保留。详见 cohort_check.json。逐日志 provenance/firmware 列表保留本地并 gitignore，脚本可重建。
