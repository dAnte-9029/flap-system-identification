# Native rate target pilot

M0 GRU-Control，C0/V0输入，A/B未来角速度目标，三折、seed17、各6epoch。只更改目标10:13；两臂共用B连续支持前缀Loss掩码，模型始终按原完整dt自主递推。第6轮为唯一主checkpoint，不选best。

从仓库根目录执行，使用既有环境，不安装依赖：

```bash
P=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
E=experiments/native_rate_target_pilot_v1
$P $E/prepare.py > $E/results/prepare.log 2>&1
$P $E/verify_packing.py > $E/results/packing_verification.log 2>&1
$P $E/preflight.py
$P $E/train.py --device cuda:1 > $E/results/training.log 2>&1
$P $E/verify_training.py > $E/results/training_verification.log 2>&1
$P $E/reuse_support.py > $E/results/support.log 2>&1
$P $E/evaluate.py --device cuda:1 > $E/results/evaluation.log 2>&1
$P $E/analyze.py > $E/results/analysis.log 2>&1
$P $E/describe.py > $E/results/description.log 2>&1
$P $E/verify_results.py > $E/results/result_verification.log 2>&1
$P $E/plot.py
```

`common.py`登记历史数据和H1000缓存路径。所需旧结果来自前序实验，原始大缓存不复制。所有读取限制train/validation，未读取test或原始ULog。运行已有完成训练时校验last哈希后跳过；中断仅用包含optimizer、scheduler、RNG与记录的last恢复，不能重训挑选。

产物：

- `results/{train,validation}_targets.npz`：B角速度、共同Loss掩码和精确窗口身份；无支持位置为原A有限占位，测试证明不参与Loss/梯度。
- `results/target_coverage*.csv`：总量、折内及逐航次监督覆盖。
- `folds/fold*/normalization*.json`：复用并核验的拟合侧尺度及原审计。
- `folds/fold*/training/{M0A,M0B}/seed17/last.pt`：6组第6轮模型，包含恢复状态；`training.jsonl`包含逐轮原A/B参考的航次等权归一化四项Loss、梯度、LR、batch身份哈希、时间与显存。
- `results/trajectory_summary.csv`、`trajectory_per_flight.csv`、`heldout_trajectory.csv.gz`：所有留出窗口、物理单位误差；参考A/B，主要共同B支持，另有原A完整支持诊断。
- `results/factual_*.csv*`：冻结A/B/D事实响应；`paired_*.csv*`：B训练相对A训练的变化。
- `results/per_group_300.csv`、`component_assignments.csv`：航次、日期、配置、11依赖分量；原背景和质量子集保留。
- `results/internal_*.csv*`：AA/AB/BA/BB内部C/H诊断，没有真实交换控制标签。
- `results/prespecified_cases_300.csv`、`figures/fixed_cases.png`：固定病例，不后选样本。
- `results/*verification.json`、`tests_pretraining.log`、`source_receipt.json`：验证及来源；历史大数据哈希沿用上轮记录，不重复全量原始核验。

普通轨迹的500/1000ms表示在该名义时域内、共同支持原生节点上的路径误差；端点为最后受支持节点，`last_supported_s`记录实际时刻，不能当作严格500/1000ms预测。响应评价沿用原冻结时间插值，不支持处保持缺失。

轨迹汇总先计算窗口MSE，再航次平均、跨航次平均后开方。响应保留逐对、航次等权、依赖分量等权。validation用于逐轮诊断、不用于选epoch；validation:44单独标为已暴露病例。历史反复使用的三折仍是开发数据探索，不是盲测。
