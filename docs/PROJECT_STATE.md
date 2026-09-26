# Project State

## 2026-09-26：真实数据控制动力学研究交付（控制未晋升）

- 工作分支 `control-dynamics-realdata-v1`，从step5 `e64a1ba`分出，保留原论文基线和数据划分。
- [阶段报告](analysis/results/control_dynamics_realdata_v1/report.md)；[计划与剩余工作](plans/control-dynamics-realdata-v1.md)。冻结三seed动力学集成改善短时预测，但PX4受限闭环、跨模型重放及可信度审计尚不支持控制晋升。原train/validation范围内探索，sealed test未打开。
- [候选包](analysis/results/control_dynamics_realdata_v1/delivery/control-dynamics-candidate-v1.tar.gz)已通过独立目录与解压后推理验证；历史残差修正未保留，经验误差余量未支持现有小幅动作的控制结论。研究产物交付不等于控制可用性目标完成。
- 追加配对、周期残差及参考辅助局部辨识均已记录。manual100ms通过训练必要筛选，但固定validation误差0.376813 rad/s，弱于冻结集成0.218773；不改候选。研究交付完成，持续舵效、时延和独立闭环有效性仍缺证据，不批准实飞或RL用途。

## 2026-07-14

- Wing-only baseline vs total effective-wrench sensitivity analysis implemented.
- Attitude-aware ground-minus-wind body-airflow rerun and diagnostic figures generated on five test windows.
