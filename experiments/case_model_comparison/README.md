# CASE 六模型多步预测对比

[全精度 CSV](results/model_comparison.csv) 和 [LaTeX 跨栏表](results/model_comparison.tex) 报告 Vanilla MLP、GRU、LSTM、TCN、Transformer 和 Transformer Slow–Fast V2 在 0.5、1.0、1.5 s 的位置、速度、姿态和角速度端点 RMSE。位置和速度在 NED 系，角速度在 FRD 系，姿态使用旋转测地角。各窗口匹配同一原始观测时间，先计算各架次内的均方误差，再对架次等权平均并开方。

数据来自 `velocity_rate_multistep_v1`。训练使用 121 架次、126,237 个窗口，验证使用 39 架次、44,524 个窗口，随机种子为 17。预测期间回放记录的控制指令，物理状态自主递推。MLP 的两层隐藏层各有 64 个单元，使用当前运动状态、控制与步长直接预测下一次观测的速度和角速度。训练共 24 轮、2,976 次更新，第 16 轮取得最低递推验证分数。[逐轮记录](results/mlp_epochs.jsonl)

本实验沿用现有 worktree 布局，输入位于主仓库的 `outputs/`。复算需要 `history_slow_fast_20261119_cache`、`controlled_slow_fast_20261119_tapes`、`controlled_slow_fast_20261119_run1`、`controlled_baselines_20261119_run1` 和 `transformer_slowfast_latent_v2_20261124`。制表时使用 `transformer_slowfast_direct_v3_20261124/all_models_comparison.csv` 核对已归档的 0.5 和 1.0 s 指标。

从本实验所在的 worktree 根目录运行。训练入口为新运行创建 `mlp/`，已有最佳权重时可直接执行评分命令。

```bash
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
EXP=experiments/case_model_comparison
mkdir -p ../../outputs/case_model_comparison_20261124
"$PY" "$EXP/vanilla.py"
"$PY" "$EXP/score_table.py" existing
"$PY" "$EXP/score_table.py" mlp
"$PY" "$EXP/score_table.py" table
```
