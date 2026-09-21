# Phase A — Baseline comparison audit

The research target is short-horizon real-flight dynamics prediction, with 0.5 s primary. This is not an RL, uncertainty, OOD or long-rollout project.

## Reuse and required additions

- B0: reuse `models/trajectory.py::ConstantTwistPredictor` directly; no fitting.
- B1: existing `MLPRegressor(16,7,(64,64))` is suitable in size. Its historical trainer used one-step derivatives and different data/statistics, so historical checkpoints are not comparable. Reuse the network through a memoryless Main V1 rollout adapter and train on the same 28,293 windows.
- B2: reuse `CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True)` directly. Old September GRU checkpoints used the smaller v2 dataset and cannot be reused as matched baselines.
- B3: reuse the completed expanded September actuator-aware checkpoint, unchanged. The old August Main V2 and later joint-control experiments are not the proposed-method checkpoint for this pilot.
- Reuse window assembly, train-only normalization fitting, rollout integration, rollout loss, increment training loop and geodesic evaluator. Add only benchmark orchestration, flight aggregation, provenance and plots.

## Frozen method and training mismatch

Ours is **not** an end-to-end control-conditioned GRU: its history-only GRU64 is trained for 40 epochs (seed 17, LR 0.0003), then frozen while actuator residuals train for 25 epochs (seed 29, LR 0.0005). Last epoch is used in both stages. The second stage adds frequency MSE and actuator-specific regularization. Historical records report 1827.4 s total on GPU 1. B1/B2 are expected to require roughly 15–30 min each; the pilot trains these two only.

A standard controlled GRU cannot simultaneously match that frozen parameter set and differ only in the control path. Minimal proposed solution: use the same 40+25 epoch budget, optimizer resets, sampling seeds, learning rates and physical prediction loss; add the same second-stage frequency loss, omit inapplicable actuator regularizers. Disclose that this is a matched-budget comparison, not a strict single-factor ablation. A 40-epoch backbone-budget alternative is supported explicitly. Do not change/retrain Ours to conceal the mismatch.

The replicate label 17 must retain actual Ours seeds 17/29 in all metadata. A second and third proposed-method replicate would require a separately agreed stage-seed mapping and authorization to reproduce frozen Ours training; this run does not do so.

## Data and causality

Authority: registered `trajectory_v3_september_expanded`, same manifest and four train/validation artifacts as frozen Ours. 41 training flights / 28,293 windows; 17 validation flights / 2,582 windows (Sep7: 9 flights, Sep17: 8). Sep17 is validation, not training. Training days: Sep6, 11–16, 18. No other dataset, raw ULog, test metric or reserved prediction is loaded.

Every origin includes 26 samples with nominal 0.5 s history and 50 future native transitions. Nominal dt is 0.02 s; actual logged dt is integrated unchanged. Validation history spans 0.4889–0.5093 s, and the 25-step endpoint spans 0.4887–0.5189 s. Horizon labels refer to 5/10/25/50 native steps, not exactly resampled wall-clock intervals; full duration distributions are in the timing CSVs. Training stride is 10 samples, validation stride 50. No origin or flight is dropped by model/error. State: position/velocity NED, body rate FRD, wxyz FRD-to-NED quaternion, relative phase and flap frequency. Dynamics features: body velocity, body rate, body gravity, origin-relative phase sin/cos, frequency. Seven derivative targets: body-expressed inertial acceleration, angular acceleration and frequency derivative. Controls: motor, left, right, rudder.

Historical extraction uses causal publication-time zero-order hold. This pilot preserves that preprocessing, checks available source timestamps, verifies contiguous identities and dt, and refits statistics on train only to verify bitwise float32 equality with frozen buffers. Future logged commands are known-input replay; future measured states/frequency/phase are labels only. Feedback-generated future commands do not constitute independent causal interventions.

## Evaluation and statistical units

Nominal endpoint horizons 0.1/0.2/0.5/1.0 s; 0.5 s primary, 1 s extended. Per-flight vector RMSE is sqrt(mean(sum(error_xyz²))), not component-averaged RMSE. Attitude is RMS quaternion geodesic angle, not quaternion-component RMSE. Macro values average per-flight RMSE equally; standard deviation uses flights (ddof=1), never thousands of windows as independent trials. Single replicate has no across-seed SD or significance claim.

The requested Δv/Δω errors algebraically equal endpoint velocity/rate errors when both start at the same true t0. Report them transparently, not as independent evidence for dynamics learning. Representative figure uses the middle lexicographically sorted validation-origin identity, chosen without viewing predictions.

## Parameter counts

| model | total | trainable | base_stage_trainable |
| --- | --- | --- | --- |
| B0_ConstantVelocity | 0 | 0 |  |
| B1_MLP | 5703 | 5703 |  |
| B2_StandardGRU | 21383 | 21383 |  |
| B3_ActuatorAwareGRU | 21376 | 1017 | 20359.00000 |

For B3, `trainable` denotes actuator-stage parameters, `base_stage_trainable` the earlier backbone stage; `total` is deployment size. In evaluation no parameters are optimized.

## Complete admitted flight list

| partition | log_id | n_windows | cohort |
| --- | --- | --- | --- |
| train | 2026.9.6/log_17_2026-9-6-18-34-06.ulg | 895 | train |
| train | 2026.9.6/log_18_2026-9-6-18-46-20.ulg | 695 | train |
| train | 2026.9.6/log_19_2026-9-6-18-52-30.ulg | 774 | train |
| train | 2026.9.6/log_20_2026-9-6-18-58-10.ulg | 588 | train |
| train | 2026.9.6/log_24_2026-9-6-06-28-38.ulg | 702 | train |
| train | 2026.9.6/log_28_2026-9-6-07-01-38.ulg | 727 | train |
| train | 9.11数据/log_13_2026-9-11-18-28-08.ulg | 447 | train |
| train | 9.11数据/log_14_2026-9-11-18-34-58.ulg | 592 | train |
| train | 9.11数据/log_15_2026-9-11-18-40-48.ulg | 440 | train |
| train | 9.12数据/log_10_2026-9-12-06-13-40.ulg | 758 | train |
| train | 9.12数据/log_11_2026-9-12-06-21-12.ulg | 720 | train |
| train | 9.12数据/log_12_2026-9-12-06-29-44.ulg | 683 | train |
| train | 9.12数据/log_2_2026-9-12-18-32-54.ulg | 753 | train |
| train | 9.12数据/log_3_2026-9-12-18-40-02.ulg | 792 | train |
| train | 9.12数据/log_7_2026-9-12-05-49-02.ulg | 655 | train |
| train | 9.12数据/log_8_2026-9-12-05-55-30.ulg | 815 | train |
| train | 9.12数据/log_9_2026-9-12-06-03-32.ulg | 1017 | train |
| train | 9.13数据/log_1_2026-9-13-06-11-06.ulg | 712 | train |
| train | 9.13数据/log_2_2026-9-13-06-20-06.ulg | 710 | train |
| train | 9.13数据/log_3_2026-9-13-06-31-26.ulg | 710 | train |
| train | 9.13数据/log_4_2026-9-13-06-36-26.ulg | 696 | train |
| train | 9.13数据/log_5_2026-9-13-06-42-40.ulg | 730 | train |
| train | 9.13数据/log_7_2026-9-13-18-12-52.ulg | 450 | train |
| train | 9.14数据/log_11_2026-9-14-18-13-22.ulg | 756 | train |
| train | 9.14数据/log_12_2026-9-14-18-20-30.ulg | 768 | train |
| train | 9.14数据/log_13_2026-9-14-18-28-32.ulg | 883 | train |
| train | 9.14数据/log_14_2026-9-14-18-34-22.ulg | 600 | train |
| train | 9.15数据/log_10_2026-9-15-17-44-04.ulg | 594 | train |
| train | 9.15数据/log_11_2026-9-15-17-51-56.ulg | 538 | train |
| train | 9.15数据/log_12_2026-9-15-18-03-58.ulg | 704 | train |
| train | 9.15数据/log_13_2026-9-15-18-10-58.ulg | 690 | train |
| train | 9.15数据/log_14_2026-9-15-18-17-26.ulg | 685 | train |
| train | 9.15数据/log_15_2026-9-15-18-23-44.ulg | 702 | train |
| train | 9.16数据/log_1_2026-9-16-18-02-34.ulg | 576 | train |
| train | 9.16数据/log_2_2026-9-16-18-08-46.ulg | 691 | train |
| train | 9.16数据/log_3_2026-9-16-18-13-58.ulg | 521 | train |
| train | 9.16数据/log_4_2026-9-16-18-20-50.ulg | 689 | train |
| train | 9.16数据/log_5_2026-9-16-18-26-48.ulg | 658 | train |
| train | 9.18数据/log_0_2026-9-18-18-15-06.ulg | 742 | train |
| train | 9.18数据/log_1_2026-9-18-18-23-36.ulg | 693 | train |
| train | 9.18数据/log_2_2026-9-18-18-29-40.ulg | 742 | train |
| validation | 2026.9.7/log_21_2026-9-7-05-54-56.ulg | 168 | Sep7 |
| validation | 2026.9.7/log_22_2026-9-7-06-01-40.ulg | 182 | Sep7 |
| validation | 2026.9.7/log_23_2026-9-7-06-09-22.ulg | 160 | Sep7 |
| validation | 2026.9.7/log_24_2026-9-7-06-16-04.ulg | 161 | Sep7 |
| validation | 2026.9.7/log_26_2026-9-7-06-30-26.ulg | 150 | Sep7 |
| validation | 2026.9.7/log_27_2026-9-7-06-37-24.ulg | 155 | Sep7 |
| validation | 2026.9.7/log_28_2026-9-7-06-44-02.ulg | 140 | Sep7 |
| validation | 2026.9.7/log_29_2026-9-7-06-51-04.ulg | 201 | Sep7 |
| validation | 2026.9.7/log_30_2026-9-7-06-58-22.ulg | 164 | Sep7 |
| validation | 9.17数据/log_10_2026-9-17-06-34-34.ulg | 130 | Sep17 |
| validation | 9.17数据/log_11_2026-9-17-06-40-30.ulg | 145 | Sep17 |
| validation | 9.17数据/log_12_2026-9-17-06-46-02.ulg | 144 | Sep17 |
| validation | 9.17数据/log_13_2026-9-17-06-52-30.ulg | 153 | Sep17 |
| validation | 9.17数据/log_6_2026-9-17-06-02-24.ulg | 138 | Sep17 |
| validation | 9.17数据/log_7_2026-9-17-06-12-16.ulg | 131 | Sep17 |
| validation | 9.17数据/log_8_2026-9-17-06-17-52.ulg | 146 | Sep17 |
| validation | 9.17数据/log_9_2026-9-17-06-28-10.ulg | 114 | Sep17 |

## Excluded/sealed assignments (metadata only)


### sealed_test

- `2026.9.8/log_0_2026-9-8-05-42-48.ulg`
- `2026.9.8/log_1_2026-9-8-05-49-28.ulg`
- `2026.9.8/log_2_2026-9-8-05-56-46.ulg`
- `2026.9.8/log_4_2026-9-8-06-38-56.ulg`
- `2026.9.8/log_6_2026-9-8-06-46-18.ulg`
- `2026.9.8/log_7_2026-9-8-06-55-26.ulg`

### reserved_evaluation

- `9.19-2数据/log_0_2026-9-19-14-31-44.ulg`
- `9.19-2数据/log_1_2026-9-19-14-40-12.ulg`
- `9.19-2数据/log_2_2026-9-19-14-45-56.ulg`
- `9.19-2数据/log_3_2026-9-19-14-56-32.ulg`
- `9.19-2数据/log_4_2026-9-19-15-15-06.ulg`
- `9.19-2数据/log_5_2026-9-19-15-23-48.ulg`
- `9.19-2数据/log_6_2026-9-19-15-28-00.ulg`
- `9.19-2数据/log_7_2026-9-19-15-39-08.ulg`
- `9.19-2数据/log_8_2026-9-19-15-47-10.ulg`
- `9.19-3数据/log_0_2026-9-19-17-14-26.ulg`
- `9.19-3数据/log_1_2026-9-19-17-21-44.ulg`
- `9.19-3数据/log_2_2026-9-19-17-26-42.ulg`
- `9.19-3数据/log_3_2026-9-19-17-38-46.ulg`
- `9.19-3数据/log_4_2026-9-19-17-44-06.ulg`
- `9.19-3数据/log_5_2026-9-19-17-49-50.ulg`
- `9.19-3数据/log_9_2026-9-19-17-05-08.ulg`
- `9.19数据/log_0_2026-9-19-05-53-40.ulg`
- `9.19数据/log_1_2026-9-19-06-01-30.ulg`
- `9.19数据/log_2_2026-9-19-06-07-52.ulg`
- `9.19数据/log_3_2026-9-19-06-14-30.ulg`
- `9.19数据/log_4_2026-9-19-06-19-40.ulg`
- `9.19数据/log_5_2026-9-19-06-26-44.ulg`
- `9.19数据/log_6_2026-9-19-06-31-46.ulg`

Six Sep8 sealed flights and 23 Sep19 reserved flights remain excluded. Reading their names from split metadata is the only access in this run. Existing raw-inventory metadata in the repository predates this task; the pilot does not assert that no historical metadata inventory ever occurred. No sealed/reserved raw samples, evaluations or predictions are opened here.

Registry, instructions and many unrelated artifacts were already dirty at startup. This task does not modify them. `protocol.json` records the current commit plus source/data/checkpoint hashes because commit identity alone cannot reproduce a dirty workspace.
