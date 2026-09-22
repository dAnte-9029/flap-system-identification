# Frozen three-seed MLP comparison


## Goal and scope

Overall Prediction Performance: assess repeatability of the frozen model-family gap, not a one-factor history ablation.


## Frozen methods and reproducibility

B1 MemorylessTrajectoryModel: current state/control input16 → ReLU64 → ReLU64 →7; 5,703 trainable parameters; original output zero initialization, derivative scales/clipping and physical integrator. It ignores history but uses H26-eligible origins. B2 Standard GRU64/H26:21,383 parameters, unchanged. B0:constant inertial velocity and body rate with the original integrator. All learned models use40 epochs AdamW lr3e-4 plus25 epochs fresh AdamW lr5e-4; batch256, weight decay1e-5, clip5, 111 updates/epoch,7,215 total; full50-step rollout, frozen lag2 increment supervision, continuation frequency MSE0.2. Last epoch65 only. Base seeds17/23/42; continuation29/35/54. No validation selection. Native dt;5/10/25/50 endpoints; vector RMSE and quaternion geodesic RMS. The protocol aggregation field defines this three-seed analysis; metrics.uncertainty is inherited Step1 single-pilot metadata, clarified before evaluation in metric_metadata_clarification.json. The old factory hardcodes seed17; the new helper calls the same class with the requested seed and preserves constructor draw order. Same-seed repeatability, different-seed random layers and seed17 factory parity are tested. Frozen six normalization buffers are loaded once, never fitted.

Source HEAD: `c0057c226065718ddc0089ead61a1c7ac76c36ab`. Branch: `step5-actuator-aware-trajectory-main-v2`. Exact paths, versions, seeds, hashes and settings: protocol.json and per-run runtime/config.json.


## Data and reuse

41 training flights /28,293 origins;17 validation flights /2,582 origins (Sep7:9 flights/1,481; Sep17:8 flights/1,101). Actual commands only. Identical cached labels, control tapes, dt, origins and normalization for all runs. Only B1 seeds23/42 trained from fresh initialization. B1 seed17, all H26 seeds and deterministic B0 reused; hashes rechecked. Original B1 seed17 full prediction metrics reproduce stored results; a fixed128-origin GPU replay passes predeclared tolerance. Original logs for B1 seed17 did not record final validation loss; it is left blank rather than inventing a value. New final validation losses are post-training reporting only. Source metrics include endpoint increments but they are algebraically redundant and not separate evidence.


## 500 ms ALL

| Cohort | Horizon [s] | Model | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.5 | B0 Kinematic | 0.4182 | 1.1707 | 23.2455 | 1.0660 |
| ALL | 0.5 | B1 MLP | 0.2407 ± 0.0019 | 0.5422 ± 0.0097 | 5.4859 ± 0.0451 | 0.7526 ± 0.0015 |
| ALL | 0.5 | B2 Standard GRU / H26 | 0.1924 ± 0.0019 | 0.3573 ± 0.0072 | 4.0784 ± 0.0461 | 0.5966 ± 0.0028 |


## Complete cohort/horizon results

| Cohort | Horizon [s] | Model | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.1 | B0 Kinematic | 0.0587 | 0.8196 | 4.1554 | 1.2121 |
| ALL | 0.1 | B1 MLP | 0.0494 ± 1.36e-05 | 0.5308 ± 0.0013 | 2.0366 ± 0.0095 | 0.7184 ± 0.0031 |
| ALL | 0.1 | B2 Standard GRU / H26 | 0.0353 ± 0.0002 | 0.1449 ± 0.0046 | 1.4074 ± 0.0206 | 0.5453 ± 0.0026 |
| ALL | 0.2 | B0 Kinematic | 0.1378 | 0.7480 | 9.3133 | 1.2005 |
| ALL | 0.2 | B1 MLP | 0.0982 ± 0.0002 | 0.4289 ± 0.0037 | 3.1704 ± 0.0144 | 0.7553 ± 0.0066 |
| ALL | 0.2 | B2 Standard GRU / H26 | 0.0704 ± 0.0005 | 0.1954 ± 0.0038 | 2.1965 ± 0.0385 | 0.5663 ± 0.0028 |
| ALL | 0.5 | B0 Kinematic | 0.4182 | 1.1707 | 23.2455 | 1.0660 |
| ALL | 0.5 | B1 MLP | 0.2407 ± 0.0019 | 0.5422 ± 0.0097 | 5.4859 ± 0.0451 | 0.7526 ± 0.0015 |
| ALL | 0.5 | B2 Standard GRU / H26 | 0.1924 ± 0.0019 | 0.3573 ± 0.0072 | 4.0784 ± 0.0461 | 0.5966 ± 0.0028 |
| ALL | 1 | B0 Kinematic | 1.2249 | 2.0733 | 46.7779 | 1.1262 |
| ALL | 1 | B1 MLP | 0.5675 ± 0.0071 | 0.8618 ± 0.0143 | 8.5328 ± 0.1523 | 0.7687 ± 0.0048 |
| ALL | 1 | B2 Standard GRU / H26 | 0.4740 ± 0.0063 | 0.6477 ± 0.0070 | 6.6925 ± 0.1321 | 0.6211 ± 0.0108 |
| Sep17 | 0.1 | B0 Kinematic | 0.0473 | 0.8067 | 4.0547 | 1.2504 |
| Sep17 | 0.1 | B1 MLP | 0.0368 ± 0.0002 | 0.5470 ± 0.0018 | 2.2452 ± 0.0272 | 0.7317 ± 0.0070 |
| Sep17 | 0.1 | B2 Standard GRU / H26 | 0.0188 ± 0.0003 | 0.1533 ± 0.0048 | 1.4698 ± 0.0125 | 0.5274 ± 0.0027 |
| Sep17 | 0.2 | B0 Kinematic | 0.1187 | 0.7733 | 9.3406 | 1.1662 |
| Sep17 | 0.2 | B1 MLP | 0.0748 ± 0.0002 | 0.4477 ± 0.0059 | 3.6687 ± 0.0360 | 0.7661 ± 0.0060 |
| Sep17 | 0.2 | B2 Standard GRU / H26 | 0.0404 ± 0.0008 | 0.2205 ± 0.0009 | 2.3762 ± 0.0567 | 0.5370 ± 0.0007 |
| Sep17 | 0.5 | B0 Kinematic | 0.3985 | 1.2780 | 23.4183 | 1.0921 |
| Sep17 | 0.5 | B1 MLP | 0.1909 ± 0.0020 | 0.5775 ± 0.0067 | 6.5473 ± 0.0512 | 0.7494 ± 0.0025 |
| Sep17 | 0.5 | B2 Standard GRU / H26 | 0.1293 ± 0.0040 | 0.3736 ± 0.0143 | 4.5695 ± 0.1054 | 0.5609 ± 0.0046 |
| Sep17 | 1 | B0 Kinematic | 1.2449 | 2.2176 | 47.3359 | 1.1321 |
| Sep17 | 1 | B1 MLP | 0.4970 ± 0.0070 | 0.9556 ± 0.0181 | 10.0683 ± 0.1660 | 0.7813 ± 0.0056 |
| Sep17 | 1 | B2 Standard GRU / H26 | 0.3682 ± 0.0141 | 0.6882 ± 0.0119 | 7.5654 ± 0.3508 | 0.6174 ± 0.0161 |
| Sep7 | 0.1 | B0 Kinematic | 0.0689 | 0.8311 | 4.2450 | 1.1780 |
| Sep7 | 0.1 | B1 MLP | 0.0607 ± 0.0001 | 0.5164 ± 0.0016 | 1.8512 ± 0.0063 | 0.7066 ± 0.0016 |
| Sep7 | 0.1 | B2 Standard GRU / H26 | 0.0501 ± 0.0001 | 0.1375 ± 0.0045 | 1.3519 ± 0.0288 | 0.5613 ± 0.0029 |
| Sep7 | 0.2 | B0 Kinematic | 0.1547 | 0.7254 | 9.2891 | 1.2309 |
| Sep7 | 0.2 | B1 MLP | 0.1190 ± 0.0002 | 0.4121 ± 0.0019 | 2.7275 ± 0.0145 | 0.7457 ± 0.0072 |
| Sep7 | 0.2 | B2 Standard GRU / H26 | 0.0970 ± 0.0003 | 0.1731 ± 0.0063 | 2.0368 ± 0.0230 | 0.5923 ± 0.0050 |
| Sep7 | 0.5 | B0 Kinematic | 0.4357 | 1.0754 | 23.0919 | 1.0428 |
| Sep7 | 0.5 | B1 MLP | 0.2849 ± 0.0025 | 0.5108 ± 0.0132 | 4.5424 ± 0.0500 | 0.7554 ± 0.0024 |
| Sep7 | 0.5 | B2 Standard GRU / H26 | 0.2485 ± 0.0008 | 0.3428 ± 0.0010 | 3.6419 ± 0.0567 | 0.6283 ± 0.0026 |
| Sep7 | 1 | B0 Kinematic | 1.2072 | 1.9450 | 46.2818 | 1.1209 |
| Sep7 | 1 | B1 MLP | 0.6301 ± 0.0097 | 0.7785 ± 0.0113 | 7.1679 ± 0.1413 | 0.7576 ± 0.0042 |
| Sep7 | 1 | B2 Standard GRU / H26 | 0.5681 ± 0.0035 | 0.6118 ± 0.0035 | 5.9166 ± 0.0650 | 0.6244 ± 0.0073 |


## Paired directions and gains

Absolute gain = MLP − GRU; relative gain =100×(MLP − GRU)/MLP on reported macro errors; zero denominator undefined. Each flight direction compares its3-seed mean errors. Positive favors GRU. No window-independence tests.

| cohort | horizon_s | metric | relative_gain_pct | seeds_gru_better | seeds_mlp_better | flights_gru_better | flights_mlp_better | n_flights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 0.1 | attitude_error_deg | 30.895 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.1 | body_rate_rmse_rad_s | 24.092 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.1 | position_rmse_m | 28.502 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.1 | velocity_rmse_m_s | 72.697 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | attitude_error_deg | 30.719 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | body_rate_rmse_rad_s | 25.022 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | position_rmse_m | 28.335 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.2 | velocity_rmse_m_s | 54.43 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | attitude_error_deg | 25.656 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | body_rate_rmse_rad_s | 20.732 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | position_rmse_m | 20.057 | 3 | 0 | 17 | 0 | 17 |
| ALL | 0.5 | velocity_rmse_m_s | 34.102 | 3 | 0 | 17 | 0 | 17 |
| ALL | 1 | attitude_error_deg | 21.568 | 3 | 0 | 17 | 0 | 17 |
| ALL | 1 | body_rate_rmse_rad_s | 19.21 | 3 | 0 | 17 | 0 | 17 |
| ALL | 1 | position_rmse_m | 16.473 | 3 | 0 | 16 | 1 | 17 |
| ALL | 1 | velocity_rmse_m_s | 24.839 | 3 | 0 | 17 | 0 | 17 |
| Sep17 | 0.1 | attitude_error_deg | 34.535 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.1 | body_rate_rmse_rad_s | 27.925 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.1 | position_rmse_m | 48.957 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.1 | velocity_rmse_m_s | 71.98 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | attitude_error_deg | 35.232 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | body_rate_rmse_rad_s | 29.898 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | position_rmse_m | 46.011 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.2 | velocity_rmse_m_s | 50.744 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | attitude_error_deg | 30.207 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | body_rate_rmse_rad_s | 25.151 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | position_rmse_m | 32.262 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 0.5 | velocity_rmse_m_s | 35.31 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | attitude_error_deg | 24.859 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | body_rate_rmse_rad_s | 20.983 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | position_rmse_m | 25.928 | 3 | 0 | 8 | 0 | 8 |
| Sep17 | 1 | velocity_rmse_m_s | 27.986 | 3 | 0 | 8 | 0 | 8 |
| Sep7 | 0.1 | attitude_error_deg | 26.971 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.1 | body_rate_rmse_rad_s | 20.564 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.1 | position_rmse_m | 17.474 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.1 | velocity_rmse_m_s | 73.372 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | attitude_error_deg | 25.323 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | body_rate_rmse_rad_s | 20.569 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | position_rmse_m | 18.462 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.2 | velocity_rmse_m_s | 57.989 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | attitude_error_deg | 19.825 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | body_rate_rmse_rad_s | 16.834 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | position_rmse_m | 12.791 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 0.5 | velocity_rmse_m_s | 32.888 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 1 | attitude_error_deg | 17.458 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 1 | body_rate_rmse_rad_s | 17.584 | 3 | 0 | 9 | 0 | 9 |
| Sep7 | 1 | position_rmse_m | 9.8432 | 3 | 0 | 8 | 1 | 9 |
| Sep7 | 1 | velocity_rmse_m_s | 21.406 | 3 | 0 | 9 | 0 | 9 |


## Exceptions

0 nonpositive seed/cohort/horizon/metric comparisons and 1 nonpositive flight/horizon/metric comparisons are retained without filtering.

No 500 ms flight exceptions.

All exceptions, including other horizons: flight_exceptions.csv and seed_exceptions.csv. Full seed comparisons: paired_seed_differences.csv.

| flight_id | cohort | horizon_s | metric | absolute_gain | relative_gain_pct |
| --- | --- | --- | --- | --- | --- |
| 2026.9.7/log_30_2026-9-7-06-58-22.ulg | Sep7 | 1 | position_rmse_m | -0.00055103 | -0.039619 |

Endpoint error need not grow monotonically: the MLP velocity error decreases from100 to200 ms, then rises at500 ms and1 s. This pattern is retained; initial alignment, prefix and native-timing checks pass. The current experiment does not isolate its physical or architectural cause.


## Training completion and engineering checks

| model | seed | status | final_epoch | final_train_loss | training_time_s | checkpoint_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| B1_MLP | 17 | reused | 65 | 0.72186 | 1396.7 | 2e82f6b6a5380883bfc78eaa419c4f1a258a9fd06cba31017f4151610a94bef7 |
| B1_MLP | 23 | complete | 65 | 0.71851 | 1348.8 | c9fca6b6435d8046ed64e7f9154314f7b9e8917dbc2c02720c5f63646787ae57 |
| B1_MLP | 42 | complete | 65 | 0.72172 | 1345 | 0483807dacbae37b849f4e8a14445522d050c2349e0956fb8c9334fe4d396dae |
| B2_StandardGRU | 17 | reused | 65 | 0.34286 | 1816.4 | f9895418b7dd2a2757dde8d740109ab42dae258c593fd5902f0076a3faeb0e26 |
| B2_StandardGRU | 23 | reused | 65 | 0.34216 | 5353.4 | 1cfbb865db55e3fc7c248eff4de95892fa9ee9155ee8e81c0318a9021b7d8b03 |
| B2_StandardGRU | 42 | reused | 65 | 0.34177 | 5334.5 | 98fd6266bdc8926055c7f8b0dc2f37a4333fa92f55f336fb4cba4d3480324f86 |

Both new runs must complete65 epochs without retries. Tests cover initialization, original baseline/increment behavior and descriptive statistics. Sanity checks cover finite losses/gradients/weights/predictions, normalization immutability, exact origin pairing, native timing, unit quaternions, future-label poisoning, prediction prefix consistency, fixed updates and final-epoch selection. Full details: tests.json, preflight.json, sanity_checks.json and completion.json.


## Paper Results

Under the frozen matched-budget protocol, the Standard GRU/H26 and memoryless MLP were evaluated using three training seeds (17, 23, 42) on the same 2,582 validation origins from 17 flights. At the nominal 500 ms horizon, the GRU reduced the mean velocity, attitude, and body-rate errors by 34.10%, 25.66%, and 20.73%, respectively, relative to the MLP. Errors were first computed within each flight, averaged equally across flights, and then summarized across seeds. Reported dispersions are sample standard deviations across three seeds. These are descriptive validation results for the frozen model families.


## Limitations / Discussion

The MLP and GRU differ in architecture and parameter count (5,703 versus 21,383); this comparison therefore cannot isolate history as the cause of all performance differences. The separate H1–H26 experiment provides the controlled history-length evidence. Matching seed identifiers pairs training repetitions, not initial weights or identical random trajectories. Three seeds offer limited coverage of optimization variability, and overlapping windows are not independent trials. All models replay logged future commands; these validation errors do not establish causal responses to arbitrary actions, independent test generalization, long-horizon simulation validity, or closed-loop control performance. No sealed or reserved test was accessed.


## Scientific interpretation and next step

Use the directions and magnitudes above rather than imposing a post-hoc success threshold. The validation Overall Prediction Performance table is now complete for B0/MLP/GRU. This does not complete independent-test evidence. Next candidate:freeze the final evaluation protocol and prepare independent testing, without opening test data in this task. H26 remains main model; prior history classification remains Mixed; future-control conclusions remain unchanged. Unrelated working-tree/registry differences are recorded, not repaired. No automatic commit/push.


## Figures

Error-vs-horizon figures show all four metrics and ALL/Sep7/Sep17. Paired-flight500ms figure includes all17 flights; flight key is saved. Shading denotes ±1 sample SD across3seeds, not confidence intervals or model uncertainty. B0 has no artificial seedSD.


## Exact split

Training flights:

- 2026.9.6/log_24_2026-9-6-06-28-38.ulg
- 2026.9.6/log_28_2026-9-6-07-01-38.ulg
- 2026.9.6/log_17_2026-9-6-18-34-06.ulg
- 2026.9.6/log_18_2026-9-6-18-46-20.ulg
- 2026.9.6/log_19_2026-9-6-18-52-30.ulg
- 2026.9.6/log_20_2026-9-6-18-58-10.ulg
- 9.11数据/log_13_2026-9-11-18-28-08.ulg
- 9.11数据/log_14_2026-9-11-18-34-58.ulg
- 9.11数据/log_15_2026-9-11-18-40-48.ulg
- 9.12数据/log_10_2026-9-12-06-13-40.ulg
- 9.12数据/log_11_2026-9-12-06-21-12.ulg
- 9.12数据/log_12_2026-9-12-06-29-44.ulg
- 9.12数据/log_2_2026-9-12-18-32-54.ulg
- 9.12数据/log_3_2026-9-12-18-40-02.ulg
- 9.12数据/log_7_2026-9-12-05-49-02.ulg
- 9.12数据/log_8_2026-9-12-05-55-30.ulg
- 9.12数据/log_9_2026-9-12-06-03-32.ulg
- 9.13数据/log_1_2026-9-13-06-11-06.ulg
- 9.13数据/log_2_2026-9-13-06-20-06.ulg
- 9.13数据/log_3_2026-9-13-06-31-26.ulg
- 9.13数据/log_4_2026-9-13-06-36-26.ulg
- 9.13数据/log_5_2026-9-13-06-42-40.ulg
- 9.13数据/log_7_2026-9-13-18-12-52.ulg
- 9.14数据/log_11_2026-9-14-18-13-22.ulg
- 9.14数据/log_12_2026-9-14-18-20-30.ulg
- 9.14数据/log_13_2026-9-14-18-28-32.ulg
- 9.14数据/log_14_2026-9-14-18-34-22.ulg
- 9.15数据/log_10_2026-9-15-17-44-04.ulg
- 9.15数据/log_11_2026-9-15-17-51-56.ulg
- 9.15数据/log_12_2026-9-15-18-03-58.ulg
- 9.15数据/log_13_2026-9-15-18-10-58.ulg
- 9.15数据/log_14_2026-9-15-18-17-26.ulg
- 9.15数据/log_15_2026-9-15-18-23-44.ulg
- 9.16数据/log_1_2026-9-16-18-02-34.ulg
- 9.16数据/log_2_2026-9-16-18-08-46.ulg
- 9.16数据/log_3_2026-9-16-18-13-58.ulg
- 9.16数据/log_4_2026-9-16-18-20-50.ulg
- 9.16数据/log_5_2026-9-16-18-26-48.ulg
- 9.18数据/log_0_2026-9-18-18-15-06.ulg
- 9.18数据/log_1_2026-9-18-18-23-36.ulg
- 9.18数据/log_2_2026-9-18-18-29-40.ulg


Validation flights:

- 2026.9.7/log_21_2026-9-7-05-54-56.ulg
- 2026.9.7/log_22_2026-9-7-06-01-40.ulg
- 2026.9.7/log_23_2026-9-7-06-09-22.ulg
- 2026.9.7/log_24_2026-9-7-06-16-04.ulg
- 2026.9.7/log_26_2026-9-7-06-30-26.ulg
- 2026.9.7/log_27_2026-9-7-06-37-24.ulg
- 2026.9.7/log_28_2026-9-7-06-44-02.ulg
- 2026.9.7/log_29_2026-9-7-06-51-04.ulg
- 2026.9.7/log_30_2026-9-7-06-58-22.ulg
- 9.17数据/log_10_2026-9-17-06-34-34.ulg
- 9.17数据/log_11_2026-9-17-06-40-30.ulg
- 9.17数据/log_12_2026-9-17-06-46-02.ulg
- 9.17数据/log_13_2026-9-17-06-52-30.ulg
- 9.17数据/log_6_2026-9-17-06-02-24.ulg
- 9.17数据/log_7_2026-9-17-06-12-16.ulg
- 9.17数据/log_8_2026-9-17-06-17-52.ulg
- 9.17数据/log_9_2026-9-17-06-28-10.ulg
