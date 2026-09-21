# Paper Step 3 — Standard GRU multi-seed robustness

## 1. Goal

Standard GRU (Step1 B2) is the current paper main model. Test initialization/minibatch robustness on open validation data only. Actuator-aware remains a historical negative ablation. No other models or ablations trained.

## 2. Frozen model definition

CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True); GRUCell(16,64), head Linear(80,64)/Tanh/Linear(64,7). Trainable parameters: 21,383. History26; same state/control targets, same frozen buffers and unchanged integrator. Exact source hashes in protocol.json.

40 epochs AdamW lr0.0003, then25 epochs lr0.0005 with a new AdamW optimizer. Batch256; all28,293 windows per epoch (111 updates). Weight decay1e-5, gradient clip5, full50-step loss. Same frozen two-step increment scales/weights; continuation adds0.2 frequency MSE through the existing zero-regularizer adapter. No validation checkpoint selection: last epoch65 for every seed.

## 3. Data/split contract

41 training flights /28,293 windows;17 validation flights /2,582 origins: Sep7 nine flights/1,481 origins; Sep17 eight flights/1,101 origins. Same windows at every horizon/seed. Freshly assembled arrays exactly match the Step1 cached histories, targets, controls, dt and identity order. No normalization fitting. Full flight names/exclusions and hashes are in protocol.json.

Manifest SHA256: `46b50d0b23a32b40339a4c07526711d0ff88b55584d309a56807c56905d2817e`; train-origin CSV: `3a17f717f89367d0f673a2f73f3df246c4e332b9fac88385ce269df28854845b`; validation-origin CSV: `d9677c58fe0f9f23b12699a35c78e1b0961ddcb3ec5e5814e9f18392c501e3d3`; normalization canonical SHA256: `f10a427d04d54b98e5d1e4e0995b3b8562347f54313c90cd64147fa5ae2710c8`.

Frames: position/velocity NED; body rates FRD; quaternion wxyz body-to-NED. Control order motor/left/right/rudder, normalized allocation commands before PWM. Relative encoder phase is reanchored at each origin. Nominal horizons are native steps5/10/25/50, integrating original dt, not resampling.

## 4. Seeds

Replicates17/23/42 use initialization and base-stage seed s, continuation seed s+12: (17,29), (23,35), (42,54). The fixed offset preserves the actual historical pilot definition. Seed17 checkpoint/predictions reused after hash and metric parity checks; no repeat training.

| seed | checkpoint_path | checkpoint_sha256 | final_epoch | best_epoch | final_train_loss | final_val_loss | training_time_s | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | artifacts/paper_baseline_comparison_v1/B2_StandardGRU.pt | f9895418b7dd2a2757dde8d740109ab42dae258c593fd5902f0076a3faeb0e26 | 65 | nan | 0.342859 | 0.375745 | 1816.394105 | reused_complete |
| 23 | artifacts/paper_standard_gru_multiseed_v1/seed23/model.pt | 1cfbb865db55e3fc7c248eff4de95892fa9ee9155ee8e81c0318a9021b7d8b03 | 65 | nan | 0.342157 | 0.384439 | 5353.385571 | complete |
| 42 | artifacts/paper_standard_gru_multiseed_v1/seed42/model.pt | 98fd6266bdc8926055c7f8b0dc2f37a4333fa92f55f336fb4cba4d3480324f86 | 65 | nan | 0.341766 | 0.378518 | 5334.545480 | complete |

## 5. Training reproducibility

torch CPU/CUDA init17;base17;continuation29;samplers17/29. Python/NumPy not explicitly seeded historically; unused by stochastic training path. Historical runtime versions not recorded; do not claim bitwise cross-version reproducibility.

New workers explicitly seed Python, NumPy and PyTorch CPU/CUDA at each stage. The existing CPU torch.Generator controls minibatch permutations; no DataLoader or worker seeds. Runtime versions/device/determinism/TF32 settings are saved in each seed runtime.json. Deterministic algorithms enabled;4torchthreads;cudnn benchmark disabled. These controls do not prove bitwise reproducibility across software versions/devices.

## 6. Per-seed results

Each flight first computes endpoint vector RMSE (no division by3); attitude uses RMS quaternion geodesic degrees. Then flights are averaged equally. Increments share the measured origin and therefore Δv/Δω error equals the endpoint velocity/rate error. Results below are never pooled-origin RMSE.

| seed | cohort | horizon_s | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | ALL | 0.100000 | 0.035362 | 0.140359 | 1.384508 | 0.543891 |
| 17 | ALL | 0.200000 | 0.070372 | 0.191337 | 2.153925 | 0.564247 |
| 17 | ALL | 0.500000 | 0.191818 | 0.351210 | 4.033740 | 0.598711 |
| 17 | ALL | 1.000000 | 0.471012 | 0.639739 | 6.743670 | 0.617962 |
| 17 | Sep17 | 0.100000 | 0.018732 | 0.149156 | 1.455401 | 0.527138 |
| 17 | Sep17 | 0.200000 | 0.040139 | 0.219564 | 2.315408 | 0.536216 |
| 17 | Sep17 | 0.500000 | 0.127037 | 0.361042 | 4.526343 | 0.562090 |
| 17 | Sep17 | 1.000000 | 0.357599 | 0.674454 | 7.675610 | 0.608763 |
| 17 | Sep7 | 0.100000 | 0.050145 | 0.132538 | 1.321493 | 0.558782 |
| 17 | Sep7 | 0.200000 | 0.097245 | 0.166247 | 2.010384 | 0.589163 |
| 17 | Sep7 | 0.500000 | 0.249401 | 0.342470 | 3.595870 | 0.631263 |
| 17 | Sep7 | 1.000000 | 0.571823 | 0.608881 | 5.915280 | 0.626138 |
| 23 | ALL | 0.100000 | 0.035487 | 0.149476 | 1.424629 | 0.543716 |
| 23 | ALL | 0.200000 | 0.070875 | 0.198765 | 2.206595 | 0.569534 |
| 23 | ALL | 0.500000 | 0.194473 | 0.365188 | 4.075688 | 0.597631 |
| 23 | ALL | 1.000000 | 0.481303 | 0.652809 | 6.542445 | 0.633125 |
| 23 | Sep17 | 0.100000 | 0.019090 | 0.158593 | 1.476157 | 0.524790 |
| 23 | Sep17 | 0.200000 | 0.041302 | 0.221340 | 2.385471 | 0.537434 |
| 23 | Sep17 | 0.500000 | 0.133924 | 0.389180 | 4.492540 | 0.564812 |
| 23 | Sep17 | 1.000000 | 0.384240 | 0.694538 | 7.172736 | 0.635915 |
| 23 | Sep7 | 0.100000 | 0.050062 | 0.141372 | 1.378827 | 0.560539 |
| 23 | Sep7 | 0.200000 | 0.097163 | 0.178698 | 2.047594 | 0.598067 |
| 23 | Sep7 | 0.500000 | 0.248293 | 0.343861 | 3.705154 | 0.626803 |
| 23 | Sep7 | 1.000000 | 0.567581 | 0.615717 | 5.982187 | 0.630645 |
| 42 | ALL | 0.100000 | 0.035170 | 0.144952 | 1.413026 | 0.548379 |
| 42 | ALL | 0.200000 | 0.069880 | 0.196195 | 2.228935 | 0.565069 |
| 42 | ALL | 0.500000 | 0.190896 | 0.355465 | 4.125764 | 0.593388 |
| 42 | ALL | 1.000000 | 0.469726 | 0.650697 | 6.791357 | 0.612132 |
| 42 | Sep17 | 0.100000 | 0.018519 | 0.152101 | 1.477926 | 0.530246 |
| 42 | Sep17 | 0.200000 | 0.039687 | 0.220679 | 2.427612 | 0.537454 |
| 42 | Sep17 | 0.500000 | 0.126894 | 0.370608 | 4.689644 | 0.555853 |
| 42 | Sep17 | 1.000000 | 0.362668 | 0.695512 | 7.847913 | 0.607418 |
| 42 | Sep7 | 0.100000 | 0.049970 | 0.138598 | 1.355336 | 0.564498 |
| 42 | Sep7 | 0.200000 | 0.096719 | 0.174431 | 2.052333 | 0.589615 |
| 42 | Sep7 | 0.500000 | 0.247787 | 0.342006 | 3.624537 | 0.626751 |
| 42 | Sep7 | 1.000000 | 0.564889 | 0.610861 | 5.852196 | 0.616323 |

## 7. Across-seed mean/std

Sample SD(ddof=1), min/max across exactly three seed-specific macro means. Full all-horizon statistics: multiseed_summary.csv. Primary500ms:

| cohort | metric | mean ± std | min | max |
| --- | --- | --- | --- | --- |
| ALL | attitude_error_deg | 4.078397 ± 0.046072 | 4.033740 | 4.125764 |
| ALL | body_rate_rmse_rad_s | 0.596577 ± 0.002814 | 0.593388 | 0.598711 |
| ALL | position_rmse_m | 0.192395 ± 0.001857 | 0.190896 | 0.194473 |
| ALL | velocity_rmse_m_s | 0.357288 ± 0.007165 | 0.351210 | 0.365188 |
| Sep17 | attitude_error_deg | 4.569509 ± 0.105404 | 4.492540 | 4.689644 |
| Sep17 | body_rate_rmse_rad_s | 0.560919 ± 0.004593 | 0.555853 | 0.564812 |
| Sep17 | position_rmse_m | 0.129285 ± 0.004018 | 0.126894 | 0.133924 |
| Sep17 | velocity_rmse_m_s | 0.373610 ± 0.014307 | 0.361042 | 0.389180 |
| Sep7 | attitude_error_deg | 3.641854 ± 0.056663 | 3.595870 | 3.705154 |
| Sep7 | body_rate_rmse_rad_s | 0.628272 ± 0.002590 | 0.626751 | 0.631263 |
| Sep7 | position_rmse_m | 0.248494 ± 0.000825 | 0.247787 | 0.249401 |
| Sep7 | velocity_rmse_m_s | 0.342779 ± 0.000965 | 0.342006 | 0.343861 |

## 8. Sep7/Sep17 robustness

| metric | consistent_gap_sign | seed17 | seed23 | seed42 |
| --- | --- | --- | --- | --- |
| position_rmse_m | True | -0.122364 | -0.114369 | -0.120893 |
| velocity_rmse_m_s | True | 0.018573 | 0.045320 | 0.028602 |
| attitude_error_deg | True | 0.930474 | 0.787386 | 1.065107 |
| body_rate_rmse_rad_s | True | -0.069172 | -0.061991 | -0.070898 |

Gap above is Sep17 minus Sep7. Inspect absolute cohort errors and seed SD together; only two sessions are represented.

## 9. Failure/anomaly check

All65epochs/7,215 updates per seed, finite logged losses and gradients, strict checkpoint loading, bitwise fixed normalization, identical origins/native timing, finite predictions and unit quaternions are checked. Future-label poisoning and500ms prefix parity pass. No automatic retry is permitted. final_val_loss is evaluated after final checkpoint solely for reporting.

## 10. Interpretation

Multi-seed robustness passed.

Max primary CV across metrics/cohorts: 3.83%; max1s CV: 4.64%. Greatest relative primary sensitivity: {'metric': 'velocity_rmse_m_s', 'cohort': 'Sep17', 'cv': 0.03829489013512584}.

Operational screen, frozen before new results: primary and1s CV≤20% in each cohort/metric; consistent500ms cohort-gap sign;1s error above100ms in all seeds/cohorts. This is a descriptive stability screen, not a significance test. Detailed1s/500ms and1s/100ms ratios: error_growth.csv.

## 11. Limitations

Only three seeds and17 correlated validation flights; no window-level confidence intervals. Baseline comparators remain seed17 pilots, so this does not establish statistically significant superiority over multi-seed B1/B3. B2 was designated main model after observing this same validation set; no unbiased test-generalization claim. Future logged commands are feedback-conditioned, not interventions. No long-rollout or simulation-readiness claim.

## 12. Sealed-test status

Sealed Sep8 and reserved Sep19 were not opened. Only train/validation Parquet allowlist and existing validation artifacts were accessed. No new seed selection, ablation, or test evaluation follows automatically. Next candidate only if robust: History Length Ablation; not executed.

Suggested commit: `feat: freeze Standard GRU three-seed robustness experiment`
