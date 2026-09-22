# Paper Step4 — History length ablation

## 1. Goal

Is past state/control context useful for short-horizon real-flight dynamics prediction, and where does its benefit saturate? Only H changes. H26 remains the paper main model regardless of this diagnostic.

## 2. History semantics

State tensor [N,H,12], control tensor [N,H,4], mask [N,H]. Samples range from origin-(H-1) through origin, inclusive. H26=current+25past; H1=current-only single-sample context. State/control use identical sample indices. Hidden state starts at zero per window, consumes allH history tokens including t; first rollout head then receives current state/control plus that hidden state, preserving the original architecture. H1 still has recurrent predicted-state transitions during rollout. Phase stays reanchored at origin; no re-anchoring at shortened-history start.

## 3. Fair comparison protocol

Every H uses exactly28,293 train and2,582 validation origins, with unchanged future targets/control tape/native dt. Short histories are suffix views of frozen H26 arrays; the trajectory object is shared. Full histories have no padding. Independent original-loader reconstruction of fixed origins checks suffix/phase/timestamp semantics. No extra short-history windows. Frozen buffers and increment scales/weights reused without fitting.

Normalization SHA256 `f10a427d04d54b98e5d1e4e0995b3b8562347f54313c90cd64147fa5ae2710c8`. Train origins `3a17f717f89367d0f673a2f73f3df246c4e332b9fac88385ce269df28854845b`. Validation origins `d9677c58fe0f9f23b12699a35c78e1b0961ddcb3ec5e5814e9f18392c501e3d3`. Source model commit `19546ada11c72f640ef31fc557664e6d02734c6f`. All12 models have21,383 trainable parameters.

## 4. Actual history durations

Span=(timestamp(origin)-timestamp(origin-H+1))*1e-6; H1=0. Native timestamps, no assumption of exact50Hz. Origin-weighted durations below; mean/median/sampleSD/p05/p95/min/max in seconds.

| history_steps | partition | cohort | mean_s | median_s | std_s | p05_s | p95_s | min_s | max_s | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | train | ALL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 28293 |
| 5 | train | ALL | 0.079972 | 0.079794 | 0.003402 | 0.078560 | 0.088750 | 0.068496 | 0.108060 | 28293 |
| 13 | train | ALL | 0.239958 | 0.239504 | 0.004361 | 0.236224 | 0.249455 | 0.229098 | 0.266141 | 28293 |
| 26 | train | ALL | 0.499970 | 0.499023 | 0.004051 | 0.492777 | 0.508943 | 0.488601 | 0.522376 | 28293 |
| 1 | validation | ALL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2582 |
| 1 | validation | Sep17 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1101 |
| 1 | validation | Sep7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1481 |
| 5 | validation | ALL | 0.080005 | 0.079718 | 0.003535 | 0.078525 | 0.088739 | 0.068666 | 0.090181 | 2582 |
| 5 | validation | Sep17 | 0.080059 | 0.079752 | 0.003512 | 0.078571 | 0.088744 | 0.068724 | 0.090051 | 1101 |
| 5 | validation | Sep7 | 0.079964 | 0.079656 | 0.003553 | 0.078491 | 0.088733 | 0.068666 | 0.090181 | 1481 |
| 13 | validation | ALL | 0.240032 | 0.239483 | 0.004496 | 0.236259 | 0.249377 | 0.229272 | 0.249795 | 2582 |
| 13 | validation | Sep17 | 0.240022 | 0.239470 | 0.004587 | 0.236226 | 0.249452 | 0.229272 | 0.249795 | 1101 |
| 13 | validation | Sep7 | 0.240039 | 0.239495 | 0.004429 | 0.236379 | 0.249251 | 0.229362 | 0.249694 | 1481 |
| 26 | validation | ALL | 0.499960 | 0.499077 | 0.004152 | 0.492750 | 0.508883 | 0.488856 | 0.509316 | 2582 |
| 26 | validation | Sep17 | 0.499969 | 0.499050 | 0.004178 | 0.492783 | 0.508938 | 0.488856 | 0.509316 | 1101 |
| 26 | validation | Sep7 | 0.499953 | 0.499135 | 0.004135 | 0.492735 | 0.508813 | 0.488919 | 0.509298 | 1481 |

## 5. Training protocol

Frozen Step3:40epochs AdamW lr3e-4, then25epochs lr5e-4 with optimizer reset; batch256, weight_decay1e-5, gradient_clip5; full50-step rollout plus frozen two-step increment objective; continuation adds0.2frequencyMSE. Last epoch65, no validation selection. Seeds17/23/42 use stage pairs17/29,23/35,42/54. New runs initialize exactly as H26 for the same seed. Runtime settings/versions in each artifact runtime.json. No retries based on unfavorable errors. H26 checkpoint/predictions/evaluation are reused, not trained or copied.

| history_steps | seed | checkpoint_path | sha256 | status | final_epoch | best_epoch | final_train_loss | final_val_loss | training_time_s | checkpoint_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 17 | artifacts/paper_history_length_ablation_v1/H1/seed17/model.pt | 60cd56efa894716297df992f2e41e56bbfdcba3de75913b42955c5902b89bad9 | complete | 65 | nan | 0.547452 | 0.571220 | 1791.238654 | nan |
| 1 | 23 | artifacts/paper_history_length_ablation_v1/H1/seed23/model.pt | b553932c90375126cb294777fc0516efdddd6847f97c33bc3f15bde8effb9346 | complete | 65 | nan | 0.544978 | 0.558998 | 1792.408602 | nan |
| 1 | 42 | artifacts/paper_history_length_ablation_v1/H1/seed42/model.pt | 3d2daa36cda770f01eb8c61fe49ccd950be44f6082567018e83926ecb370daf0 | complete | 65 | nan | 0.548483 | 0.565817 | 1649.146170 | nan |
| 5 | 17 | artifacts/paper_history_length_ablation_v1/H5/seed17/model.pt | c80b540cf053530837edfe7f89102d867035255ed848d6118d9eee5112c2ccc8 | complete | 65 | nan | 0.380259 | 0.428213 | 1659.158868 | nan |
| 5 | 23 | artifacts/paper_history_length_ablation_v1/H5/seed23/model.pt | bc5dffb44a77478c416306c3aae8bff07ea85055ccd8f1ae5b76595408f04038 | complete | 65 | nan | 0.382470 | 0.426009 | 1648.463519 | nan |
| 5 | 42 | artifacts/paper_history_length_ablation_v1/H5/seed42/model.pt | 2f9daae4538b3d5d47cf368607b25ae2b31dd12509f14b71cf4c37ed5dfe6b83 | complete | 65 | nan | 0.385427 | 0.419572 | 1626.910520 | nan |
| 13 | 17 | artifacts/paper_history_length_ablation_v1/H13/seed17/model.pt | 1f685777f019c3f86eef07f4e304cc19ca79959d26f9aa1cb94f555735103c6e | complete | 65 | nan | 0.349018 | 0.396036 | 1673.376818 | nan |
| 13 | 23 | artifacts/paper_history_length_ablation_v1/H13/seed23/model.pt | 33dd843386802c31982a3092d43a6f5a3beec34ad6ea38c26b7ac0bb434dc285 | complete | 65 | nan | 0.350801 | 0.385978 | 1662.297813 | nan |
| 13 | 42 | artifacts/paper_history_length_ablation_v1/H13/seed42/model.pt | ed7b81e19e28ec30ef7751767a7add74a984b7f30b2fe7e0b47b0ad5bf07fe93 | complete | 65 | nan | 0.349922 | 0.385437 | 1627.365645 | nan |
| 26 | 17 | artifacts/paper_baseline_comparison_v1/B2_StandardGRU.pt | f9895418b7dd2a2757dde8d740109ab42dae258c593fd5902f0076a3faeb0e26 | reused | 65 | nan | 0.342859 | 0.375745 | 1816.394105 | f9895418b7dd2a2757dde8d740109ab42dae258c593fd5902f0076a3faeb0e26 |
| 26 | 23 | artifacts/paper_standard_gru_multiseed_v1/seed23/model.pt | 1cfbb865db55e3fc7c248eff4de95892fa9ee9155ee8e81c0318a9021b7d8b03 | reused | 65 | nan | 0.342157 | 0.384439 | 5353.385571 | 1cfbb865db55e3fc7c248eff4de95892fa9ee9155ee8e81c0318a9021b7d8b03 |
| 26 | 42 | artifacts/paper_standard_gru_multiseed_v1/seed42/model.pt | 98fd6266bdc8926055c7f8b0dc2f37a4333fa92f55f336fb4cba4d3480324f86 | reused | 65 | nan | 0.341766 | 0.378518 | 5334.545480 | 98fd6266bdc8926055c7f8b0dc2f37a4333fa92f55f336fb4cba4d3480324f86 |

## 6. Main results

Per flight, compute endpoint vector RMSE (not divided by3), or RMS quaternion geodesic degrees. Average flights equally, then compute mean and sampleSD across three seeds. Delta-v/delta-omega endpoint increment errors equal velocity/body-rate endpoint errors because the same measured origin is subtracted. Primary500ms:

| cohort | history_steps | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- |
| ALL | 1 | 0.21209 ± 0.00313 | 0.39534 ± 0.01053 | 4.56002 ± 0.03141 | 0.64112 ± 0.00556 |
| ALL | 5 | 0.19841 ± 0.00100 | 0.38624 ± 0.00314 | 4.38324 ± 0.00604 | 0.62085 ± 0.00695 |
| ALL | 13 | 0.19378 ± 0.00143 | 0.36470 ± 0.00519 | 4.13435 ± 0.02738 | 0.59984 ± 0.00620 |
| ALL | 26 | 0.19240 ± 0.00186 | 0.35729 ± 0.00716 | 4.07840 ± 0.04607 | 0.59658 ± 0.00281 |
| Sep17 | 1 | 0.15015 ± 0.00258 | 0.39800 ± 0.01448 | 5.33518 ± 0.00279 | 0.61388 ± 0.00711 |
| Sep17 | 5 | 0.13298 ± 0.00110 | 0.39141 ± 0.00983 | 4.91406 ± 0.08295 | 0.58445 ± 0.00256 |
| Sep17 | 13 | 0.12945 ± 0.00049 | 0.37664 ± 0.00331 | 4.61853 ± 0.07827 | 0.56284 ± 0.00420 |
| Sep17 | 26 | 0.12928 ± 0.00402 | 0.37361 ± 0.01431 | 4.56951 ± 0.10540 | 0.56092 ± 0.00459 |
| Sep7 | 1 | 0.26715 ± 0.00423 | 0.39297 ± 0.01489 | 3.87099 ± 0.06178 | 0.66532 ± 0.00477 |
| Sep7 | 5 | 0.25656 ± 0.00202 | 0.38165 ± 0.00583 | 3.91140 ± 0.06269 | 0.65321 ± 0.01108 |
| Sep7 | 13 | 0.25096 ± 0.00314 | 0.35409 ± 0.01240 | 3.70397 ± 0.02494 | 0.63273 ± 0.00869 |
| Sep7 | 26 | 0.24849 ± 0.00083 | 0.34278 ± 0.00097 | 3.64185 ± 0.05666 | 0.62827 ± 0.00259 |

## 7. Error vs history length

Four error_vs_history figures use500ms and ±1seedSD. Positive relative gain means shorter-reference error decreased. Largest adjacent-range mean improvement for each dynamics metric (negative values would mean regression):

| metric | reference_history_steps | history_steps | relative_improvement_pct |
| --- | --- | --- | --- |
| attitude_error_deg | 5 | 13 | 5.678239 |
| body_rate_rmse_rad_s | 5 | 13 | 3.385088 |
| velocity_rmse_m_s | 5 | 13 | 5.578114 |

## 8. Error vs prediction horizon

Complete nominal0.1/0.2/0.5/1.0s mean±SD below. Native5/10/25/50steps, same measured dt as Step3. History-by-horizon figures separately plot velocity/attitude/body rate.

| cohort | history_steps | horizon_s | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 1 | 0.100000 | 0.04796 ± 0.00009 | 0.46441 ± 0.00209 | 1.89822 ± 0.00250 | 0.68311 ± 0.00517 |
| ALL | 1 | 0.200000 | 0.09296 ± 0.00059 | 0.34250 ± 0.00346 | 2.73503 ± 0.02336 | 0.68596 ± 0.00058 |
| ALL | 1 | 0.500000 | 0.21209 ± 0.00313 | 0.39534 ± 0.01053 | 4.56002 ± 0.03141 | 0.64112 ± 0.00556 |
| ALL | 1 | 1.000000 | 0.48900 ± 0.01549 | 0.66462 ± 0.02200 | 6.93358 ± 0.15846 | 0.63876 ± 0.00303 |
| ALL | 5 | 0.100000 | 0.03624 ± 0.00011 | 0.19257 ± 0.00257 | 1.51298 ± 0.01264 | 0.58271 ± 0.00669 |
| ALL | 5 | 0.200000 | 0.07331 ± 0.00007 | 0.23870 ± 0.00719 | 2.38046 ± 0.03169 | 0.59816 ± 0.00545 |
| ALL | 5 | 0.500000 | 0.19841 ± 0.00100 | 0.38624 ± 0.00314 | 4.38324 ± 0.00604 | 0.62085 ± 0.00695 |
| ALL | 5 | 1.000000 | 0.49269 ± 0.00959 | 0.69965 ± 0.02382 | 6.97329 ± 0.11653 | 0.64509 ± 0.00496 |
| ALL | 13 | 0.100000 | 0.03552 ± 0.00008 | 0.15268 ± 0.00325 | 1.42269 ± 0.00757 | 0.55168 ± 0.00273 |
| ALL | 13 | 0.200000 | 0.07089 ± 0.00019 | 0.20238 ± 0.00491 | 2.23638 ± 0.01682 | 0.57292 ± 0.00420 |
| ALL | 13 | 0.500000 | 0.19378 ± 0.00143 | 0.36470 ± 0.00519 | 4.13435 ± 0.02738 | 0.59984 ± 0.00620 |
| ALL | 13 | 1.000000 | 0.48365 ± 0.01274 | 0.67312 ± 0.02944 | 6.77160 ± 0.17485 | 0.62419 ± 0.00661 |
| ALL | 26 | 0.100000 | 0.03534 ± 0.00016 | 0.14493 ± 0.00456 | 1.40739 ± 0.02065 | 0.54533 ± 0.00264 |
| ALL | 26 | 0.200000 | 0.07038 ± 0.00050 | 0.19543 ± 0.00377 | 2.19649 ± 0.03851 | 0.56628 ± 0.00285 |
| ALL | 26 | 0.500000 | 0.19240 ± 0.00186 | 0.35729 ± 0.00716 | 4.07840 ± 0.04607 | 0.59658 ± 0.00281 |
| ALL | 26 | 1.000000 | 0.47401 ± 0.00635 | 0.64775 ± 0.00702 | 6.69249 ± 0.13211 | 0.62107 ± 0.01084 |
| Sep17 | 1 | 0.100000 | 0.03465 ± 0.00017 | 0.49250 ± 0.00503 | 2.13408 ± 0.00198 | 0.67537 ± 0.00653 |
| Sep17 | 1 | 0.200000 | 0.06872 ± 0.00053 | 0.35633 ± 0.00732 | 3.11461 ± 0.02113 | 0.67842 ± 0.00471 |
| Sep17 | 1 | 0.500000 | 0.15015 ± 0.00258 | 0.39800 ± 0.01448 | 5.33518 ± 0.00279 | 0.61388 ± 0.00711 |
| Sep17 | 1 | 1.000000 | 0.37753 ± 0.01299 | 0.69675 ± 0.02699 | 8.00463 ± 0.20117 | 0.62689 ± 0.00925 |
| Sep17 | 5 | 0.100000 | 0.01933 ± 0.00006 | 0.19559 ± 0.00399 | 1.58789 ± 0.02502 | 0.55925 ± 0.00587 |
| Sep17 | 5 | 0.200000 | 0.04295 ± 0.00051 | 0.25662 ± 0.00551 | 2.57682 ± 0.01386 | 0.56845 ± 0.00407 |
| Sep17 | 5 | 0.500000 | 0.13298 ± 0.00110 | 0.39141 ± 0.00983 | 4.91406 ± 0.08295 | 0.58445 ± 0.00256 |
| Sep17 | 5 | 1.000000 | 0.37839 ± 0.00517 | 0.72860 ± 0.01692 | 7.83663 ± 0.26886 | 0.63356 ± 0.00497 |
| Sep17 | 13 | 0.100000 | 0.01899 ± 0.00012 | 0.15943 ± 0.00296 | 1.50253 ± 0.02033 | 0.53439 ± 0.00086 |
| Sep17 | 13 | 0.200000 | 0.04075 ± 0.00027 | 0.22459 ± 0.00403 | 2.42421 ± 0.02938 | 0.54260 ± 0.00227 |
| Sep17 | 13 | 0.500000 | 0.12945 ± 0.00049 | 0.37664 ± 0.00331 | 4.61853 ± 0.07827 | 0.56284 ± 0.00420 |
| Sep17 | 13 | 1.000000 | 0.37449 ± 0.00423 | 0.70913 ± 0.01385 | 7.57731 ± 0.36949 | 0.61880 ± 0.00711 |
| Sep17 | 26 | 0.100000 | 0.01878 ± 0.00029 | 0.15328 ± 0.00483 | 1.46983 ± 0.01253 | 0.52739 ± 0.00274 |
| Sep17 | 26 | 0.200000 | 0.04038 ± 0.00083 | 0.22053 ± 0.00090 | 2.37616 ± 0.05668 | 0.53703 ± 0.00071 |
| Sep17 | 26 | 0.500000 | 0.12928 ± 0.00402 | 0.37361 ± 0.01431 | 4.56951 ± 0.10540 | 0.56092 ± 0.00459 |
| Sep17 | 26 | 1.000000 | 0.36817 ± 0.01415 | 0.68817 ± 0.01189 | 7.56542 ± 0.35082 | 0.61737 ± 0.01608 |
| Sep7 | 1 | 0.100000 | 0.05979 ± 0.00010 | 0.43944 ± 0.00144 | 1.68856 ± 0.00299 | 0.69000 ± 0.00407 |
| Sep7 | 1 | 0.200000 | 0.11452 ± 0.00110 | 0.33020 ± 0.00277 | 2.39762 ± 0.02537 | 0.69266 ± 0.00383 |
| Sep7 | 1 | 0.500000 | 0.26715 ± 0.00423 | 0.39297 ± 0.01489 | 3.87099 ± 0.06178 | 0.66532 ± 0.00477 |
| Sep7 | 1 | 1.000000 | 0.58808 ± 0.01991 | 0.63605 ± 0.01869 | 5.98155 ± 0.13781 | 0.64932 ± 0.00251 |
| Sep7 | 5 | 0.100000 | 0.05127 ± 0.00025 | 0.18989 ± 0.00237 | 1.44640 ± 0.02855 | 0.60356 ± 0.01058 |
| Sep7 | 5 | 0.200000 | 0.10029 ± 0.00054 | 0.22276 ± 0.00975 | 2.20592 ± 0.05377 | 0.62456 ± 0.00676 |
| Sep7 | 5 | 0.500000 | 0.25656 ± 0.00202 | 0.38165 ± 0.00583 | 3.91140 ± 0.06269 | 0.65321 ± 0.01108 |
| Sep7 | 5 | 1.000000 | 0.59429 ± 0.01428 | 0.67391 ± 0.03176 | 6.20589 ± 0.02432 | 0.65535 ± 0.00551 |
| Sep7 | 13 | 0.100000 | 0.05022 ± 0.00013 | 0.14669 ± 0.00371 | 1.35172 ± 0.00448 | 0.56705 ± 0.00465 |
| Sep7 | 13 | 0.200000 | 0.09768 ± 0.00048 | 0.18263 ± 0.00763 | 2.06943 ± 0.00573 | 0.59986 ± 0.00616 |
| Sep7 | 13 | 0.500000 | 0.25096 ± 0.00314 | 0.35409 ± 0.01240 | 3.70397 ± 0.02494 | 0.63273 ± 0.00869 |
| Sep7 | 13 | 1.000000 | 0.58068 ± 0.02071 | 0.64111 ± 0.04350 | 6.05542 ± 0.04352 | 0.62899 ± 0.00669 |
| Sep7 | 26 | 0.100000 | 0.05006 ± 0.00009 | 0.13750 ± 0.00452 | 1.35189 ± 0.02882 | 0.56127 ± 0.00293 |
| Sep7 | 26 | 0.200000 | 0.09704 ± 0.00028 | 0.17313 ± 0.00633 | 2.03677 ± 0.02297 | 0.59228 ± 0.00502 |
| Sep7 | 26 | 0.500000 | 0.24849 ± 0.00083 | 0.34278 ± 0.00097 | 3.64185 ± 0.05666 | 0.62827 ± 0.00259 |
| Sep7 | 26 | 1.000000 | 0.56810 ± 0.00350 | 0.61182 ± 0.00352 | 5.91655 ± 0.06500 | 0.62437 ± 0.00732 |

## 9. Cohort consistency

Cohort panels and ALL/Sep7/Sep17 tables preserve session differences; rankings for each metric/horizon/cohort are in interpretation.json. Per-flight paired differences first average the three seed-specific per-flight RMSEs; no windows treated as independent samples. Primary ALL flight directions:

| reference_history_steps | history_steps | cohort | horizon_s | metric | n_flights | flights_improved | flights_worse | mean_paired_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | ALL | 0.500000 | attitude_error_deg | 17 | 11 | 6 | -0.176777 |
| 1 | 5 | ALL | 0.500000 | body_rate_rmse_rad_s | 17 | 14 | 3 | -0.020261 |
| 1 | 5 | ALL | 0.500000 | velocity_rmse_m_s | 17 | 12 | 5 | -0.009094 |
| 1 | 13 | ALL | 0.500000 | attitude_error_deg | 17 | 17 | 0 | -0.425668 |
| 1 | 13 | ALL | 0.500000 | body_rate_rmse_rad_s | 17 | 16 | 1 | -0.041277 |
| 1 | 13 | ALL | 0.500000 | velocity_rmse_m_s | 17 | 16 | 1 | -0.030639 |
| 1 | 26 | ALL | 0.500000 | attitude_error_deg | 17 | 17 | 0 | -0.481621 |
| 1 | 26 | ALL | 0.500000 | body_rate_rmse_rad_s | 17 | 17 | 0 | -0.044539 |
| 1 | 26 | ALL | 0.500000 | velocity_rmse_m_s | 17 | 16 | 1 | -0.038050 |
| 5 | 13 | ALL | 0.500000 | attitude_error_deg | 17 | 17 | 0 | -0.248891 |
| 5 | 13 | ALL | 0.500000 | body_rate_rmse_rad_s | 17 | 17 | 0 | -0.021016 |
| 5 | 13 | ALL | 0.500000 | velocity_rmse_m_s | 17 | 15 | 2 | -0.021545 |
| 5 | 26 | ALL | 0.500000 | attitude_error_deg | 17 | 17 | 0 | -0.304844 |
| 5 | 26 | ALL | 0.500000 | body_rate_rmse_rad_s | 17 | 17 | 0 | -0.024278 |
| 5 | 26 | ALL | 0.500000 | velocity_rmse_m_s | 17 | 17 | 0 | -0.028955 |
| 13 | 26 | ALL | 0.500000 | attitude_error_deg | 17 | 14 | 3 | -0.055953 |
| 13 | 26 | ALL | 0.500000 | body_rate_rmse_rad_s | 17 | 11 | 6 | -0.003261 |
| 13 | 26 | ALL | 0.500000 | velocity_rmse_m_s | 17 | 14 | 3 | -0.007410 |

## 10. Seed robustness

Full coefficient-of-variation and min/max: seed_sensitivity.csv. Paired seed directions below count how often the larger-H error decreases; each seed uses the same initialization/minibatch seed policy across H.

| reference_history_steps | history_steps | cohort | horizon_s | metric | n_seeds | seeds_improved | mean_paired_difference | min_paired_difference | max_paired_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | ALL | 0.500000 | velocity_rmse_m_s | 3 | 3 | -0.009094 | -0.016474 | -0.000911 |
| 1 | 5 | ALL | 0.500000 | attitude_error_deg | 3 | 3 | -0.176777 | -0.197076 | -0.142060 |
| 1 | 5 | ALL | 0.500000 | body_rate_rmse_rad_s | 3 | 3 | -0.020261 | -0.024112 | -0.018325 |
| 1 | 13 | ALL | 0.500000 | velocity_rmse_m_s | 3 | 3 | -0.030639 | -0.034291 | -0.023810 |
| 1 | 13 | ALL | 0.500000 | attitude_error_deg | 3 | 3 | -0.425668 | -0.462451 | -0.358804 |
| 1 | 13 | ALL | 0.500000 | body_rate_rmse_rad_s | 3 | 3 | -0.041277 | -0.047193 | -0.031679 |
| 1 | 26 | ALL | 0.500000 | velocity_rmse_m_s | 3 | 3 | -0.038050 | -0.052722 | -0.018398 |
| 1 | 26 | ALL | 0.500000 | attitude_error_deg | 3 | 3 | -0.481621 | -0.547308 | -0.398149 |
| 1 | 26 | ALL | 0.500000 | body_rate_rmse_rad_s | 3 | 3 | -0.044539 | -0.049515 | -0.037491 |
| 13 | 26 | ALL | 0.500000 | velocity_rmse_m_s | 3 | 2 | -0.007410 | -0.018905 | 0.005412 |
| 13 | 26 | ALL | 0.500000 | attitude_error_deg | 3 | 3 | -0.055953 | -0.091558 | -0.036955 |
| 13 | 26 | ALL | 0.500000 | body_rate_rmse_rad_s | 3 | 2 | -0.003261 | -0.005812 | 0.000583 |
| 5 | 13 | ALL | 0.500000 | velocity_rmse_m_s | 3 | 3 | -0.021545 | -0.024393 | -0.017343 |
| 5 | 13 | ALL | 0.500000 | attitude_error_deg | 3 | 3 | -0.248891 | -0.265376 | -0.216744 |
| 5 | 13 | ALL | 0.500000 | body_rate_rmse_rad_s | 3 | 3 | -0.021016 | -0.026614 | -0.013354 |
| 5 | 26 | ALL | 0.500000 | velocity_rmse_m_s | 3 | 3 | -0.028955 | -0.036249 | -0.017487 |
| 5 | 26 | ALL | 0.500000 | attitude_error_deg | 3 | 3 | -0.304844 | -0.356111 | -0.256089 |
| 5 | 26 | ALL | 0.500000 | body_rate_rmse_rad_s | 3 | 3 | -0.024278 | -0.031169 | -0.019166 |

Largest500ms relative seed variation: {'history_steps': 26, 'cohort': 'Sep17', 'metric': 'velocity_rmse_m_s', 'cv': 0.03829489013512584}. No claim that smaller mean automatically implies stable improvement.

## 11. Relative gains

Formula100*(reference_error-error_H)/reference_error on three-seed macro means, not mean of per-window ratios. Includes H1 comparisons, adjacent-range gains, and H26 relative to H13; all horizons/cohorts in relative_improvement.csv.

| reference_history_steps | history_steps | metric | relative_improvement_pct |
| --- | --- | --- | --- |
| 1 | 5 | attitude_error_deg | 3.876682 |
| 1 | 13 | attitude_error_deg | 9.334793 |
| 1 | 26 | attitude_error_deg | 10.561824 |
| 13 | 26 | attitude_error_deg | 1.353364 |
| 5 | 13 | attitude_error_deg | 5.678239 |
| 5 | 26 | attitude_error_deg | 6.954756 |
| 1 | 5 | body_rate_rmse_rad_s | 3.160266 |
| 1 | 13 | body_rate_rmse_rad_s | 6.438377 |
| 1 | 26 | body_rate_rmse_rad_s | 6.947074 |
| 13 | 26 | body_rate_rmse_rad_s | 0.543703 |
| 5 | 13 | body_rate_rmse_rad_s | 3.385088 |
| 5 | 26 | body_rate_rmse_rad_s | 3.910387 |
| 1 | 5 | velocity_rmse_m_s | 2.300359 |
| 1 | 13 | velocity_rmse_m_s | 7.750156 |
| 1 | 26 | velocity_rmse_m_s | 9.624606 |
| 13 | 26 | velocity_rmse_m_s | 2.031928 |
| 5 | 13 | velocity_rmse_m_s | 5.578114 |
| 5 | 26 | velocity_rmse_m_s | 7.496698 |

## 12. Interpretation

Outcome Mixed: Metric-dependent differences do not cleanly support a single A/B/C outcome; preserve the full results and keep H26 main model unchanged.

Best primary ALL history per metric: {'attitude_error_deg': 26, 'body_rate_rmse_rad_s': 26, 'position_rmse_m': 26, 'velocity_rmse_m_s': 26}. H13->H26 gains(%): {'attitude_error_deg': 1.3533644865641115, 'body_rate_rmse_rad_s': 0.5437030753512222, 'velocity_rmse_m_s': 2.031927792357414}.

Classification is only a descriptive aid: a predeclared2% mean-error band describes near-equal performance, not statistical equivalence. OutcomeA also requires all3seeds to favorH26 on all3dynamics metrics. Mixed tradeoffs remain mixed. Review cohort and flight directions rather than claiming a uniformly optimal H. Do not switch the main model automatically.

## 13. Limitations

Only3seeds,17flights and2open validation sessions; no window-level significance tests. History effects can mix physical memory with estimator/filter state and closed-loop correlations. Logged future controls are feedback-conditioned; this is prediction ablation, not causal actuator identification. Shorter history does not remove recurrence from future rollout. Phase pose provenance remains unchanged/unconfirmed. Reused H26 runtime provenance limitations from Step3 still apply. Representative seed17 window is frozen from Step1, never chosen by error; Euler curves illustrate orientation, while all primary attitude metrics are geodesic.

## 14. Sealed-test status

Sealed Sep8 and reserved Sep19 remain unopened. Only explicit train/validation files and previously opened validation artifacts are used. No phase/frequency/loss ablation or new model selection follows. H26 remains frozen pending user decision.

Suggested commit: `feat: add frozen three-seed history length ablation`
