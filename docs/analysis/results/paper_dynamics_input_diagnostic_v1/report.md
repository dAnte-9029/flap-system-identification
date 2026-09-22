# Frozen Future-Control Information Diagnostic

## Research question and protocol

Under identical initial states, H26 context and frozen GRU64 dynamics, does replaying realized future commands predict observed motion better than holding the origin command? No training or checkpoint selection. All3seeds17/23/42 retained. Full protocol is frozen before Hold inference, including train-only thresholds and case identities.

Actual reuses the original4-channel future tape; Hold repeats its first command50times. Controls are normalized motor/left/right/rudder allocation commands before PWM; not physical angles/frequency. History, true origin, zero hidden initialization before history encoding, origin phase reference, future labels and native dt are unchanged.

## Coverage and control grouping

| horizon_s | step | train_q25 | train_q75 | grouping_available | n_train_origins |
| --- | --- | --- | --- | --- | --- |
| 0.100000 | 5 | 0.069504 | 0.125937 | True | 28293 |
| 0.200000 | 10 | 0.110089 | 0.193606 | True | 28293 |
| 0.500000 | 25 | 0.173570 | 0.321677 | True | 28293 |
| 1.000000 | 50 | 0.254164 | 0.502741 | True | 28293 |

| step | horizon_s | group | cohort | grouping_available | n_origins | n_flights | E_mean | E_std | E_min | E_p05 | E_p25 | E_median | E_p75 | E_p95 | E_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 0.500000 | ALL | ALL | True | 2582 | 17 | 0.243320 | 0.148965 | 0.053858 | 0.100108 | 0.148140 | 0.211485 | 0.283405 | 0.530446 | 1.385652 |
| 25 | 0.500000 | ALL | Sep7 | True | 1481 | 9 | 0.215088 | 0.135728 | 0.053858 | 0.091273 | 0.129708 | 0.181135 | 0.252447 | 0.488635 | 1.356100 |
| 25 | 0.500000 | ALL | Sep17 | True | 1101 | 8 | 0.281296 | 0.157366 | 0.084109 | 0.127002 | 0.191051 | 0.241152 | 0.320538 | 0.574079 | 1.385652 |
| 25 | 0.500000 | low | ALL | True | 888 | 17 | 0.129367 | 0.026461 | 0.053858 | 0.084487 | 0.109886 | 0.130879 | 0.150085 | 0.169141 | 0.173434 |
| 25 | 0.500000 | low | Sep7 | True | 687 | 9 | 0.126095 | 0.026921 | 0.053858 | 0.081760 | 0.105036 | 0.126867 | 0.147372 | 0.169107 | 0.173434 |
| 25 | 0.500000 | low | Sep17 | True | 201 | 8 | 0.140551 | 0.021376 | 0.084109 | 0.103814 | 0.124389 | 0.142983 | 0.157723 | 0.169706 | 0.173114 |
| 25 | 0.500000 | middle | ALL | True | 1240 | 17 | 0.235321 | 0.039742 | 0.173764 | 0.179171 | 0.202727 | 0.230001 | 0.265175 | 0.307120 | 0.321499 |
| 25 | 0.500000 | middle | Sep7 | True | 609 | 9 | 0.231349 | 0.040459 | 0.173764 | 0.178931 | 0.196650 | 0.221687 | 0.263506 | 0.305914 | 0.320939 |
| 25 | 0.500000 | middle | Sep17 | True | 631 | 8 | 0.239154 | 0.038684 | 0.173857 | 0.179669 | 0.209307 | 0.235392 | 0.267420 | 0.307588 | 0.321499 |
| 25 | 0.500000 | high | ALL | True | 454 | 17 | 0.488053 | 0.187310 | 0.321971 | 0.328017 | 0.361846 | 0.420190 | 0.554363 | 0.842672 | 1.385652 |
| 25 | 0.500000 | high | Sep7 | True | 185 | 9 | 0.492032 | 0.180299 | 0.321971 | 0.327097 | 0.358649 | 0.427928 | 0.574782 | 0.821190 | 1.356100 |
| 25 | 0.500000 | high | Sep17 | True | 269 | 8 | 0.485316 | 0.192268 | 0.322593 | 0.329700 | 0.363653 | 0.414407 | 0.535945 | 0.854780 | 1.385652 |

E_K is RMS across time and channels of (u[t+k]-u[t])/frozen_control_std, k=0..K-1. Train25th/75th linear quantiles define low<=q25, middle(q25,q75], high>q75. Degenerate thresholds disable grouping. Horizon-specific groups may differ; all evolution group plots keep the500ms membership fixed. Full per-flight zero-inclusive coverage: control_group_per_flight.csv. These are offline groups using realized future commands, not an online foreknowledge classifier.

## Primary500ms results

| condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- |
| Actual | 0.19240 ± 0.00186 | 0.35729 ± 0.00716 | 4.07840 ± 0.04607 | 0.59658 ± 0.00281 |
| Hold | 0.20259 ± 0.00228 | 0.43269 ± 0.00802 | 4.70068 ± 0.06310 | 0.64385 ± 0.00929 |

| group | condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- |
| high | Actual | 0.22688 ± 0.00220 | 0.50810 ± 0.01441 | 5.06294 ± 0.02050 | 0.66879 ± 0.00277 |
| high | Hold | 0.24185 ± 0.00245 | 0.61156 ± 0.00777 | 6.14294 ± 0.00360 | 0.72290 ± 0.00826 |
| low | Actual | 0.18298 ± 0.00320 | 0.29291 ± 0.01103 | 3.40530 ± 0.04863 | 0.55851 ± 0.00483 |
| low | Hold | 0.18998 ± 0.00422 | 0.33363 ± 0.01735 | 3.76577 ± 0.10836 | 0.58094 ± 0.01154 |
| middle | Actual | 0.18354 ± 0.00133 | 0.32167 ± 0.00384 | 3.97694 ± 0.04722 | 0.57555 ± 0.00416 |
| middle | Hold | 0.19231 ± 0.00185 | 0.39621 ± 0.00539 | 4.51025 ± 0.06788 | 0.62876 ± 0.00960 |

## Paired information gains

| cohort | group | metric | actual | hold | absolute_gain | relative_gain_pct | seeds_improved | flights_improved | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | attitude_error_deg | 4.078397 | 4.700676 | 0.622279 | 13.238070 | 3.000000 | 17.000000 | 17.000000 | 2582 |
| Sep17 | ALL | attitude_error_deg | 4.569509 | 5.323367 | 0.753858 | 14.161294 | 3.000000 | 8.000000 | 8.000000 | 1101 |
| Sep7 | ALL | attitude_error_deg | 3.641854 | 4.147173 | 0.505320 | 12.184682 | 3.000000 | 9.000000 | 9.000000 | 1481 |
| ALL | ALL | body_rate_rmse_rad_s | 0.596577 | 0.643854 | 0.047277 | 7.342886 | 3.000000 | 17.000000 | 17.000000 | 2582 |
| Sep17 | ALL | body_rate_rmse_rad_s | 0.560919 | 0.636586 | 0.075667 | 11.886352 | 3.000000 | 8.000000 | 8.000000 | 1101 |
| Sep7 | ALL | body_rate_rmse_rad_s | 0.628272 | 0.650315 | 0.022043 | 3.389512 | 3.000000 | 9.000000 | 9.000000 | 1481 |
| ALL | ALL | velocity_rmse_m_s | 0.357288 | 0.432695 | 0.075407 | 17.427311 | 3.000000 | 17.000000 | 17.000000 | 2582 |
| Sep17 | ALL | velocity_rmse_m_s | 0.373610 | 0.487218 | 0.113608 | 23.317730 | 3.000000 | 8.000000 | 8.000000 | 1101 |
| Sep7 | ALL | velocity_rmse_m_s | 0.342779 | 0.384229 | 0.041450 | 10.787936 | 3.000000 | 9.000000 | 9.000000 | 1481 |
| ALL | high | attitude_error_deg | 5.062936 | 6.142940 | 1.080004 | 17.581230 | 3.000000 | 17.000000 | 17.000000 | 454 |
| Sep17 | high | attitude_error_deg | 5.771198 | 7.002653 | 1.231456 | 17.585558 | 3.000000 | 8.000000 | 8.000000 | 269 |
| Sep7 | high | attitude_error_deg | 4.433369 | 5.378751 | 0.945381 | 17.576222 | 3.000000 | 9.000000 | 9.000000 | 185 |
| ALL | high | body_rate_rmse_rad_s | 0.668793 | 0.722897 | 0.054104 | 7.484390 | 3.000000 | 14.000000 | 17.000000 | 454 |
| Sep17 | high | body_rate_rmse_rad_s | 0.626949 | 0.709076 | 0.082128 | 11.582343 | 3.000000 | 7.000000 | 8.000000 | 269 |
| Sep7 | high | body_rate_rmse_rad_s | 0.705987 | 0.735182 | 0.029195 | 3.971111 | 3.000000 | 7.000000 | 9.000000 | 185 |
| ALL | high | velocity_rmse_m_s | 0.508097 | 0.611564 | 0.103467 | 16.918454 | 3.000000 | 15.000000 | 17.000000 | 454 |
| Sep17 | high | velocity_rmse_m_s | 0.506143 | 0.667533 | 0.161390 | 24.177080 | 3.000000 | 8.000000 | 8.000000 | 269 |
| Sep7 | high | velocity_rmse_m_s | 0.509834 | 0.561814 | 0.051980 | 9.252222 | 3.000000 | 7.000000 | 9.000000 | 185 |
| ALL | low | attitude_error_deg | 3.405298 | 3.765772 | 0.360474 | 9.572378 | 3.000000 | 17.000000 | 17.000000 | 888 |
| Sep17 | low | attitude_error_deg | 3.461300 | 3.794084 | 0.332784 | 8.771136 | 3.000000 | 8.000000 | 8.000000 | 201 |
| Sep7 | low | attitude_error_deg | 3.355519 | 3.740606 | 0.385087 | 10.294775 | 3.000000 | 9.000000 | 9.000000 | 687 |
| ALL | low | body_rate_rmse_rad_s | 0.558512 | 0.580943 | 0.022431 | 3.861101 | 3.000000 | 14.000000 | 17.000000 | 888 |
| Sep17 | low | body_rate_rmse_rad_s | 0.498926 | 0.523763 | 0.024837 | 4.742085 | 3.000000 | 6.000000 | 8.000000 | 201 |
| Sep7 | low | body_rate_rmse_rad_s | 0.611477 | 0.631769 | 0.020292 | 3.211881 | 3.000000 | 8.000000 | 9.000000 | 687 |
| ALL | low | velocity_rmse_m_s | 0.292911 | 0.333635 | 0.040724 | 12.206036 | 3.000000 | 17.000000 | 17.000000 | 888 |
| Sep17 | low | velocity_rmse_m_s | 0.303337 | 0.360247 | 0.056910 | 15.797437 | 3.000000 | 8.000000 | 8.000000 | 201 |
| Sep7 | low | velocity_rmse_m_s | 0.283644 | 0.309979 | 0.026336 | 8.495994 | 3.000000 | 9.000000 | 9.000000 | 687 |
| ALL | middle | attitude_error_deg | 3.976936 | 4.510248 | 0.533311 | 11.824439 | 3.000000 | 17.000000 | 17.000000 | 1240 |
| Sep17 | middle | attitude_error_deg | 4.269051 | 4.884423 | 0.615372 | 12.598667 | 3.000000 | 8.000000 | 8.000000 | 631 |
| Sep7 | middle | attitude_error_deg | 3.717279 | 4.177648 | 0.460369 | 11.019807 | 3.000000 | 9.000000 | 9.000000 | 609 |
| ALL | middle | body_rate_rmse_rad_s | 0.575552 | 0.628762 | 0.053209 | 8.462582 | 3.000000 | 16.000000 | 17.000000 | 1240 |
| Sep17 | middle | body_rate_rmse_rad_s | 0.544304 | 0.628170 | 0.083865 | 13.350707 | 3.000000 | 8.000000 | 8.000000 | 631 |
| Sep7 | middle | body_rate_rmse_rad_s | 0.603329 | 0.629289 | 0.025960 | 4.125308 | 3.000000 | 8.000000 | 9.000000 | 609 |
| ALL | middle | velocity_rmse_m_s | 0.321673 | 0.396213 | 0.074540 | 18.813019 | 3.000000 | 17.000000 | 17.000000 | 1240 |
| Sep17 | middle | velocity_rmse_m_s | 0.320689 | 0.423857 | 0.103168 | 24.340311 | 3.000000 | 8.000000 | 8.000000 | 631 |
| Sep7 | middle | velocity_rmse_m_s | 0.322548 | 0.371640 | 0.049092 | 13.209549 | 3.000000 | 9.000000 | 9.000000 | 609 |

Positive gain=Hold−Actual. Relative gain=100*(Hold−Actual)/Hold at the reported macro error level; zero denominators are undefined (blank CSV/NaN), never epsilon-adjusted. Seed rows pair identical seeds; flight rows first average the3seed flight errors. No window-independence tests.

## All endpoint horizons

| cohort | group | step | condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | 5 | Actual | 0.03534 ± 0.00016 | 0.14493 ± 0.00456 | 1.40739 ± 0.02065 | 0.54533 ± 0.00264 |
| ALL | ALL | 5 | Hold | 0.03538 ± 0.00016 | 0.14837 ± 0.00472 | 1.52534 ± 0.01911 | 0.54439 ± 0.00123 |
| ALL | ALL | 10 | Actual | 0.07038 ± 0.00050 | 0.19543 ± 0.00377 | 2.19649 ± 0.03851 | 0.56628 ± 0.00285 |
| ALL | ALL | 10 | Hold | 0.07096 ± 0.00053 | 0.20832 ± 0.00436 | 2.39426 ± 0.02490 | 0.58221 ± 0.00539 |
| ALL | ALL | 25 | Actual | 0.19240 ± 0.00186 | 0.35729 ± 0.00716 | 4.07840 ± 0.04607 | 0.59658 ± 0.00281 |
| ALL | ALL | 25 | Hold | 0.20259 ± 0.00228 | 0.43269 ± 0.00802 | 4.70068 ± 0.06310 | 0.64385 ± 0.00929 |
| ALL | ALL | 50 | Actual | 0.47401 ± 0.00635 | 0.64775 ± 0.00702 | 6.69249 ± 0.13211 | 0.62107 ± 0.01084 |
| ALL | ALL | 50 | Hold | 0.53543 ± 0.00758 | 0.84772 ± 0.00877 | 8.10351 ± 0.15571 | 0.79634 ± 0.00729 |
| ALL | high | 5 | Actual | 0.04069 ± 0.00022 | 0.19277 ± 0.00564 | 1.63623 ± 0.02169 | 0.58600 ± 0.00405 |
| ALL | high | 5 | Hold | 0.04076 ± 0.00021 | 0.19644 ± 0.00563 | 1.86780 ± 0.01228 | 0.58372 ± 0.00499 |
| ALL | high | 10 | Actual | 0.08478 ± 0.00087 | 0.27384 ± 0.00363 | 2.76887 ± 0.06452 | 0.61515 ± 0.00193 |
| ALL | high | 10 | Hold | 0.08565 ± 0.00086 | 0.28617 ± 0.00414 | 3.10241 ± 0.05121 | 0.63367 ± 0.00339 |
| ALL | high | 25 | Actual | 0.22688 ± 0.00220 | 0.50810 ± 0.01441 | 5.06294 ± 0.02050 | 0.66879 ± 0.00277 |
| ALL | high | 25 | Hold | 0.24185 ± 0.00245 | 0.61156 ± 0.00777 | 6.14294 ± 0.00360 | 0.72290 ± 0.00826 |
| ALL | high | 50 | Actual | 0.59410 ± 0.00697 | 0.83303 ± 0.02669 | 8.10050 ± 0.06090 | 0.65913 ± 0.00792 |
| ALL | high | 50 | Hold | 0.67903 ± 0.00299 | 1.13170 ± 0.02197 | 10.49564 ± 0.08554 | 0.87891 ± 0.00530 |
| ALL | low | 5 | Actual | 0.02989 ± 0.00009 | 0.11578 ± 0.00396 | 1.24759 ± 0.03171 | 0.51550 ± 0.00209 |
| ALL | low | 5 | Hold | 0.02989 ± 0.00009 | 0.11742 ± 0.00381 | 1.26962 ± 0.02997 | 0.51802 ± 0.00094 |
| ALL | low | 10 | Actual | 0.06654 ± 0.00032 | 0.15708 ± 0.00237 | 1.91262 ± 0.03495 | 0.56635 ± 0.00359 |
| ALL | low | 10 | Hold | 0.06693 ± 0.00037 | 0.16584 ± 0.00193 | 1.97321 ± 0.02730 | 0.57666 ± 0.00617 |
| ALL | low | 25 | Actual | 0.18298 ± 0.00320 | 0.29291 ± 0.01103 | 3.40530 ± 0.04863 | 0.55851 ± 0.00483 |
| ALL | low | 25 | Hold | 0.18998 ± 0.00422 | 0.33363 ± 0.01735 | 3.76577 ± 0.10836 | 0.58094 ± 0.01154 |
| ALL | low | 50 | Actual | 0.45336 ± 0.00980 | 0.57435 ± 0.01559 | 5.76448 ± 0.19380 | 0.59792 ± 0.01318 |
| ALL | low | 50 | Hold | 0.49201 ± 0.01750 | 0.67576 ± 0.03970 | 6.84211 ± 0.28934 | 0.67043 ± 0.01960 |
| ALL | middle | 5 | Actual | 0.03442 ± 0.00014 | 0.12096 ± 0.00441 | 1.35065 ± 0.02717 | 0.53126 ± 0.00337 |
| ALL | middle | 5 | Hold | 0.03446 ± 0.00015 | 0.12487 ± 0.00469 | 1.44281 ± 0.02889 | 0.52785 ± 0.00178 |
| ALL | middle | 10 | Actual | 0.06321 ± 0.00051 | 0.16759 ± 0.00651 | 2.06117 ± 0.02038 | 0.54110 ± 0.00471 |
| ALL | middle | 10 | Hold | 0.06374 ± 0.00056 | 0.18259 ± 0.00670 | 2.23834 ± 0.00629 | 0.55804 ± 0.00672 |
| ALL | middle | 25 | Actual | 0.18354 ± 0.00133 | 0.32167 ± 0.00384 | 3.97694 ± 0.04722 | 0.57555 ± 0.00416 |
| ALL | middle | 25 | Hold | 0.19231 ± 0.00185 | 0.39621 ± 0.00539 | 4.51025 ± 0.06788 | 0.62876 ± 0.00960 |
| ALL | middle | 50 | Actual | 0.41895 ± 0.00589 | 0.60572 ± 0.00632 | 6.63321 ± 0.15117 | 0.61010 ± 0.01353 |
| ALL | middle | 50 | Hold | 0.47777 ± 0.00729 | 0.80278 ± 0.01068 | 7.73444 ± 0.17421 | 0.81754 ± 0.00417 |
| Sep17 | ALL | 5 | Actual | 0.01878 ± 0.00029 | 0.15328 ± 0.00483 | 1.46983 ± 0.01253 | 0.52739 ± 0.00274 |
| Sep17 | ALL | 5 | Hold | 0.01888 ± 0.00030 | 0.15992 ± 0.00562 | 1.61630 ± 0.01134 | 0.53535 ± 0.00073 |
| Sep17 | ALL | 10 | Actual | 0.04038 ± 0.00083 | 0.22053 ± 0.00090 | 2.37616 ± 0.05668 | 0.53703 ± 0.00071 |
| Sep17 | ALL | 10 | Hold | 0.04160 ± 0.00091 | 0.24609 ± 0.00336 | 2.68623 ± 0.04133 | 0.55581 ± 0.00320 |
| Sep17 | ALL | 25 | Actual | 0.12928 ± 0.00402 | 0.37361 ± 0.01431 | 4.56951 ± 0.10540 | 0.56092 ± 0.00459 |
| Sep17 | ALL | 25 | Hold | 0.14690 ± 0.00449 | 0.48722 ± 0.01366 | 5.32337 ± 0.09602 | 0.63659 ± 0.01228 |
| Sep17 | ALL | 50 | Actual | 0.36817 ± 0.01415 | 0.68817 ± 0.01189 | 7.56542 ± 0.35082 | 0.61737 ± 0.01608 |
| Sep17 | ALL | 50 | Hold | 0.46170 ± 0.01386 | 0.96991 ± 0.01129 | 8.82203 ± 0.20027 | 0.87253 ± 0.00481 |
| Sep17 | high | 5 | Actual | 0.02117 ± 0.00033 | 0.20684 ± 0.00850 | 1.81106 ± 0.01715 | 0.61906 ± 0.00495 |
| Sep17 | high | 5 | Hold | 0.02135 ± 0.00034 | 0.21469 ± 0.00920 | 2.08407 ± 0.00760 | 0.63107 ± 0.00648 |
| Sep17 | high | 10 | Actual | 0.05037 ± 0.00107 | 0.31172 ± 0.00543 | 3.11785 ± 0.11855 | 0.59837 ± 0.00451 |
| Sep17 | high | 10 | Hold | 0.05215 ± 0.00109 | 0.34116 ± 0.00639 | 3.65293 ± 0.08653 | 0.62685 ± 0.00745 |
| Sep17 | high | 25 | Actual | 0.16796 ± 0.00499 | 0.50614 ± 0.02808 | 5.77120 ± 0.13157 | 0.62695 ± 0.00089 |
| Sep17 | high | 25 | Hold | 0.19454 ± 0.00399 | 0.66753 ± 0.01109 | 7.00265 ± 0.09460 | 0.70908 ± 0.00777 |
| Sep17 | high | 50 | Actual | 0.47566 ± 0.02352 | 0.88454 ± 0.04038 | 8.95938 ± 0.14472 | 0.67256 ± 0.01229 |
| Sep17 | high | 50 | Hold | 0.60638 ± 0.00680 | 1.27078 ± 0.02915 | 11.10173 ± 0.05581 | 0.96783 ± 0.00526 |
| Sep17 | low | 5 | Actual | 0.01715 ± 0.00029 | 0.12115 ± 0.00437 | 1.21566 ± 0.02988 | 0.46547 ± 0.00265 |
| Sep17 | low | 5 | Hold | 0.01714 ± 0.00028 | 0.12439 ± 0.00437 | 1.23862 ± 0.02753 | 0.46736 ± 0.00314 |
| Sep17 | low | 10 | Actual | 0.03598 ± 0.00073 | 0.16701 ± 0.00124 | 1.88234 ± 0.04619 | 0.52666 ± 0.00303 |
| Sep17 | low | 10 | Hold | 0.03683 ± 0.00082 | 0.18513 ± 0.00098 | 1.96910 ± 0.04828 | 0.54195 ± 0.00536 |
| Sep17 | low | 25 | Actual | 0.11685 ± 0.00652 | 0.30334 ± 0.02143 | 3.46130 ± 0.11216 | 0.49893 ± 0.00767 |
| Sep17 | low | 25 | Hold | 0.12832 ± 0.00845 | 0.36025 ± 0.03277 | 3.79408 ± 0.17437 | 0.52376 ± 0.01582 |
| Sep17 | low | 50 | Actual | 0.32469 ± 0.02090 | 0.58846 ± 0.03475 | 6.27346 ± 0.44063 | 0.58048 ± 0.02340 |
| Sep17 | low | 50 | Hold | 0.37920 ± 0.03454 | 0.72693 ± 0.07245 | 7.06019 ± 0.42926 | 0.68449 ± 0.02185 |
| Sep17 | middle | 5 | Actual | 0.01777 ± 0.00029 | 0.12748 ± 0.00329 | 1.36909 ± 0.01023 | 0.49625 ± 0.00383 |
| Sep17 | middle | 5 | Hold | 0.01785 ± 0.00031 | 0.13472 ± 0.00455 | 1.47376 ± 0.01342 | 0.50394 ± 0.00310 |
| Sep17 | middle | 10 | Actual | 0.03549 ± 0.00074 | 0.17731 ± 0.00280 | 2.13952 ± 0.02129 | 0.50803 ± 0.00312 |
| Sep17 | middle | 10 | Hold | 0.03648 ± 0.00084 | 0.20349 ± 0.00386 | 2.38339 ± 0.01262 | 0.52244 ± 0.00062 |
| Sep17 | middle | 25 | Actual | 0.10994 ± 0.00285 | 0.32069 ± 0.00508 | 4.26905 ± 0.10051 | 0.54430 ± 0.00562 |
| Sep17 | middle | 25 | Hold | 0.12403 ± 0.00372 | 0.42386 ± 0.00993 | 4.88442 ± 0.08719 | 0.62817 ± 0.01488 |
| Sep17 | middle | 50 | Actual | 0.31776 ± 0.01020 | 0.61124 ± 0.00300 | 7.20165 ± 0.42753 | 0.59829 ± 0.01804 |
| Sep17 | middle | 50 | Hold | 0.40052 ± 0.01416 | 0.86536 ± 0.01748 | 8.09822 ± 0.24263 | 0.88383 ± 0.00740 |
| Sep7 | ALL | 5 | Actual | 0.05006 ± 0.00009 | 0.13750 ± 0.00452 | 1.35189 ± 0.02882 | 0.56127 ± 0.00293 |
| Sep7 | ALL | 5 | Hold | 0.05006 ± 0.00008 | 0.13810 ± 0.00420 | 1.44449 ± 0.02617 | 0.55243 ± 0.00233 |
| Sep7 | ALL | 10 | Actual | 0.09704 ± 0.00028 | 0.17313 ± 0.00633 | 2.03677 ± 0.02297 | 0.59228 ± 0.00502 |
| Sep7 | ALL | 10 | Hold | 0.09705 ± 0.00029 | 0.17475 ± 0.00532 | 2.13473 ± 0.01161 | 0.60567 ± 0.00747 |
| Sep7 | ALL | 25 | Actual | 0.24849 ± 0.00083 | 0.34278 ± 0.00097 | 3.64185 ± 0.05666 | 0.62827 ± 0.00259 |
| Sep7 | ALL | 25 | Hold | 0.25210 ± 0.00128 | 0.38423 ± 0.00600 | 4.14717 ± 0.03392 | 0.65031 ± 0.00745 |
| Sep7 | ALL | 50 | Actual | 0.56810 ± 0.00350 | 0.61182 ± 0.00352 | 5.91655 ± 0.06500 | 0.62437 ± 0.00732 |
| Sep7 | ALL | 50 | Hold | 0.60097 ± 0.00689 | 0.73911 ± 0.01170 | 7.46483 ± 0.12172 | 0.72861 ± 0.01446 |
| Sep7 | high | 5 | Actual | 0.05804 ± 0.00012 | 0.18026 ± 0.00390 | 1.48083 ± 0.02646 | 0.55660 ± 0.00327 |
| Sep7 | high | 5 | Hold | 0.05801 ± 0.00010 | 0.18021 ± 0.00381 | 1.67557 ± 0.01981 | 0.54163 ± 0.00383 |
| Sep7 | high | 10 | Actual | 0.11536 ± 0.00075 | 0.24016 ± 0.00208 | 2.45867 ± 0.01915 | 0.63007 ± 0.00412 |
| Sep7 | high | 10 | Hold | 0.11542 ± 0.00068 | 0.23729 ± 0.00217 | 2.61307 ± 0.02695 | 0.63973 ± 0.00367 |
| Sep7 | high | 25 | Actual | 0.27925 ± 0.00251 | 0.50983 ± 0.00271 | 4.43337 ± 0.09031 | 0.70599 ± 0.00597 |
| Sep7 | high | 25 | Hold | 0.28390 ± 0.00304 | 0.56181 ± 0.01001 | 5.37875 ± 0.07730 | 0.73518 ± 0.00890 |
| Sep7 | high | 50 | Actual | 0.69938 ± 0.00791 | 0.78724 ± 0.01581 | 7.33706 ± 0.02577 | 0.64720 ± 0.00854 |
| Sep7 | high | 50 | Hold | 0.74360 ± 0.01136 | 1.00806 ± 0.01872 | 9.95691 ± 0.13166 | 0.79987 ± 0.00815 |
| Sep7 | low | 5 | Actual | 0.04122 ± 0.00013 | 0.11100 ± 0.00392 | 1.27598 ± 0.03467 | 0.55997 ± 0.00205 |
| Sep7 | low | 5 | Hold | 0.04123 ± 0.00013 | 0.11123 ± 0.00360 | 1.29719 ± 0.03350 | 0.56304 ± 0.00129 |
| Sep7 | low | 10 | Actual | 0.09370 ± 0.00022 | 0.14826 ± 0.00459 | 1.93954 ± 0.02752 | 0.60164 ± 0.00427 |
| Sep7 | low | 10 | Hold | 0.09369 ± 0.00024 | 0.14869 ± 0.00345 | 1.97686 ± 0.02075 | 0.60751 ± 0.00711 |
| Sep7 | low | 25 | Actual | 0.24176 ± 0.00061 | 0.28364 ± 0.00282 | 3.35552 ± 0.07363 | 0.61148 ± 0.00416 |
| Sep7 | low | 25 | Hold | 0.24479 ± 0.00111 | 0.30998 ± 0.00441 | 3.74061 ± 0.04994 | 0.63177 ± 0.00870 |
| Sep7 | low | 50 | Actual | 0.56773 ± 0.00572 | 0.56181 ± 0.00610 | 5.31204 ± 0.04044 | 0.61342 ± 0.00618 |
| Sep7 | low | 50 | Hold | 0.59228 ± 0.00814 | 0.63028 ± 0.01985 | 6.64827 ± 0.16522 | 0.65793 ± 0.01937 |
| Sep7 | middle | 5 | Actual | 0.04923 ± 0.00006 | 0.11516 ± 0.00596 | 1.33426 ± 0.04260 | 0.56238 ± 0.00415 |
| Sep7 | middle | 5 | Hold | 0.04922 ± 0.00006 | 0.11611 ± 0.00537 | 1.41531 ± 0.04274 | 0.54911 ± 0.00331 |
| Sep7 | middle | 10 | Actual | 0.08785 ± 0.00031 | 0.15895 ± 0.01252 | 1.99151 ± 0.02606 | 0.57050 ± 0.01125 |
| Sep7 | middle | 10 | Hold | 0.08796 ± 0.00032 | 0.16401 ± 0.01039 | 2.10940 ± 0.00293 | 0.58968 ± 0.01325 |
| Sep7 | middle | 25 | Actual | 0.24896 ± 0.00033 | 0.32255 ± 0.00287 | 3.71728 ± 0.04854 | 0.60333 ± 0.00338 |
| Sep7 | middle | 25 | Hold | 0.25300 ± 0.00087 | 0.37164 ± 0.00491 | 4.17765 ± 0.05501 | 0.62929 ± 0.00566 |
| Sep7 | middle | 50 | Actual | 0.50889 ± 0.00212 | 0.60081 ± 0.01221 | 6.12792 ± 0.10455 | 0.62059 ± 0.00978 |
| Sep7 | middle | 50 | Hold | 0.54643 ± 0.00392 | 0.74716 ± 0.00668 | 7.41108 ± 0.12005 | 0.75862 ± 0.01354 |

## Error evolution and interval metric

Error_evolution.csv retains steps1..50, nominal step*.02 labels, and native cumulative elapsed-time distributions. Figure x is median elapsed time over the participating origins at each step, not a common timestamp or resampled trajectory. Mean and shaded±1SD use equal-flight error followed by3seeds. The median time distribution is origin-weighted; errors are flight-weighted.

| condition | step | metric | mean | std | elapsed_median_s | elapsed_min_s | elapsed_max_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Actual | 1 | attitude_error_deg | 0.382484 | 0.001334 | 0.019868 | 0.009439 | 0.030167 |
| Actual | 1 | body_rate_rmse_rad_s | 0.506497 | 0.003566 | 0.019868 | 0.009439 | 0.030167 |
| Actual | 1 | position_rmse_m | 0.008603 | 0.000004 | 0.019868 | 0.009439 | 0.030167 |
| Actual | 1 | velocity_rmse_m_s | 0.051333 | 0.000797 | 0.019868 | 0.009439 | 0.030167 |
| Actual | 5 | attitude_error_deg | 1.407388 | 0.020646 | 0.099633 | 0.088430 | 0.109965 |
| Actual | 5 | body_rate_rmse_rad_s | 0.545329 | 0.002643 | 0.099633 | 0.088430 | 0.109965 |
| Actual | 5 | position_rmse_m | 0.035340 | 0.000160 | 0.099633 | 0.088430 | 0.109965 |
| Actual | 5 | velocity_rmse_m_s | 0.144929 | 0.004559 | 0.099633 | 0.088430 | 0.109965 |
| Actual | 10 | attitude_error_deg | 2.196485 | 0.038513 | 0.199567 | 0.189358 | 0.209794 |
| Actual | 10 | body_rate_rmse_rad_s | 0.566283 | 0.002845 | 0.199567 | 0.189358 | 0.209794 |
| Actual | 10 | position_rmse_m | 0.070376 | 0.000498 | 0.199567 | 0.189358 | 0.209794 |
| Actual | 10 | velocity_rmse_m_s | 0.195432 | 0.003772 | 0.199567 | 0.189358 | 0.209794 |
| Actual | 25 | attitude_error_deg | 4.078397 | 0.046072 | 0.499094 | 0.488664 | 0.518876 |
| Actual | 25 | body_rate_rmse_rad_s | 0.596577 | 0.002814 | 0.499094 | 0.488664 | 0.518876 |
| Actual | 25 | position_rmse_m | 0.192395 | 0.001857 | 0.499094 | 0.488664 | 0.518876 |
| Actual | 25 | velocity_rmse_m_s | 0.357288 | 0.007165 | 0.499094 | 0.488664 | 0.518876 |
| Actual | 50 | attitude_error_deg | 6.692491 | 0.132113 | 0.997967 | 0.987696 | 1.027915 |
| Actual | 50 | body_rate_rmse_rad_s | 0.621073 | 0.010837 | 0.997967 | 0.987696 | 1.027915 |
| Actual | 50 | position_rmse_m | 0.474014 | 0.006345 | 0.997967 | 0.987696 | 1.027915 |
| Actual | 50 | velocity_rmse_m_s | 0.647748 | 0.007016 | 0.997967 | 0.987696 | 1.027915 |
| Hold | 1 | attitude_error_deg | 0.382484 | 0.001334 | 0.019868 | 0.009439 | 0.030167 |
| Hold | 1 | body_rate_rmse_rad_s | 0.506497 | 0.003566 | 0.019868 | 0.009439 | 0.030167 |
| Hold | 1 | position_rmse_m | 0.008603 | 0.000004 | 0.019868 | 0.009439 | 0.030167 |
| Hold | 1 | velocity_rmse_m_s | 0.051333 | 0.000797 | 0.019868 | 0.009439 | 0.030167 |
| Hold | 5 | attitude_error_deg | 1.525339 | 0.019112 | 0.099633 | 0.088430 | 0.109965 |
| Hold | 5 | body_rate_rmse_rad_s | 0.544392 | 0.001226 | 0.099633 | 0.088430 | 0.109965 |
| Hold | 5 | position_rmse_m | 0.035385 | 0.000164 | 0.099633 | 0.088430 | 0.109965 |
| Hold | 5 | velocity_rmse_m_s | 0.148370 | 0.004716 | 0.099633 | 0.088430 | 0.109965 |
| Hold | 10 | attitude_error_deg | 2.394262 | 0.024903 | 0.199567 | 0.189358 | 0.209794 |
| Hold | 10 | body_rate_rmse_rad_s | 0.582207 | 0.005394 | 0.199567 | 0.189358 | 0.209794 |
| Hold | 10 | position_rmse_m | 0.070956 | 0.000530 | 0.199567 | 0.189358 | 0.209794 |
| Hold | 10 | velocity_rmse_m_s | 0.208319 | 0.004361 | 0.199567 | 0.189358 | 0.209794 |
| Hold | 25 | attitude_error_deg | 4.700676 | 0.063101 | 0.499094 | 0.488664 | 0.518876 |
| Hold | 25 | body_rate_rmse_rad_s | 0.643854 | 0.009295 | 0.499094 | 0.488664 | 0.518876 |
| Hold | 25 | position_rmse_m | 0.202593 | 0.002282 | 0.499094 | 0.488664 | 0.518876 |
| Hold | 25 | velocity_rmse_m_s | 0.432695 | 0.008023 | 0.499094 | 0.488664 | 0.518876 |
| Hold | 50 | attitude_error_deg | 8.103512 | 0.155713 | 0.997967 | 0.987696 | 1.027915 |
| Hold | 50 | body_rate_rmse_rad_s | 0.796340 | 0.007287 | 0.997967 | 0.987696 | 1.027915 |
| Hold | 50 | position_rmse_m | 0.535432 | 0.007583 | 0.997967 | 0.987696 | 1.027915 |
| Hold | 50 | velocity_rmse_m_s | 0.847724 | 0.008767 | 0.997967 | 0.987696 | 1.027915 |

Auxiliary interval metric: I_i,K² = sum_{k=1..K} dt_i,k-1 * e_i,k² / sum_{k=1..K}dt_i,k-1. This uses right-endpoint errors, excludes t0, then averages I_i,K² equally over origins in each flight before square root. Flights are averaged equally, then seeds summarized. No duration pooling across origins and no replacement of frozen endpoint metrics. See trajectory_interval_errors.csv.

| step | condition | position_rmse_m | velocity_rmse_m_s | attitude_error_deg | body_rate_rmse_rad_s |
| --- | --- | --- | --- | --- | --- |
| 5 | Actual | 0.02417 ± 0.00008 | 0.10725 ± 0.00310 | 1.02725 ± 0.01239 | 0.53943 ± 0.00158 |
| 5 | Hold | 0.02419 ± 0.00008 | 0.10903 ± 0.00321 | 1.08775 ± 0.01222 | 0.54024 ± 0.00148 |
| 10 | Actual | 0.04388 ± 0.00027 | 0.14943 ± 0.00373 | 1.52573 ± 0.02403 | 0.55165 ± 0.00148 |
| 10 | Hold | 0.04411 ± 0.00028 | 0.15538 ± 0.00403 | 1.65427 ± 0.01778 | 0.55640 ± 0.00150 |
| 25 | Actual | 0.11041 ± 0.00091 | 0.23937 ± 0.00434 | 2.72076 ± 0.02962 | 0.56654 ± 0.00066 |
| 25 | Hold | 0.11433 ± 0.00107 | 0.27197 ± 0.00449 | 3.05520 ± 0.03063 | 0.58797 ± 0.00427 |
| 50 | Actual | 0.25253 ± 0.00320 | 0.39847 ± 0.00630 | 4.32718 ± 0.04634 | 0.58477 ± 0.00297 |
| 50 | Hold | 0.27809 ± 0.00375 | 0.50018 ± 0.00652 | 5.12906 ± 0.08253 | 0.65897 ± 0.00628 |

## Dynamic response cases

All three predetermined cases are retained: original Step1 representative plus identity-median high-E25 origins from Sep7 and Sep17. Seed17 fixed; no best-seed selection. Each figure shows all4commands, vN/vE/vD, illustrative Euler angles and p/q/r over25native steps. Quantitative attitude remains quaternion geodesic error. A case may favor Hold; it is never omitted.

| case | window_id | seed | step | metric | actual | hold | absolute_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | position_rmse_m | 0.397782 | 0.391630 | -0.006152 |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | velocity_rmse_m_s | 0.250229 | 0.188781 | -0.061448 |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | attitude_error_deg | 2.649345 | 2.303632 | -0.345714 |
| fixed_step1 | validation:2026.9.7/log_29_2026-9-7-06-51-04.ulg:4:3975 | 17 | 25 | body_rate_rmse_rad_s | 0.387868 | 0.377270 | -0.010597 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | position_rmse_m | 0.116554 | 0.161476 | 0.044922 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | velocity_rmse_m_s | 0.319508 | 0.548358 | 0.228849 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | attitude_error_deg | 4.927704 | 6.108864 | 1.181159 |
| Sep7_high_median_identity | validation:2026.9.7/log_24_2026-9-7-06-16-04.ulg:1:825 | 17 | 25 | body_rate_rmse_rad_s | 1.386037 | 1.407350 | 0.021313 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | position_rmse_m | 0.084275 | 0.065705 | -0.018570 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | velocity_rmse_m_s | 0.462596 | 0.243905 | -0.218691 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | attitude_error_deg | 5.621190 | 5.295020 | -0.326170 |
| Sep17_high_median_identity | validation:9.17数据/log_13_2026-9-17-06-52-30.ulg:1:5175 | 17 | 25 | body_rate_rmse_rad_s | 0.529446 | 0.590129 | 0.060683 |

## Scientific interpretation

Positive aggregate gains are not universal. At100ms, body-rate error slightly favors Hold (−0.17% relative gain). The fixed Step1 case favors Hold on all four500ms endpoints; the Sep17 high-control case favors Hold for position, velocity and attitude. All cases are retained. Relative gains do not increase monotonically across activity groups: middle-group velocity/body-rate percentage gains exceed high-group gains. Both cohorts nevertheless support500ms aggregate gains on all three dynamics metrics.



At the nominal 500 ms horizon, replaying the logged future commands reduced velocity RMSE from 0.4327 to 0.3573 m/s (17.43%), attitude geodesic RMS error from 4.7007° to 4.0784° (13.24%), and body-rate RMSE from 0.6439 to 0.5966 rad/s (7.34%) compared with holding the origin command. All three seeds favored actual-command replay on these aggregate metrics. After averaging seeds within each flight, the improvement direction was also consistent across all 17 validation flights. In the high-control-change subset (454 origins spanning all 17 flights), the corresponding reductions were 16.92%, 17.58% and 7.48%. These findings support incremental predictive information from future logged commands within the observed flight distribution; they do not imply improvement for every individual trajectory.

## Limitations

Both conditions were evaluated against the same observed trajectory generated under the logged commands. Hold is therefore an information-limited forecast control, not a counterfactual flight with matched ground truth. Closed-loop feedback and correlated state estimates can contribute to the predictive association. These results do not establish causal accuracy for arbitrary control actions, closed-loop improvement, or suitability for long-horizon simulation/RL. Control groups use realized future commands offline; three-seed SD describes initialization sensitivity, not predictive uncertainty or a confidence interval.

History horizon-dependence remains the frozen Step4 Mixed result; no H1/H5/H13 inference or new history conclusions were generated. Dedicated flight input-excitation data would be needed to validate action-response direction, timing and amplitude under interventions.

## Engineering verification and sealed status

Tests, actual replay parity on first128origins per seed, constant-command equality, shared first-step prediction, label poisoning/prefix checks, finite values/unit quaternions/native timing, identical coverage and bitwise unchanged model buffers passed. Required explicit artifacts are hash-pinned; unrelated registry/workspace differences are recorded rather than repaired. Only train/open-validation are used. Sealed Sep8/reserved Sep19 remain unopened. No automatic next experiment or commit/push.
