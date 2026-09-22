# Frozen held-out flight evaluation

## Registration and access

Registered 2026-09-22T06:09:10.808392+00:00; first read this run: {'Sep8': '2026-09-22T06:09:24.239195+00:00', 'Sep19': '2026-09-22T06:09:32.901130+00:00'}. HEAD `63ad78f269eb882d9b311c517274b5a3b9732939`. Prior access disclosures are retained; neither date is now unopened.

Sep8: 6/6 flights admitted; 871 fixed origins.
Sep19: 22/23 flights admitted; 3202 fixed origins.

See evaluation_registration.md, protocol.json and append-only test_access_log.jsonl. No training, parameter tuning, checkpoint selection or ensemble.

## Data quality

| date | session | flight_id | status | raw_samples | valid_samples | final_origins | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | 2026.9.8 | 2026.9.8/log_0_2026-9-8-05-42-48.ulg | admitted | 7813 | 5669 | 111 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_1_2026-9-8-05-49-28.ulg | admitted | 10899 | 8411 | 165 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_2_2026-9-8-05-56-46.ulg | admitted | 10857 | 8174 | 156 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_4_2026-9-8-06-38-56.ulg | admitted | 10162 | 7682 | 149 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_6_2026-9-8-06-46-18.ulg | admitted | 10290 | 7392 | 142 | — |
| Sep8 | 2026.9.8 | 2026.9.8/log_7_2026-9-8-06-55-26.ulg | admitted | 9504 | 7530 | 148 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_0_2026-9-19-14-31-44.ulg | admitted | 10831 | 8174 | 160 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_1_2026-9-19-14-40-12.ulg | admitted | 11103 | 6455 | 126 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_2_2026-9-19-14-45-56.ulg | excluded_by_frozen_gate | 1309 | 14 | 0 | insufficient contiguous airborne data: duration<30 or stride10 windows<10 |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_3_2026-9-19-14-56-32.ulg | admitted | 10496 | 7214 | 141 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_4_2026-9-19-15-15-06.ulg | admitted | 9459 | 7321 | 140 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_5_2026-9-19-15-23-48.ulg | admitted | 15929 | 6795 | 133 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_6_2026-9-19-15-28-00.ulg | admitted | 8165 | 5999 | 119 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_7_2026-9-19-15-39-08.ulg | admitted | 14876 | 7624 | 150 | — |
| Sep19 | 9.19-2数据 | 9.19-2数据/log_8_2026-9-19-15-47-10.ulg | admitted | 10778 | 8386 | 164 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_0_2026-9-19-17-14-26.ulg | admitted | 10968 | 7186 | 141 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_1_2026-9-19-17-21-44.ulg | admitted | 9974 | 7231 | 142 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_2_2026-9-19-17-26-42.ulg | admitted | 9574 | 7234 | 141 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_3_2026-9-19-17-38-46.ulg | admitted | 8784 | 6457 | 126 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_4_2026-9-19-17-44-06.ulg | admitted | 10324 | 7562 | 148 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_5_2026-9-19-17-49-50.ulg | admitted | 9859 | 7899 | 155 | — |
| Sep19 | 9.19-3数据 | 9.19-3数据/log_9_2026-9-19-17-05-08.ulg | admitted | 10324 | 7931 | 154 | — |
| Sep19 | 9.19数据 | 9.19数据/log_0_2026-9-19-05-53-40.ulg | admitted | 11619 | 7966 | 153 | — |
| Sep19 | 9.19数据 | 9.19数据/log_1_2026-9-19-06-01-30.ulg | admitted | 10696 | 7423 | 144 | — |
| Sep19 | 9.19数据 | 9.19数据/log_2_2026-9-19-06-07-52.ulg | admitted | 9694 | 7382 | 145 | — |
| Sep19 | 9.19数据 | 9.19数据/log_3_2026-9-19-06-14-30.ulg | admitted | 11383 | 8000 | 156 | — |
| Sep19 | 9.19数据 | 9.19数据/log_4_2026-9-19-06-19-40.ulg | admitted | 9944 | 7916 | 153 | — |
| Sep19 | 9.19数据 | 9.19数据/log_5_2026-9-19-06-26-44.ulg | admitted | 9688 | 7669 | 150 | — |
| Sep19 | 9.19数据 | 9.19数据/log_6_2026-9-19-06-31-46.ulg | admitted | 9947 | 8194 | 161 | — |

Exact gates and exclusion counters are in dataset_quality_report.md. Sep8 inherits the original v2 flight gate; Sep19 inherits the expanded v3 gate. All admitted models share longest-history valid origins and native dt. Raw possible counts ignore gaps; quality counts apply original valid_core. No error-based filtering.

## 500 ms main performance

| date | model | condition | group | horizon_s | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | B0 kinematic | Actual | ALL | 0.5 | 0.3628 | 1.1462 | 23.9791 | 0.9779 |
| Sep8 | MLP | Actual | ALL | 0.5 | 0.1636 ± 0.0037 | 0.4921 ± 0.0159 | 4.5510 ± 0.0677 | 0.6576 ± 0.0060 |
| Sep8 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.1205 ± 0.0036 | 0.3533 ± 0.0132 | 3.6977 ± 0.1867 | 0.5298 ± 0.0082 |
| Sep19 | B0 kinematic | Actual | ALL | 0.5 | 0.4769 | 1.4227 | 22.8819 | 1.1018 |
| Sep19 | MLP | Actual | ALL | 0.5 | 0.2847 ± 0.0013 | 0.7036 ± 0.0034 | 6.9392 ± 0.0204 | 0.7459 ± 0.0020 |
| Sep19 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.2097 ± 0.0016 | 0.4610 ± 0.0112 | 4.9615 ± 0.0364 | 0.5286 ± 0.0043 |

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | attitude_error_deg | 6.9392 | 4.9615 | 28.5 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.74595 | 0.52859 | 29.138 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | position_rmse_m | 0.28466 | 0.20969 | 26.335 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.70362 | 0.46098 | 34.485 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | attitude_error_deg | 4.551 | 3.6977 | 18.75 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.65764 | 0.5298 | 19.44 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | position_rmse_m | 0.16357 | 0.12052 | 26.315 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.49209 | 0.35328 | 28.208 | 3 | 0 | 6 | 0 | 6 | 871 |

## History

| date | model | condition | group | horizon_s | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | GRU/H1 | Actual | ALL | 0.5 | 0.1380 ± 0.0022 | 0.3797 ± 0.0010 | 3.7222 ± 0.0665 | 0.5456 ± 0.0024 |
| Sep8 | GRU/H5 | Actual | ALL | 0.5 | 0.1272 ± 0.0042 | 0.3651 ± 0.0089 | 3.9062 ± 0.0747 | 0.5340 ± 0.0023 |
| Sep8 | GRU/H13 | Actual | ALL | 0.5 | 0.1227 ± 0.0009 | 0.3494 ± 0.0051 | 3.7016 ± 0.1078 | 0.5268 ± 0.0074 |
| Sep8 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.1205 ± 0.0036 | 0.3533 ± 0.0132 | 3.6977 ± 0.1867 | 0.5298 ± 0.0082 |
| Sep19 | GRU/H1 | Actual | ALL | 0.5 | 0.2495 ± 0.0010 | 0.5302 ± 0.0015 | 5.5766 ± 0.0457 | 0.5749 ± 0.0058 |
| Sep19 | GRU/H5 | Actual | ALL | 0.5 | 0.2260 ± 0.0012 | 0.5056 ± 0.0022 | 5.1653 ± 0.0313 | 0.5561 ± 0.0051 |
| Sep19 | GRU/H13 | Actual | ALL | 0.5 | 0.2142 ± 0.0006 | 0.4748 ± 0.0056 | 4.9652 ± 0.0475 | 0.5335 ± 0.0051 |
| Sep19 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.2097 ± 0.0016 | 0.4610 ± 0.0112 | 4.9615 ± 0.0364 | 0.5286 ± 0.0043 |

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H13_to_H26 | Sep19 | ALL | 0.5 | attitude_error_deg | 4.9652 | 4.9615 | 0.073583 | 2 | 1 | 11 | 11 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.53349 | 0.52859 | 0.9177 | 2 | 1 | 16 | 6 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | position_rmse_m | 0.21418 | 0.20969 | 2.0944 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.47478 | 0.46098 | 2.9063 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H13_to_H26 | Sep8 | ALL | 0.5 | attitude_error_deg | 3.7016 | 3.6977 | 0.10508 | 2 | 1 | 4 | 2 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.52681 | 0.5298 | -0.56667 | 1 | 2 | 1 | 5 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | position_rmse_m | 0.12268 | 0.12052 | 1.7582 | 2 | 1 | 5 | 1 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.34942 | 0.35328 | -1.104 | 1 | 2 | 4 | 2 | 6 | 871 |
| H1_to_H26 | Sep19 | ALL | 0.5 | attitude_error_deg | 5.5766 | 4.9615 | 11.03 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.57493 | 0.52859 | 8.0597 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | position_rmse_m | 0.24951 | 0.20969 | 15.958 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.53016 | 0.46098 | 13.049 | 3 | 0 | 21 | 1 | 22 | 3202 |
| H1_to_H26 | Sep8 | ALL | 0.5 | attitude_error_deg | 3.7222 | 3.6977 | 0.65858 | 2 | 1 | 4 | 2 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.54564 | 0.5298 | 2.9036 | 3 | 0 | 5 | 1 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | position_rmse_m | 0.13795 | 0.12052 | 12.635 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.37975 | 0.35328 | 6.9688 | 3 | 0 | 5 | 1 | 6 | 871 |

H1 is current-sample context with recurrent future transitions. H26 remains main even if a shorter context is better. No physical memory time constant or replacement of validation Mixed classification is inferred.

## Future command information

| date | model | condition | group | horizon_s | Position RMSE [m] | Velocity RMSE [m/s] | Attitude geodesic RMS [deg] | Body-rate RMSE [rad/s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.1205 ± 0.0036 | 0.3533 ± 0.0132 | 3.6977 ± 0.1867 | 0.5298 ± 0.0082 |
| Sep8 | Standard GRU/H26 | Actual | low | 0.5 | 0.1023 ± 0.0024 | 0.2712 ± 0.0140 | 3.3242 ± 0.2472 | 0.5214 ± 0.0077 |
| Sep8 | Standard GRU/H26 | Actual | middle | 0.5 | 0.1138 ± 0.0021 | 0.3547 ± 0.0065 | 3.5281 ± 0.1961 | 0.5137 ± 0.0110 |
| Sep8 | Standard GRU/H26 | Actual | high | 0.5 | 0.1649 ± 0.0115 | 0.4651 ± 0.0383 | 4.8100 ± 0.0709 | 0.5449 ± 0.0093 |
| Sep8 | Standard GRU/H26 | Hold | ALL | 0.5 | 0.1368 ± 0.0039 | 0.4381 ± 0.0163 | 4.4201 ± 0.2186 | 0.5831 ± 0.0118 |
| Sep8 | Standard GRU/H26 | Hold | low | 0.5 | 0.1204 ± 0.0037 | 0.3574 ± 0.0214 | 3.7431 ± 0.2891 | 0.5591 ± 0.0109 |
| Sep8 | Standard GRU/H26 | Hold | middle | 0.5 | 0.1317 ± 0.0032 | 0.4491 ± 0.0099 | 4.2230 ± 0.2315 | 0.5684 ± 0.0114 |
| Sep8 | Standard GRU/H26 | Hold | high | 0.5 | 0.1775 ± 0.0092 | 0.5494 ± 0.0339 | 6.0050 ± 0.1015 | 0.6293 ± 0.0143 |
| Sep19 | Standard GRU/H26 | Actual | ALL | 0.5 | 0.2097 ± 0.0016 | 0.4610 ± 0.0112 | 4.9615 ± 0.0364 | 0.5286 ± 0.0043 |
| Sep19 | Standard GRU/H26 | Actual | low | 0.5 | 0.1974 ± 0.0058 | 0.3949 ± 0.0300 | 3.9179 ± 0.0543 | 0.4665 ± 0.0082 |
| Sep19 | Standard GRU/H26 | Actual | middle | 0.5 | 0.1956 ± 0.0018 | 0.4118 ± 0.0099 | 4.5606 ± 0.0438 | 0.5093 ± 0.0050 |
| Sep19 | Standard GRU/H26 | Actual | high | 0.5 | 0.2238 ± 0.0038 | 0.5655 ± 0.0174 | 6.2085 ± 0.0349 | 0.6063 ± 0.0044 |
| Sep19 | Standard GRU/H26 | Hold | ALL | 0.5 | 0.2254 ± 0.0010 | 0.5638 ± 0.0077 | 5.8669 ± 0.0459 | 0.6392 ± 0.0025 |
| Sep19 | Standard GRU/H26 | Hold | low | 0.5 | 0.2054 ± 0.0060 | 0.4267 ± 0.0321 | 4.2742 ± 0.0949 | 0.5219 ± 0.0029 |
| Sep19 | Standard GRU/H26 | Hold | middle | 0.5 | 0.2086 ± 0.0013 | 0.4982 ± 0.0089 | 5.2524 ± 0.0624 | 0.6177 ± 0.0040 |
| Sep19 | Standard GRU/H26 | Hold | high | 0.5 | 0.2468 ± 0.0044 | 0.7252 ± 0.0228 | 7.6533 ± 0.0327 | 0.7434 ± 0.0028 |

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | attitude_error_deg | 5.8669 | 4.9615 | 15.432 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.63916 | 0.52859 | 17.299 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | position_rmse_m | 0.22545 | 0.20969 | 6.9879 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.56382 | 0.46098 | 18.241 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | high | 0.5 | attitude_error_deg | 7.6533 | 6.2085 | 18.878 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | body_rate_rmse_rad_s | 0.74339 | 0.60632 | 18.438 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | position_rmse_m | 0.2468 | 0.22384 | 9.3034 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | velocity_rmse_m_s | 0.72519 | 0.56546 | 22.026 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | low | 0.5 | attitude_error_deg | 4.2742 | 3.9179 | 8.3359 | 3 | 0 | 21 | 1 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | body_rate_rmse_rad_s | 0.52194 | 0.46654 | 10.613 | 3 | 0 | 22 | 0 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | position_rmse_m | 0.20537 | 0.19735 | 3.9019 | 3 | 0 | 21 | 1 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | velocity_rmse_m_s | 0.42668 | 0.39494 | 7.4393 | 3 | 0 | 22 | 0 | 22 | 603 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | attitude_error_deg | 5.2524 | 4.5606 | 13.171 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | body_rate_rmse_rad_s | 0.61773 | 0.50931 | 17.551 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | position_rmse_m | 0.20864 | 0.19559 | 6.2527 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | velocity_rmse_m_s | 0.49818 | 0.41184 | 17.331 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | attitude_error_deg | 4.4201 | 3.6977 | 16.343 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.58312 | 0.5298 | 9.1455 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | position_rmse_m | 0.13683 | 0.12052 | 11.92 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.43807 | 0.35328 | 19.355 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | high | 0.5 | attitude_error_deg | 6.005 | 4.81 | 19.9 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | body_rate_rmse_rad_s | 0.62932 | 0.54489 | 13.417 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | position_rmse_m | 0.17747 | 0.16491 | 7.0761 | 3 | 0 | 5 | 1 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | velocity_rmse_m_s | 0.54943 | 0.46513 | 15.343 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | low | 0.5 | attitude_error_deg | 3.7431 | 3.3242 | 11.191 | 3 | 0 | 5 | 1 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | body_rate_rmse_rad_s | 0.55911 | 0.52141 | 6.7434 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | position_rmse_m | 0.12038 | 0.10232 | 15.003 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | velocity_rmse_m_s | 0.35738 | 0.27124 | 24.104 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | attitude_error_deg | 4.223 | 3.5281 | 16.455 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | body_rate_rmse_rad_s | 0.56835 | 0.51375 | 9.6077 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | position_rmse_m | 0.13172 | 0.11379 | 13.617 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | velocity_rmse_m_s | 0.44908 | 0.35471 | 21.013 | 3 | 0 | 5 | 1 | 6 | 379 |

Train quantiles are reused exactly. Group proportions need not be25/50/25. Empty groups are explicit unavailable rows/zero coverage. Horizon-specific groups change membership; full curves fixK25 membership.

| date | step | horizon_s | group | flight_id | n_origins | n_flights | E_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | 25 | 0.5 | ALL | ALL | 871 | 6 | 0.19157 |
| Sep8 | 25 | 0.5 | low | ALL | 348 | 6 | 0.13812 |
| Sep8 | 25 | 0.5 | middle | ALL | 379 | 6 | 0.2189 |
| Sep8 | 25 | 0.5 | high | ALL | 144 | 6 | 0.44096 |
| Sep19 | 25 | 0.5 | ALL | ALL | 3202 | 22 | 0.25172 |
| Sep19 | 25 | 0.5 | low | ALL | 603 | 22 | 0.13924 |
| Sep19 | 25 | 0.5 | middle | ALL | 1694 | 22 | 0.24086 |
| Sep19 | 25 | 0.5 | high | ALL | 905 | 22 | 0.41145 |

## All horizons and directions

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Actual_vs_Hold | Sep19 | ALL | 0.1 | attitude_error_deg | 1.7365 | 1.6323 | 6.0017 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.1 | body_rate_rmse_rad_s | 0.48667 | 0.46827 | 3.7798 | 3 | 0 | 20 | 2 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.1 | position_rmse_m | 0.034946 | 0.034894 | 0.1486 | 3 | 0 | 16 | 6 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.1 | velocity_rmse_m_s | 0.17947 | 0.17627 | 1.7824 | 3 | 0 | 21 | 1 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.2 | attitude_error_deg | 2.9498 | 2.6755 | 9.2965 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.2 | body_rate_rmse_rad_s | 0.54378 | 0.50005 | 8.0415 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.2 | position_rmse_m | 0.072667 | 0.071851 | 1.123 | 3 | 0 | 21 | 1 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.2 | velocity_rmse_m_s | 0.26054 | 0.24902 | 4.4214 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | attitude_error_deg | 5.8669 | 4.9615 | 15.432 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.63916 | 0.52859 | 17.299 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | position_rmse_m | 0.22545 | 0.20969 | 6.9879 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.56382 | 0.46098 | 18.241 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 1 | attitude_error_deg | 9.7622 | 7.7146 | 20.975 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 1 | body_rate_rmse_rad_s | 0.87121 | 0.55472 | 36.327 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 1 | position_rmse_m | 0.64671 | 0.54763 | 15.32 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | ALL | 1 | velocity_rmse_m_s | 1.1282 | 0.81217 | 28.015 | 3 | 0 | 22 | 0 | 22 | 3202 |
| Actual_vs_Hold | Sep19 | high | 0.1 | attitude_error_deg | 2.1192 | 1.9017 | 10.266 | 3 | 0 | 22 | 0 | 22 | 866 |
| Actual_vs_Hold | Sep19 | high | 0.1 | body_rate_rmse_rad_s | 0.5675 | 0.5387 | 5.0749 | 3 | 0 | 17 | 5 | 22 | 866 |
| Actual_vs_Hold | Sep19 | high | 0.1 | position_rmse_m | 0.038807 | 0.03871 | 0.25071 | 3 | 0 | 18 | 4 | 22 | 866 |
| Actual_vs_Hold | Sep19 | high | 0.1 | velocity_rmse_m_s | 0.22984 | 0.22498 | 2.1122 | 3 | 0 | 19 | 3 | 22 | 866 |
| Actual_vs_Hold | Sep19 | high | 0.2 | attitude_error_deg | 3.8349 | 3.3055 | 13.806 | 3 | 0 | 22 | 0 | 22 | 923 |
| Actual_vs_Hold | Sep19 | high | 0.2 | body_rate_rmse_rad_s | 0.63054 | 0.56436 | 10.495 | 3 | 0 | 22 | 0 | 22 | 923 |
| Actual_vs_Hold | Sep19 | high | 0.2 | position_rmse_m | 0.074351 | 0.073348 | 1.3489 | 3 | 0 | 19 | 3 | 22 | 923 |
| Actual_vs_Hold | Sep19 | high | 0.2 | velocity_rmse_m_s | 0.31698 | 0.29969 | 5.4571 | 3 | 0 | 19 | 3 | 22 | 923 |
| Actual_vs_Hold | Sep19 | high | 0.5 | attitude_error_deg | 7.6533 | 6.2085 | 18.878 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | body_rate_rmse_rad_s | 0.74339 | 0.60632 | 18.438 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | position_rmse_m | 0.2468 | 0.22384 | 9.3034 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 0.5 | velocity_rmse_m_s | 0.72519 | 0.56546 | 22.026 | 3 | 0 | 22 | 0 | 22 | 905 |
| Actual_vs_Hold | Sep19 | high | 1 | attitude_error_deg | 12.229 | 9.3664 | 23.405 | 3 | 0 | 22 | 0 | 22 | 921 |
| Actual_vs_Hold | Sep19 | high | 1 | body_rate_rmse_rad_s | 1.0014 | 0.60964 | 39.122 | 3 | 0 | 22 | 0 | 22 | 921 |
| Actual_vs_Hold | Sep19 | high | 1 | position_rmse_m | 0.73471 | 0.57408 | 21.863 | 3 | 0 | 22 | 0 | 22 | 921 |
| Actual_vs_Hold | Sep19 | high | 1 | velocity_rmse_m_s | 1.4389 | 0.95452 | 33.663 | 3 | 0 | 22 | 0 | 22 | 921 |
| Actual_vs_Hold | Sep19 | low | 0.1 | attitude_error_deg | 1.4746 | 1.4605 | 0.96089 | 3 | 0 | 19 | 3 | 22 | 788 |
| Actual_vs_Hold | Sep19 | low | 0.1 | body_rate_rmse_rad_s | 0.43703 | 0.43279 | 0.97076 | 3 | 0 | 18 | 4 | 22 | 788 |
| Actual_vs_Hold | Sep19 | low | 0.1 | position_rmse_m | 0.0318 | 0.031773 | 0.083401 | 3 | 0 | 16 | 6 | 22 | 788 |
| Actual_vs_Hold | Sep19 | low | 0.1 | velocity_rmse_m_s | 0.14868 | 0.14711 | 1.0585 | 3 | 0 | 18 | 4 | 22 | 788 |
| Actual_vs_Hold | Sep19 | low | 0.2 | attitude_error_deg | 2.3763 | 2.3097 | 2.8 | 3 | 0 | 21 | 1 | 22 | 694 |
| Actual_vs_Hold | Sep19 | low | 0.2 | body_rate_rmse_rad_s | 0.46554 | 0.44906 | 3.5404 | 3 | 0 | 21 | 1 | 22 | 694 |
| Actual_vs_Hold | Sep19 | low | 0.2 | position_rmse_m | 0.072499 | 0.072027 | 0.65036 | 3 | 0 | 19 | 3 | 22 | 694 |
| Actual_vs_Hold | Sep19 | low | 0.2 | velocity_rmse_m_s | 0.22704 | 0.22438 | 1.1695 | 3 | 0 | 15 | 7 | 22 | 694 |
| Actual_vs_Hold | Sep19 | low | 0.5 | attitude_error_deg | 4.2742 | 3.9179 | 8.3359 | 3 | 0 | 21 | 1 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | body_rate_rmse_rad_s | 0.52194 | 0.46654 | 10.613 | 3 | 0 | 22 | 0 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | position_rmse_m | 0.20537 | 0.19735 | 3.9019 | 3 | 0 | 21 | 1 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 0.5 | velocity_rmse_m_s | 0.42668 | 0.39494 | 7.4393 | 3 | 0 | 22 | 0 | 22 | 603 |
| Actual_vs_Hold | Sep19 | low | 1 | attitude_error_deg | 7.5614 | 6.1642 | 18.478 | 3 | 0 | 22 | 0 | 22 | 642 |
| Actual_vs_Hold | Sep19 | low | 1 | body_rate_rmse_rad_s | 0.61978 | 0.49817 | 19.621 | 3 | 0 | 22 | 0 | 22 | 642 |
| Actual_vs_Hold | Sep19 | low | 1 | position_rmse_m | 0.51378 | 0.47446 | 7.6535 | 3 | 0 | 21 | 1 | 22 | 642 |
| Actual_vs_Hold | Sep19 | low | 1 | velocity_rmse_m_s | 0.80082 | 0.7009 | 12.478 | 3 | 0 | 21 | 1 | 22 | 642 |
| Actual_vs_Hold | Sep19 | middle | 0.1 | attitude_error_deg | 1.6204 | 1.5539 | 4.1027 | 3 | 0 | 21 | 1 | 22 | 1548 |
| Actual_vs_Hold | Sep19 | middle | 0.1 | body_rate_rmse_rad_s | 0.45874 | 0.44068 | 3.936 | 3 | 0 | 20 | 2 | 22 | 1548 |
| Actual_vs_Hold | Sep19 | middle | 0.1 | position_rmse_m | 0.034351 | 0.034311 | 0.11487 | 3 | 0 | 16 | 6 | 22 | 1548 |
| Actual_vs_Hold | Sep19 | middle | 0.1 | velocity_rmse_m_s | 0.15673 | 0.1537 | 1.9326 | 3 | 0 | 19 | 3 | 22 | 1548 |
| Actual_vs_Hold | Sep19 | middle | 0.2 | attitude_error_deg | 2.5674 | 2.4092 | 6.1599 | 3 | 0 | 22 | 0 | 22 | 1585 |
| Actual_vs_Hold | Sep19 | middle | 0.2 | body_rate_rmse_rad_s | 0.51557 | 0.47523 | 7.8242 | 3 | 0 | 22 | 0 | 22 | 1585 |
| Actual_vs_Hold | Sep19 | middle | 0.2 | position_rmse_m | 0.0688 | 0.068013 | 1.143 | 3 | 0 | 22 | 0 | 22 | 1585 |
| Actual_vs_Hold | Sep19 | middle | 0.2 | velocity_rmse_m_s | 0.23739 | 0.22581 | 4.8799 | 3 | 0 | 22 | 0 | 22 | 1585 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | attitude_error_deg | 5.2524 | 4.5606 | 13.171 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | body_rate_rmse_rad_s | 0.61773 | 0.50931 | 17.551 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | position_rmse_m | 0.20864 | 0.19559 | 6.2527 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 0.5 | velocity_rmse_m_s | 0.49818 | 0.41184 | 17.331 | 3 | 0 | 22 | 0 | 22 | 1694 |
| Actual_vs_Hold | Sep19 | middle | 1 | attitude_error_deg | 9.0641 | 7.2696 | 19.797 | 3 | 0 | 22 | 0 | 22 | 1639 |
| Actual_vs_Hold | Sep19 | middle | 1 | body_rate_rmse_rad_s | 0.87194 | 0.5419 | 37.851 | 3 | 0 | 22 | 0 | 22 | 1639 |
| Actual_vs_Hold | Sep19 | middle | 1 | position_rmse_m | 0.60557 | 0.52971 | 12.528 | 3 | 0 | 22 | 0 | 22 | 1639 |
| Actual_vs_Hold | Sep19 | middle | 1 | velocity_rmse_m_s | 1.0402 | 0.77043 | 25.935 | 3 | 0 | 22 | 0 | 22 | 1639 |
| Actual_vs_Hold | Sep8 | ALL | 0.1 | attitude_error_deg | 1.3982 | 1.3069 | 6.5302 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.1 | body_rate_rmse_rad_s | 0.54588 | 0.5474 | -0.27813 | 1 | 2 | 4 | 2 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.1 | position_rmse_m | 0.019648 | 0.019486 | 0.82447 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.1 | velocity_rmse_m_s | 0.15677 | 0.14878 | 5.094 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.2 | attitude_error_deg | 2.2869 | 2.0644 | 9.7307 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.2 | body_rate_rmse_rad_s | 0.5424 | 0.5337 | 1.6033 | 3 | 0 | 5 | 1 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.2 | position_rmse_m | 0.041035 | 0.039355 | 4.0959 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.2 | velocity_rmse_m_s | 0.21676 | 0.19527 | 9.912 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | attitude_error_deg | 4.4201 | 3.6977 | 16.343 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.58312 | 0.5298 | 9.1455 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | position_rmse_m | 0.13683 | 0.12052 | 11.92 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.43807 | 0.35328 | 19.355 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 1 | attitude_error_deg | 8.4941 | 5.8066 | 31.64 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.76637 | 0.60658 | 20.85 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 1 | position_rmse_m | 0.42846 | 0.33676 | 21.402 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | ALL | 1 | velocity_rmse_m_s | 0.86714 | 0.60238 | 30.533 | 3 | 0 | 6 | 0 | 6 | 871 |
| Actual_vs_Hold | Sep8 | high | 0.1 | attitude_error_deg | 1.5867 | 1.4001 | 11.763 | 3 | 0 | 6 | 0 | 6 | 228 |
| Actual_vs_Hold | Sep8 | high | 0.1 | body_rate_rmse_rad_s | 0.59533 | 0.59873 | -0.57079 | 1 | 2 | 2 | 4 | 6 | 228 |
| Actual_vs_Hold | Sep8 | high | 0.1 | position_rmse_m | 0.0226 | 0.022423 | 0.7814 | 3 | 0 | 6 | 0 | 6 | 228 |
| Actual_vs_Hold | Sep8 | high | 0.1 | velocity_rmse_m_s | 0.20812 | 0.19928 | 4.2453 | 3 | 0 | 6 | 0 | 6 | 228 |
| Actual_vs_Hold | Sep8 | high | 0.2 | attitude_error_deg | 2.8938 | 2.5107 | 13.241 | 3 | 0 | 6 | 0 | 6 | 183 |
| Actual_vs_Hold | Sep8 | high | 0.2 | body_rate_rmse_rad_s | 0.56993 | 0.54465 | 4.4356 | 3 | 0 | 6 | 0 | 6 | 183 |
| Actual_vs_Hold | Sep8 | high | 0.2 | position_rmse_m | 0.049921 | 0.048364 | 3.1185 | 3 | 0 | 6 | 0 | 6 | 183 |
| Actual_vs_Hold | Sep8 | high | 0.2 | velocity_rmse_m_s | 0.28252 | 0.25904 | 8.3093 | 3 | 0 | 5 | 1 | 6 | 183 |
| Actual_vs_Hold | Sep8 | high | 0.5 | attitude_error_deg | 6.005 | 4.81 | 19.9 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | body_rate_rmse_rad_s | 0.62932 | 0.54489 | 13.417 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | position_rmse_m | 0.17747 | 0.16491 | 7.0761 | 3 | 0 | 5 | 1 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 0.5 | velocity_rmse_m_s | 0.54943 | 0.46513 | 15.343 | 3 | 0 | 6 | 0 | 6 | 144 |
| Actual_vs_Hold | Sep8 | high | 1 | attitude_error_deg | 11.73 | 7.7131 | 34.248 | 3 | 0 | 6 | 0 | 6 | 139 |
| Actual_vs_Hold | Sep8 | high | 1 | body_rate_rmse_rad_s | 0.82784 | 0.60378 | 27.066 | 3 | 0 | 6 | 0 | 6 | 139 |
| Actual_vs_Hold | Sep8 | high | 1 | position_rmse_m | 0.52974 | 0.41064 | 22.484 | 3 | 0 | 6 | 0 | 6 | 139 |
| Actual_vs_Hold | Sep8 | high | 1 | velocity_rmse_m_s | 1.1544 | 0.75266 | 34.8 | 3 | 0 | 6 | 0 | 6 | 139 |
| Actual_vs_Hold | Sep8 | low | 0.1 | attitude_error_deg | 1.2724 | 1.2505 | 1.7165 | 3 | 0 | 6 | 0 | 6 | 227 |
| Actual_vs_Hold | Sep8 | low | 0.1 | body_rate_rmse_rad_s | 0.51397 | 0.5077 | 1.2194 | 3 | 0 | 5 | 1 | 6 | 227 |
| Actual_vs_Hold | Sep8 | low | 0.1 | position_rmse_m | 0.018975 | 0.018906 | 0.36732 | 3 | 0 | 5 | 1 | 6 | 227 |
| Actual_vs_Hold | Sep8 | low | 0.1 | velocity_rmse_m_s | 0.13136 | 0.12679 | 3.4783 | 3 | 0 | 6 | 0 | 6 | 227 |
| Actual_vs_Hold | Sep8 | low | 0.2 | attitude_error_deg | 2.0569 | 1.9713 | 4.1615 | 3 | 0 | 6 | 0 | 6 | 248 |
| Actual_vs_Hold | Sep8 | low | 0.2 | body_rate_rmse_rad_s | 0.54193 | 0.53754 | 0.81187 | 3 | 0 | 4 | 2 | 6 | 248 |
| Actual_vs_Hold | Sep8 | low | 0.2 | position_rmse_m | 0.039742 | 0.038645 | 2.7598 | 3 | 0 | 6 | 0 | 6 | 248 |
| Actual_vs_Hold | Sep8 | low | 0.2 | velocity_rmse_m_s | 0.18434 | 0.1697 | 7.9389 | 3 | 0 | 6 | 0 | 6 | 248 |
| Actual_vs_Hold | Sep8 | low | 0.5 | attitude_error_deg | 3.7431 | 3.3242 | 11.191 | 3 | 0 | 5 | 1 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | body_rate_rmse_rad_s | 0.55911 | 0.52141 | 6.7434 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | position_rmse_m | 0.12038 | 0.10232 | 15.003 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 0.5 | velocity_rmse_m_s | 0.35738 | 0.27124 | 24.104 | 3 | 0 | 6 | 0 | 6 | 348 |
| Actual_vs_Hold | Sep8 | low | 1 | attitude_error_deg | 7.2707 | 5.0448 | 30.614 | 3 | 0 | 6 | 0 | 6 | 377 |
| Actual_vs_Hold | Sep8 | low | 1 | body_rate_rmse_rad_s | 0.70925 | 0.60643 | 14.498 | 3 | 0 | 6 | 0 | 6 | 377 |
| Actual_vs_Hold | Sep8 | low | 1 | position_rmse_m | 0.36786 | 0.27339 | 25.681 | 3 | 0 | 6 | 0 | 6 | 377 |
| Actual_vs_Hold | Sep8 | low | 1 | velocity_rmse_m_s | 0.69492 | 0.48211 | 30.624 | 3 | 0 | 6 | 0 | 6 | 377 |
| Actual_vs_Hold | Sep8 | middle | 0.1 | attitude_error_deg | 1.3482 | 1.2827 | 4.8604 | 3 | 0 | 6 | 0 | 6 | 416 |
| Actual_vs_Hold | Sep8 | middle | 0.1 | body_rate_rmse_rad_s | 0.51616 | 0.51685 | -0.13291 | 2 | 1 | 3 | 3 | 6 | 416 |
| Actual_vs_Hold | Sep8 | middle | 0.1 | position_rmse_m | 0.017685 | 0.017479 | 1.1643 | 3 | 0 | 6 | 0 | 6 | 416 |
| Actual_vs_Hold | Sep8 | middle | 0.1 | velocity_rmse_m_s | 0.1235 | 0.11293 | 8.5564 | 3 | 0 | 6 | 0 | 6 | 416 |
| Actual_vs_Hold | Sep8 | middle | 0.2 | attitude_error_deg | 2.0841 | 1.8772 | 9.929 | 3 | 0 | 6 | 0 | 6 | 440 |
| Actual_vs_Hold | Sep8 | middle | 0.2 | body_rate_rmse_rad_s | 0.52563 | 0.52289 | 0.52141 | 2 | 1 | 4 | 2 | 6 | 440 |
| Actual_vs_Hold | Sep8 | middle | 0.2 | position_rmse_m | 0.036431 | 0.034218 | 6.0761 | 3 | 0 | 6 | 0 | 6 | 440 |
| Actual_vs_Hold | Sep8 | middle | 0.2 | velocity_rmse_m_s | 0.19633 | 0.17031 | 13.249 | 3 | 0 | 6 | 0 | 6 | 440 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | attitude_error_deg | 4.223 | 3.5281 | 16.455 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | body_rate_rmse_rad_s | 0.56835 | 0.51375 | 9.6077 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | position_rmse_m | 0.13172 | 0.11379 | 13.617 | 3 | 0 | 6 | 0 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 0.5 | velocity_rmse_m_s | 0.44908 | 0.35471 | 21.013 | 3 | 0 | 5 | 1 | 6 | 379 |
| Actual_vs_Hold | Sep8 | middle | 1 | attitude_error_deg | 8.1504 | 5.6462 | 30.725 | 3 | 0 | 6 | 0 | 6 | 355 |
| Actual_vs_Hold | Sep8 | middle | 1 | body_rate_rmse_rad_s | 0.79841 | 0.59828 | 25.066 | 3 | 0 | 6 | 0 | 6 | 355 |
| Actual_vs_Hold | Sep8 | middle | 1 | position_rmse_m | 0.43748 | 0.35735 | 18.315 | 3 | 0 | 6 | 0 | 6 | 355 |
| Actual_vs_Hold | Sep8 | middle | 1 | velocity_rmse_m_s | 0.89503 | 0.63224 | 29.36 | 3 | 0 | 6 | 0 | 6 | 355 |
| GRU_vs_MLP | Sep19 | ALL | 0.1 | attitude_error_deg | 2.398 | 1.6323 | 31.931 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.1 | body_rate_rmse_rad_s | 0.69304 | 0.46827 | 32.433 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.1 | position_rmse_m | 0.051141 | 0.034894 | 31.769 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.1 | velocity_rmse_m_s | 0.56678 | 0.17627 | 68.899 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.2 | attitude_error_deg | 3.9316 | 2.6755 | 31.947 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.2 | body_rate_rmse_rad_s | 0.75339 | 0.50005 | 33.626 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.2 | position_rmse_m | 0.10883 | 0.071851 | 33.976 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.2 | velocity_rmse_m_s | 0.49845 | 0.24902 | 50.04 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | attitude_error_deg | 6.9392 | 4.9615 | 28.5 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.74595 | 0.52859 | 29.138 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | position_rmse_m | 0.28466 | 0.20969 | 26.335 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.70362 | 0.46098 | 34.485 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 1 | attitude_error_deg | 9.8867 | 7.7146 | 21.97 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 1 | body_rate_rmse_rad_s | 0.77141 | 0.55472 | 28.09 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 1 | position_rmse_m | 0.71015 | 0.54763 | 22.886 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep19 | ALL | 1 | velocity_rmse_m_s | 1.1136 | 0.81217 | 27.068 | 3 | 0 | 22 | 0 | 22 | 3202 |
| GRU_vs_MLP | Sep8 | ALL | 0.1 | attitude_error_deg | 1.9901 | 1.3069 | 34.328 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.1 | body_rate_rmse_rad_s | 0.64864 | 0.5474 | 15.607 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.1 | position_rmse_m | 0.035546 | 0.019486 | 45.18 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.1 | velocity_rmse_m_s | 0.50373 | 0.14878 | 70.464 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.2 | attitude_error_deg | 2.7725 | 2.0644 | 25.541 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.2 | body_rate_rmse_rad_s | 0.67276 | 0.5337 | 20.669 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.2 | position_rmse_m | 0.07011 | 0.039355 | 43.867 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.2 | velocity_rmse_m_s | 0.37888 | 0.19527 | 48.46 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | attitude_error_deg | 4.551 | 3.6977 | 18.75 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.65764 | 0.5298 | 19.44 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | position_rmse_m | 0.16357 | 0.12052 | 26.315 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.49209 | 0.35328 | 28.208 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 1 | attitude_error_deg | 6.8975 | 5.8066 | 15.816 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.71351 | 0.60658 | 14.986 | 3 | 0 | 6 | 0 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 1 | position_rmse_m | 0.41836 | 0.33676 | 19.504 | 3 | 0 | 5 | 1 | 6 | 871 |
| GRU_vs_MLP | Sep8 | ALL | 1 | velocity_rmse_m_s | 0.79833 | 0.60238 | 24.546 | 3 | 0 | 6 | 0 | 6 | 871 |
| H13_to_H26 | Sep19 | ALL | 0.1 | attitude_error_deg | 1.6405 | 1.6323 | 0.49575 | 2 | 1 | 10 | 12 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.1 | body_rate_rmse_rad_s | 0.47222 | 0.46827 | 0.83702 | 2 | 1 | 18 | 4 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.1 | position_rmse_m | 0.035083 | 0.034894 | 0.53843 | 3 | 0 | 15 | 7 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.1 | velocity_rmse_m_s | 0.18623 | 0.17627 | 5.3481 | 3 | 0 | 20 | 2 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.2 | attitude_error_deg | 2.6842 | 2.6755 | 0.32375 | 2 | 1 | 13 | 9 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.2 | body_rate_rmse_rad_s | 0.50012 | 0.50005 | 0.013146 | 1 | 2 | 8 | 14 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.2 | position_rmse_m | 0.072727 | 0.071851 | 1.204 | 3 | 0 | 18 | 4 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.2 | velocity_rmse_m_s | 0.26534 | 0.24902 | 6.1473 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | attitude_error_deg | 4.9652 | 4.9615 | 0.073583 | 2 | 1 | 11 | 11 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.53349 | 0.52859 | 0.9177 | 2 | 1 | 16 | 6 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | position_rmse_m | 0.21418 | 0.20969 | 2.0944 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.47478 | 0.46098 | 2.9063 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 1 | attitude_error_deg | 7.7422 | 7.7146 | 0.35605 | 2 | 1 | 12 | 10 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 1 | body_rate_rmse_rad_s | 0.56475 | 0.55472 | 1.7759 | 3 | 0 | 20 | 2 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 1 | position_rmse_m | 0.56219 | 0.54763 | 2.5896 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H13_to_H26 | Sep19 | ALL | 1 | velocity_rmse_m_s | 0.83887 | 0.81217 | 3.183 | 3 | 0 | 21 | 1 | 22 | 3202 |
| H13_to_H26 | Sep8 | ALL | 0.1 | attitude_error_deg | 1.3227 | 1.3069 | 1.1909 | 2 | 1 | 5 | 1 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.1 | body_rate_rmse_rad_s | 0.55526 | 0.5474 | 1.4142 | 3 | 0 | 6 | 0 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.1 | position_rmse_m | 0.019755 | 0.019486 | 1.3592 | 3 | 0 | 5 | 1 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.1 | velocity_rmse_m_s | 0.1529 | 0.14878 | 2.6943 | 3 | 0 | 6 | 0 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.2 | attitude_error_deg | 2.0697 | 2.0644 | 0.2591 | 2 | 1 | 4 | 2 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.2 | body_rate_rmse_rad_s | 0.53351 | 0.5337 | -0.035689 | 2 | 1 | 3 | 3 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.2 | position_rmse_m | 0.039963 | 0.039355 | 1.5223 | 3 | 0 | 5 | 1 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.2 | velocity_rmse_m_s | 0.19733 | 0.19527 | 1.0407 | 2 | 1 | 3 | 3 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | attitude_error_deg | 3.7016 | 3.6977 | 0.10508 | 2 | 1 | 4 | 2 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.52681 | 0.5298 | -0.56667 | 1 | 2 | 1 | 5 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | position_rmse_m | 0.12268 | 0.12052 | 1.7582 | 2 | 1 | 5 | 1 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.34942 | 0.35328 | -1.104 | 1 | 2 | 4 | 2 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 1 | attitude_error_deg | 5.9123 | 5.8066 | 1.7881 | 2 | 1 | 6 | 0 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.60405 | 0.60658 | -0.41821 | 2 | 1 | 2 | 4 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 1 | position_rmse_m | 0.34152 | 0.33676 | 1.3933 | 2 | 1 | 5 | 1 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 1 | velocity_rmse_m_s | 0.60745 | 0.60238 | 0.83509 | 2 | 1 | 5 | 1 | 6 | 871 |
| H1_to_H26 | Sep19 | ALL | 0.1 | attitude_error_deg | 2.1969 | 1.6323 | 25.697 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.1 | body_rate_rmse_rad_s | 0.63251 | 0.46827 | 25.966 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.1 | position_rmse_m | 0.050475 | 0.034894 | 30.87 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.1 | velocity_rmse_m_s | 0.51618 | 0.17627 | 65.851 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.2 | attitude_error_deg | 3.3101 | 2.6755 | 19.171 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.2 | body_rate_rmse_rad_s | 0.64142 | 0.50005 | 22.039 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.2 | position_rmse_m | 0.10267 | 0.071851 | 30.016 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.2 | velocity_rmse_m_s | 0.43845 | 0.24902 | 43.203 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | attitude_error_deg | 5.5766 | 4.9615 | 11.03 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | body_rate_rmse_rad_s | 0.57493 | 0.52859 | 8.0597 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | position_rmse_m | 0.24951 | 0.20969 | 15.958 | 3 | 0 | 22 | 0 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 0.5 | velocity_rmse_m_s | 0.53016 | 0.46098 | 13.049 | 3 | 0 | 21 | 1 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 1 | attitude_error_deg | 7.7911 | 7.7146 | 0.98162 | 2 | 1 | 11 | 11 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 1 | body_rate_rmse_rad_s | 0.57832 | 0.55472 | 4.0808 | 3 | 0 | 19 | 3 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 1 | position_rmse_m | 0.58974 | 0.54763 | 7.1402 | 3 | 0 | 20 | 2 | 22 | 3202 |
| H1_to_H26 | Sep19 | ALL | 1 | velocity_rmse_m_s | 0.81498 | 0.81217 | 0.34483 | 2 | 1 | 10 | 12 | 22 | 3202 |
| H1_to_H26 | Sep8 | ALL | 0.1 | attitude_error_deg | 1.9223 | 1.3069 | 32.014 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.1 | body_rate_rmse_rad_s | 0.60985 | 0.5474 | 10.24 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.1 | position_rmse_m | 0.034178 | 0.019486 | 42.986 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.1 | velocity_rmse_m_s | 0.44741 | 0.14878 | 66.746 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.2 | attitude_error_deg | 2.3641 | 2.0644 | 12.679 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.2 | body_rate_rmse_rad_s | 0.6301 | 0.5337 | 15.298 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.2 | position_rmse_m | 0.063654 | 0.039355 | 38.174 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.2 | velocity_rmse_m_s | 0.30174 | 0.19527 | 35.284 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | attitude_error_deg | 3.7222 | 3.6977 | 0.65858 | 2 | 1 | 4 | 2 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.54564 | 0.5298 | 2.9036 | 3 | 0 | 5 | 1 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | position_rmse_m | 0.13795 | 0.12052 | 12.635 | 3 | 0 | 6 | 0 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.37975 | 0.35328 | 6.9688 | 3 | 0 | 5 | 1 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | attitude_error_deg | 5.8228 | 5.8066 | 0.27765 | 2 | 1 | 3 | 3 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.58864 | 0.60658 | -3.0472 | 1 | 2 | 1 | 5 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | position_rmse_m | 0.34792 | 0.33676 | 3.206 | 2 | 1 | 5 | 1 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | velocity_rmse_m_s | 0.59678 | 0.60238 | -0.93839 | 2 | 1 | 5 | 1 | 6 | 871 |

Positive gain favors comparison. Flight directions compare per-flight three-seed error means, not all windows. Relative gains use aggregated errors. No pooled-date mean; no window independent tests; sampleSD is not confidence/uncertainty.

## Validation versus test

| test_date | validation_cohort | model | horizon_s | metric | test_mean | test_seed_sd | validation_mean | validation_seed_sd | test_over_validation_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sep8 | ALL | B0 | 0.5 | position_rmse_m | 0.3628 | — | 0.41818 | — | 0.86757 |
| Sep8 | ALL | B0 | 0.5 | velocity_rmse_m_s | 1.1462 | — | 1.1707 | — | 0.97905 |
| Sep8 | ALL | B0 | 0.5 | attitude_error_deg | 23.979 | — | 23.245 | — | 1.0316 |
| Sep8 | ALL | B0 | 0.5 | body_rate_rmse_rad_s | 0.97789 | — | 1.066 | — | 0.91736 |
| Sep8 | ALL | MLP | 0.5 | position_rmse_m | 0.16357 | 0.003654 | 0.24067 | 0.0018522 | 0.67964 |
| Sep8 | ALL | MLP | 0.5 | velocity_rmse_m_s | 0.49209 | 0.015907 | 0.54218 | 0.009692 | 0.9076 |
| Sep8 | ALL | MLP | 0.5 | attitude_error_deg | 4.551 | 0.067684 | 5.4859 | 0.045136 | 0.82959 |
| Sep8 | ALL | MLP | 0.5 | body_rate_rmse_rad_s | 0.65764 | 0.0060157 | 0.7526 | 0.0014554 | 0.87382 |
| Sep8 | ALL | H26 | 0.5 | position_rmse_m | 0.12052 | 0.0036177 | 0.1924 | 0.0018568 | 0.62643 |
| Sep8 | ALL | H26 | 0.5 | velocity_rmse_m_s | 0.35328 | 0.013161 | 0.35729 | 0.0071649 | 0.98879 |
| Sep8 | ALL | H26 | 0.5 | attitude_error_deg | 3.6977 | 0.18666 | 4.0784 | 0.046072 | 0.90666 |
| Sep8 | ALL | H26 | 0.5 | body_rate_rmse_rad_s | 0.5298 | 0.0082155 | 0.59658 | 0.0028141 | 0.88806 |
| Sep19 | ALL | B0 | 0.5 | position_rmse_m | 0.47688 | — | 0.41818 | — | 1.1404 |
| Sep19 | ALL | B0 | 0.5 | velocity_rmse_m_s | 1.4227 | — | 1.1707 | — | 1.2153 |
| Sep19 | ALL | B0 | 0.5 | attitude_error_deg | 22.882 | — | 23.245 | — | 0.98436 |
| Sep19 | ALL | B0 | 0.5 | body_rate_rmse_rad_s | 1.1018 | — | 1.066 | — | 1.0336 |
| Sep19 | ALL | MLP | 0.5 | position_rmse_m | 0.28466 | 0.0012817 | 0.24067 | 0.0018522 | 1.1828 |
| Sep19 | ALL | MLP | 0.5 | velocity_rmse_m_s | 0.70362 | 0.0033878 | 0.54218 | 0.009692 | 1.2978 |
| Sep19 | ALL | MLP | 0.5 | attitude_error_deg | 6.9392 | 0.020443 | 5.4859 | 0.045136 | 1.2649 |
| Sep19 | ALL | MLP | 0.5 | body_rate_rmse_rad_s | 0.74595 | 0.0019967 | 0.7526 | 0.0014554 | 0.99116 |
| Sep19 | ALL | H26 | 0.5 | position_rmse_m | 0.20969 | 0.0016474 | 0.1924 | 0.0018568 | 1.0899 |
| Sep19 | ALL | H26 | 0.5 | velocity_rmse_m_s | 0.46098 | 0.011184 | 0.35729 | 0.0071649 | 1.2902 |
| Sep19 | ALL | H26 | 0.5 | attitude_error_deg | 4.9615 | 0.036435 | 4.0784 | 0.046072 | 1.2165 |
| Sep19 | ALL | H26 | 0.5 | body_rate_rmse_rad_s | 0.52859 | 0.0042755 | 0.59658 | 0.0028141 | 0.88604 |

## Exceptions and engineering failures

0 invalid model/date/condition runs. 255 nonpositive flight comparisons and 36 nonpositive seed comparisons retained in CSVs.

| comparison | date | group | horizon_s | metric | reference_error | comparison_error | relative_gain_pct | seeds_improved | seeds_worse | flights_improved | flights_worse | n_flights | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Actual_vs_Hold | Sep8 | ALL | 0.1 | body_rate_rmse_rad_s | 0.54588 | 0.5474 | -0.27813 | 1 | 2 | 4 | 2 | 6 | 871 |
| Actual_vs_Hold | Sep8 | high | 0.1 | body_rate_rmse_rad_s | 0.59533 | 0.59873 | -0.57079 | 1 | 2 | 2 | 4 | 6 | 228 |
| Actual_vs_Hold | Sep8 | middle | 0.1 | body_rate_rmse_rad_s | 0.51616 | 0.51685 | -0.13291 | 2 | 1 | 3 | 3 | 6 | 416 |
| H13_to_H26 | Sep8 | ALL | 0.2 | body_rate_rmse_rad_s | 0.53351 | 0.5337 | -0.035689 | 2 | 1 | 3 | 3 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | body_rate_rmse_rad_s | 0.52681 | 0.5298 | -0.56667 | 1 | 2 | 1 | 5 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 0.5 | velocity_rmse_m_s | 0.34942 | 0.35328 | -1.104 | 1 | 2 | 4 | 2 | 6 | 871 |
| H13_to_H26 | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.60405 | 0.60658 | -0.41821 | 2 | 1 | 2 | 4 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | body_rate_rmse_rad_s | 0.58864 | 0.60658 | -3.0472 | 1 | 2 | 1 | 5 | 6 | 871 |
| H1_to_H26 | Sep8 | ALL | 1 | velocity_rmse_m_s | 0.59678 | 0.60238 | -0.93839 | 2 | 1 | 5 | 1 | 6 | 871 |

## Results text

On Sep8, the signed mean reductions in velocity, attitude and body-rate error of Standard GRU/H26 relative to the MLP were 28.21%, 18.75% and 19.44%, respectively, at the nominal500ms horizon (positive denotes lower GRU error). On Sep19, the signed mean reductions in velocity, attitude and body-rate error of Standard GRU/H26 relative to the MLP were 34.49%, 28.50% and 29.14%, respectively, at the nominal500ms horizon (positive denotes lower GRU error). Errors were computed within each flight, averaged equally across flights within each date, and summarized across three seeds. These are descriptive held-out-date results; no window-independence significance test was used.

## Discussion

The test dates were selected and the model checkpoints, preprocessing gates and analysis rules were frozen before this evaluation. Sep8 had prior descriptive quality-audit exposure and Sep19 had prior file-inventory exposure; we do not claim first-ever access to the raw logs. The inherited v2 and v3 flight-admission gates differ, and results apply to eligible windows of these dates, not all recordings or the flight envelope. The MLP/GRU comparison changes capacity and architecture; controlled history comparisons supply separate evidence. Both Actual and Hold forecasts use the same logged-trajectory truth, without matched counterfactual Hold flights. Neither relative accuracy nor small seed SD establishes arbitrary-action causal fidelity, generalization to arbitrary winds or airframes, long-horizon stability, or closed-loop benefits. Sep8 and Sep19 are now used test data for this model version and cannot subsequently be treated as unopened independent tests after development on their results.

## Next step

Prediction experiments can be organized for writing with all admission/negative-result caveats and independent date tables. The next research stage would separately validate input-response direction, delay/amplitude, timing/actuator constraints and bounded-horizon controller robustness, then plan real-flight validation. None is executed here. No commit/push.

## Focused interpretation

主性能结论在两个留出日期得到支持：500 ms三项动力学指标均有3/3配对seed支持GRU；按每flight先平均三seed误差后，Sep8的6/6和Sep19的22/22架flight均支持GRU。这不是每个窗口都改善，也不是基于独立窗口假设的显著性结论。

History结论需要收窄到日期、指标与时域：500 ms下H1→H26在Sep8速度/姿态/角速度改善6.97%/0.66%/2.90%，Sep19改善13.05%/11.03%/8.06%。Sep8姿态仅2/3seed、4/6flight同方向，不能称稳定全面改善。H13→H26在Sep8速度和角速度分别为−1.10%和−0.57%，即H13略好；Sep19三项边际改善为2.91%/0.07%/0.92%，姿态flight方向11胜11负。约240 ms上下文已获取大部分500 ms收益的叙述仍可保留，但更长历史持续稳定更好的说法不受支持。到1 s，Sep8 H1→H26速度和角速度分别−0.94%和−3.05%，对应结论在该条件下未得到支持；Sep19的速度/姿态均值收益也缩小至不足1%。不切换H26，不修改旧Mixed分类，不把历史跨度解释为物理时间常数。

500 ms未来命令信息价值在两日期ALL和high组均得到支持；三项动力学均为3/3seed且每flight三seed均值方向全部正向。ALL速度/姿态/角速度改善为Sep8的19.36%/16.34%/9.15%及Sep19的18.24%/15.43%/17.30%；high组分别15.34%/19.90%/13.42%及22.03%/18.88%/18.44%。不能扩大为所有时域、组别和flight皆改善：Sep8 100 ms ALL角速度由Hold略好0.28%，high组也有0.57%的反向均值；500 ms low组两日期各有姿态flight反例，Sep8 middle速度也有局部反例。Hold无对应反事实实测轨迹，结论仍是增量预测信息。

相对优势与绝对日期变化必须分开：H26在Sep19的500 ms速度、姿态误差较validation ALL分别高29.02%和21.65%，但角速度误差反而低11.40%。因此不能声称所有状态都同等跨日期稳定或全部退化。独立测试主表可作为论文性能叙述的主要依据，适用范围限定于原质量规则准入的两日期日志。
