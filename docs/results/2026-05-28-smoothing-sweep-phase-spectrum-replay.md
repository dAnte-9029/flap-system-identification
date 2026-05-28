# Smoothing Sweep, Removed-Content Structure, and Replay Diagnostics

## Scope

This report compares Savitzky-Golay derivative smoothing windows of `0.03 s`, `0.06 s`, and `0.12 s` for effective-wrench label reconstruction. It addresses whether smoothing mostly removes unreliable derivative artifacts or erases real flapping-wing dynamics.

Input raw split:

```text
dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1
```

Candidate splits:

```text
dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p03_v1
dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p06_v1
dataset/canonical_v0.2_training_ready_split_hq_v5_smoothed_time_aligned_wrench_sg0p12_v1
```

Artifact root:

```text
artifacts/20260528_smoothing_sweep_wrench_labels_v1
```

## Diagnostic 1: Raw-Clean Label Similarity and Spike Reduction

Test split summary:

| candidate | channel | corr clean/raw | RMSE clean vs raw | p99 clean | p99 jump clean | clean high-pass frac 8 Hz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sg0p03` | `fx_b` | 0.965 | 1.264 | 16.47 | 6.00 | 0.309 |
| `sg0p03` | `fy_b` | 0.685 | 1.059 | 2.98 | 3.46 | 0.711 |
| `sg0p03` | `fz_b` | 0.974 | 2.125 | 27.06 | 8.03 | 0.056 |
| `sg0p03` | `mx_b` | 0.756 | 0.00221 | 0.00831 | 0.01006 | 0.701 |
| `sg0p03` | `my_b` | 0.933 | 0.00157 | 0.00940 | 0.00608 | 0.578 |
| `sg0p03` | `mz_b` | 0.803 | 0.00031 | 0.00139 | 0.00124 | 0.820 |
| `sg0p06` | `fx_b` | 0.948 | 1.639 | 13.80 | 3.46 | 0.191 |
| `sg0p06` | `fy_b` | 0.441 | 1.297 | 1.59 | 0.76 | 0.312 |
| `sg0p06` | `fz_b` | 0.963 | 2.596 | 23.56 | 4.31 | 0.011 |
| `sg0p06` | `mx_b` | 0.525 | 0.00283 | 0.00423 | 0.00206 | 0.248 |
| `sg0p06` | `my_b` | 0.902 | 0.00199 | 0.00640 | 0.00317 | 0.460 |
| `sg0p06` | `mz_b` | 0.681 | 0.00039 | 0.00072 | 0.00041 | 0.512 |
| `sg0p12` | `fx_b` | 0.854 | 2.779 | 9.60 | 1.58 | 0.032 |
| `sg0p12` | `fy_b` | 0.337 | 1.362 | 1.07 | 0.28 | 0.069 |
| `sg0p12` | `fz_b` | 0.945 | 3.702 | 19.68 | 3.35 | 0.002 |
| `sg0p12` | `mx_b` | 0.477 | 0.00296 | 0.00281 | 0.00090 | 0.039 |
| `sg0p12` | `my_b` | 0.768 | 0.00323 | 0.00354 | 0.00124 | 0.150 |
| `sg0p12` | `mz_b` | 0.389 | 0.00047 | 0.00040 | 0.00014 | 0.085 |

Interpretation: `0.03 s` preserves raw-label structure best, but leaves substantial high-frequency content in `fy_b` and moment channels. `0.12 s` suppresses spikes strongly, but changes several channels too much. `0.06 s` is intermediate.

## Diagnostic 2: Phase and Spectrum of Removed Content

Removed content is defined as:

```text
removed = raw_label - clean_label
```

Phase-coherent fraction is the RMSE of phase-binned removed medians divided by the RMSE of the removed signal. Larger values mean the removed part is more wingbeat-phase-locked and therefore more likely to contain real repeatable flapping dynamics.

| candidate | channel | removed RMSE | phase coherent fraction | phase peak-to-peak |
| --- | --- | ---: | ---: | ---: |
| `sg0p03` | `fx_b` | 1.264 | 0.708 | 3.196 |
| `sg0p03` | `fy_b` | 1.059 | 0.175 | 0.765 |
| `sg0p03` | `fz_b` | 2.125 | 0.581 | 4.428 |
| `sg0p03` | `mx_b` | 0.00221 | 0.285 | 0.00388 |
| `sg0p03` | `my_b` | 0.00157 | 0.673 | 0.00367 |
| `sg0p03` | `mz_b` | 0.00031 | 0.290 | 0.00030 |
| `sg0p06` | `fx_b` | 1.639 | 0.751 | 4.963 |
| `sg0p06` | `fy_b` | 1.297 | 0.220 | 1.330 |
| `sg0p06` | `fz_b` | 2.596 | 0.574 | 5.402 |
| `sg0p06` | `mx_b` | 0.00283 | 0.294 | 0.00374 |
| `sg0p06` | `my_b` | 0.00199 | 0.626 | 0.00378 |
| `sg0p06` | `mz_b` | 0.00039 | 0.326 | 0.00043 |
| `sg0p12` | `fx_b` | 2.779 | 0.872 | 9.532 |
| `sg0p12` | `fy_b` | 1.362 | 0.264 | 1.767 |
| `sg0p12` | `fz_b` | 3.702 | 0.673 | 10.51 |
| `sg0p12` | `mx_b` | 0.00296 | 0.342 | 0.00457 |
| `sg0p12` | `my_b` | 0.00323 | 0.722 | 0.00709 |
| `sg0p12` | `mz_b` | 0.00047 | 0.419 | 0.00074 |

Selected frequency fractions for removed content:

| candidate | channel | 1f fraction | 2f fraction | broadband high 8-25 Hz excluding 1f/2f |
| --- | --- | ---: | ---: | ---: |
| `sg0p03` | `fx_b` | 0.156 | 0.266 | 0.274 |
| `sg0p03` | `fy_b` | 0.006 | 0.015 | 0.185 |
| `sg0p03` | `fz_b` | 0.339 | 0.007 | 0.238 |
| `sg0p03` | `mx_b` | 0.015 | 0.019 | 0.240 |
| `sg0p03` | `my_b` | 0.043 | 0.369 | 0.212 |
| `sg0p03` | `mz_b` | 0.010 | 0.033 | 0.466 |
| `sg0p06` | `fx_b` | 0.108 | 0.287 | 0.378 |
| `sg0p06` | `fy_b` | 0.005 | 0.019 | 0.295 |
| `sg0p06` | `fz_b` | 0.265 | 0.007 | 0.368 |
| `sg0p06` | `my_b` | 0.029 | 0.345 | 0.247 |
| `sg0p12` | `fx_b` | 0.138 | 0.462 | 0.273 |
| `sg0p12` | `fy_b` | 0.016 | 0.077 | 0.312 |
| `sg0p12` | `fz_b` | 0.467 | 0.016 | 0.342 |
| `sg0p12` | `my_b` | 0.027 | 0.568 | 0.172 |

Interpretation:

- `fy_b`: removed content has weak phase coherence and very little 1f/2f energy. This supports the hypothesis that a large part of raw `fy_b` is low-SNR or artifact-dominated.
- `fx_b`, `fz_b`, and `my_b`: removed content is strongly phase-structured. Larger windows remove content that may include real repeatable flapping dynamics.
- `mz_b`: removed content is mostly broadband high-frequency, consistent with a fragile low-SNR yaw-moment label.

## Diagnostic 3: Short-Horizon Replay

Teacher-forced oracle replay uses logged attitude for force integration. Coupled oracle replay integrates attitude and body rate as well.

| mode | candidate | horizon | median pos err | median vel err | median att err | median rate err |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| teacher-forced | raw | 0.25 | 0.0528 | 0.0640 | 5.323 | 0.428 |
| teacher-forced | `sg0p03` | 0.25 | 0.0467 | 0.0305 | 2.650 | 0.213 |
| teacher-forced | `sg0p06` | 0.25 | 0.0477 | 0.0386 | 3.609 | 0.292 |
| teacher-forced | `sg0p12` | 0.25 | 0.0513 | 0.0637 | 6.296 | 0.434 |
| teacher-forced | raw | 0.50 | 0.1046 | 0.1019 | 10.70 | 0.536 |
| teacher-forced | `sg0p03` | 0.50 | 0.0899 | 0.0493 | 5.223 | 0.256 |
| teacher-forced | `sg0p06` | 0.50 | 0.0913 | 0.0597 | 7.142 | 0.343 |
| teacher-forced | `sg0p12` | 0.50 | 0.0999 | 0.1023 | 12.58 | 0.567 |
| coupled | raw | 0.25 | 0.0540 | 0.1424 | 5.322 | 0.432 |
| coupled | `sg0p03` | 0.25 | 0.0472 | 0.0740 | 2.651 | 0.213 |
| coupled | `sg0p06` | 0.25 | 0.0481 | 0.0965 | 3.611 | 0.294 |
| coupled | `sg0p12` | 0.25 | 0.0534 | 0.1558 | 6.306 | 0.439 |
| coupled | raw | 0.50 | 0.1359 | 0.4661 | 10.77 | 0.539 |
| coupled | `sg0p03` | 0.50 | 0.1016 | 0.2313 | 5.245 | 0.257 |
| coupled | `sg0p06` | 0.50 | 0.1116 | 0.3225 | 7.176 | 0.345 |
| coupled | `sg0p12` | 0.50 | 0.1420 | 0.5570 | 12.60 | 0.570 |

Interpretation: `sg0p03` is the best replay candidate among the tested windows. It improves both translational and rotational oracle replay substantially relative to raw labels. `sg0p06` still improves replay, but less. `sg0p12` is too aggressive and can be worse than raw.

## Overall Judgment

The data support the following channel-level interpretation:

- `fy_b` is likely low-SNR in the raw inverse-dynamics label. Its removed content is weakly phase-coherent and weakly concentrated at flapping harmonics, while smoothing improves replay. This suggests that smoothing removes mostly unreliable lateral-force content, though `0.12 s` still looks too aggressive.
- `fx_b` and `fz_b` contain real phase-locked removed content. Large smoothing windows remove repeatable wingbeat structure, so they should use the most conservative window that improves replay.
- `my_b` also contains phase/harmonic structure in the removed content. Treat moment smoothing cautiously even though replay improves.
- `mx_b` and `mz_b` look more noise/artifact dominated, but rotational labels remain tied to inertia, CG, and angular-acceleration assumptions.

Recommended next working choice:

```text
Use sg0p03 as the primary clean-label candidate for downstream B+C correction and paper diagnostics.
Keep sg0p06 as a sensitivity comparison.
Do not use sg0p12 as the main label set.
```

For the paper, the safe claim is:

```text
Short-window derivative smoothing reduces replay-inconsistent spikes while preserving most raw-label structure. Longer smoothing windows remove more high-frequency content but begin to suppress repeatable wingbeat-phase structure, especially in fx_b, fz_b, and my_b.
```

