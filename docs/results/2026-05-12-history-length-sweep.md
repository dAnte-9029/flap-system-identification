# Supplementary Experiment 1: History Length Sweep

Date: 2026-05-12

## Purpose

This experiment checks whether the temporal backbones improve because the aerodynamic wrench depends on multi-step flight history, rather than because the Transformer is simply a newer architecture.

The sweep is a validation-first screen, not a locked final test comparison.

This result should be used as the first supporting experiment for the paper claim that the effective aerodynamic wrench is history-dependent. It is not intended to be the final model-ranking table.

## Protocol

- Split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Stage: `history_length`
- Output: `artifacts/20260512_history_length_screen/`
- Feature set: `paper_no_accel_v2`
- Sequence features: `phase_actuator_airdata`
- Current features: `remaining_current`
- No acceleration inputs
- No past wrench / target history
- Selection metric: validation metrics only

History lengths:

```text
H = 1, 16, 32, 64, 128, 256
```

Backbones:

```text
MLP current-time anchor
GRU
TCN
Transformer
```

The MLP is included as an instantaneous/current-time anchor. The temporal models are run at each history length.

## Compact Results

Validation-only summary:

| backbone | H | val RMSE | val R2 | fx R2 | fy R2 | fz R2 | mx R2 | my R2 | mz R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 1 | 1.296 | 0.610 | 0.872 | 0.228 | 0.937 | 0.410 | 0.828 | 0.383 |
| GRU | 1 | 1.306 | 0.594 | 0.871 | 0.213 | 0.936 | 0.354 | 0.834 | 0.353 |
| GRU | 16 | 1.204 | 0.615 | 0.912 | 0.259 | 0.941 | 0.403 | 0.846 | 0.327 |
| GRU | 32 | 1.262 | 0.604 | 0.880 | 0.240 | 0.941 | 0.402 | 0.842 | 0.321 |
| GRU | 64 | 1.169 | 0.616 | 0.928 | 0.235 | 0.941 | 0.403 | 0.840 | 0.346 |
| GRU | 128 | 1.102 | 0.623 | 0.946 | 0.250 | 0.945 | 0.407 | 0.845 | 0.345 |
| GRU | 256 | 1.111 | 0.607 | 0.945 | 0.197 | 0.945 | 0.382 | 0.837 | 0.336 |
| TCN | 1 | 1.251 | 0.604 | 0.885 | 0.226 | 0.941 | 0.393 | 0.843 | 0.336 |
| TCN | 16 | 1.223 | 0.615 | 0.903 | 0.242 | 0.941 | 0.422 | 0.849 | 0.333 |
| TCN | 32 | 1.252 | 0.615 | 0.890 | 0.241 | 0.940 | 0.415 | 0.853 | 0.349 |
| TCN | 64 | 1.147 | 0.622 | 0.929 | 0.236 | 0.944 | 0.416 | 0.853 | 0.353 |
| TCN | 128 | 1.128 | 0.618 | 0.943 | 0.216 | 0.943 | 0.395 | 0.843 | 0.369 |
| TCN | 256 | 1.143 | 0.605 | 0.941 | 0.203 | 0.941 | 0.381 | 0.840 | 0.325 |
| Transformer | 1 | 1.285 | 0.606 | 0.878 | 0.236 | 0.937 | 0.388 | 0.840 | 0.358 |
| Transformer | 16 | 1.133 | 0.641 | 0.920 | 0.279 | 0.950 | 0.450 | 0.865 | 0.382 |
| Transformer | 32 | 1.212 | 0.628 | 0.882 | 0.262 | 0.948 | 0.437 | 0.857 | 0.382 |
| Transformer | 64 | 1.077 | 0.637 | 0.937 | 0.248 | 0.952 | 0.435 | 0.866 | 0.385 |
| Transformer | 128 | 1.059 | 0.636 | 0.952 | 0.259 | 0.950 | 0.425 | 0.865 | 0.362 |
| Transformer | 256 | 1.054 | 0.622 | 0.951 | 0.226 | 0.952 | 0.388 | 0.860 | 0.356 |

Full CSV:

```text
artifacts/20260512_history_length_screen/history_length_compact_summary.csv
```

Figures:

```text
artifacts/20260512_history_length_screen/history_length_val_rmse_r2.png
artifacts/20260512_history_length_screen/history_length_transformer_lateral_r2.png
```

## Interpretation

The result supports the need for temporal context:

- At `H=1`, GRU/TCN/Transformer are close to the MLP anchor.
- Increasing history generally improves validation RMSE for all temporal backbones.
- The best validation RMSE for each temporal family occurs at longer history:
  - GRU: `H=128`
  - TCN: `H=128`
  - Transformer: `H=256` by RMSE, with `H=64/128` close and often better on lateral/hard targets.

The effect is not monotonic. Very long history can hurt some lateral-directional metrics:

```text
Transformer fy_b R2:
H=16  -> 0.279
H=128 -> 0.259
H=256 -> 0.226
```

Therefore the correct claim is not "longer is always better." The stronger claim is:

> The model benefits from a finite multi-step history window, with diminishing returns and possible lateral-channel degradation when the history is too long.

There is also a metric-dependent ranking difference. The best Transformer by validation RMSE is `H=256` (`val RMSE=1.054`), while the best Transformer by aggregate validation R2 is `H=16` (`val R2=0.641`). This means the history-length conclusion should not rely on a single scalar metric. For this project, aggregate RMSE, aggregate R2, and per-target R2 should be reported together because the six wrench channels have very different scales and noise levels.

## Why H=32 Can Be Worse Than H=16

The `H=32` result should be treated as a non-monotonic screening result, not as evidence that 32 samples is physically worse than 16 samples. Several practical factors can explain the dip.

First, this was a validation screen with fixed sampled subsets and early stopping, not a multi-seed group K-fold estimate. Small differences between nearby history lengths can therefore reflect sampling and optimization variance.

Second, changing history length changes the effective training examples. A causal sequence model cannot use the first `H-1` samples of each valid segment in the same way, so larger `H` changes which time points are available after window construction. Because the data contain segment boundaries and skipped invalid regions, this can slightly change the regime mixture seen by the model.

Third, `H=32` may be an awkward intermediate context length for this dataset. The median flapping frequency is roughly 4.36 Hz, so one wingbeat period is about 0.229 s. If the sequence samples are near 100 Hz, then one wingbeat is about 23 samples. Under that approximation, `H=16` captures short-term delay within one wingbeat, `H=32` covers about 1.4 wingbeats, `H=64` covers about 2.8 wingbeats, and `H=128` covers about 5.6 wingbeats. A window around `H=32` adds extra context beyond the immediate phase-local neighborhood, but may not yet provide stable multi-cycle information. This can increase optimization burden without adding consistently useful memory.

Fourth, the lateral-directional targets, especially `fy_b`, contain low-confidence broadband transients and weaker excitation than the longitudinal channels. Longer windows can expose the model to more lateral high-frequency variation that is not fully predictable from the measured inputs. This can reduce per-target R2 even when the longitudinal channels benefit.

The practical conclusion is that history length is a hyperparameter with a non-monotonic response. The useful paper claim is not that performance increases smoothly with `H`, but that temporal context matters and that the Transformer advantage emerges once a causal history window is available.

## Why This Supports The Transformer Story

The Transformer is not better just because it is a more modern architecture. In this screen:

- At `H=1`, the Transformer is not clearly better than the MLP anchor.
- With nontrivial history (`H=16/64/128`), the Transformer becomes the strongest validation model overall.
- This is consistent with the aerodynamic wrench depending on ordered multi-step context, flapping phase history, control-state delay, and nonlocal temporal correlations.

Useful paper wording:

> The history-length sweep shows that instantaneous inputs are insufficient to fully explain the effective aerodynamic wrench. Temporal models improve as the history window covers multiple recent samples, and the Transformer benefits most from this setting. This suggests that the attention-based backbone is useful because it can select informative time instants across the causal history, rather than because of architectural complexity alone.

More concise manuscript wording:

> A history-length sweep was used to test whether the improvement of temporal backbones was associated with flight-history information. When the history length was reduced to one sample, the Transformer was close to the MLP anchor, indicating that the gain was not simply due to model class. With a causal multi-step history, the Transformer achieved the strongest validation performance, supporting the hypothesis that effective aerodynamic wrench estimation benefits from nonlocal temporal context, flapping phase history, and control-state delays.

Recommended figure caption:

> Validation performance across causal history lengths for MLP, GRU, TCN, and Transformer backbones. The Transformer does not dominate in the current-time setting (`H=1`), but improves when multi-step history is available, suggesting that attention is useful for selecting informative samples across recent flight history. The non-monotonic response indicates that history length is a tuned modeling choice rather than a parameter that should be maximized blindly.

## Caveats

This is still a screen:

- It used validation metrics only.
- It did not run group K-fold.
- It did not include temporal order ablations.
- It did not include frequency-resolved backbone metrics.

The next most important check is temporal order ablation:

```text
normal ordered history
no positional encoding
shuffle history order
reverse history order
H=1 only
```

That will test whether the model uses ordered history rather than only the distribution of features inside the window.
