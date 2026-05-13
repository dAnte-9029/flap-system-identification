# Supplementary Experiment 3: Frequency-Resolved Backbone Comparison

Date: 2026-05-12

## Purpose

This experiment compares the main backbones by frequency component instead of using only raw aggregate metrics. The goal is to check whether Transformer-style models are better because they capture structured aerodynamic content, especially wingbeat-periodic components, rather than only improving a global R2 number.

## Protocol

- Split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Evaluation splits: `val`, `test`
- Feature protocol: no acceleration inputs, whole-log split, no past wrench history
- Output: `artifacts/20260512_frequency_resolved_backbone_comparison_v2/`
- Script: `scripts/frequency_resolved_backbone_comparison.py`

The compared models are:

```text
MLP
GRU
TCN
TCN+GRU
Transformer
Head FiLM Transformer
```

The frequency components are:

```text
0-1 Hz
1-3 Hz
flap-main band: median local flapping frequency +/- 0.75 Hz
high-frequency residual: raw signal minus 0-1 Hz, 1-3 Hz, and flap-main components
```

The main table below uses mean per-target R2 across all six wrench channels. Per-target tables are also saved because the six channels have very different scales and low-frequency variances.

## Main Mean R2 Results

Validation:

| model | 0-1 Hz | 1-3 Hz | flap-main | high-frequency residual |
|---|---:|---:|---:|---:|
| MLP | -10.103 | 0.163 | 0.729 | 0.553 |
| GRU | -10.506 | 0.343 | 0.753 | 0.616 |
| TCN | -11.882 | 0.078 | 0.693 | 0.551 |
| TCN+GRU | -15.462 | 0.212 | 0.718 | 0.551 |
| Transformer | -6.851 | 0.419 | 0.714 | 0.638 |
| Head FiLM | -6.074 | 0.436 | 0.721 | 0.655 |

Test:

| model | 0-1 Hz | 1-3 Hz | flap-main | high-frequency residual |
|---|---:|---:|---:|---:|
| MLP | -7.175 | 0.078 | 0.812 | 0.651 |
| GRU | -6.288 | 0.376 | 0.812 | 0.744 |
| TCN | -5.715 | -0.062 | 0.828 | 0.638 |
| TCN+GRU | -7.234 | 0.041 | 0.784 | 0.634 |
| Transformer | -3.374 | 0.310 | 0.896 | 0.761 |
| Head FiLM | -2.354 | 0.396 | 0.881 | 0.769 |

Figures:

```text
artifacts/20260512_frequency_resolved_backbone_comparison_v2/frequency_resolved_mean_r2.png
artifacts/20260512_frequency_resolved_backbone_comparison_v2/frequency_resolved_mean_rmse.png
```

CSV outputs:

```text
artifacts/20260512_frequency_resolved_backbone_comparison_v2/frequency_resolved_backbone_summary.csv
artifacts/20260512_frequency_resolved_backbone_comparison_v2/frequency_resolved_backbone_mean_summary.csv
artifacts/20260512_frequency_resolved_backbone_comparison_v2/test_per_target_r2_table_for_md.csv
```

## Interpretation

The most important result is that the Transformer-family models are strongest on structured periodic content. On the test split, the baseline Transformer has the best mean flap-main R2 (`0.896`), and Head FiLM is close (`0.881`). Both are clearly above MLP, GRU, TCN, and TCN+GRU in this band. This supports the claim that the Transformer is useful for wingbeat-related aerodynamic structure, not merely for global curve fitting.

Head FiLM is strongest on the high-frequency residual mean R2 (`0.769` test), with the baseline Transformer close (`0.761`) and GRU also competitive (`0.744`). This does not mean all residual high-frequency content is physically reliable. It means that after removing low-frequency, mid-frequency, and flap-main components, some remaining structure is still predictable, especially by the Transformer-family models. This is consistent with earlier `fy_b` diagnostics: high-frequency content includes both phase-locked aerodynamic information and broadband low-confidence transients.

The `0-1 Hz` mean R2 is negative for all models and should not be used alone as a headline metric. Per-target inspection shows that major force channels can have reasonable low-frequency R2, but very low-variance moment channels such as `mx_b` and `my_b` produce large negative R2 values and dominate the simple six-channel average. Therefore, for this band, RMSE and per-target R2 are more informative than mean R2.

## Control-Relevant Test Channels

For the more control-relevant longitudinal channels, Transformer-family models are consistently strong. On the test split:

```text
fx_b flap-main R2:
Transformer 0.997
Head FiLM   0.997

fz_b flap-main R2:
Transformer 0.997
Head FiLM   0.998

my_b flap-main R2:
Transformer 0.965
Head FiLM   0.958
```

For lateral-directional channels, Head FiLM improves or remains competitive:

```text
fy_b flap-main R2:
Transformer 0.809
Head FiLM   0.847

mx_b high-frequency residual R2:
Transformer 0.715
Head FiLM   0.713

mz_b high-frequency residual R2:
Transformer 0.686
Head FiLM   0.701
```

This is a useful paper point: the model is strongest where the signal has structured periodic content, and the remaining weaknesses are concentrated in low-variance low-frequency moment components and noisy lateral-directional residuals.

## Caveats

The model sources are not perfectly identical in training budget. The main `Transformer` and `Head FiLM` models come from the no-suspect full-final phase-FiLM experiment. MLP and GRU were rerun on the no-suspect split for this comparison. TCN was taken from the no-suspect history-length screen, and TCN+GRU was rerun as a no-suspect screen-level model because full-budget TCN/TCN+GRU sequence construction was too slow in this session. Therefore, this result is best used as a frequency-resolved diagnostic, not as the final locked model-ranking table.

A screen-matched diagnostic was also produced:

```text
artifacts/20260512_frequency_resolved_backbone_comparison_screen_matched/
```

It shows the same broad pattern: Transformer-family models remain strong on flap-main and high-frequency residual components. The full-final comparison should be used for the main narrative, while the screen-matched comparison can be used as a robustness check.

## Paper Wording

Recommended concise wording:

> A frequency-resolved backbone comparison was performed to determine whether the temporal models improved specific aerodynamic components rather than only aggregate error. The Transformer-family models achieved the strongest performance around the flapping fundamental frequency and remained competitive on the high-frequency residual component. This supports the interpretation that attention-based temporal modeling is beneficial for structured wingbeat-periodic aerodynamic content. The low-frequency mean R2 was not used as a primary conclusion because several moment channels have very small low-frequency variance, making R2 unstable; per-channel diagnostics were therefore reported alongside aggregate values.

