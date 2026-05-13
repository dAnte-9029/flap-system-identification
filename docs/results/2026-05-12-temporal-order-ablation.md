# Supplementary Experiment 2: Temporal Order Ablation

Date: 2026-05-12

## Purpose

This experiment checks whether the locked Transformer uses ordered causal history, rather than only the unordered distribution of features inside the history window.

It complements the history-length sweep. The history-length sweep showed that multi-step history helps. This temporal-order ablation asks a stricter question: if the same history samples are present but their order is broken, does the model fail?

## Protocol

- Split: `dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log`
- Stage: `temporal_order`
- Output: `artifacts/20260512_temporal_order_ablation/`
- Feature set: `paper_no_accel_v2`
- Sequence features: `phase_actuator_airdata`
- Current features: `remaining_current`
- No acceleration inputs
- No past wrench / target history
- Training screen selection: validation metrics only
- Device: `cuda:0`

The locked Transformer setting is:

```text
backbone: causal Transformer
history: 128
d_model: 64
layers: 2
heads: 4
dropout: 0.05
```

Two types of ablation were used.

First, training-time ablations retrained the model under:

```text
normal order, H=128
no positional encoding, H=128
H=1 only
```

Second, evaluation-time ablations reused the normal-order trained model and changed only the order of the history window during evaluation:

```text
normal order
reverse history order
shuffle history order
```

This second test is the cleanest evidence for temporal-order dependence because the model weights are identical and only the input order is perturbed.

## Training-Time Results

Validation-only summary:

| config | H | positional encoding | val RMSE | val R2 | fx R2 | fy R2 | fz R2 | mx R2 | my R2 | mz R2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 128 | yes | 1.059 | 0.636 | 0.952 | 0.259 | 0.950 | 0.425 | 0.865 | 0.362 |
| no positional encoding | 128 | no | 1.087 | 0.625 | 0.949 | 0.231 | 0.947 | 0.393 | 0.859 | 0.370 |
| H=1 only | 1 | yes | 1.285 | 0.606 | 0.878 | 0.236 | 0.937 | 0.388 | 0.840 | 0.358 |

Compact CSV:

```text
artifacts/20260512_temporal_order_ablation/temporal_order_training_compact_summary.csv
```

The normal `H=128` model is best. Removing positional encoding degrades validation RMSE from `1.059` to `1.087`, and reducing the model to current-time history (`H=1`) degrades validation RMSE further to `1.285`. This supports the claim that the Transformer benefits from multi-step causal history and from explicit position information.

The no-positional-encoding result is not catastrophic because the sequence features include phase, actuator, and airdata histories. Those features themselves contain temporal structure, and the causal mask plus final-token readout also impose some asymmetry. Therefore, the expected result is degradation, not necessarily complete failure.

## Evaluation-Time Order Perturbation

The normal `H=128` Transformer was evaluated on the full validation and test splits after changing the order of each history window.

| split | order | RMSE | R2 | fx R2 | fy R2 | fz R2 | mx R2 | my R2 | mz R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | normal | 1.062 | 0.632 | 0.953 | 0.253 | 0.950 | 0.418 | 0.865 | 0.354 |
| val | reverse | 5.913 | -0.638 | -1.063 | -0.273 | -0.843 | -0.520 | -0.801 | -0.327 |
| val | shuffle | 5.805 | -0.628 | -0.986 | -0.287 | -0.776 | -0.555 | -0.829 | -0.334 |
| test | normal | 0.978 | 0.742 | 0.963 | 0.433 | 0.956 | 0.596 | 0.895 | 0.611 |
| test | reverse | 5.933 | -0.657 | -1.015 | -0.278 | -0.936 | -0.701 | -0.679 | -0.332 |
| test | shuffle | 5.788 | -0.611 | -0.954 | -0.245 | -0.831 | -0.646 | -0.705 | -0.283 |

Compact CSV:

```text
artifacts/20260512_temporal_order_ablation/temporal_order_eval_compact_summary.csv
```

This is strong evidence that the model relies on ordered temporal context. On the same trained model, reversing or shuffling the history window collapses the aggregate R2 from positive values to negative values on both validation and test. The longitudinal force channels are especially sensitive, with `fx_b` and `fz_b` R2 becoming strongly negative under order perturbation.

The result means the model is not merely averaging over a bag of recent phase/control/airdata samples. It is using the sequence order to interpret where each sample lies relative to the prediction time.

## Interpretation

This experiment supports a more specific Transformer contribution than "Transformer is more advanced." The useful claim is:

> The effective aerodynamic wrench depends on ordered causal flight history. A Transformer can exploit this because attention operates over the full history window while retaining positional information. When temporal order is removed by reversing or shuffling the history, performance collapses, confirming that the model uses ordered temporal context rather than only instantaneous or unordered feature statistics.

For the manuscript, this experiment should be presented together with the history-length sweep:

```text
history-length sweep:
  multi-step history helps

temporal-order ablation:
  the order of that history matters
```

Together, these two experiments justify using a causal temporal backbone and make the Transformer explanation more defensible.

## Caveats

This result does not prove physical causality by itself. It shows that the trained model depends on ordered history. The physical interpretation still comes from the leakage-resistant protocol, the whole-log split, no acceleration inputs, no past target inputs, phase/frequency features, and frequency/phase diagnostics.

The shuffle/reverse ablations are intentionally harsh out-of-distribution perturbations. Their role is not to estimate deployment performance, but to test whether the model has learned a temporal-order-sensitive representation.

## Paper Wording

Recommended concise wording:

> To verify that the temporal backbone used ordered history rather than only a set of recent feature values, we performed a temporal-order ablation. Removing positional encoding moderately degraded validation performance, while reducing the history to a single sample produced a larger drop. More importantly, when the trained Transformer was evaluated with reversed or shuffled history windows, performance collapsed on both validation and test splits. This confirms that the model exploits ordered causal history, supporting the use of an attention-based temporal backbone for effective aerodynamic wrench estimation.

