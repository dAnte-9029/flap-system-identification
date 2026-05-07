# Transformer Training Tricks Trial

## Protocol

This trial starts from the current best Transformer:

```text
history=128
d_model=64
num_layers=2
num_heads=4
dim_feedforward=128
dropout=0.05
```

Added training tricks:

```text
lr_scheduler=warmup_cosine
lr_warmup_ratio=0.05
gradient_clip_norm=1.0
ema_decay=0.999
```

Command:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260507_transformer_training_tricks_final \
  --recipes causal_transformer_paper_no_accel_v2_phase_actuator_airdata \
  --hidden-sizes 64,128 \
  --dropout 0.05 \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --sequence-history-size 128 \
  --sequence-feature-mode phase_actuator_airdata \
  --current-feature-mode remaining_current \
  --transformer-d-model 64 \
  --transformer-num-layers 2 \
  --transformer-num-heads 4 \
  --transformer-dim-feedforward 128 \
  --lr-scheduler warmup_cosine \
  --lr-warmup-ratio 0.05 \
  --gradient-clip-norm 1.0 \
  --ema-decay 0.999 \
  --device cuda:0
```

## Result

```text
config                         val RMSE  val R2    test RMSE  test R2   fy_b R2  mz_b R2  best_epoch
dropout0.05 baseline            0.935145  0.680652  0.892095   0.736392  0.343493 0.605771 14
training tricks combo           0.938424  0.661799  0.889547   0.729475  0.340416 0.599685 12
```

## Decision

Do not promote this exact combo.

It slightly improves test overall RMSE:

```text
(0.892095 - 0.889547) / 0.892095 = 0.29%
```

But it makes validation metrics worse and slightly hurts `fy_b`, `mz_b`, and test overall R2. Under the validation-first protocol, this is not a reliable improvement.

## Interpretation

The result suggests that training tricks are worth exploring, but this exact combination is too heavy-handed or mismatched:

```text
EMA decay 0.999 may be too slow for this training length.
Warmup cosine may need a lower base LR or a shorter warmup.
Gradient clipping did not obviously help by itself.
```

Recommended next small sweep:

```text
baseline dropout0.05
lr_scheduler: none / warmup_cosine
lr: 3e-4 / 7e-4 / 1e-3
ema_decay: 0.0 / 0.99 / 0.995
gradient_clip_norm: none / 1.0
```

Use validation ranking first, then test only locked configs.

## Artifact

```text
artifacts/20260507_transformer_training_tricks_final/baseline_comparison_summary.csv
```
