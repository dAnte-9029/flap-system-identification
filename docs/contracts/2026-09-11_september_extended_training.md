# Fixed 120-epoch full-2s training diagnostic

User authorized extending optimization with frozen 40/80/120 epoch evaluations.
Keep the September registered dataset, 4,198 training windows, normalization,
64-unit controlled GRU, 26 history samples, 2-second objective, seed 17, batch 256,
AdamW settings and GPU 1 unchanged. Do not tune from validation or open September 8.

The original checkpoint lacks optimizer/RNG state. Replay from the same seed in
one uninterrupted 120-epoch training run; never restart AdamW on old weights and
call it exact continuation. At epoch 40 require bitwise equality of all model
state tensors with the original full_2s checkpoint; fail closed on mismatch.
Save model, optimizer and RNG state at epochs 40/80/120. A passive epoch callback
must leave the original optimizer updates unchanged. Evaluation runs after all
training completes so it cannot perturb the training trajectory.

Evaluate each checkpoint on all original train origins at 1/2 seconds, on the
same 3,742 five-second-eligible train origins at 1/2/3/5 seconds, and on the same
6,835 validation origins. Reuse previous train-only speed/body-z-rate bins and
per-log/equal-log metrics, save nonfinite and large-error flags. Body z angular
rate is an origin-condition proxy, not actual flight-path curvature. Differences
between training and validation also include operating-condition shift.

This single-seed experiment distinguishes additional optimization benefits from
training saturation or emerging validation degradation. No automatic promotion.
Output: artifacts/september_extended_training_20260911; sibling .run.log and
.launch.json. No overwrite, retries or interactive monitoring.
