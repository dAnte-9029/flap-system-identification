# Baseline Training Design

## Context

The dataset packaging and cycle-block shuffled split are now available, but the repository still lacks a first trainable model baseline. The immediate goal is not to finalize the research model architecture. The goal is to make the project capable of running an end-to-end first training pass against the current dataset.

## Candidate approaches

### Option 1: Add a PyTorch training stack now

Pros:
- aligns with likely long-term neural-network work
- flexible for custom architectures
- can use the available RTX 4090 GPUs

Cons:
- requires a dedicated environment setup because CUDA wheels are large
- slightly more implementation work than a purely CPU baseline

### Option 2: Add a scikit-learn MLP baseline first

Pros:
- `scikit-learn` is already available
- minimal code and dependency surface
- enough to validate features, labels, split wiring, metrics, and artifact format

Cons:
- not the final training stack
- less flexible than PyTorch for future sequence or physics-informed models

### Option 3: Use a non-neural baseline first

Pros:
- very fast to implement
- good as a numeric sanity check

Cons:
- does not satisfy the user's immediate goal of starting neural-network fitting

## Selected design

Use Option 1 now, but keep the implementation conservative and simple.

## Model scope

Train a first multi-output regression baseline that predicts:

- `fx_b`
- `fy_b`
- `fz_b`
- `mx_b`
- `my_b`
- `mz_b`

from a conservative numeric feature set derived from:

- body translational velocity and acceleration
- body angular velocity and angular acceleration
- attitude quaternion
- corrected phase encoding
- wing stroke angle and flap frequency
- actuator commands
- airspeed, wind, and air density

Absolute position and ad hoc debug fields are excluded from the first baseline.

## Preprocessing

- use only numeric features
- derive `phase_corrected_sin` and `phase_corrected_cos`
- median-impute missing feature values
- standardize input features
- standardize output targets

Target scaling is mandatory because force and moment magnitudes differ by orders of magnitude. Without target scaling, the network would mostly optimize force channels and underfit moment channels.

## Training flow

1. load `train_samples.parquet`, `val_samples.parquet`, `test_samples.parquet`
2. build derived feature columns
3. fit imputer + input scaler + target scaler on train only
4. fit a plain PyTorch MLP on scaled train data
5. evaluate on train/val/test
6. save model bundle, feature spec, config, and metrics

## Outputs

Create:

- training module under `src/system_identification`
- CLI training script under `scripts`
- tests for feature preparation and smoke training
- output artifact directory containing:
  - serialized PyTorch model bundle
  - metrics JSON
  - training config JSON

## Verification

- targeted unit tests pass
- a real smoke training run on the current split completes
- metrics file is written
- serialized model bundle can be loaded for inference later
