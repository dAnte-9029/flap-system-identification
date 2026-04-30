# Cycle-Block Split Design

## Context

The canonical parquet packaging step is complete for the currently accepted flight logs, but the repository does not yet contain a train/validation/test split that matches the agreed data policy:

- do not split by whole flight batch only
- do not randomly shuffle individual adjacent samples
- reduce leakage from near-neighbor cycles inside the same flight

The user already approved a cycle-block shuffled strategy for the first training-ready dataset.

## Candidate approaches

### Option 1: Whole-log split

Assign each accepted log entirely to train, validation, or test.

Pros:
- simplest bookkeeping
- no temporal leakage within a log

Cons:
- too sensitive to between-log hardware drift
- wastes useful variation because each split may miss some operating states
- explicitly rejected by the user

### Option 2: Fully random sample-level split

Shuffle individual valid rows globally, then assign rows to train, validation, and test.

Pros:
- maximally mixes operating conditions
- very easy to implement

Cons:
- leaks adjacent states across splits
- gives overly optimistic validation metrics
- not acceptable for time-correlated flight data

### Option 3: Global shuffled cycle-block split

Within each accepted log, keep cycles contiguous. Build fixed-size blocks of consecutive valid cycles, shuffle those blocks globally, assign blocks to train/validation/test, and purge nearby cycles around validation/test blocks from train.

Pros:
- respects local time structure better than sample-level shuffle
- still mixes data across logs and hardware states
- directly matches the user-approved policy

Cons:
- slightly more bookkeeping
- some train data is intentionally dropped by the purge rule

## Selected design

Use Option 3.

## Scope

### Included in the first training-ready split

- all accepted wing-phase cohorts:
  - `canonical_v0.2_seed_labels_2026_4_12`
  - `canonical_v0.2_seed_labels_2026_4_14_3algorithms_weakwing`
  - `canonical_v0.2_seed_labels_2026_4_14_3algorithms`
  - `canonical_v0.2_seed_labels_2026_4_15`

### Not included in the first main split

- `canonical_v0.2_seed_labels_2026_4_13_air_encoder_fallback`

Reason: it is a valid packaged cohort, but its phase is reconstructed from `encoder_count_fallback` instead of logged `wing_phase`. Keeping it separate avoids mixing two phase-source regimes in the very first baseline training run. It can later be added as a supplemental robustness cohort or a second split variant.

## Split policy

- valid row mask:
  - `label_valid == True`
  - `cycle_valid == True`
  - `flap_active == True`
  - `cycle_id >= 0`
- block unit: consecutive valid cycles from the same log
- default block size: 60 cycles
- split ratios: 70% train, 15% validation, 15% test
- random seed: fixed and recorded in manifest
- purge gap: 8 cycles around validation/test blocks, removed from train if from the same log

## Outputs

Create a new dataset root:

- `dataset/canonical_v0.2_training_ready_split_v1`

Artifacts:

- `dataset_manifest.json`
- `all_blocks.csv`
- `train_blocks.csv`
- `val_blocks.csv`
- `test_blocks.csv`
- `train_samples.parquet`
- `val_samples.parquet`
- `test_samples.parquet`

Each materialized sample row keeps provenance fields such as dataset id, log id, split name, block id, and source parquet path.

## Verification

- no block overlap across train/validation/test
- purge removes nearby train cycles around validation/test ranges within the same log
- sample counts and cycle counts are recorded per split
- only rows passing the valid-row mask appear in split parquets

